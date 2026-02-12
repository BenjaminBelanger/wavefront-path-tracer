#include "reservoir.cuh"
#include "../../core/memory/device_buffer.cuh"
#include "../../geometry/bvh/bvh.cuh"
#include "../../app/scene.cuh"

namespace lumina {

// =============================================================================
// ReSTIR Direct Illumination Implementation
// Based on "Spatiotemporal reservoir resampling for real-time ray tracing
// with dynamic direct lighting" (Bitterli et al., SIGGRAPH 2020)
// =============================================================================

// =============================================================================
// Initial Candidate Generation
// =============================================================================

__global__ void generate_initial_samples_kernel(
    ReservoirView reservoirs,
    const float3* __restrict__ positions,
    const float3* __restrict__ normals,
    const float3* __restrict__ albedos,
    const Light* __restrict__ lights,
    const AliasTableView alias_table,
    PCG32* __restrict__ rngs,
    int num_pixels,
    int num_lights,
    int num_candidates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float3 pos = positions[idx];
    float3 normal = normals[idx];
    float3 albedo = albedos[idx];

    PCG32& rng = rngs[idx];
    Reservoir reservoir;
    reservoir.reset();

    // Generate M candidate samples
    for (int i = 0; i < num_candidates; i++) {
        // Sample light from alias table
        float pdf_light;
        int light_idx = alias_table.sample(rng.next_float(), rng.next_float(), pdf_light);

        if (light_idx < 0 || light_idx >= num_lights) continue;

        const Light& light = lights[light_idx];

        // Sample point on light
        LightSample sample;
        sample.light_idx = light_idx;

        switch (light.type) {
            case LightType::Point: {
                sample.point_on_light = light.position;
                sample.light_normal = normalize(pos - light.position);
                sample.emission = light.emission;
                sample.pdf = pdf_light;
                break;
            }
            case LightType::Sphere: {
                // Sample visible portion of sphere
                float3 to_center = light.position - pos;
                float dist = length(to_center);

                if (dist < light.radius) {
                    // Inside sphere - sample uniformly
                    float3 dir = sample_sphere_uniform(rng.next_float(), rng.next_float());
                    sample.point_on_light = light.position + light.radius * dir;
                    sample.light_normal = dir;
                } else {
                    // Outside - sample visible hemisphere
                    float sin_theta_max = light.radius / dist;
                    float cos_theta_max = sqrtf(1.0f - sin_theta_max * sin_theta_max);

                    float3 dir = sample_cone_uniform(rng.next_float(), rng.next_float(), cos_theta_max);

                    // Transform to world space
                    Frame frame(normalize(to_center));
                    dir = frame.to_world(dir);

                    // Intersect with sphere
                    float3 oc = pos - light.position;
                    float b = dot(oc, dir);
                    float c = dot(oc, oc) - light.radius * light.radius;
                    float discriminant = b * b - c;
                    float t = -b - sqrtf(fmaxf(0.0f, discriminant));

                    sample.point_on_light = pos + t * dir;
                    sample.light_normal = normalize(sample.point_on_light - light.position);
                }

                sample.emission = light.emission;
                float area = 4.0f * PI * light.radius * light.radius;
                sample.pdf = pdf_light / area;
                break;
            }
            case LightType::Directional: {
                // Directional lights have no position - treat as very far away
                sample.point_on_light = pos - light.direction * 1e6f;
                sample.light_normal = light.direction;
                sample.emission = light.emission;
                sample.pdf = pdf_light;
                break;
            }
            default:
                continue;
        }

        // Compute target PDF
        float p_hat = compute_target_pdf(sample, pos, normal, albedo);

        // Weight = p_hat / pdf
        float w = (sample.pdf > 0.0f) ? p_hat / sample.pdf : 0.0f;

        reservoir.update(sample, w, rng);
    }

    // Finalize reservoir
    float final_p_hat = compute_target_pdf(reservoir.y, pos, normal, albedo);
    reservoir.finalize(final_p_hat);

    reservoirs.set(idx, reservoir);
}

// =============================================================================
// Temporal Resampling
// =============================================================================

__global__ void temporal_resampling_kernel(
    ReservoirView current_reservoirs,
    const ReservoirView prev_reservoirs,
    const float3* __restrict__ positions,
    const float3* __restrict__ normals,
    const float3* __restrict__ albedos,
    const float2* __restrict__ motion_vectors,
    const float* __restrict__ depths,
    const float* __restrict__ prev_depths,
    PCG32* __restrict__ rngs,
    int width, int height,
    float depth_threshold,
    float normal_threshold,
    int max_temporal_M
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float3 pos = positions[idx];
    float3 normal = normals[idx];
    float3 albedo = albedos[idx];
    float depth = depths[idx];

    PCG32& rng = rngs[idx];
    Reservoir reservoir = current_reservoirs.get(idx);

    // Get motion vector and find previous pixel
    float2 mv = motion_vectors[idx];
    int prev_x = int(x - mv.x * width + 0.5f);
    int prev_y = int(y - mv.y * height + 0.5f);

    // Check if previous pixel is valid
    if (prev_x >= 0 && prev_x < width && prev_y >= 0 && prev_y < height) {
        int prev_idx = prev_y * width + prev_x;
        float prev_depth = prev_depths[prev_idx];

        // Depth test
        bool valid = fabsf(depth - prev_depth) / depth < depth_threshold;

        if (valid) {
            Reservoir prev_reservoir = prev_reservoirs.get(prev_idx);

            // Clamp temporal history to prevent unbounded growth
            if (prev_reservoir.M > max_temporal_M) {
                float scale = float(max_temporal_M) / prev_reservoir.M;
                prev_reservoir.w_sum *= scale;
                prev_reservoir.M = max_temporal_M;
            }

            // Compute target PDF for previous sample at current shading point
            float p_hat_prev = compute_target_pdf(prev_reservoir.y, pos, normal, albedo);

            // Merge previous reservoir
            reservoir.merge(prev_reservoir, p_hat_prev, rng);
        }
    }

    // Finalize
    float final_p_hat = compute_target_pdf(reservoir.y, pos, normal, albedo);
    reservoir.finalize(final_p_hat);

    current_reservoirs.set(idx, reservoir);
}

// =============================================================================
// Spatial Resampling
// =============================================================================

__global__ void spatial_resampling_kernel(
    ReservoirView output_reservoirs,
    const ReservoirView input_reservoirs,
    const float3* __restrict__ positions,
    const float3* __restrict__ normals,
    const float3* __restrict__ albedos,
    const float* __restrict__ depths,
    PCG32* __restrict__ rngs,
    int width, int height,
    float spatial_radius,
    int num_spatial_samples,
    float depth_threshold,
    float normal_threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float3 pos = positions[idx];
    float3 normal = normals[idx];
    float3 albedo = albedos[idx];
    float depth = depths[idx];

    PCG32& rng = rngs[idx];
    Reservoir reservoir = input_reservoirs.get(idx);

    // Sample neighboring pixels
    for (int i = 0; i < num_spatial_samples; i++) {
        // Random offset within radius
        float angle = rng.next_float() * TWO_PI;
        float r = sqrtf(rng.next_float()) * spatial_radius;
        int nx = x + int(r * cosf(angle));
        int ny = y + int(r * sinf(angle));

        // Bounds check
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        if (nx == x && ny == y) continue;

        int neighbor_idx = ny * width + nx;
        float neighbor_depth = depths[neighbor_idx];
        float3 neighbor_normal = normals[neighbor_idx];

        // Geometry similarity test
        bool depth_similar = fabsf(depth - neighbor_depth) / depth < depth_threshold;
        bool normal_similar = dot(normal, neighbor_normal) > normal_threshold;

        if (!depth_similar || !normal_similar) continue;

        Reservoir neighbor = input_reservoirs.get(neighbor_idx);

        // Compute target PDF at current point for neighbor's sample
        float p_hat = compute_target_pdf(neighbor.y, pos, normal, albedo);

        // Merge with appropriate weight
        float merge_weight = p_hat * neighbor.W * neighbor.M;
        reservoir.update(neighbor.y, merge_weight, rng);
    }

    // Finalize with bias correction (simplified - full version uses MIS)
    float final_p_hat = compute_target_pdf(reservoir.y, pos, normal, albedo);
    reservoir.finalize(final_p_hat);

    output_reservoirs.set(idx, reservoir);
}

// =============================================================================
// Visibility Reuse (Shadow Ray)
// =============================================================================

__global__ void shade_with_reservoir_kernel(
    float3* __restrict__ output,
    const ReservoirView reservoirs,
    const float3* __restrict__ positions,
    const float3* __restrict__ normals,
    const float3* __restrict__ albedos,
    const BVHNode* __restrict__ bvh_nodes,
    const Triangle* __restrict__ triangles,
    int num_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float3 pos = positions[idx];
    float3 normal = normals[idx];
    float3 albedo = albedos[idx];

    Reservoir reservoir = reservoirs.get(idx);

    if (!reservoir.is_valid() || reservoir.W <= 0.0f) {
        output[idx] = make_float3(0.0f);
        return;
    }

    const LightSample& sample = reservoir.y;

    // Direction to light
    float3 to_light = sample.point_on_light - pos;
    float dist = length(to_light);
    float3 wi = to_light / dist;

    // Check visibility with shadow ray
    Ray shadow_ray;
    shadow_ray.origin = pos + normal * RAY_EPSILON;
    shadow_ray.direction = wi;
    shadow_ray.t_min = RAY_EPSILON;
    shadow_ray.t_max = dist - RAY_EPSILON;

    bool occluded = traverse_bvh_shadow(bvh_nodes, triangles, shadow_ray);

    if (occluded) {
        output[idx] = make_float3(0.0f);
        return;
    }

    // Compute final contribution: f * L * G * W
    float cos_theta_i = dot(normal, wi);
    float cos_theta_o = dot(sample.light_normal, -wi);

    if (cos_theta_i <= 0.0f || cos_theta_o <= 0.0f) {
        output[idx] = make_float3(0.0f);
        return;
    }

    float3 brdf = albedo * INV_PI;
    float G = cos_theta_i * cos_theta_o / (dist * dist);
    float3 L = sample.emission;

    output[idx] = L * brdf * G * reservoir.W;
}

// =============================================================================
// ReSTIR DI Manager Class
// =============================================================================

class ReSTIRDI {
public:
    struct Settings {
        int num_initial_candidates = 32;
        int num_spatial_samples = 5;
        float spatial_radius = 30.0f;
        float depth_threshold = 0.1f;
        float normal_threshold = 0.9f;
        int max_temporal_M = 20;
        bool enable_temporal = true;
        bool enable_spatial = true;
    };

    void initialize(int width, int height) {
        width_ = width;
        height_ = height;
        int num_pixels = width * height;

        current_reservoirs_.resize(num_pixels);
        prev_reservoirs_.resize(num_pixels);
        output_.resize(num_pixels);

        current_reservoirs_.clear();
        prev_reservoirs_.clear();
    }

    void render(
        const float3* positions,
        const float3* normals,
        const float3* albedos,
        const float* depths,
        const float2* motion_vectors,
        const Light* lights,
        int num_lights,
        const AliasTable& light_alias,
        const BVHNode* bvh_nodes,
        const Triangle* triangles,
        PCG32* rngs
    ) {
        int num_pixels = width_ * height_;
        int block_size = 256;
        int grid_size = (num_pixels + block_size - 1) / block_size;

        dim3 block_2d(16, 16);
        dim3 grid_2d((width_ + 15) / 16, (height_ + 15) / 16);

        // Step 1: Generate initial candidates
        generate_initial_samples_kernel<<<grid_size, block_size>>>(
            make_view(current_reservoirs_),
            positions, normals, albedos,
            lights, make_view(light_alias),
            rngs,
            num_pixels, num_lights,
            settings_.num_initial_candidates
        );

        // Step 2: Temporal resampling
        if (settings_.enable_temporal && frame_count_ > 0) {
            temporal_resampling_kernel<<<grid_2d, block_2d>>>(
                make_view(current_reservoirs_),
                make_view(prev_reservoirs_),
                positions, normals, albedos,
                motion_vectors,
                depths, prev_depths_.data(),
                rngs,
                width_, height_,
                settings_.depth_threshold,
                settings_.normal_threshold,
                settings_.max_temporal_M
            );
        }

        // Step 3: Spatial resampling
        if (settings_.enable_spatial) {
            spatial_resampling_kernel<<<grid_2d, block_2d>>>(
                make_view(prev_reservoirs_),  // Use as temp output
                make_view(current_reservoirs_),
                positions, normals, albedos,
                depths, rngs,
                width_, height_,
                settings_.spatial_radius,
                settings_.num_spatial_samples,
                settings_.depth_threshold,
                settings_.normal_threshold
            );

            // Swap so current has the final result
            std::swap(current_reservoirs_, prev_reservoirs_);
        }

        // Step 4: Shade with visibility
        shade_with_reservoir_kernel<<<grid_size, block_size>>>(
            output_.data(),
            make_view(current_reservoirs_),
            positions, normals, albedos,
            bvh_nodes, triangles,
            num_pixels
        );

        // Save for next frame
        std::swap(current_reservoirs_, prev_reservoirs_);
        prev_depths_.resize(num_pixels);
        cudaMemcpy(prev_depths_.data(), depths, num_pixels * sizeof(float), cudaMemcpyDeviceToDevice);

        frame_count_++;
    }

    float3* output() { return output_.data(); }
    Settings& settings() { return settings_; }

private:
    int width_ = 0;
    int height_ = 0;
    int frame_count_ = 0;

    Settings settings_;

    ReservoirSoA current_reservoirs_;
    ReservoirSoA prev_reservoirs_;
    DeviceBuffer<float> prev_depths_;
    DeviceBuffer<float3> output_;
};

} // namespace lumina
