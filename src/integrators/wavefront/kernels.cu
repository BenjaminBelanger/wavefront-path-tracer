#include "path_state.cuh"
#include "ray_queue.cuh"
#include "../../core/math/spectral.cuh"
#include "../../geometry/bvh/bvh.cuh"
#include "../../geometry/primitives/triangle.cuh"
#include "../../materials/bsdf/lambert.cuh"
#include "../../materials/bsdf/ggx.cuh"
#include "../../app/camera.cuh"

namespace lumina {

// =============================================================================
// Generate Primary Rays Kernel
// =============================================================================

__global__ void generate_rays_kernel(
    PathStateView paths,
    const Camera camera,
    int width, int height,
    int frame_number,
    int samples_per_pixel,
    int current_sample
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;

    // Initialize RNG for this path
    paths.rng[pixel_idx].init_from_pixel(x, y, frame_number, current_sample);
    PCG32& rng = paths.rng[pixel_idx];

    // Store pixel coordinates
    paths.pixel_x[pixel_idx] = x;
    paths.pixel_y[pixel_idx] = y;

    // Jittered sample within pixel
    float u = (x + rng.next_float()) / float(width);
    float v = (y + rng.next_float()) / float(height);

    // Generate ray (with optional DOF)
    Ray ray;
    if (camera.aperture > 0.0f) {
        ray = camera.generate_ray_dof(u, v, rng.next_float(), rng.next_float());
    } else {
        ray = camera.generate_ray(u, v);
    }

    paths.set_ray(pixel_idx, ray);

    // Initialize path state
    paths.set_throughput(pixel_idx, make_float3(1.0f));
    paths.radiance_x[pixel_idx] = 0.0f;
    paths.radiance_y[pixel_idx] = 0.0f;
    paths.radiance_z[pixel_idx] = 0.0f;
    paths.depth[pixel_idx] = 0;
    paths.flags[pixel_idx] = PATH_ACTIVE;
    paths.material_id[pixel_idx] = -1;

    // Initialize spectral data with hero wavelength sampling
    SpectralSample wavelengths = sample_hero_wavelength(rng.next_float());
    for (int i = 0; i < NUM_WAVELENGTHS; i++) {
        paths.wavelengths[i][pixel_idx] = wavelengths.lambda[i];
        paths.spectral_throughput[i][pixel_idx] = 1.0f;
        paths.spectral_radiance[i][pixel_idx] = 0.0f;
    }
}

// =============================================================================
// Ray-Scene Intersection Kernel
// =============================================================================

__global__ void intersect_kernel(
    PathStateView paths,
    HitInfoView hits,
    const BVHNode* __restrict__ bvh_nodes,
    const Triangle* __restrict__ triangles,
    const int* __restrict__ active_paths,
    int active_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    int path_idx = active_paths[idx];

    if (!paths.is_active(path_idx)) {
        hits.t[path_idx] = -1.0f;
        hits.prim_id[path_idx] = -1;
        return;
    }

    Ray ray = paths.get_ray(path_idx);

    float t_hit, u_hit, v_hit;
    int prim_id, mat_id;

    bool hit = traverse_bvh(bvh_nodes, triangles, ray, t_hit, u_hit, v_hit, prim_id, mat_id);

    if (hit) {
        hits.t[path_idx] = t_hit;
        hits.prim_id[path_idx] = prim_id;
        hits.material_id[path_idx] = mat_id;
        hits.u[path_idx] = u_hit;
        hits.v[path_idx] = v_hit;

        // Compute hit position and interpolated normal
        const Triangle& tri = triangles[prim_id];
        float3 pos = ray.at(t_hit);
        float3 normal = tri.interpolate_normal(u_hit, v_hit);
        float3 geom_normal = tri.geometric_normal();

        hits.pos_x[path_idx] = pos.x;
        hits.pos_y[path_idx] = pos.y;
        hits.pos_z[path_idx] = pos.z;
        hits.normal_x[path_idx] = normal.x;
        hits.normal_y[path_idx] = normal.y;
        hits.normal_z[path_idx] = normal.z;
        hits.geom_normal_x[path_idx] = geom_normal.x;
        hits.geom_normal_y[path_idx] = geom_normal.y;
        hits.geom_normal_z[path_idx] = geom_normal.z;

        paths.material_id[path_idx] = mat_id;
    } else {
        hits.t[path_idx] = -1.0f;
        hits.prim_id[path_idx] = -1;
    }
}

// =============================================================================
// Miss Shader (Environment)
// =============================================================================

__device__ float3 environment_color(const float3& direction) {
    // Simple sky gradient
    float t = 0.5f * (direction.y + 1.0f);
    float3 sky_blue = make_float3(0.5f, 0.7f, 1.0f);
    float3 white = make_float3(1.0f);
    return lerp(white, sky_blue, t) * 0.5f;
}

__global__ void shade_miss_kernel(
    PathStateView paths,
    const HitInfoView hits,
    const int* __restrict__ active_paths,
    int active_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    int path_idx = active_paths[idx];

    if (!paths.is_active(path_idx) || hits.has_hit(path_idx)) {
        return;
    }

    // Get ray direction
    float3 dir = make_float3(
        paths.ray_dir_x[path_idx],
        paths.ray_dir_y[path_idx],
        paths.ray_dir_z[path_idx]
    );

    // Evaluate environment
    float3 env_color = environment_color(dir);
    float3 throughput = paths.get_throughput(path_idx);

    // Add contribution
    paths.add_radiance(path_idx, throughput * env_color);

    // Terminate path
    paths.set_active(path_idx, false);
}

// =============================================================================
// Surface Shading Kernel
// =============================================================================

__global__ void shade_surface_kernel(
    PathStateView paths,
    const HitInfoView hits,
    const Material* __restrict__ materials,
    const int* __restrict__ active_paths,
    unsigned int* __restrict__ next_count,
    int* __restrict__ next_paths,
    int active_count,
    int max_depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    int path_idx = active_paths[idx];

    if (!paths.is_active(path_idx) || !hits.has_hit(path_idx)) {
        return;
    }

    int depth = paths.depth[path_idx];

    // Check max depth
    if (depth >= max_depth) {
        paths.set_active(path_idx, false);
        return;
    }

    // Get hit information
    int mat_id = hits.material_id[path_idx];
    const Material& material = materials[mat_id];

    float3 hit_pos = hits.get_position(path_idx);
    float3 normal = hits.get_normal(path_idx);
    float3 geom_normal = hits.get_geom_normal(path_idx);

    float3 ray_dir = make_float3(
        paths.ray_dir_x[path_idx],
        paths.ray_dir_y[path_idx],
        paths.ray_dir_z[path_idx]
    );
    float3 wo = -ray_dir;

    // Check for emission
    if (material.is_emissive()) {
        float3 throughput = paths.get_throughput(path_idx);
        float3 emission = material.get_emission();
        // Always accumulate emissive hits. This integrator currently does not
        // perform explicit light sampling, so gating this to only primary/specular
        // paths makes diffuse transport go black.
        paths.add_radiance(path_idx, throughput * emission);

        paths.set_active(path_idx, false);
        return;
    }

    // Setup shading context
    ShadingContext ctx;
    ctx.position = hit_pos;
    // Orient normals so dot(n, wo) > 0 for BSDF evaluation.
    // Our faceforward helper returns n when dot(n, v) < 0, so use -wo.
    ctx.normal = faceforward(normal, -wo);
    ctx.geometric_normal = faceforward(geom_normal, -wo);
    ctx.wo = wo;
    ctx.build_frame();

    // Get RNG
    PCG32& rng = paths.rng[path_idx];

    // Russian roulette (after depth 3)
    float3 throughput = paths.get_throughput(path_idx);
    if (depth > 3) {
        float continue_prob = russian_roulette_prob(throughput);
        if (rng.next_float() > continue_prob) {
            paths.set_active(path_idx, false);
            return;
        }
        throughput = throughput / continue_prob;
        paths.set_throughput(path_idx, throughput);
    }

    // Sample BSDF
    BSDFSample sample = sample_bsdf(material, ctx, rng.next_float(), rng.next_float());

    if (!sample.is_valid()) {
        paths.set_active(path_idx, false);
        return;
    }

    // Update throughput: throughput *= f * |cos| / pdf
    // sample.f already includes |cos|
    float3 bsdf_weight = sample.f / sample.pdf;
    paths.multiply_throughput(path_idx, bsdf_weight);

    // Spawn new ray
    Ray new_ray;
    new_ray.origin = hit_pos + ctx.geometric_normal * RAY_EPSILON;
    new_ray.direction = sample.wi;
    new_ray.t_min = RAY_EPSILON;
    new_ray.t_max = INFINITY_F;

    paths.set_ray(path_idx, new_ray);

    // Update flags
    if (sample.is_specular) {
        paths.flags[path_idx] |= PATH_SPECULAR;
    } else {
        paths.flags[path_idx] &= ~PATH_SPECULAR;
    }

    paths.depth[path_idx] = depth + 1;

    // Add to next iteration queue
    unsigned int slot = atomicAdd(next_count, 1);
    next_paths[slot] = path_idx;
}

// =============================================================================
// Shadow Ray Kernel (for explicit light sampling)
// =============================================================================

__global__ void trace_shadow_kernel(
    PathStateView paths,
    const BVHNode* __restrict__ bvh_nodes,
    const Triangle* __restrict__ triangles,
    const float3* __restrict__ shadow_origins,
    const float3* __restrict__ shadow_directions,
    const float* __restrict__ shadow_max_t,
    const float3* __restrict__ shadow_contributions,
    const int* __restrict__ shadow_path_indices,
    int shadow_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shadow_count) return;

    int path_idx = shadow_path_indices[idx];

    Ray shadow_ray;
    shadow_ray.origin = shadow_origins[idx];
    shadow_ray.direction = shadow_directions[idx];
    shadow_ray.t_min = RAY_EPSILON;
    shadow_ray.t_max = shadow_max_t[idx] - RAY_EPSILON;

    bool occluded = traverse_bvh_shadow(bvh_nodes, triangles, shadow_ray);

    if (!occluded) {
        // Add light contribution
        float3 contribution = shadow_contributions[idx];
        paths.add_radiance(path_idx, contribution);
    }
}

// =============================================================================
// Accumulate Results to Framebuffer
// =============================================================================

__global__ void accumulate_kernel(
    const PathStateView paths,
    float4* __restrict__ accumulation_buffer,
    int* __restrict__ sample_count,
    int width, int height,
    int num_paths
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    int x = paths.pixel_x[idx];
    int y = paths.pixel_y[idx];
    int pixel_idx = y * width + x;

    float3 radiance = paths.get_radiance(idx);

    // Clamp fireflies
    float lum = luminance(radiance);
    if (lum > 100.0f) {
        radiance = radiance * (100.0f / lum);
    }

    // Accumulate
    float4 old_value = accumulation_buffer[pixel_idx];
    int old_count = sample_count[pixel_idx];

    float new_count = float(old_count + 1);
    float weight = 1.0f / new_count;

    float4 new_value = make_float4(
        old_value.x + (radiance.x - old_value.x) * weight,
        old_value.y + (radiance.y - old_value.y) * weight,
        old_value.z + (radiance.z - old_value.z) * weight,
        1.0f
    );

    accumulation_buffer[pixel_idx] = new_value;
    sample_count[pixel_idx] = old_count + 1;
}

// =============================================================================
// Spectral to RGB Conversion Kernel
// =============================================================================

__global__ void spectral_to_rgb_kernel(
    const PathStateView paths,
    float4* __restrict__ accumulation_buffer,
    int* __restrict__ sample_count,
    int width, int height,
    int num_paths
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    int x = paths.pixel_x[idx];
    int y = paths.pixel_y[idx];
    int pixel_idx = y * width + x;

    // Gather spectral radiance
    SpectralRadiance radiance;
    SpectralSample wavelengths;
    for (int i = 0; i < NUM_WAVELENGTHS; i++) {
        radiance.L[i] = paths.spectral_radiance[i][idx];
        wavelengths.lambda[i] = paths.wavelengths[i][idx];
    }

    // Convert to RGB
    float3 rgb = spectral_to_rgb(radiance, wavelengths);

    // Clamp negative values (can happen due to spectral mismatch)
    rgb = max(rgb, make_float3(0.0f));

    // Clamp fireflies
    float lum = luminance(rgb);
    if (lum > 100.0f) {
        rgb = rgb * (100.0f / lum);
    }

    // Accumulate
    float4 old_value = accumulation_buffer[pixel_idx];
    int old_count = sample_count[pixel_idx];

    float new_count = float(old_count + 1);
    float weight = 1.0f / new_count;

    float4 new_value = make_float4(
        old_value.x + (rgb.x - old_value.x) * weight,
        old_value.y + (rgb.y - old_value.y) * weight,
        old_value.z + (rgb.z - old_value.z) * weight,
        1.0f
    );

    accumulation_buffer[pixel_idx] = new_value;
    sample_count[pixel_idx] = old_count + 1;
}

// =============================================================================
// Tonemap and Convert to Display Kernel
// =============================================================================

__device__ float3 aces_tonemap(float3 x) {
    // ACES filmic tone mapping
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((x * (a * x + make_float3(b))) / (x * (c * x + make_float3(d)) + make_float3(e)), 0.0f, 1.0f);
}

__global__ void tonemap_kernel(
    const float4* __restrict__ accumulation_buffer,
    uchar4* __restrict__ display_buffer,
    int width, int height,
    float exposure
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;

    float4 hdr = accumulation_buffer[pixel_idx];
    float3 color = make_float3(hdr.x, hdr.y, hdr.z);

    // Exposure
    color = color * exposure;

    // Tonemap
    color = aces_tonemap(color);

    // Gamma correction (linear to sRGB)
    color = linear_to_srgb(color);

    // Convert to 8-bit
    display_buffer[pixel_idx] = make_uchar4(
        static_cast<unsigned char>(color.x * 255.0f + 0.5f),
        static_cast<unsigned char>(color.y * 255.0f + 0.5f),
        static_cast<unsigned char>(color.z * 255.0f + 0.5f),
        255
    );
}

// =============================================================================
// Host-Callable Wrapper Functions for Cross-TU Kernel Launches
// =============================================================================

void launch_generate_rays(
    PathStateView paths,
    const Camera& camera,
    int width, int height,
    int frame_number,
    int samples_per_pixel,
    int current_sample
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    generate_rays_kernel<<<grid, block>>>(paths, camera, width, height,
                                          frame_number, samples_per_pixel, current_sample);
}

void launch_intersect(
    PathStateView paths,
    HitInfoView hits,
    const BVHNode* bvh_nodes,
    const Triangle* triangles,
    const int* active_paths,
    int active_count
) {
    if (active_count == 0) return;
    int block = 256;
    int grid = (active_count + block - 1) / block;
    intersect_kernel<<<grid, block>>>(paths, hits, bvh_nodes, triangles, active_paths, active_count);
}

void launch_shade_miss(
    PathStateView paths,
    const HitInfoView& hits,
    const int* active_paths,
    int active_count
) {
    if (active_count == 0) return;
    int block = 256;
    int grid = (active_count + block - 1) / block;
    shade_miss_kernel<<<grid, block>>>(paths, hits, active_paths, active_count);
}

void launch_shade_surface(
    PathStateView paths,
    const HitInfoView& hits,
    const Material* materials,
    const int* active_paths,
    unsigned int* next_count,
    int* next_paths,
    int active_count,
    int max_depth
) {
    if (active_count == 0) return;
    int block = 256;
    int grid = (active_count + block - 1) / block;
    shade_surface_kernel<<<grid, block>>>(paths, hits, materials, active_paths,
                                          next_count, next_paths, active_count, max_depth);
}

void launch_accumulate(
    const PathStateView& paths,
    float4* accumulation_buffer,
    int* sample_count,
    int width, int height,
    int num_paths
) {
    int block = 256;
    int grid = (num_paths + block - 1) / block;
    accumulate_kernel<<<grid, block>>>(paths, accumulation_buffer, sample_count, width, height, num_paths);
}

void launch_tonemap(
    const float4* accumulation_buffer,
    uchar4* display_buffer,
    int width, int height,
    float exposure
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    tonemap_kernel<<<grid, block>>>(accumulation_buffer, display_buffer, width, height, exposure);
}

} // namespace lumina
