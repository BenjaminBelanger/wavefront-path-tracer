#include "path_state.cuh"
#include "ray_queue.cuh"
#include "../../core/math/spectral.cuh"
#include "../../geometry/bvh/bvh.cuh"
#include "../../geometry/primitives/triangle.cuh"
#include "../../materials/bsdf/lambert.cuh"
#include "../../materials/bsdf/ggx.cuh"
#include "../../app/camera.cuh"

namespace lumina
{

    __global__ void generate_rays_kernel(
        PathStateView paths,
        const Camera camera,
        int width, int height,
        int frame_number,
        int samples_per_pixel,
        int current_sample)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int pixel_idx = y * width + x;

        paths.rng[pixel_idx].init_from_pixel(x, y, frame_number, current_sample);
        PCG32 &rng = paths.rng[pixel_idx];

        paths.pixel_x[pixel_idx] = x;
        paths.pixel_y[pixel_idx] = y;

        float u = (x + rng.next_float()) / float(width);
        float v = (y + rng.next_float()) / float(height);

        Ray ray;
        if (camera.aperture > 0.0f)
        {
            ray = camera.generate_ray_dof(u, v, rng.next_float(), rng.next_float());
        }
        else
        {
            ray = camera.generate_ray(u, v);
        }

        paths.set_ray(pixel_idx, ray);

        paths.set_throughput(pixel_idx, make_float3(1.0f));
        paths.radiance_x[pixel_idx] = 0.0f;
        paths.radiance_y[pixel_idx] = 0.0f;
        paths.radiance_z[pixel_idx] = 0.0f;
        paths.depth[pixel_idx] = 0;
        paths.flags[pixel_idx] = PATH_ACTIVE;
        paths.material_id[pixel_idx] = -1;

        SpectralSample wavelengths = sample_hero_wavelength(rng.next_float());
        for (int i = 0; i < NUM_WAVELENGTHS; i++)
        {
            paths.wavelengths[i][pixel_idx] = wavelengths.lambda[i];
            paths.spectral_throughput[i][pixel_idx] = 1.0f;
            paths.spectral_radiance[i][pixel_idx] = 0.0f;
        }
    }

    __global__ void intersect_kernel(
        PathStateView paths,
        HitInfoView hits,
        const BVHNode *__restrict__ bvh_nodes,
        const TrianglePrecomputed *__restrict__ precomputed,
        const Triangle *__restrict__ triangles,
        const int *__restrict__ active_paths,
        const unsigned int *__restrict__ active_count_ptr)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= *active_count_ptr)
            return;

        int path_idx = active_paths[idx];

        if (!paths.is_active(path_idx))
        {
            hits.t[path_idx] = -1.0f;
            hits.prim_id[path_idx] = -1;
            return;
        }

        Ray ray = paths.get_ray(path_idx);

        float t_hit, u_hit, v_hit;
        int prim_id, mat_id;

        bool hit = traverse_bvh(bvh_nodes, precomputed, ray, t_hit, u_hit, v_hit, prim_id, mat_id);

        if (hit)
        {
            hits.t[path_idx] = t_hit;
            hits.prim_id[path_idx] = prim_id;
            hits.material_id[path_idx] = mat_id;
            hits.u[path_idx] = u_hit;
            hits.v[path_idx] = v_hit;

            const Triangle &tri = triangles[prim_id];
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

            float2 tex_uv = tri.interpolate_uv(u_hit, v_hit);
            hits.tex_u[path_idx] = tex_uv.x;
            hits.tex_v[path_idx] = tex_uv.y;

            float3 edge1 = tri.v1 - tri.v0;
            float3 edge2 = tri.v2 - tri.v0;
            float2 duv1 = tri.uv1 - tri.uv0;
            float2 duv2 = tri.uv2 - tri.uv0;
            float det = duv1.x * duv2.y - duv2.x * duv1.y;
            float3 tangent;
            if (fabsf(det) > 1e-8f)
            {
                float inv_det = 1.0f / det;
                tangent = normalize((edge1 * duv2.y - edge2 * duv1.y) * inv_det);
            }
            else
            {
                Frame fallback(normal);
                tangent = fallback.tangent;
            }
            hits.tangent_x[path_idx] = tangent.x;
            hits.tangent_y[path_idx] = tangent.y;
            hits.tangent_z[path_idx] = tangent.z;

            paths.material_id[path_idx] = mat_id;
        }
        else
        {
            hits.t[path_idx] = -1.0f;
            hits.prim_id[path_idx] = -1;
        }
    }

    __device__ float3 environment_color(const float3 &direction, cudaTextureObject_t env_map, float env_intensity)
    {
        if (env_map)
        {
            float u = atan2f(direction.z, direction.x) * INV_TWO_PI + 0.5f;
            float v = asinf(fmaxf(-1.0f, fminf(1.0f, direction.y))) * INV_PI + 0.5f;
            float4 sample = tex2D<float4>(env_map, u, v);
            return make_float3(sample.x, sample.y, sample.z) * env_intensity;
        }

        float t = 0.5f * (direction.y + 1.0f);
        float3 sky_blue = make_float3(0.5f, 0.7f, 1.0f);
        float3 white = make_float3(1.0f);
        return lerp(white, sky_blue, t) * 0.5f;
    }

    __global__ void shade_miss_kernel(
        PathStateView paths,
        const HitInfoView hits,
        const int *__restrict__ active_paths,
        const unsigned int *__restrict__ active_count_ptr,
        cudaTextureObject_t env_map,
        float env_intensity)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= *active_count_ptr)
            return;

        int path_idx = active_paths[idx];

        if (!paths.is_active(path_idx) || hits.has_hit(path_idx))
        {
            return;
        }

        float3 dir = make_float3(
            paths.ray_dir_x[path_idx],
            paths.ray_dir_y[path_idx],
            paths.ray_dir_z[path_idx]);

        float3 env_color = environment_color(dir, env_map, env_intensity);
        float3 throughput = paths.get_throughput(path_idx);

        paths.add_radiance(path_idx, throughput * env_color);

        paths.set_active(path_idx, false);
    }

    __global__ void shade_surface_kernel(
        PathStateView paths,
        const HitInfoView hits,
        const Material *__restrict__ materials,
        const cudaTextureObject_t *__restrict__ textures,
        const int *__restrict__ active_paths,
        unsigned int *__restrict__ next_count,
        int *__restrict__ next_paths,
        const unsigned int *__restrict__ active_count_ptr,
        int max_depth)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= *active_count_ptr)
            return;

        int path_idx = active_paths[idx];

        if (!paths.is_active(path_idx) || !hits.has_hit(path_idx))
        {
            return;
        }

        int depth = paths.depth[path_idx];

        if (depth >= max_depth)
        {
            paths.set_active(path_idx, false);
            return;
        }

        int mat_id = hits.material_id[path_idx];
        Material material = materials[mat_id];

        if (material.albedo_tex >= 0 && textures != nullptr)
        {
            float2 uv = hits.get_tex_uv(path_idx);
            float4 tex_color = tex2D<float4>(textures[material.albedo_tex], uv.x, uv.y);
            material.albedo = make_float3(tex_color.x, tex_color.y, tex_color.z);
        }

        if (material.roughness_tex >= 0 && textures != nullptr)
        {
            float2 uv = hits.get_tex_uv(path_idx);
            float4 tex_rough = tex2D<float4>(textures[material.roughness_tex], uv.x, uv.y);
            material.roughness = tex_rough.x;
        }

        float3 hit_pos = hits.get_position(path_idx);
        float3 normal = hits.get_normal(path_idx);
        float3 geom_normal = hits.get_geom_normal(path_idx);

        if (material.normal_tex >= 0 && textures != nullptr)
        {
            float2 uv = hits.get_tex_uv(path_idx);
            float4 tex_n = tex2D<float4>(textures[material.normal_tex], uv.x, uv.y);
            float3 ts_normal = make_float3(tex_n.x * 2.0f - 1.0f, tex_n.y * 2.0f - 1.0f, tex_n.z * 2.0f - 1.0f);

            float3 T = normalize(hits.get_tangent(path_idx));
            float3 N = normal;
            T = normalize(T - N * dot(T, N));
            float3 B = cross(N, T);

            normal = normalize(T * ts_normal.x + B * ts_normal.y + N * ts_normal.z);
        }

        float3 ray_dir = make_float3(
            paths.ray_dir_x[path_idx],
            paths.ray_dir_y[path_idx],
            paths.ray_dir_z[path_idx]);
        float3 wo = -ray_dir;

        if (material.is_emissive())
        {
            float3 throughput = paths.get_throughput(path_idx);
            float3 emission = material.get_emission();

            paths.add_radiance(path_idx, throughput * emission);

            paths.set_active(path_idx, false);
            return;
        }

        ShadingContext ctx;
        ctx.position = hit_pos;

        ctx.normal = faceforward(normal, -wo);
        ctx.geometric_normal = faceforward(geom_normal, -wo);
        ctx.wo = wo;
        ctx.build_frame();

        PCG32 &rng = paths.rng[path_idx];

        float3 throughput = paths.get_throughput(path_idx);
        if (depth > 3)
        {
            float continue_prob = russian_roulette_prob(throughput);
            if (rng.next_float() > continue_prob)
            {
                paths.set_active(path_idx, false);
                return;
            }
            throughput = throughput / continue_prob;
            paths.set_throughput(path_idx, throughput);
        }

        BSDFSample sample = sample_bsdf(material, ctx, rng.next_float(), rng.next_float(), rng.next_float());

        if (!sample.is_valid())
        {
            paths.set_active(path_idx, false);
            return;
        }

        float3 bsdf_weight = sample.f / sample.pdf;
        bsdf_weight = clamp(bsdf_weight, 0.0f, 100.0f);
        paths.multiply_throughput(path_idx, bsdf_weight);

        Ray new_ray;

        float origin_sign = (dot(sample.wi, ctx.geometric_normal) >= 0.0f) ? 1.0f : -1.0f;
        new_ray.origin = hit_pos + ctx.geometric_normal * (RAY_EPSILON * origin_sign);
        new_ray.direction = sample.wi;
        new_ray.t_min = RAY_EPSILON;
        new_ray.t_max = INFINITY_F;

        paths.set_ray(path_idx, new_ray);

        if (sample.is_specular)
        {
            paths.flags[path_idx] |= PATH_SPECULAR;
        }
        else
        {
            paths.flags[path_idx] &= ~PATH_SPECULAR;
        }

        paths.depth[path_idx] = depth + 1;

        unsigned int slot = atomicAdd(next_count, 1);
        next_paths[slot] = path_idx;
    }

    __global__ void trace_shadow_kernel(
        PathStateView paths,
        const BVHNode *__restrict__ bvh_nodes,
        const TrianglePrecomputed *__restrict__ precomputed,
        const float3 *__restrict__ shadow_origins,
        const float3 *__restrict__ shadow_directions,
        const float *__restrict__ shadow_max_t,
        const float3 *__restrict__ shadow_contributions,
        const int *__restrict__ shadow_path_indices,
        int shadow_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= shadow_count)
            return;

        int path_idx = shadow_path_indices[idx];

        Ray shadow_ray;
        shadow_ray.origin = shadow_origins[idx];
        shadow_ray.direction = shadow_directions[idx];
        shadow_ray.t_min = RAY_EPSILON;
        shadow_ray.t_max = shadow_max_t[idx] - RAY_EPSILON;

        bool occluded = traverse_bvh_shadow(bvh_nodes, precomputed, shadow_ray);

        if (!occluded)
        {

            float3 contribution = shadow_contributions[idx];
            paths.add_radiance(path_idx, contribution);
        }
    }

    __global__ void accumulate_kernel(
        const PathStateView paths,
        float4 *__restrict__ accumulation_buffer,
        int *__restrict__ sample_count,
        int width, int height,
        int num_paths)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_paths)
            return;

        int x = paths.pixel_x[idx];
        int y = paths.pixel_y[idx];
        int pixel_idx = y * width + x;

        float3 radiance = paths.get_radiance(idx);

        float lum = luminance(radiance);
        if (lum > 100.0f)
        {
            radiance = radiance * (100.0f / lum);
        }

        float4 old_value = accumulation_buffer[pixel_idx];
        int old_count = sample_count[pixel_idx];

        float new_count = float(old_count + 1);
        float weight = 1.0f / new_count;

        float4 new_value = make_float4(
            old_value.x + (radiance.x - old_value.x) * weight,
            old_value.y + (radiance.y - old_value.y) * weight,
            old_value.z + (radiance.z - old_value.z) * weight,
            1.0f);

        accumulation_buffer[pixel_idx] = new_value;
        sample_count[pixel_idx] = old_count + 1;
    }

    __global__ void spectral_to_rgb_kernel(
        const PathStateView paths,
        float4 *__restrict__ accumulation_buffer,
        int *__restrict__ sample_count,
        int width, int height,
        int num_paths)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_paths)
            return;

        int x = paths.pixel_x[idx];
        int y = paths.pixel_y[idx];
        int pixel_idx = y * width + x;

        SpectralRadiance radiance;
        SpectralSample wavelengths;
        for (int i = 0; i < NUM_WAVELENGTHS; i++)
        {
            radiance.L[i] = paths.spectral_radiance[i][idx];
            wavelengths.lambda[i] = paths.wavelengths[i][idx];
        }

        float3 rgb = spectral_to_rgb(radiance, wavelengths);

        rgb = max(rgb, make_float3(0.0f));

        float lum = luminance(rgb);
        if (lum > 100.0f)
        {
            rgb = rgb * (100.0f / lum);
        }

        float4 old_value = accumulation_buffer[pixel_idx];
        int old_count = sample_count[pixel_idx];

        float new_count = float(old_count + 1);
        float weight = 1.0f / new_count;

        float4 new_value = make_float4(
            old_value.x + (rgb.x - old_value.x) * weight,
            old_value.y + (rgb.y - old_value.y) * weight,
            old_value.z + (rgb.z - old_value.z) * weight,
            1.0f);

        accumulation_buffer[pixel_idx] = new_value;
        sample_count[pixel_idx] = old_count + 1;
    }

    __device__ float3 aces_tonemap(float3 x)
    {

        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        return clamp((x * (a * x + make_float3(b))) / (x * (c * x + make_float3(d)) + make_float3(e)), 0.0f, 1.0f);
    }

    __global__ void tonemap_kernel(
        const float4 *__restrict__ accumulation_buffer,
        uchar4 *__restrict__ display_buffer,
        int width, int height,
        float exposure)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int pixel_idx = y * width + x;

        float4 hdr = accumulation_buffer[pixel_idx];
        float3 color = make_float3(hdr.x, hdr.y, hdr.z);

        color = color * exposure;

        color = aces_tonemap(color);

        color = linear_to_srgb(color);

        display_buffer[pixel_idx] = make_uchar4(
            static_cast<unsigned char>(color.x * 255.0f + 0.5f),
            static_cast<unsigned char>(color.y * 255.0f + 0.5f),
            static_cast<unsigned char>(color.z * 255.0f + 0.5f),
            255);
    }

    void launch_generate_rays(
        PathStateView paths,
        const Camera &camera,
        int width, int height,
        int frame_number,
        int samples_per_pixel,
        int current_sample)
    {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        generate_rays_kernel<<<grid, block>>>(paths, camera, width, height,
                                              frame_number, samples_per_pixel, current_sample);
    }

    void launch_intersect(
        PathStateView paths,
        HitInfoView hits,
        const BVHNode *bvh_nodes,
        const TrianglePrecomputed *precomputed,
        const Triangle *triangles,
        const int *active_paths,
        const unsigned int *active_count_ptr,
        int max_threads)
    {
        int block = 256;
        int grid = (max_threads + block - 1) / block;
        intersect_kernel<<<grid, block>>>(paths, hits, bvh_nodes, precomputed, triangles, active_paths, active_count_ptr);
    }

    void launch_shade_miss(
        PathStateView paths,
        const HitInfoView &hits,
        const int *active_paths,
        const unsigned int *active_count_ptr,
        int max_threads,
        cudaTextureObject_t env_map,
        float env_intensity)
    {
        int block = 256;
        int grid = (max_threads + block - 1) / block;
        shade_miss_kernel<<<grid, block>>>(paths, hits, active_paths, active_count_ptr, env_map, env_intensity);
    }

    void launch_shade_surface(
        PathStateView paths,
        const HitInfoView &hits,
        const Material *materials,
        const cudaTextureObject_t *textures,
        const int *active_paths,
        unsigned int *next_count,
        int *next_paths,
        const unsigned int *active_count_ptr,
        int max_threads,
        int max_depth)
    {
        int block = 256;
        int grid = (max_threads + block - 1) / block;
        shade_surface_kernel<<<grid, block>>>(paths, hits, materials, textures, active_paths,
                                              next_count, next_paths, active_count_ptr, max_depth);
    }

    void launch_accumulate(
        const PathStateView &paths,
        float4 *accumulation_buffer,
        int *sample_count,
        int width, int height,
        int num_paths)
    {
        int block = 256;
        int grid = (num_paths + block - 1) / block;
        accumulate_kernel<<<grid, block>>>(paths, accumulation_buffer, sample_count, width, height, num_paths);
    }

    void launch_tonemap(
        const float4 *accumulation_buffer,
        uchar4 *display_buffer,
        int width, int height,
        float exposure)
    {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        tonemap_kernel<<<grid, block>>>(accumulation_buffer, display_buffer, width, height, exposure);
    }

}
