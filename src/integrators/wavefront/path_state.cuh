#pragma once

#include "../../core/math/vector.cuh"
#include "../../core/math/spectral.cuh"
#include "../../core/memory/device_buffer.cuh"
#include "../../core/random/pcg.cuh"

namespace lumina {






constexpr int MAX_PATH_DEPTH = 16;


enum PathFlags : uint32_t {
    PATH_ACTIVE = 1 << 0,
    PATH_SPECULAR = 1 << 1,
    PATH_INSIDE_MEDIUM = 1 << 2,
    PATH_HIT_LIGHT = 1 << 3,
};


struct PathStateSoA {
    
    DeviceBuffer<int> pixel_x;
    DeviceBuffer<int> pixel_y;

    
    DeviceBuffer<float> ray_origin_x;
    DeviceBuffer<float> ray_origin_y;
    DeviceBuffer<float> ray_origin_z;
    DeviceBuffer<float> ray_dir_x;
    DeviceBuffer<float> ray_dir_y;
    DeviceBuffer<float> ray_dir_z;

    
    DeviceBuffer<float> throughput_x;
    DeviceBuffer<float> throughput_y;
    DeviceBuffer<float> throughput_z;

    
    DeviceBuffer<float> radiance_x;
    DeviceBuffer<float> radiance_y;
    DeviceBuffer<float> radiance_z;

    
    DeviceBuffer<int> depth;
    DeviceBuffer<uint32_t> flags;
    DeviceBuffer<int> material_id;

    
    DeviceBuffer<PCG32> rng;

    
    DeviceBuffer<float> wavelengths[NUM_WAVELENGTHS];
    DeviceBuffer<float> spectral_throughput[NUM_WAVELENGTHS];
    DeviceBuffer<float> spectral_radiance[NUM_WAVELENGTHS];

    void resize(size_t count) {
        pixel_x.resize(count);
        pixel_y.resize(count);

        ray_origin_x.resize(count);
        ray_origin_y.resize(count);
        ray_origin_z.resize(count);
        ray_dir_x.resize(count);
        ray_dir_y.resize(count);
        ray_dir_z.resize(count);

        throughput_x.resize(count);
        throughput_y.resize(count);
        throughput_z.resize(count);

        radiance_x.resize(count);
        radiance_y.resize(count);
        radiance_z.resize(count);

        depth.resize(count);
        flags.resize(count);
        material_id.resize(count);

        rng.resize(count);

        for (int i = 0; i < NUM_WAVELENGTHS; i++) {
            wavelengths[i].resize(count);
            spectral_throughput[i].resize(count);
            spectral_radiance[i].resize(count);
        }
    }

    size_t size() const { return pixel_x.size(); }
};


struct PathStateView {
    int* __restrict__ pixel_x;
    int* __restrict__ pixel_y;

    float* __restrict__ ray_origin_x;
    float* __restrict__ ray_origin_y;
    float* __restrict__ ray_origin_z;
    float* __restrict__ ray_dir_x;
    float* __restrict__ ray_dir_y;
    float* __restrict__ ray_dir_z;

    float* __restrict__ throughput_x;
    float* __restrict__ throughput_y;
    float* __restrict__ throughput_z;

    float* __restrict__ radiance_x;
    float* __restrict__ radiance_y;
    float* __restrict__ radiance_z;

    int* __restrict__ depth;
    uint32_t* __restrict__ flags;
    int* __restrict__ material_id;

    PCG32* __restrict__ rng;

    
    float* __restrict__ wavelengths[NUM_WAVELENGTHS];
    float* __restrict__ spectral_throughput[NUM_WAVELENGTHS];
    float* __restrict__ spectral_radiance[NUM_WAVELENGTHS];

    __device__ bool is_active(int idx) const {
        return (flags[idx] & PATH_ACTIVE) != 0;
    }

    __device__ void set_active(int idx, bool active) {
        if (active) {
            flags[idx] |= PATH_ACTIVE;
        } else {
            flags[idx] &= ~PATH_ACTIVE;
        }
    }

    __device__ Ray get_ray(int idx) const {
        Ray ray;
        ray.origin = make_float3(ray_origin_x[idx], ray_origin_y[idx], ray_origin_z[idx]);
        ray.direction = make_float3(ray_dir_x[idx], ray_dir_y[idx], ray_dir_z[idx]);
        ray.t_min = RAY_EPSILON;
        ray.t_max = INFINITY_F;
        return ray;
    }

    __device__ void set_ray(int idx, const Ray& ray) {
        ray_origin_x[idx] = ray.origin.x;
        ray_origin_y[idx] = ray.origin.y;
        ray_origin_z[idx] = ray.origin.z;
        ray_dir_x[idx] = ray.direction.x;
        ray_dir_y[idx] = ray.direction.y;
        ray_dir_z[idx] = ray.direction.z;
    }

    __device__ float3 get_throughput(int idx) const {
        return make_float3(throughput_x[idx], throughput_y[idx], throughput_z[idx]);
    }

    __device__ void set_throughput(int idx, const float3& t) {
        throughput_x[idx] = t.x;
        throughput_y[idx] = t.y;
        throughput_z[idx] = t.z;
    }

    __device__ void multiply_throughput(int idx, const float3& f) {
        throughput_x[idx] *= f.x;
        throughput_y[idx] *= f.y;
        throughput_z[idx] *= f.z;
    }

    __device__ float3 get_radiance(int idx) const {
        return make_float3(radiance_x[idx], radiance_y[idx], radiance_z[idx]);
    }

    __device__ void add_radiance(int idx, const float3& L) {
        radiance_x[idx] += L.x;
        radiance_y[idx] += L.y;
        radiance_z[idx] += L.z;
    }

    __device__ SpectralSample get_wavelengths(int idx) const {
        SpectralSample s;
        for (int i = 0; i < NUM_WAVELENGTHS; i++) {
            s.lambda[i] = wavelengths[i][idx];
        }
        return s;
    }

    __device__ SpectralRadiance get_spectral_throughput(int idx) const {
        SpectralRadiance t;
        for (int i = 0; i < NUM_WAVELENGTHS; i++) {
            t.L[i] = spectral_throughput[i][idx];
        }
        return t;
    }

    __device__ void set_spectral_throughput(int idx, const SpectralRadiance& t) {
        for (int i = 0; i < NUM_WAVELENGTHS; i++) {
            spectral_throughput[i][idx] = t.L[i];
        }
    }

    __device__ void add_spectral_radiance(int idx, const SpectralRadiance& L) {
        for (int i = 0; i < NUM_WAVELENGTHS; i++) {
            spectral_radiance[i][idx] += L.L[i];
        }
    }
};

inline PathStateView make_view(PathStateSoA& state) {
    PathStateView view;
    view.pixel_x = state.pixel_x.data();
    view.pixel_y = state.pixel_y.data();
    view.ray_origin_x = state.ray_origin_x.data();
    view.ray_origin_y = state.ray_origin_y.data();
    view.ray_origin_z = state.ray_origin_z.data();
    view.ray_dir_x = state.ray_dir_x.data();
    view.ray_dir_y = state.ray_dir_y.data();
    view.ray_dir_z = state.ray_dir_z.data();
    view.throughput_x = state.throughput_x.data();
    view.throughput_y = state.throughput_y.data();
    view.throughput_z = state.throughput_z.data();
    view.radiance_x = state.radiance_x.data();
    view.radiance_y = state.radiance_y.data();
    view.radiance_z = state.radiance_z.data();
    view.depth = state.depth.data();
    view.flags = state.flags.data();
    view.material_id = state.material_id.data();
    view.rng = state.rng.data();

    for (int i = 0; i < NUM_WAVELENGTHS; i++) {
        view.wavelengths[i] = state.wavelengths[i].data();
        view.spectral_throughput[i] = state.spectral_throughput[i].data();
        view.spectral_radiance[i] = state.spectral_radiance[i].data();
    }

    return view;
}





struct HitInfoSoA {
    DeviceBuffer<float> t;
    DeviceBuffer<int> prim_id;
    DeviceBuffer<int> material_id;
    DeviceBuffer<float> u;
    DeviceBuffer<float> v;

    
    DeviceBuffer<float> pos_x, pos_y, pos_z;
    DeviceBuffer<float> normal_x, normal_y, normal_z;
    DeviceBuffer<float> geom_normal_x, geom_normal_y, geom_normal_z;

    void resize(size_t count) {
        t.resize(count);
        prim_id.resize(count);
        material_id.resize(count);
        u.resize(count);
        v.resize(count);
        pos_x.resize(count);
        pos_y.resize(count);
        pos_z.resize(count);
        normal_x.resize(count);
        normal_y.resize(count);
        normal_z.resize(count);
        geom_normal_x.resize(count);
        geom_normal_y.resize(count);
        geom_normal_z.resize(count);
    }

    size_t size() const { return t.size(); }
};

struct HitInfoView {
    float* __restrict__ t;
    int* __restrict__ prim_id;
    int* __restrict__ material_id;
    float* __restrict__ u;
    float* __restrict__ v;
    float* __restrict__ pos_x;
    float* __restrict__ pos_y;
    float* __restrict__ pos_z;
    float* __restrict__ normal_x;
    float* __restrict__ normal_y;
    float* __restrict__ normal_z;
    float* __restrict__ geom_normal_x;
    float* __restrict__ geom_normal_y;
    float* __restrict__ geom_normal_z;

    __device__ bool has_hit(int idx) const {
        return t[idx] > 0.0f && prim_id[idx] >= 0;
    }

    __device__ float3 get_position(int idx) const {
        return make_float3(pos_x[idx], pos_y[idx], pos_z[idx]);
    }

    __device__ float3 get_normal(int idx) const {
        return make_float3(normal_x[idx], normal_y[idx], normal_z[idx]);
    }

    __device__ float3 get_geom_normal(int idx) const {
        return make_float3(geom_normal_x[idx], geom_normal_y[idx], geom_normal_z[idx]);
    }
};

inline HitInfoView make_view(HitInfoSoA& hits) {
    HitInfoView view;
    view.t = hits.t.data();
    view.prim_id = hits.prim_id.data();
    view.material_id = hits.material_id.data();
    view.u = hits.u.data();
    view.v = hits.v.data();
    view.pos_x = hits.pos_x.data();
    view.pos_y = hits.pos_y.data();
    view.pos_z = hits.pos_z.data();
    view.normal_x = hits.normal_x.data();
    view.normal_y = hits.normal_y.data();
    view.normal_z = hits.normal_z.data();
    view.geom_normal_x = hits.geom_normal_x.data();
    view.geom_normal_y = hits.geom_normal_y.data();
    view.geom_normal_z = hits.geom_normal_z.data();
    return view;
}

} 
