#pragma once

#include "device_buffer.cuh"
#include "../math/vector.cuh"

namespace wpt {






struct Float3SoA {
    DeviceBuffer<float> x;
    DeviceBuffer<float> y;
    DeviceBuffer<float> z;

    void resize(size_t count) {
        x.resize(count);
        y.resize(count);
        z.resize(count);
    }

    void resize_and_clear(size_t count) {
        x.resize_and_clear(count);
        y.resize_and_clear(count);
        z.resize_and_clear(count);
    }

    size_t size() const { return x.size(); }
};


struct Float3SoAView {
    float* __restrict__ x;
    float* __restrict__ y;
    float* __restrict__ z;

    __device__ float3 get(int idx) const {
        return make_float3(x[idx], y[idx], z[idx]);
    }

    __device__ void set(int idx, const float3& v) {
        x[idx] = v.x;
        y[idx] = v.y;
        z[idx] = v.z;
    }

    __device__ void add(int idx, const float3& v) {
        x[idx] += v.x;
        y[idx] += v.y;
        z[idx] += v.z;
    }

    __device__ void atomic_add(int idx, const float3& v) {
        atomicAdd(&x[idx], v.x);
        atomicAdd(&y[idx], v.y);
        atomicAdd(&z[idx], v.z);
    }
};

inline Float3SoAView make_view(Float3SoA& soa) {
    return Float3SoAView{soa.x.data(), soa.y.data(), soa.z.data()};
}


struct Float4SoA {
    DeviceBuffer<float> x;
    DeviceBuffer<float> y;
    DeviceBuffer<float> z;
    DeviceBuffer<float> w;

    void resize(size_t count) {
        x.resize(count);
        y.resize(count);
        z.resize(count);
        w.resize(count);
    }

    void resize_and_clear(size_t count) {
        x.resize_and_clear(count);
        y.resize_and_clear(count);
        z.resize_and_clear(count);
        w.resize_and_clear(count);
    }

    size_t size() const { return x.size(); }
};

struct Float4SoAView {
    float* __restrict__ x;
    float* __restrict__ y;
    float* __restrict__ z;
    float* __restrict__ w;

    __device__ float4 get(int idx) const {
        return make_float4(x[idx], y[idx], z[idx], w[idx]);
    }

    __device__ void set(int idx, const float4& v) {
        x[idx] = v.x;
        y[idx] = v.y;
        z[idx] = v.z;
        w[idx] = v.w;
    }
};

inline Float4SoAView make_view(Float4SoA& soa) {
    return Float4SoAView{soa.x.data(), soa.y.data(), soa.z.data(), soa.w.data()};
}





struct RaysSoA {
    
    DeviceBuffer<float> origin_x;
    DeviceBuffer<float> origin_y;
    DeviceBuffer<float> origin_z;

    
    DeviceBuffer<float> dir_x;
    DeviceBuffer<float> dir_y;
    DeviceBuffer<float> dir_z;

    
    DeviceBuffer<float> t_min;
    DeviceBuffer<float> t_max;

    void resize(size_t count) {
        origin_x.resize(count);
        origin_y.resize(count);
        origin_z.resize(count);
        dir_x.resize(count);
        dir_y.resize(count);
        dir_z.resize(count);
        t_min.resize(count);
        t_max.resize(count);
    }

    size_t size() const { return origin_x.size(); }
};

struct RaysSoAView {
    float* __restrict__ origin_x;
    float* __restrict__ origin_y;
    float* __restrict__ origin_z;
    float* __restrict__ dir_x;
    float* __restrict__ dir_y;
    float* __restrict__ dir_z;
    float* __restrict__ t_min;
    float* __restrict__ t_max;

    __device__ Ray get(int idx) const {
        Ray ray;
        ray.origin = make_float3(origin_x[idx], origin_y[idx], origin_z[idx]);
        ray.direction = make_float3(dir_x[idx], dir_y[idx], dir_z[idx]);
        ray.t_min = t_min[idx];
        ray.t_max = t_max[idx];
        return ray;
    }

    __device__ void set(int idx, const Ray& ray) {
        origin_x[idx] = ray.origin.x;
        origin_y[idx] = ray.origin.y;
        origin_z[idx] = ray.origin.z;
        dir_x[idx] = ray.direction.x;
        dir_y[idx] = ray.direction.y;
        dir_z[idx] = ray.direction.z;
        t_min[idx] = ray.t_min;
        t_max[idx] = ray.t_max;
    }

    __device__ float3 get_origin(int idx) const {
        return make_float3(origin_x[idx], origin_y[idx], origin_z[idx]);
    }

    __device__ float3 get_direction(int idx) const {
        return make_float3(dir_x[idx], dir_y[idx], dir_z[idx]);
    }
};

inline RaysSoAView make_view(RaysSoA& rays) {
    return RaysSoAView{
        rays.origin_x.data(), rays.origin_y.data(), rays.origin_z.data(),
        rays.dir_x.data(), rays.dir_y.data(), rays.dir_z.data(),
        rays.t_min.data(), rays.t_max.data()
    };
}





struct HitsSoA {
    DeviceBuffer<float> t;          
    DeviceBuffer<int> prim_id;      
    DeviceBuffer<int> material_id;  
    DeviceBuffer<float> u;          
    DeviceBuffer<float> v;          

    void resize(size_t count) {
        t.resize(count);
        prim_id.resize(count);
        material_id.resize(count);
        u.resize(count);
        v.resize(count);
    }

    void resize_and_clear(size_t count) {
        t.resize_and_clear(count);
        prim_id.resize_and_clear(count);
        material_id.resize_and_clear(count);
        u.resize_and_clear(count);
        v.resize_and_clear(count);
    }

    size_t size() const { return t.size(); }
};

struct HitInfo {
    float t;
    int prim_id;
    int material_id;
    float u, v;

    __device__ bool hit() const { return t > 0.0f; }
};

struct HitsSoAView {
    float* __restrict__ t;
    int* __restrict__ prim_id;
    int* __restrict__ material_id;
    float* __restrict__ u;
    float* __restrict__ v;

    __device__ HitInfo get(int idx) const {
        HitInfo hit;
        hit.t = t[idx];
        hit.prim_id = prim_id[idx];
        hit.material_id = material_id[idx];
        hit.u = u[idx];
        hit.v = v[idx];
        return hit;
    }

    __device__ void set(int idx, const HitInfo& hit) {
        t[idx] = hit.t;
        prim_id[idx] = hit.prim_id;
        material_id[idx] = hit.material_id;
        u[idx] = hit.u;
        v[idx] = hit.v;
    }

    __device__ void set_miss(int idx) {
        t[idx] = -1.0f;
        prim_id[idx] = -1;
        material_id[idx] = -1;
    }
};

inline HitsSoAView make_view(HitsSoA& hits) {
    return HitsSoAView{
        hits.t.data(), hits.prim_id.data(), hits.material_id.data(),
        hits.u.data(), hits.v.data()
    };
}





class AtomicCounter {
public:
    AtomicCounter() : counter_(nullptr) {
        CUDA_CHECK(cudaMalloc(&counter_, sizeof(unsigned int)));
        reset();
    }

    ~AtomicCounter() {
        if (counter_) cudaFree(counter_);
    }

    AtomicCounter(const AtomicCounter&) = delete;
    AtomicCounter& operator=(const AtomicCounter&) = delete;

    void reset() {
        CUDA_CHECK(cudaMemset(counter_, 0, sizeof(unsigned int)));
    }

    unsigned int get() const {
        unsigned int value;
        CUDA_CHECK(cudaMemcpy(&value, counter_, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        return value;
    }

    unsigned int* ptr() { return counter_; }

    
    __device__ unsigned int increment() {
        return atomicAdd(counter_, 1);
    }

private:
    unsigned int* counter_;
};

} 
