#pragma once

#include "../../core/math/vector.cuh"

namespace lumina {





struct AABB {
    float3 min_bound;
    float3 max_bound;

    __host__ __device__ AABB()
        : min_bound(make_float3(INFINITY_F))
        , max_bound(make_float3(-INFINITY_F)) {}

    __host__ __device__ AABB(const float3& min_b, const float3& max_b)
        : min_bound(min_b), max_bound(max_b) {}

    __host__ __device__ static AABB empty() {
        return AABB();
    }

    __host__ __device__ static AABB unit() {
        return AABB(make_float3(0.0f), make_float3(1.0f));
    }

    __host__ __device__ bool is_valid() const {
        return min_bound.x <= max_bound.x &&
               min_bound.y <= max_bound.y &&
               min_bound.z <= max_bound.z;
    }

    __host__ __device__ float3 center() const {
        return (min_bound + max_bound) * 0.5f;
    }

    __host__ __device__ float3 extent() const {
        return max_bound - min_bound;
    }

    __host__ __device__ float3 diagonal() const {
        return extent();
    }

    __host__ __device__ float surface_area() const {
        float3 d = extent();
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    __host__ __device__ float volume() const {
        float3 d = extent();
        return d.x * d.y * d.z;
    }

    __host__ __device__ int largest_axis() const {
        float3 d = extent();
        if (d.x > d.y && d.x > d.z) return 0;
        if (d.y > d.z) return 1;
        return 2;
    }

    __host__ __device__ float get_axis(int axis, bool is_max) const {
        if (axis == 0) return is_max ? max_bound.x : min_bound.x;
        if (axis == 1) return is_max ? max_bound.y : min_bound.y;
        return is_max ? max_bound.z : min_bound.z;
    }

    __host__ __device__ void expand(const float3& point) {
        min_bound = min(min_bound, point);
        max_bound = max(max_bound, point);
    }

    __host__ __device__ void expand(const AABB& other) {
        min_bound = min(min_bound, other.min_bound);
        max_bound = max(max_bound, other.max_bound);
    }

    __host__ __device__ void expand(float delta) {
        min_bound = min_bound - make_float3(delta);
        max_bound = max_bound + make_float3(delta);
    }

    __host__ __device__ bool contains(const float3& point) const {
        return point.x >= min_bound.x && point.x <= max_bound.x &&
               point.y >= min_bound.y && point.y <= max_bound.y &&
               point.z >= min_bound.z && point.z <= max_bound.z;
    }

    __host__ __device__ bool intersects(const AABB& other) const {
        return min_bound.x <= other.max_bound.x && max_bound.x >= other.min_bound.x &&
               min_bound.y <= other.max_bound.y && max_bound.y >= other.min_bound.y &&
               min_bound.z <= other.max_bound.z && max_bound.z >= other.min_bound.z;
    }

    
    
    __host__ __device__ bool intersect(const Ray& ray, float& t_near, float& t_far) const {
        float3 inv_dir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

        float3 t0 = (min_bound - ray.origin) * inv_dir;
        float3 t1 = (max_bound - ray.origin) * inv_dir;

        float3 t_min_vec = min(t0, t1);
        float3 t_max_vec = max(t0, t1);

        t_near = fmaxf(fmaxf(t_min_vec.x, t_min_vec.y), fmaxf(t_min_vec.z, ray.t_min));
        t_far = fminf(fminf(t_max_vec.x, t_max_vec.y), fminf(t_max_vec.z, ray.t_max));

        return t_near <= t_far;
    }

    
    __host__ __device__ bool intersect_fast(const float3& origin, const float3& inv_dir,
                                            float t_min, float t_max) const {
        float3 t0 = (min_bound - origin) * inv_dir;
        float3 t1 = (max_bound - origin) * inv_dir;

        float3 t_min_vec = min(t0, t1);
        float3 t_max_vec = max(t0, t1);

        float t_near = fmaxf(fmaxf(t_min_vec.x, t_min_vec.y), fmaxf(t_min_vec.z, t_min));
        float t_far = fminf(fminf(t_max_vec.x, t_max_vec.y), fminf(t_max_vec.z, t_max));

        return t_near <= t_far;
    }
};

__host__ __device__ inline AABB union_aabb(const AABB& a, const AABB& b) {
    return AABB(min(a.min_bound, b.min_bound), max(a.max_bound, b.max_bound));
}

__host__ __device__ inline AABB intersection_aabb(const AABB& a, const AABB& b) {
    return AABB(max(a.min_bound, b.min_bound), min(a.max_bound, b.max_bound));
}

} 
