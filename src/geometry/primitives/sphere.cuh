#pragma once

#include "../../core/math/vector.cuh"
#include "../../core/math/sampling.cuh"
#include "aabb.cuh"

namespace wpt
{

    struct Sphere
    {
        float3 center;
        float radius;
        int material_id;

        __host__ __device__ Sphere() : center(make_float3(0.0f)), radius(1.0f), material_id(0) {}

        __host__ __device__ Sphere(const float3 &c, float r, int mat_id = 0)
            : center(c), radius(r), material_id(mat_id) {}

        __host__ __device__ AABB bounds() const
        {
            float3 r = make_float3(radius);
            return AABB(center - r, center + r);
        }

        __host__ __device__ float surface_area() const
        {
            return 4.0f * PI * radius * radius;
        }

        __host__ __device__ float volume() const
        {
            return (4.0f / 3.0f) * PI * radius * radius * radius;
        }

        __host__ __device__ float3 normal_at(const float3 &point) const
        {
            return normalize(point - center);
        }

        __host__ __device__ float2 uv_at(const float3 &point) const
        {
            float3 n = normalize(point - center);
            float theta = acosf(clamp(n.y, -1.0f, 1.0f));
            float phi = atan2f(n.z, n.x);

            float u = (phi + PI) * INV_TWO_PI;
            float v = theta * INV_PI;
            return make_float2(u, v);
        }

        __host__ __device__ bool intersect(const Ray &ray, float &t) const
        {
            float3 oc = ray.origin - center;

            float a = dot(ray.direction, ray.direction);
            float half_b = dot(oc, ray.direction);
            float c = dot(oc, oc) - radius * radius;

            float discriminant = half_b * half_b - a * c;

            if (discriminant < 0.0f)
            {
                return false;
            }

            float sqrt_d = sqrtf(discriminant);
            float inv_a = 1.0f / a;

            float t_hit = (-half_b - sqrt_d) * inv_a;
            if (t_hit < ray.t_min || t_hit > ray.t_max)
            {

                t_hit = (-half_b + sqrt_d) * inv_a;
                if (t_hit < ray.t_min || t_hit > ray.t_max)
                {
                    return false;
                }
            }

            t = t_hit;
            return true;
        }

        __host__ __device__ bool intersect(const Ray &ray, float &t, float3 &normal) const
        {
            if (!intersect(ray, t))
            {
                return false;
            }
            normal = normal_at(ray.at(t));
            return true;
        }

        __host__ __device__ bool intersects(const Ray &ray) const
        {
            float3 oc = ray.origin - center;
            float a = dot(ray.direction, ray.direction);
            float half_b = dot(oc, ray.direction);
            float c = dot(oc, oc) - radius * radius;
            float discriminant = half_b * half_b - a * c;
            return discriminant >= 0.0f;
        }

        __host__ __device__ float3 sample_uniform(float u1, float u2) const
        {
            float3 dir = sample_sphere_uniform(u1, u2);
            return center + radius * dir;
        }

        __host__ __device__ float sample_pdf_uniform() const
        {
            return 1.0f / surface_area();
        }

        __host__ __device__ float3 sample_solid_angle(const float3 &ref_point, float u1, float u2, float &pdf) const
        {
            float3 to_center = center - ref_point;
            float dist_sq = length_squared(to_center);
            float dist = sqrtf(dist_sq);

            if (dist < radius)
            {

                float3 sample = sample_uniform(u1, u2);
                float3 dir = normalize(sample - ref_point);
                pdf = sample_pdf_uniform() * length_squared(sample - ref_point) / fabsf(dot(normal_at(sample), -dir));
                return sample;
            }

            float sin_theta_max_sq = radius * radius / dist_sq;
            float cos_theta_max = sqrtf(fmaxf(0.0f, 1.0f - sin_theta_max_sq));

            float3 dir = sample_cone_uniform(u1, u2, cos_theta_max);

            Frame frame(normalize(to_center));
            dir = frame.to_world(dir);

            Ray ray(ref_point, dir);
            float t;
            if (!intersect(ray, t))
            {

                t = dist;
            }

            float3 sample = ray.at(t);
            pdf = pdf_cone_uniform(cos_theta_max);
            return sample;
        }

        __host__ __device__ float sample_pdf_solid_angle(const float3 &ref_point) const
        {
            float3 to_center = center - ref_point;
            float dist_sq = length_squared(to_center);

            if (dist_sq <= radius * radius)
            {

                return sample_pdf_uniform();
            }

            float sin_theta_max_sq = radius * radius / dist_sq;
            float cos_theta_max = sqrtf(fmaxf(0.0f, 1.0f - sin_theta_max_sq));
            return pdf_cone_uniform(cos_theta_max);
        }
    };

}
