#pragma once

#include "../../core/math/vector.cuh"
#include "aabb.cuh"

namespace lumina {

// =============================================================================
// Triangle Primitive
// =============================================================================

struct Triangle {
    float3 v0, v1, v2;  // Vertices
    float3 n0, n1, n2;  // Per-vertex normals (for smooth shading)
    float2 uv0, uv1, uv2;  // Texture coordinates
    int material_id;

    __host__ __device__ Triangle() : material_id(0) {}

    __host__ __device__ Triangle(const float3& a, const float3& b, const float3& c, int mat_id = 0)
        : v0(a), v1(b), v2(c), material_id(mat_id) {
        // Compute face normal for all vertices
        float3 face_normal = geometric_normal();
        n0 = n1 = n2 = face_normal;
        uv0 = make_float2(0.0f, 0.0f);
        uv1 = make_float2(1.0f, 0.0f);
        uv2 = make_float2(0.0f, 1.0f);
    }

    __host__ __device__ float3 geometric_normal() const {
        return normalize(cross(v1 - v0, v2 - v0));
    }

    __host__ __device__ float3 centroid() const {
        return (v0 + v1 + v2) * (1.0f / 3.0f);
    }

    __host__ __device__ AABB bounds() const {
        AABB box;
        box.expand(v0);
        box.expand(v1);
        box.expand(v2);
        return box;
    }

    __host__ __device__ float area() const {
        return 0.5f * length(cross(v1 - v0, v2 - v0));
    }

    // Get interpolated normal at barycentric coordinates (u, v)
    // Barycentric: P = (1-u-v)*v0 + u*v1 + v*v2
    __host__ __device__ float3 interpolate_normal(float u, float v) const {
        float w = 1.0f - u - v;
        return normalize(w * n0 + u * n1 + v * n2);
    }

    // Get interpolated UV at barycentric coordinates
    __host__ __device__ float2 interpolate_uv(float u, float v) const {
        float w = 1.0f - u - v;
        return make_float2(
            w * uv0.x + u * uv1.x + v * uv2.x,
            w * uv0.y + u * uv1.y + v * uv2.y
        );
    }

    // Get interpolated position at barycentric coordinates
    __host__ __device__ float3 interpolate_position(float u, float v) const {
        float w = 1.0f - u - v;
        return w * v0 + u * v1 + v * v2;
    }

    // Moller-Trumbore ray-triangle intersection
    // Returns true if hit, sets t, u, v (barycentric coordinates)
    __host__ __device__ bool intersect(const Ray& ray, float& t, float& u, float& v) const {
        const float3 edge1 = v1 - v0;
        const float3 edge2 = v2 - v0;

        const float3 h = cross(ray.direction, edge2);
        const float a = dot(edge1, h);

        // Check if ray is parallel to triangle
        if (fabsf(a) < EPSILON) {
            return false;
        }

        const float f = 1.0f / a;
        const float3 s = ray.origin - v0;
        u = f * dot(s, h);

        // Check barycentric coordinate u
        if (u < 0.0f || u > 1.0f) {
            return false;
        }

        const float3 q = cross(s, edge1);
        v = f * dot(ray.direction, q);

        // Check barycentric coordinate v and u+v
        if (v < 0.0f || u + v > 1.0f) {
            return false;
        }

        // Compute t and check bounds
        t = f * dot(edge2, q);

        return t >= ray.t_min && t <= ray.t_max;
    }

    // Watertight ray-triangle intersection (Woop et al.)
    // More robust at edges and corners
    __host__ __device__ bool intersect_watertight(const Ray& ray, float& t, float& u, float& v) const {
        // Calculate dimension where ray direction is maximal
        int kz = max_dimension(abs(ray.direction));
        int kx = kz + 1; if (kx == 3) kx = 0;
        int ky = kx + 1; if (ky == 3) ky = 0;

        // Swap kx and ky to preserve winding direction of triangles
        if ((&ray.direction.x)[kz] < 0.0f) {
            int temp = kx;
            kx = ky;
            ky = temp;
        }

        // Calculate shear constants
        float Sx = (&ray.direction.x)[kx] / (&ray.direction.x)[kz];
        float Sy = (&ray.direction.x)[ky] / (&ray.direction.x)[kz];
        float Sz = 1.0f / (&ray.direction.x)[kz];

        // Calculate vertices relative to ray origin
        float3 A = v0 - ray.origin;
        float3 B = v1 - ray.origin;
        float3 C = v2 - ray.origin;

        // Perform shear and scale
        float Ax = (&A.x)[kx] - Sx * (&A.x)[kz];
        float Ay = (&A.x)[ky] - Sy * (&A.x)[kz];
        float Bx = (&B.x)[kx] - Sx * (&B.x)[kz];
        float By = (&B.x)[ky] - Sy * (&B.x)[kz];
        float Cx = (&C.x)[kx] - Sx * (&C.x)[kz];
        float Cy = (&C.x)[ky] - Sy * (&C.x)[kz];

        // Calculate scaled barycentric coordinates
        float U = Cx * By - Cy * Bx;
        float V = Ax * Cy - Ay * Cx;
        float W = Bx * Ay - By * Ax;

        // Fall back to double precision for edge tests
        if (U == 0.0f || V == 0.0f || W == 0.0f) {
            double CxBy = static_cast<double>(Cx) * static_cast<double>(By);
            double CyBx = static_cast<double>(Cy) * static_cast<double>(Bx);
            U = static_cast<float>(CxBy - CyBx);
            double AxCy = static_cast<double>(Ax) * static_cast<double>(Cy);
            double AyCx = static_cast<double>(Ay) * static_cast<double>(Cx);
            V = static_cast<float>(AxCy - AyCx);
            double BxAy = static_cast<double>(Bx) * static_cast<double>(Ay);
            double ByAx = static_cast<double>(By) * static_cast<double>(Ax);
            W = static_cast<float>(BxAy - ByAx);
        }

        // Check if point is on triangle or edges
        if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) {
            return false;
        }

        // Calculate determinant
        float det = U + V + W;
        if (det == 0.0f) {
            return false;
        }

        // Calculate scaled z-coordinates and use them to compute t
        float Az = Sz * (&A.x)[kz];
        float Bz = Sz * (&B.x)[kz];
        float Cz = Sz * (&C.x)[kz];
        float T = U * Az + V * Bz + W * Cz;

        // Check t bounds
        if (det > 0.0f) {
            if (T < ray.t_min * det || T > ray.t_max * det) {
                return false;
            }
        } else {
            if (T > ray.t_min * det || T < ray.t_max * det) {
                return false;
            }
        }

        // Normalize barycentric coordinates
        float inv_det = 1.0f / det;
        u = V * inv_det;
        v = W * inv_det;
        t = T * inv_det;

        return true;
    }

    // Sample a random point on the triangle uniformly
    __host__ __device__ float3 sample_point(float u1, float u2) const {
        float su1 = sqrtf(u1);
        float u = 1.0f - su1;
        float v = u2 * su1;
        return interpolate_position(u, v);
    }

    // PDF for uniform sampling (1 / area)
    __host__ __device__ float sample_pdf() const {
        return 1.0f / area();
    }
};

// =============================================================================
// Compact Triangle Storage for BVH
// =============================================================================

// Store only vertex indices, actual vertices in separate array
struct TriangleIndices {
    uint32_t v0, v1, v2;
    uint32_t material_id;
};

// Precomputed triangle data for faster intersection
struct TrianglePrecomputed {
    float3 v0;
    float3 edge1;  // v1 - v0
    float3 edge2;  // v2 - v0
    int material_id;

    __host__ __device__ void from_triangle(const Triangle& tri) {
        v0 = tri.v0;
        edge1 = tri.v1 - tri.v0;
        edge2 = tri.v2 - tri.v0;
        material_id = tri.material_id;
    }

    __host__ __device__ bool intersect(const Ray& ray, float& t, float& u, float& v) const {
        const float3 h = cross(ray.direction, edge2);
        const float a = dot(edge1, h);

        if (fabsf(a) < EPSILON) {
            return false;
        }

        const float f = 1.0f / a;
        const float3 s = ray.origin - v0;
        u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f) {
            return false;
        }

        const float3 q = cross(s, edge1);
        v = f * dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f) {
            return false;
        }

        t = f * dot(edge2, q);
        return t >= ray.t_min && t <= ray.t_max;
    }
};

} // namespace lumina
