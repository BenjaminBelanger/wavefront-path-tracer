#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace wpt {

// =============================================================================
// Float3 Vector Operations
// =============================================================================

// Wrap CUDA's make_float3 to allow consistent usage within wpt namespace
__host__ __device__ inline float3 make_float3(float x, float y, float z) {
    return ::make_float3(x, y, z);
}

__host__ __device__ inline float3 make_float3(float s) {
    return ::make_float3(s, s, s);
}

__host__ __device__ inline float3 make_float3(const float4& v) {
    return ::make_float3(v.x, v.y, v.z);
}

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return ::make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
    return ::make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 operator/(const float3& a, const float3& b) {
    return ::make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ inline float3 operator*(const float3& v, float s) {
    return ::make_float3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3& v) {
    return ::make_float3(s * v.x, s * v.y, s * v.z);
}

__host__ __device__ inline float3 operator/(const float3& v, float s) {
    float inv = 1.0f / s;
    return v * inv;
}

__host__ __device__ inline float3 operator-(const float3& v) {
    return ::make_float3(-v.x, -v.y, -v.z);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__host__ __device__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

__host__ __device__ inline float3& operator*=(float3& a, float s) {
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}

__host__ __device__ inline float3& operator/=(float3& a, float s) {
    float inv = 1.0f / s;
    a.x *= inv; a.y *= inv; a.z *= inv;
    return a;
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return ::make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline float length_squared(const float3& v) {
    return dot(v, v);
}

__host__ __device__ inline float length(const float3& v) {
    return sqrtf(length_squared(v));
}

__host__ __device__ inline float3 normalize(const float3& v) {
    return v / length(v);
}

__host__ __device__ inline float3 safe_normalize(const float3& v, const float3& fallback = make_float3(0.0f, 1.0f, 0.0f)) {
    float len_sq = length_squared(v);
    if (len_sq > 1e-10f) {
        return v / sqrtf(len_sq);
    }
    return fallback;
}

__host__ __device__ inline float3 lerp(const float3& a, const float3& b, float t) {
    return a + t * (b - a);
}

__host__ __device__ inline float3 min(const float3& a, const float3& b) {
    return ::make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ inline float3 max(const float3& a, const float3& b) {
    return ::make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ inline float3 abs(const float3& v) {
    return ::make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

__host__ __device__ inline float3 clamp(const float3& v, float lo, float hi) {
    return ::make_float3(
        fminf(fmaxf(v.x, lo), hi),
        fminf(fmaxf(v.y, lo), hi),
        fminf(fmaxf(v.z, lo), hi)
    );
}

// Scalar clamp
__host__ __device__ inline float clamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// Scalar min/max for int
__host__ __device__ inline int min_int(int a, int b) {
    return (a < b) ? a : b;
}

__host__ __device__ inline int max_int(int a, int b) {
    return (a > b) ? a : b;
}

__host__ __device__ inline float max_component(const float3& v) {
    return fmaxf(fmaxf(v.x, v.y), v.z);
}

__host__ __device__ inline float min_component(const float3& v) {
    return fminf(fminf(v.x, v.y), v.z);
}

__host__ __device__ inline int max_dimension(const float3& v) {
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

__host__ __device__ inline float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline bool refract(const float3& v, const float3& n, float eta, float3& refracted) {
    float cos_i = dot(-v, n);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t > 1.0f) return false;
    float cos_t = sqrtf(1.0f - sin2_t);
    refracted = eta * v + (eta * cos_i - cos_t) * n;
    return true;
}

__host__ __device__ inline float3 faceforward(const float3& n, const float3& v) {
    return (dot(n, v) < 0.0f) ? n : -n;
}

// =============================================================================
// Float4 Vector Operations
// =============================================================================

__host__ __device__ inline float4 make_float4(float x, float y, float z, float w) {
    return ::make_float4(x, y, z, w);
}

__host__ __device__ inline float4 make_float4(const float3& v, float w) {
    return ::make_float4(v.x, v.y, v.z, w);
}

__host__ __device__ inline float4 make_float4(float s) {
    return ::make_float4(s, s, s, s);
}

__host__ __device__ inline float3 xyz(const float4& v) {
    return ::make_float3(v.x, v.y, v.z);
}

__host__ __device__ inline float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ inline float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ inline float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ inline float4 operator*(const float4& v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

__host__ __device__ inline float4 operator*(float s, const float4& v) {
    return v * s;
}

__host__ __device__ inline float4 operator/(const float4& v, float s) {
    float inv = 1.0f / s;
    return v * inv;
}

__host__ __device__ inline float4& operator+=(float4& a, const float4& b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}

__host__ __device__ inline float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// =============================================================================
// Float2 Vector Operations
// =============================================================================

__host__ __device__ inline float2 make_float2(float x, float y) {
    return ::make_float2(x, y);
}

__host__ __device__ inline float2 make_float2(float s) {
    return ::make_float2(s, s);
}

__host__ __device__ inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline float2 operator*(const float2& v, float s) {
    return make_float2(v.x * s, v.y * s);
}

__host__ __device__ inline float2 operator*(float s, const float2& v) {
    return v * s;
}

__host__ __device__ inline float2 operator/(const float2& v, float s) {
    return make_float2(v.x / s, v.y / s);
}

// =============================================================================
// Integer Vector Operations
// =============================================================================

__host__ __device__ inline int3 make_int3(int x, int y, int z) {
    return ::make_int3(x, y, z);
}

__host__ __device__ inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) {
    return ::make_uint3(x, y, z);
}

// =============================================================================
// Constants
// =============================================================================

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float INV_PI = 0.31830988618379067154f;
constexpr float INV_TWO_PI = 0.15915494309189533577f;
constexpr float INV_FOUR_PI = 0.07957747154594766788f;
constexpr float SQRT_TWO = 1.41421356237309504880f;
constexpr float INV_SQRT_TWO = 0.70710678118654752440f;
constexpr float EPSILON = 1e-6f;
constexpr float RAY_EPSILON = 1e-4f;
constexpr float INFINITY_F = 1e30f;

// =============================================================================
// Coordinate Frame / Basis
// =============================================================================

struct Frame {
    float3 tangent;
    float3 bitangent;
    float3 normal;

    __host__ __device__ Frame() {}

    __host__ __device__ Frame(const float3& n) : normal(n) {
        // Frisvad's method for building orthonormal basis
        if (n.z < -0.9999999f) {
            tangent = make_float3(0.0f, -1.0f, 0.0f);
            bitangent = make_float3(-1.0f, 0.0f, 0.0f);
        } else {
            float a = 1.0f / (1.0f + n.z);
            float b = -n.x * n.y * a;
            tangent = make_float3(1.0f - n.x * n.x * a, b, -n.x);
            bitangent = make_float3(b, 1.0f - n.y * n.y * a, -n.y);
        }
    }

    __host__ __device__ float3 to_local(const float3& v) const {
        return ::make_float3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
    }

    __host__ __device__ float3 to_world(const float3& v) const {
        return v.x * tangent + v.y * bitangent + v.z * normal;
    }
};

// =============================================================================
// Ray Structure
// =============================================================================

struct Ray {
    float3 origin;
    float3 direction;
    float t_min;
    float t_max;

    __host__ __device__ Ray() : t_min(RAY_EPSILON), t_max(INFINITY_F) {}

    __host__ __device__ Ray(const float3& o, const float3& d, float tmin = RAY_EPSILON, float tmax = INFINITY_F)
        : origin(o), direction(d), t_min(tmin), t_max(tmax) {}

    __host__ __device__ float3 at(float t) const {
        return origin + t * direction;
    }
};

} // namespace wpt
