#pragma once

#include "vector.cuh"

namespace lumina {

// =============================================================================
// Sampling Utilities for Monte Carlo Integration
// =============================================================================

// Uniform sample on unit disk
__host__ __device__ inline float2 sample_disk_uniform(float u1, float u2) {
    float r = sqrtf(u1);
    float theta = TWO_PI * u2;
    return make_float2(r * cosf(theta), r * sinf(theta));
}

// Concentric disk sampling (better distribution)
__host__ __device__ inline float2 sample_disk_concentric(float u1, float u2) {
    // Map [0,1]^2 to [-1,1]^2
    float sx = 2.0f * u1 - 1.0f;
    float sy = 2.0f * u2 - 1.0f;

    if (sx == 0.0f && sy == 0.0f) {
        return make_float2(0.0f, 0.0f);
    }

    float r, theta;
    if (fabsf(sx) > fabsf(sy)) {
        r = sx;
        theta = (PI / 4.0f) * (sy / sx);
    } else {
        r = sy;
        theta = (PI / 2.0f) - (PI / 4.0f) * (sx / sy);
    }

    return make_float2(r * cosf(theta), r * sinf(theta));
}

// Cosine-weighted hemisphere sampling (PDF = cos(theta) / PI)
__host__ __device__ inline float3 sample_hemisphere_cosine(float u1, float u2) {
    float2 d = sample_disk_concentric(u1, u2);
    float z = sqrtf(fmaxf(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return make_float3(d.x, d.y, z);
}

__host__ __device__ inline float pdf_hemisphere_cosine(float cos_theta) {
    return fmaxf(0.0f, cos_theta) * INV_PI;
}

// Uniform hemisphere sampling (PDF = 1 / 2PI)
__host__ __device__ inline float3 sample_hemisphere_uniform(float u1, float u2) {
    float z = u1;  // cos(theta)
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));  // sin(theta)
    float phi = TWO_PI * u2;
    return make_float3(r * cosf(phi), r * sinf(phi), z);
}

__host__ __device__ inline float pdf_hemisphere_uniform() {
    return INV_TWO_PI;
}

// Uniform sphere sampling (PDF = 1 / 4PI)
__host__ __device__ inline float3 sample_sphere_uniform(float u1, float u2) {
    float z = 1.0f - 2.0f * u1;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * u2;
    return make_float3(r * cosf(phi), r * sinf(phi), z);
}

__host__ __device__ inline float pdf_sphere_uniform() {
    return INV_FOUR_PI;
}

// Sample direction in cone (for spherical light sampling)
__host__ __device__ inline float3 sample_cone_uniform(float u1, float u2, float cos_theta_max) {
    float cos_theta = 1.0f - u1 * (1.0f - cos_theta_max);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = TWO_PI * u2;
    return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
}

__host__ __device__ inline float pdf_cone_uniform(float cos_theta_max) {
    return 1.0f / (TWO_PI * (1.0f - cos_theta_max));
}

// GGX/Beckmann importance sampling for microfacet models
__host__ __device__ inline float3 sample_ggx_vndf(const float3& Ve, float alpha_x, float alpha_y, float u1, float u2) {
    // Transform view direction to hemisphere configuration
    float3 Vh = normalize(make_float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));

    // Build orthonormal basis
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : make_float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    // Sample point with polar coordinates
    float r = sqrtf(u1);
    float phi = TWO_PI * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    // Reproject onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // Transform back to ellipsoid configuration
    return normalize(make_float3(alpha_x * Nh.x, alpha_y * Nh.y, fmaxf(0.0f, Nh.z)));
}

// Power heuristic for MIS (beta=2)
__host__ __device__ inline float power_heuristic(float pdf1, float pdf2) {
    float f = pdf1 * pdf1;
    float g = pdf2 * pdf2;
    return f / (f + g);
}

// Balance heuristic for MIS
__host__ __device__ inline float balance_heuristic(float pdf1, float pdf2) {
    return pdf1 / (pdf1 + pdf2);
}

// =============================================================================
// Probability Utilities
// =============================================================================

// Fresnel term (Schlick approximation)
__host__ __device__ inline float fresnel_schlick(float cos_theta, float F0) {
    float x = 1.0f - cos_theta;
    float x2 = x * x;
    return F0 + (1.0f - F0) * x2 * x2 * x;
}

__host__ __device__ inline float3 fresnel_schlick(float cos_theta, const float3& F0) {
    float x = 1.0f - cos_theta;
    float x2 = x * x;
    float factor = x2 * x2 * x;
    return F0 + (make_float3(1.0f) - F0) * factor;
}

// Exact Fresnel for dielectrics
__host__ __device__ inline float fresnel_dielectric(float cos_theta_i, float eta) {
    float sin2_theta_i = fmaxf(0.0f, 1.0f - cos_theta_i * cos_theta_i);
    float sin2_theta_t = sin2_theta_i / (eta * eta);

    if (sin2_theta_t >= 1.0f) {
        return 1.0f;  // Total internal reflection
    }

    float cos_theta_t = sqrtf(1.0f - sin2_theta_t);

    float r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    return 0.5f * (r_parl * r_parl + r_perp * r_perp);
}

// Russian roulette probability based on throughput
__host__ __device__ inline float russian_roulette_prob(const float3& throughput) {
    return fminf(0.95f, max_component(throughput));
}

// Convert spherical to cartesian
__host__ __device__ inline float3 spherical_to_cartesian(float theta, float phi) {
    float sin_theta = sinf(theta);
    return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cosf(theta));
}

// Convert cartesian to spherical
__host__ __device__ inline float2 cartesian_to_spherical(const float3& v) {
    float theta = acosf(clamp(v.z, -1.0f, 1.0f));
    float phi = atan2f(v.y, v.x);
    if (phi < 0.0f) phi += TWO_PI;
    return make_float2(theta, phi);
}

// Sample triangle uniformly
__host__ __device__ inline float2 sample_triangle_uniform(float u1, float u2) {
    float su1 = sqrtf(u1);
    return make_float2(1.0f - su1, u2 * su1);
}

// Luminance of RGB
__host__ __device__ inline float luminance(const float3& rgb) {
    return 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z;
}

} // namespace lumina
