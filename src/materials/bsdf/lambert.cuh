#pragma once

#include "bsdf.cuh"
#include "ggx.cuh"

namespace lumina {

// =============================================================================
// Lambertian (Diffuse) BSDF
// =============================================================================

// f(wo, wi) = albedo / PI
// PDF = cos(theta) / PI  (cosine-weighted hemisphere sampling)

struct LambertBSDF {
    float3 albedo;

    __host__ __device__ LambertBSDF(const float3& color) : albedo(color) {}

    // Evaluate BSDF (without cosine term)
    __host__ __device__ float3 evaluate(const float3& wo_local, const float3& wi_local) const {
        // Both directions should be in same hemisphere
        if (wo_local.z <= 0.0f || wi_local.z <= 0.0f) {
            return make_float3(0.0f);
        }
        return albedo * INV_PI;
    }

    // Sample incoming direction
    __host__ __device__ BSDFSample sample(const float3& wo_local, float u1, float u2) const {
        BSDFSample sample;

        // Check that we're on the correct side
        if (wo_local.z <= 0.0f) {
            return sample;
        }

        // Cosine-weighted hemisphere sampling
        sample.wi = sample_hemisphere_cosine(u1, u2);
        sample.pdf = pdf_hemisphere_cosine(sample.wi.z);

        // f * |cos(theta)| = (albedo / PI) * cos(theta)
        // But we sampled with PDF = cos(theta) / PI
        // So f * |cos| / PDF = albedo
        sample.f = albedo * sample.wi.z;  // Include cosine for proper weighting

        sample.is_specular = false;
        sample.is_transmission = false;

        return sample;
    }

    // PDF for a given direction
    __host__ __device__ float pdf(const float3& wo_local, const float3& wi_local) const {
        if (wo_local.z <= 0.0f || wi_local.z <= 0.0f) {
            return 0.0f;
        }
        return pdf_hemisphere_cosine(wi_local.z);
    }
};

// =============================================================================
// Spectral Lambertian BSDF
// =============================================================================

struct SpectralLambertBSDF {
    SpectralRadiance albedo;

    __host__ __device__ SpectralLambertBSDF(const SpectralRadiance& color) : albedo(color) {}

    __host__ __device__ SpectralRadiance evaluate(const float3& wo_local, const float3& wi_local) const {
        if (wo_local.z <= 0.0f || wi_local.z <= 0.0f) {
            return SpectralRadiance(0.0f);
        }
        return albedo * INV_PI;
    }

    __host__ __device__ SpectralBSDFSample sample(const float3& wo_local, float u1, float u2) const {
        SpectralBSDFSample sample;

        if (wo_local.z <= 0.0f) {
            return sample;
        }

        sample.wi = sample_hemisphere_cosine(u1, u2);
        sample.pdf = pdf_hemisphere_cosine(sample.wi.z);
        sample.f = albedo * sample.wi.z;
        sample.is_specular = false;
        sample.is_transmission = false;

        return sample;
    }

    __host__ __device__ float pdf(const float3& wo_local, const float3& wi_local) const {
        if (wo_local.z <= 0.0f || wi_local.z <= 0.0f) {
            return 0.0f;
        }
        return pdf_hemisphere_cosine(wi_local.z);
    }
};

// =============================================================================
// Oren-Nayar Diffuse (Rough Diffuse)
// =============================================================================

struct OrenNayarBSDF {
    float3 albedo;
    float sigma;  // Surface roughness (standard deviation of angle in radians)
    float A, B;   // Precomputed coefficients

    __host__ __device__ OrenNayarBSDF(const float3& color, float roughness)
        : albedo(color), sigma(roughness) {
        float sigma2 = sigma * sigma;
        A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
        B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    __host__ __device__ float3 evaluate(const float3& wo_local, const float3& wi_local) const {
        if (wo_local.z <= 0.0f || wi_local.z <= 0.0f) {
            return make_float3(0.0f);
        }

        float sin_theta_i = sqrtf(fmaxf(0.0f, 1.0f - wi_local.z * wi_local.z));
        float sin_theta_o = sqrtf(fmaxf(0.0f, 1.0f - wo_local.z * wo_local.z));

        // Compute cos(phi_i - phi_o)
        float max_cos = 0.0f;
        if (sin_theta_i > 1e-4f && sin_theta_o > 1e-4f) {
            float cos_phi_diff = (wi_local.x * wo_local.x + wi_local.y * wo_local.y) /
                                 (sin_theta_i * sin_theta_o);
            max_cos = fmaxf(0.0f, cos_phi_diff);
        }

        // Compute sin(alpha) * tan(beta)
        float sin_alpha, tan_beta;
        if (wi_local.z > wo_local.z) {
            sin_alpha = sin_theta_o;
            tan_beta = sin_theta_i / wi_local.z;
        } else {
            sin_alpha = sin_theta_i;
            tan_beta = sin_theta_o / wo_local.z;
        }

        return albedo * INV_PI * (A + B * max_cos * sin_alpha * tan_beta);
    }

    __host__ __device__ BSDFSample sample(const float3& wo_local, float u1, float u2) const {
        BSDFSample sample;

        if (wo_local.z <= 0.0f) {
            return sample;
        }

        // Use cosine-weighted sampling (not optimal but simple)
        sample.wi = sample_hemisphere_cosine(u1, u2);
        sample.pdf = pdf_hemisphere_cosine(sample.wi.z);
        sample.f = evaluate(wo_local, sample.wi) * sample.wi.z;
        sample.is_specular = false;
        sample.is_transmission = false;

        return sample;
    }

    __host__ __device__ float pdf(const float3& wo_local, const float3& wi_local) const {
        if (wo_local.z <= 0.0f || wi_local.z <= 0.0f) {
            return 0.0f;
        }
        return pdf_hemisphere_cosine(wi_local.z);
    }
};

// =============================================================================
// Helper: Dispatch BSDF evaluation based on material type
// =============================================================================

__device__ inline BSDFSample sample_bsdf(
    const Material& material,
    const ShadingContext& ctx,
    float u1, float u2
) {
    float3 wo_local = ctx.to_local(ctx.wo);

    switch (material.type) {
        case MaterialType::Lambert: {
            LambertBSDF bsdf(material.albedo);
            BSDFSample sample = bsdf.sample(wo_local, u1, u2);
            if (sample.is_valid()) {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        case MaterialType::Metal: {
            // Metallic reflection with optional roughness
            BSDFSample sample;
            if (wo_local.z <= 0.0f) return sample;

            if (material.roughness < 0.01f) {
                // Perfect mirror
                sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                sample.f = material.albedo * wo_local.z;
                sample.pdf = 1.0f;
                sample.is_specular = true;
            } else {
                // Rough metal using GGX
                float alpha = material.roughness * material.roughness;
                float3 h = ggx_sample_vndf(wo_local, alpha, u1, u2);
                sample.wi = reflect(-wo_local, h);
                if (sample.wi.z <= 0.0f) return sample;

                float D = ggx_d(h.z, alpha);
                float G = ggx_g(wo_local.z, sample.wi.z, alpha);
                float3 F = fresnel_schlick(dot(wo_local, h), material.albedo);

                sample.f = F * D * G / (4.0f * wo_local.z);
                sample.pdf = ggx_vndf_pdf(wo_local, h, alpha) / (4.0f * dot(wo_local, h));
                sample.is_specular = false;
            }
            sample.is_transmission = false;
            if (sample.is_valid()) {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        case MaterialType::Dielectric: {
            // Glass with refraction
            BSDFSample sample;
            if (wo_local.z == 0.0f) return sample;

            bool entering = wo_local.z > 0.0f;
            float eta = entering ? (1.0f / material.ior) : material.ior;
            float3 n = make_float3(0.0f, 0.0f, entering ? 1.0f : -1.0f);

            float cos_theta = fabsf(wo_local.z);
            float F = fresnel_dielectric(cos_theta, eta);

            // Choose reflect or refract
            if (u1 < F) {
                // Reflect
                sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                sample.f = make_float3(1.0f) * fabsf(sample.wi.z);
                sample.pdf = F;
                sample.is_specular = true;
                sample.is_transmission = false;
            } else {
                // Refract
                float3 refracted;
                float3 wo_n = entering ? wo_local : -wo_local;
                if (!refract(wo_n, n, eta, refracted)) {
                    // Total internal reflection
                    sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                    sample.f = make_float3(1.0f) * fabsf(sample.wi.z);
                    sample.pdf = 1.0f;
                    sample.is_specular = true;
                    sample.is_transmission = false;
                } else {
                    sample.wi = entering ? refracted : -refracted;
                    // Account for solid angle compression
                    sample.f = make_float3(eta * eta) * fabsf(sample.wi.z);
                    sample.pdf = 1.0f - F;
                    sample.is_specular = true;
                    sample.is_transmission = true;
                }
            }
            if (sample.is_valid()) {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        default: {
            // Fallback to Lambert
            LambertBSDF bsdf(material.albedo);
            BSDFSample sample = bsdf.sample(wo_local, u1, u2);
            if (sample.is_valid()) {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
    }
}

__device__ inline float3 evaluate_bsdf(
    const Material& material,
    const ShadingContext& ctx,
    const float3& wi
) {
    float3 wo_local = ctx.to_local(ctx.wo);
    float3 wi_local = ctx.to_local(wi);

    switch (material.type) {
        case MaterialType::Lambert: {
            LambertBSDF bsdf(material.albedo);
            return bsdf.evaluate(wo_local, wi_local);
        }
        default: {
            LambertBSDF bsdf(material.albedo);
            return bsdf.evaluate(wo_local, wi_local);
        }
    }
}

__device__ inline float pdf_bsdf(
    const Material& material,
    const ShadingContext& ctx,
    const float3& wi
) {
    float3 wo_local = ctx.to_local(ctx.wo);
    float3 wi_local = ctx.to_local(wi);

    switch (material.type) {
        case MaterialType::Lambert: {
            LambertBSDF bsdf(material.albedo);
            return bsdf.pdf(wo_local, wi_local);
        }
        default: {
            LambertBSDF bsdf(material.albedo);
            return bsdf.pdf(wo_local, wi_local);
        }
    }
}

} // namespace lumina
