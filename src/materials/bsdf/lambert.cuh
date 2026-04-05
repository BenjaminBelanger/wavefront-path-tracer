#pragma once

#include "bsdf.cuh"
#include "ggx.cuh"
#include "dielectric.cuh"

namespace lumina
{

    struct LambertBSDF
    {
        float3 albedo;

        __host__ __device__ LambertBSDF(const float3 &color) : albedo(color) {}

        __host__ __device__ float3 evaluate(const float3 &wo_local, const float3 &wi_local) const
        {

            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
            {
                return make_float3(0.0f);
            }
            return albedo * INV_PI;
        }

        __host__ __device__ BSDFSample sample(const float3 &wo_local, float u1, float u2) const
        {
            BSDFSample sample;

            if (wo_local.z <= 0.0f)
            {
                return sample;
            }

            sample.wi = sample_hemisphere_cosine(u1, u2);
            sample.pdf = pdf_hemisphere_cosine(sample.wi.z);

            sample.f = albedo * sample.wi.z;

            sample.is_specular = false;
            sample.is_transmission = false;

            return sample;
        }

        __host__ __device__ float pdf(const float3 &wo_local, const float3 &wi_local) const
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
            {
                return 0.0f;
            }
            return pdf_hemisphere_cosine(wi_local.z);
        }
    };

    struct SpectralLambertBSDF
    {
        SpectralRadiance albedo;

        __host__ __device__ SpectralLambertBSDF(const SpectralRadiance &color) : albedo(color) {}

        __host__ __device__ SpectralRadiance evaluate(const float3 &wo_local, const float3 &wi_local) const
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
            {
                return SpectralRadiance(0.0f);
            }
            return albedo * INV_PI;
        }

        __host__ __device__ SpectralBSDFSample sample(const float3 &wo_local, float u1, float u2) const
        {
            SpectralBSDFSample sample;

            if (wo_local.z <= 0.0f)
            {
                return sample;
            }

            sample.wi = sample_hemisphere_cosine(u1, u2);
            sample.pdf = pdf_hemisphere_cosine(sample.wi.z);
            sample.f = albedo * sample.wi.z;
            sample.is_specular = false;
            sample.is_transmission = false;

            return sample;
        }

        __host__ __device__ float pdf(const float3 &wo_local, const float3 &wi_local) const
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
            {
                return 0.0f;
            }
            return pdf_hemisphere_cosine(wi_local.z);
        }
    };

    struct OrenNayarBSDF
    {
        float3 albedo;
        float sigma;
        float A, B;

        __host__ __device__ OrenNayarBSDF(const float3 &color, float roughness)
            : albedo(color), sigma(roughness)
        {
            float sigma2 = sigma * sigma;
            A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
            B = 0.45f * sigma2 / (sigma2 + 0.09f);
        }

        __host__ __device__ float3 evaluate(const float3 &wo_local, const float3 &wi_local) const
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
            {
                return make_float3(0.0f);
            }

            float sin_theta_i = sqrtf(fmaxf(0.0f, 1.0f - wi_local.z * wi_local.z));
            float sin_theta_o = sqrtf(fmaxf(0.0f, 1.0f - wo_local.z * wo_local.z));

            float max_cos = 0.0f;
            if (sin_theta_i > 1e-4f && sin_theta_o > 1e-4f)
            {
                float cos_phi_diff = (wi_local.x * wo_local.x + wi_local.y * wo_local.y) /
                                     (sin_theta_i * sin_theta_o);
                max_cos = fmaxf(0.0f, cos_phi_diff);
            }

            float sin_alpha, tan_beta;
            if (wi_local.z > wo_local.z)
            {
                sin_alpha = sin_theta_o;
                tan_beta = sin_theta_i / wi_local.z;
            }
            else
            {
                sin_alpha = sin_theta_i;
                tan_beta = sin_theta_o / wo_local.z;
            }

            return albedo * INV_PI * (A + B * max_cos * sin_alpha * tan_beta);
        }

        __host__ __device__ BSDFSample sample(const float3 &wo_local, float u1, float u2) const
        {
            BSDFSample sample;

            if (wo_local.z <= 0.0f)
            {
                return sample;
            }

            sample.wi = sample_hemisphere_cosine(u1, u2);
            sample.pdf = pdf_hemisphere_cosine(sample.wi.z);
            sample.f = evaluate(wo_local, sample.wi) * sample.wi.z;
            sample.is_specular = false;
            sample.is_transmission = false;

            return sample;
        }

        __host__ __device__ float pdf(const float3 &wo_local, const float3 &wi_local) const
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
            {
                return 0.0f;
            }
            return pdf_hemisphere_cosine(wi_local.z);
        }
    };

    __device__ inline BSDFSample sample_bsdf(
        const Material &material,
        const ShadingContext &ctx,
        float u1, float u2, float u3 = 0.5f)
    {
        float3 wo_local = ctx.to_local(ctx.wo);

        switch (material.type)
        {
        case MaterialType::Lambert:
        {
            LambertBSDF bsdf(material.albedo);
            BSDFSample sample = bsdf.sample(wo_local, u1, u2);
            if (sample.is_valid())
            {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        case MaterialType::OrenNayar:
        {
            OrenNayarBSDF bsdf(material.albedo, material.roughness);
            BSDFSample sample = bsdf.sample(wo_local, u1, u2);
            if (sample.is_valid())
            {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        case MaterialType::Metal:
        {

            BSDFSample sample;
            if (wo_local.z <= 0.0f)
                return sample;

            if (material.roughness < 0.01f)
            {

                sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                sample.f = material.albedo * wo_local.z;
                sample.pdf = 1.0f;
                sample.is_specular = true;
            }
            else
            {

                float alpha = material.roughness * material.roughness;
                float3 h = ggx_sample_vndf(wo_local, alpha, u1, u2);
                sample.wi = reflect(-wo_local, h);
                if (sample.wi.z <= 0.0f)
                    return sample;

                float D = ggx_d(h.z, alpha);
                float G = ggx_g(wo_local.z, sample.wi.z, alpha);
                float3 F = fresnel_schlick(dot(wo_local, h), material.albedo);

                sample.f = F * D * G / (4.0f * wo_local.z);
                sample.pdf = ggx_vndf_pdf(wo_local, h, alpha) / (4.0f * dot(wo_local, h));
                sample.is_specular = false;
            }
            sample.is_transmission = false;
            if (sample.is_valid())
            {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        case MaterialType::Plastic:
        {
            BSDFSample sample;
            if (wo_local.z <= 0.0f)
                return sample;

            float cos_theta = wo_local.z;
            float F0 = ((material.ior - 1.0f) * (material.ior - 1.0f)) /
                        ((material.ior + 1.0f) * (material.ior + 1.0f));
            float F = F0 + (1.0f - F0) * powf(1.0f - cos_theta, 5.0f);

            if (u1 < F)
            {
                float alpha = fmaxf(material.roughness * material.roughness, 0.001f);
                float3 h = ggx_sample_vndf(wo_local, alpha, u1 / F, u2);
                sample.wi = reflect(-wo_local, h);
                if (sample.wi.z <= 0.0f)
                    return sample;

                float D = ggx_d(h.z, alpha);
                float G = ggx_g(wo_local.z, sample.wi.z, alpha);

                sample.f = make_float3(D * G / (4.0f * wo_local.z));
                float spec_pdf = ggx_vndf_pdf(wo_local, h, alpha) / (4.0f * dot(wo_local, h));
                sample.pdf = F * spec_pdf;
                sample.is_specular = (material.roughness < 0.01f);
            }
            else
            {
                sample.wi = sample_hemisphere_cosine((u1 - F) / (1.0f - F), u2);
                float diff_pdf = pdf_hemisphere_cosine(sample.wi.z);
                sample.pdf = (1.0f - F) * diff_pdf;
                sample.f = material.albedo * INV_PI * sample.wi.z * (1.0f - F);
                sample.is_specular = false;
            }
            sample.is_transmission = false;
            if (sample.is_valid())
            {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        case MaterialType::ThinFilm:
        {
            BSDFSample sample;
            if (wo_local.z <= 0.0f)
                return sample;

            float cos_theta = wo_local.z;
            float3 R = thin_film_reflectance_rgb(cos_theta, material.film_ior,
                                                  material.film_thickness, material.ior);
            float avg_R = (R.x + R.y + R.z) / 3.0f;

            if (u1 < avg_R)
            {
                sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                sample.f = R * fabsf(sample.wi.z);
                sample.pdf = avg_R;
                sample.is_specular = true;
            }
            else
            {
                float remapped_u1 = (u1 - avg_R) / (1.0f - avg_R);
                sample.wi = sample_hemisphere_cosine(remapped_u1, u2);
                float diff_pdf = pdf_hemisphere_cosine(sample.wi.z);
                float3 transmission = make_float3(1.0f - R.x, 1.0f - R.y, 1.0f - R.z);
                sample.f = material.albedo * transmission * sample.wi.z;
                sample.pdf = (1.0f - avg_R) * diff_pdf;
                sample.is_specular = false;
            }
            sample.is_transmission = false;
            if (sample.is_valid())
            {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        case MaterialType::Dielectric:
        {
            if (material.roughness >= 0.01f)
            {
                GGXDielectricBSDF bsdf(material.ior, material.roughness);
                BSDFSample sample = bsdf.sample(wo_local, u1, u2, u3);
                if (sample.is_valid())
                {
                    sample.f = sample.f * material.albedo;
                    sample.wi = ctx.to_world(sample.wi);
                }
                return sample;
            }

            BSDFSample sample;
            if (wo_local.z == 0.0f)
                return sample;

            bool entering = wo_local.z > 0.0f;
            float eta = entering ? (1.0f / material.ior) : material.ior;
            float3 n = make_float3(0.0f, 0.0f, entering ? 1.0f : -1.0f);

            float cos_theta = fabsf(wo_local.z);
            float F = fresnel_dielectric(cos_theta, eta);

            if (u1 < F)
            {

                sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                sample.f = material.albedo * fabsf(sample.wi.z);
                sample.pdf = F;
                sample.is_specular = true;
                sample.is_transmission = false;
            }
            else
            {

                float3 refracted;
                float3 wo_n = entering ? wo_local : -wo_local;
                if (!refract(wo_n, n, eta, refracted))
                {

                    sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                    sample.f = material.albedo * fabsf(sample.wi.z);
                    sample.pdf = 1.0f;
                    sample.is_specular = true;
                    sample.is_transmission = false;
                }
                else
                {
                    sample.wi = entering ? refracted : -refracted;

                    sample.f = material.albedo * (eta * eta) * fabsf(sample.wi.z);
                    sample.pdf = 1.0f - F;
                    sample.is_specular = true;
                    sample.is_transmission = true;
                }
            }
            if (sample.is_valid())
            {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        default:
        {

            LambertBSDF bsdf(material.albedo);
            BSDFSample sample = bsdf.sample(wo_local, u1, u2);
            if (sample.is_valid())
            {
                sample.wi = ctx.to_world(sample.wi);
            }
            return sample;
        }
        }
    }

    __device__ inline float3 evaluate_bsdf(
        const Material &material,
        const ShadingContext &ctx,
        const float3 &wi)
    {
        float3 wo_local = ctx.to_local(ctx.wo);
        float3 wi_local = ctx.to_local(wi);

        switch (material.type)
        {
        case MaterialType::Lambert:
        {
            LambertBSDF bsdf(material.albedo);
            return bsdf.evaluate(wo_local, wi_local);
        }
        case MaterialType::OrenNayar:
        {
            OrenNayarBSDF bsdf(material.albedo, material.roughness);
            return bsdf.evaluate(wo_local, wi_local);
        }
        case MaterialType::Plastic:
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
                return make_float3(0.0f);
            float F0 = ((material.ior - 1.0f) * (material.ior - 1.0f)) /
                        ((material.ior + 1.0f) * (material.ior + 1.0f));
            float F = F0 + (1.0f - F0) * powf(1.0f - wo_local.z, 5.0f);
            float alpha = fmaxf(material.roughness * material.roughness, 0.001f);
            float3 h = normalize(wo_local + wi_local);
            float D = ggx_d(h.z, alpha);
            float G = ggx_g(wo_local.z, wi_local.z, alpha);
            float3 spec = make_float3(F * D * G / (4.0f * wo_local.z * wi_local.z));
            float3 diff = material.albedo * INV_PI * (1.0f - F);
            return spec + diff;
        }
        case MaterialType::ThinFilm:
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
                return make_float3(0.0f);
            float3 R = thin_film_reflectance_rgb(wo_local.z, material.film_ior,
                                                  material.film_thickness, material.ior);
            float3 transmission = make_float3(1.0f - R.x, 1.0f - R.y, 1.0f - R.z);
            return material.albedo * transmission * INV_PI;
        }
        default:
        {
            LambertBSDF bsdf(material.albedo);
            return bsdf.evaluate(wo_local, wi_local);
        }
        }
    }

    __device__ inline float pdf_bsdf(
        const Material &material,
        const ShadingContext &ctx,
        const float3 &wi)
    {
        float3 wo_local = ctx.to_local(ctx.wo);
        float3 wi_local = ctx.to_local(wi);

        switch (material.type)
        {
        case MaterialType::Lambert:
        {
            LambertBSDF bsdf(material.albedo);
            return bsdf.pdf(wo_local, wi_local);
        }
        case MaterialType::OrenNayar:
        {
            OrenNayarBSDF bsdf(material.albedo, material.roughness);
            return bsdf.pdf(wo_local, wi_local);
        }
        case MaterialType::Plastic:
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
                return 0.0f;
            float F0 = ((material.ior - 1.0f) * (material.ior - 1.0f)) /
                        ((material.ior + 1.0f) * (material.ior + 1.0f));
            float F = F0 + (1.0f - F0) * powf(1.0f - wo_local.z, 5.0f);
            float alpha = fmaxf(material.roughness * material.roughness, 0.001f);
            float3 h = normalize(wo_local + wi_local);
            float spec_pdf = ggx_vndf_pdf(wo_local, h, alpha) / (4.0f * dot(wo_local, h));
            float diff_pdf = pdf_hemisphere_cosine(wi_local.z);
            return F * spec_pdf + (1.0f - F) * diff_pdf;
        }
        case MaterialType::ThinFilm:
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
                return 0.0f;
            float3 R = thin_film_reflectance_rgb(wo_local.z, material.film_ior,
                                                  material.film_thickness, material.ior);
            float avg_R = (R.x + R.y + R.z) / 3.0f;
            return (1.0f - avg_R) * pdf_hemisphere_cosine(wi_local.z);
        }
        default:
        {
            LambertBSDF bsdf(material.albedo);
            return bsdf.pdf(wo_local, wi_local);
        }
        }
    }

}
