#pragma once

#include "bsdf.cuh"

namespace lumina
{

    __host__ __device__ inline float ggx_d(float cos_theta_h, float alpha)
    {
        if (cos_theta_h <= 0.0f)
            return 0.0f;

        float alpha2 = alpha * alpha;
        float cos2 = cos_theta_h * cos_theta_h;
        float tan2 = (1.0f - cos2) / cos2;

        float denom = PI * cos2 * cos2 * (alpha2 + tan2) * (alpha2 + tan2);
        return alpha2 / denom;
    }

    __host__ __device__ inline float ggx_g1(float cos_theta, float alpha)
    {
        if (cos_theta <= 0.0f)
            return 0.0f;

        float alpha2 = alpha * alpha;
        float cos2 = cos_theta * cos_theta;
        float tan2 = (1.0f - cos2) / cos2;

        return 2.0f / (1.0f + sqrtf(1.0f + alpha2 * tan2));
    }

    __host__ __device__ inline float ggx_g(float cos_theta_o, float cos_theta_i, float alpha)
    {
        return ggx_g1(cos_theta_o, alpha) * ggx_g1(cos_theta_i, alpha);
    }

    __host__ __device__ inline float3 ggx_sample_vndf(const float3 &wo_local, float alpha, float u1, float u2)
    {

        float3 Vh = normalize(make_float3(alpha * wo_local.x, alpha * wo_local.y, wo_local.z));

        float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
        float3 T1 = lensq > 0.0f ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : make_float3(1.0f, 0.0f, 0.0f);
        float3 T2 = cross(Vh, T1);

        float r = sqrtf(u1);
        float phi = TWO_PI * u2;
        float t1 = r * cosf(phi);
        float t2 = r * sinf(phi);
        float s = 0.5f * (1.0f + Vh.z);
        t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;

        float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

        return normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
    }

    __host__ __device__ inline float ggx_vndf_pdf(const float3 &wo_local, const float3 &h_local, float alpha)
    {
        float cos_theta_o = wo_local.z;
        if (cos_theta_o <= 0.0f)
            return 0.0f;

        float D = ggx_d(h_local.z, alpha);
        float G1 = ggx_g1(cos_theta_o, alpha);

        return D * G1 * fmaxf(0.0f, dot(wo_local, h_local)) / cos_theta_o;
    }

    struct GGXConductorBSDF
    {
        float3 eta;
        float3 k;
        float alpha;

        __host__ __device__ GGXConductorBSDF(const float3 &F0, float roughness)
            : alpha(fmaxf(0.001f, roughness * roughness))
        {

            eta = F0;
            k = make_float3(1.0f);
        }

        __host__ __device__ GGXConductorBSDF(const float3 &eta_in, const float3 &k_in, float roughness)
            : eta(eta_in), k(k_in), alpha(fmaxf(0.001f, roughness * roughness)) {}

        __host__ __device__ float3 fresnel(float cos_theta) const
        {

            return fresnel_schlick(cos_theta, eta);
        }

        __host__ __device__ float3 evaluate(const float3 &wo_local, const float3 &wi_local) const
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
            {
                return make_float3(0.0f);
            }

            float3 h = normalize(wo_local + wi_local);
            if (h.z <= 0.0f)
                return make_float3(0.0f);

            float D = ggx_d(h.z, alpha);
            float G = ggx_g(wo_local.z, wi_local.z, alpha);
            float3 F = fresnel(dot(wo_local, h));

            return F * D * G / (4.0f * wo_local.z * wi_local.z);
        }

        __host__ __device__ BSDFSample sample(const float3 &wo_local, float u1, float u2) const
        {
            BSDFSample sample;

            if (wo_local.z <= 0.0f)
            {
                return sample;
            }

            float3 h = ggx_sample_vndf(wo_local, alpha, u1, u2);

            sample.wi = reflect(-wo_local, h);

            if (sample.wi.z <= 0.0f)
            {
                return sample;
            }

            float vndf_pdf = ggx_vndf_pdf(wo_local, h, alpha);
            sample.pdf = vndf_pdf / (4.0f * dot(wo_local, h));

            if (sample.pdf <= 0.0f)
            {
                return sample;
            }

            float D = ggx_d(h.z, alpha);
            float G = ggx_g(wo_local.z, sample.wi.z, alpha);
            float3 F = fresnel(dot(wo_local, h));

            sample.f = F * D * G / (4.0f * wo_local.z) * sample.wi.z;
            sample.is_specular = (alpha < 0.01f);
            sample.is_transmission = false;

            return sample;
        }

        __host__ __device__ float pdf(const float3 &wo_local, const float3 &wi_local) const
        {
            if (wo_local.z <= 0.0f || wi_local.z <= 0.0f)
            {
                return 0.0f;
            }

            float3 h = normalize(wo_local + wi_local);
            if (h.z <= 0.0f)
                return 0.0f;

            float vndf_pdf = ggx_vndf_pdf(wo_local, h, alpha);
            return vndf_pdf / (4.0f * dot(wo_local, h));
        }
    };

    struct GGXDielectricBSDF
    {
        float eta;
        float alpha;

        __host__ __device__ GGXDielectricBSDF(float ior, float roughness)
            : eta(ior), alpha(fmaxf(0.001f, roughness * roughness)) {}

        __host__ __device__ BSDFSample sample(const float3 &wo_local, float u1, float u2, float u3) const
        {
            BSDFSample sample;

            if (wo_local.z == 0.0f)
            {
                return sample;
            }

            bool entering = wo_local.z > 0.0f;
            float eta_ratio = entering ? (1.0f / eta) : eta;
            float3 n = entering ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 0.0f, -1.0f);

            float3 wo_hemi = entering ? wo_local : -wo_local;
            float3 h = ggx_sample_vndf(wo_hemi, alpha, u1, u2);
            if (!entering)
                h = -h;

            float cos_theta_o = dot(wo_local, h);

            float F = fresnel_dielectric(fabsf(cos_theta_o), eta_ratio);

            if (u3 < F)
            {

                sample.wi = reflect(-wo_local, h);
                if ((entering && sample.wi.z <= 0.0f) || (!entering && sample.wi.z >= 0.0f))
                {
                    return sample;
                }

                float vndf_pdf = ggx_vndf_pdf(wo_hemi, entering ? h : -h, alpha);
                sample.pdf = F * vndf_pdf / (4.0f * fabsf(cos_theta_o));

                float D = ggx_d(fabsf(h.z), alpha);
                float G = ggx_g(fabsf(wo_local.z), fabsf(sample.wi.z), alpha);

                sample.f = make_float3(F * D * G / (4.0f * fabsf(wo_local.z)));
                sample.is_specular = (alpha < 0.01f);
                sample.is_transmission = false;
            }
            else
            {

                float3 refracted;
                if (!refract(wo_local, h, eta_ratio, refracted))
                {
                    return sample;
                }
                sample.wi = refracted;

                if ((entering && sample.wi.z >= 0.0f) || (!entering && sample.wi.z <= 0.0f))
                {
                    return sample;
                }

                float cos_theta_i = dot(sample.wi, h);
                float denom = eta_ratio * cos_theta_o + cos_theta_i;

                float vndf_pdf = ggx_vndf_pdf(wo_hemi, entering ? h : -h, alpha);
                float jacobian = fabsf(cos_theta_i) / (denom * denom);
                sample.pdf = (1.0f - F) * vndf_pdf * jacobian;

                float D = ggx_d(fabsf(h.z), alpha);
                float G = ggx_g(fabsf(wo_local.z), fabsf(sample.wi.z), alpha);

                float btdf = fabsf(cos_theta_o * cos_theta_i) * (1.0f - F) * D * G /
                             (fabsf(wo_local.z) * denom * denom);

                sample.f = make_float3(btdf * eta_ratio * eta_ratio);
                sample.is_specular = (alpha < 0.01f);
                sample.is_transmission = true;
            }

            sample.f = sample.f * fabsf(sample.wi.z);
            return sample;
        }
    };

    struct MirrorBSDF
    {
        float3 reflectance;

        __host__ __device__ MirrorBSDF(const float3 &r = make_float3(1.0f)) : reflectance(r) {}

        __host__ __device__ BSDFSample sample(const float3 &wo_local) const
        {
            BSDFSample sample;

            if (wo_local.z <= 0.0f)
            {
                return sample;
            }

            sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
            sample.pdf = 1.0f;
            sample.f = reflectance;
            sample.is_specular = true;
            sample.is_transmission = false;

            return sample;
        }
    };

    struct PerfectGlassBSDF
    {
        float eta;

        __host__ __device__ PerfectGlassBSDF(float ior) : eta(ior) {}

        __host__ __device__ BSDFSample sample(const float3 &wo_local, float u) const
        {
            BSDFSample sample;

            bool entering = wo_local.z > 0.0f;
            float eta_ratio = entering ? (1.0f / eta) : eta;
            float3 n = make_float3(0.0f, 0.0f, entering ? 1.0f : -1.0f);

            float cos_theta_i = fabsf(wo_local.z);
            float F = fresnel_dielectric(cos_theta_i, entering ? eta : (1.0f / eta));

            if (u < F)
            {

                sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                sample.f = make_float3(1.0f);
                sample.is_transmission = false;
            }
            else
            {

                float3 refracted;
                if (!refract(-wo_local, n, eta_ratio, refracted))
                {

                    sample.wi = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
                    sample.f = make_float3(1.0f);
                    sample.is_transmission = false;
                }
                else
                {
                    sample.wi = refracted;
                    sample.f = make_float3(eta_ratio * eta_ratio);
                    sample.is_transmission = true;
                }
            }

            sample.pdf = 1.0f;
            sample.is_specular = true;

            return sample;
        }
    };

}
