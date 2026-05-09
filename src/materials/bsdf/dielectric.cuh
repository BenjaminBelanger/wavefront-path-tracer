#pragma once

#include "bsdf.cuh"
#include "ggx.cuh"
#include "../spectral/sellmeier.cuh"

namespace wpt
{

    struct SpectralDielectricBSDF
    {
        float base_ior;
        SellmeierCoeffs coeffs;
        float alpha;

        __host__ __device__ SpectralDielectricBSDF(float ior, float roughness = 0.0f)
            : base_ior(ior), alpha(fmaxf(0.001f, roughness * roughness))
        {

            coeffs = SellmeierCoeffs::bk7_glass();
        }

        __host__ __device__ SpectralDielectricBSDF(const SellmeierCoeffs &c, float roughness = 0.0f)
            : coeffs(c), alpha(fmaxf(0.001f, roughness * roughness))
        {
            base_ior = coeffs.ior(550.0f);
        }

        __host__ __device__ float get_ior(float wavelength_nm) const
        {
            return coeffs.ior(wavelength_nm);
        }

        __host__ __device__ SpectralBSDFSample sample(
            const float3 &wo_local,
            const SpectralSample &wavelengths,
            float u1, float u2, float u3) const
        {
            SpectralBSDFSample sample;

            if (wo_local.z == 0.0f)
            {
                return sample;
            }

            float hero_ior = get_ior(wavelengths.lambda[0]);
            bool entering = wo_local.z > 0.0f;
            float eta_hero = entering ? (1.0f / hero_ior) : hero_ior;

            float3 n = make_float3(0.0f, 0.0f, entering ? 1.0f : -1.0f);

            float3 h;
            if (alpha < 0.01f)
            {
                h = n;
            }
            else
            {
                float3 wo_hemi = entering ? wo_local : -wo_local;
                h = ggx_sample_vndf(wo_hemi, alpha, u1, u2);
                if (!entering)
                    h = -h;
            }

            float cos_theta_o = dot(wo_local, h);
            float F_hero = fresnel_dielectric(fabsf(cos_theta_o), eta_hero);

            bool do_reflect = (u3 < F_hero);

            if (do_reflect)
            {

                sample.wi = reflect(-wo_local, h);
                if ((entering && sample.wi.z <= 0.0f) || (!entering && sample.wi.z >= 0.0f))
                {
                    return sample;
                }

                sample.pdf = F_hero;
                sample.is_specular = (alpha < 0.01f);
                sample.is_transmission = false;

                for (int i = 0; i < NUM_WAVELENGTHS; i++)
                {
                    float ior_i = get_ior(wavelengths.lambda[i]);
                    float eta_i = entering ? (1.0f / ior_i) : ior_i;
                    float F_i = fresnel_dielectric(fabsf(cos_theta_o), eta_i);
                    sample.f.L[i] = F_i / F_hero;
                }
            }
            else
            {

                float3 refracted;
                if (!refract(wo_local, h, eta_hero, refracted))
                {

                    return sample;
                }
                sample.wi = refracted;

                if ((entering && sample.wi.z >= 0.0f) || (!entering && sample.wi.z <= 0.0f))
                {
                    return sample;
                }

                sample.pdf = 1.0f - F_hero;
                sample.is_specular = (alpha < 0.01f);
                sample.is_transmission = true;

                for (int i = 0; i < NUM_WAVELENGTHS; i++)
                {
                    float ior_i = get_ior(wavelengths.lambda[i]);
                    float eta_i = entering ? (1.0f / ior_i) : ior_i;
                    float F_i = fresnel_dielectric(fabsf(cos_theta_o), eta_i);

                    float3 refracted_i;
                    if (refract(wo_local, h, eta_i, refracted_i))
                    {

                        float dot_match = dot(normalize(refracted_i), normalize(sample.wi));

                        if (dot_match > 0.99f)
                        {
                            sample.f.L[i] = (1.0f - F_i) * eta_i * eta_i / (1.0f - F_hero);
                        }
                        else
                        {

                            sample.f.L[i] = 0.0f;
                        }
                    }
                    else
                    {
                        sample.f.L[i] = 0.0f;
                    }
                }
            }

            return sample;
        }
    };

    __host__ __device__ inline float3 thin_film_fresnel(
        float cos_theta,
        float film_ior,
        float film_thickness_nm,
        float substrate_ior,
        float wavelength_nm)
    {

        float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
        float sin_theta_film = sin_theta / film_ior;
        float cos_theta_film = sqrtf(fmaxf(0.0f, 1.0f - sin_theta_film * sin_theta_film));

        float opd = 2.0f * film_ior * film_thickness_nm * cos_theta_film;
        float phase = TWO_PI * opd / wavelength_nm;

        float r01 = (1.0f - film_ior) / (1.0f + film_ior);
        float r12 = (film_ior - substrate_ior) / (film_ior + substrate_ior);

        float r01_sq = r01 * r01;
        float r12_sq = r12 * r12;

        float cos_phase = cosf(phase);
        float R = (r01_sq + r12_sq + 2.0f * r01 * r12 * cos_phase) /
                  (1.0f + r01_sq * r12_sq + 2.0f * r01 * r12 * cos_phase);

        float hue = fmodf(wavelength_nm - 380.0f, 400.0f) / 400.0f;
        float3 color;
        if (hue < 0.33f)
        {
            color = make_float3(1.0f - hue * 3.0f, hue * 3.0f, 0.0f);
        }
        else if (hue < 0.67f)
        {
            color = make_float3(0.0f, 1.0f - (hue - 0.33f) * 3.0f, (hue - 0.33f) * 3.0f);
        }
        else
        {
            color = make_float3((hue - 0.67f) * 3.0f, 0.0f, 1.0f - (hue - 0.67f) * 3.0f);
        }

        return color * R;
    }

}
