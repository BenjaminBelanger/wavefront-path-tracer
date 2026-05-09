#pragma once

#include "vector.cuh"

namespace wpt
{

    constexpr float LAMBDA_MIN = 380.0f;
    constexpr float LAMBDA_MAX = 780.0f;
    constexpr int NUM_WAVELENGTHS = 4;

    __host__ __device__ inline float cie_x(float lambda)
    {
        float t1 = (lambda - 442.0f) * ((lambda < 442.0f) ? 0.0624f : 0.0374f);
        float t2 = (lambda - 599.8f) * ((lambda < 599.8f) ? 0.0264f : 0.0323f);
        float t3 = (lambda - 501.1f) * ((lambda < 501.1f) ? 0.0490f : 0.0382f);
        return 0.362f * expf(-0.5f * t1 * t1) + 1.056f * expf(-0.5f * t2 * t2) - 0.065f * expf(-0.5f * t3 * t3);
    }

    __host__ __device__ inline float cie_y(float lambda)
    {
        float t1 = (lambda - 568.8f) * ((lambda < 568.8f) ? 0.0213f : 0.0247f);
        float t2 = (lambda - 530.9f) * ((lambda < 530.9f) ? 0.0613f : 0.0322f);
        return 0.821f * expf(-0.5f * t1 * t1) + 0.286f * expf(-0.5f * t2 * t2);
    }

    __host__ __device__ inline float cie_z(float lambda)
    {
        float t1 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
        float t2 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);
        return 1.217f * expf(-0.5f * t1 * t1) + 0.681f * expf(-0.5f * t2 * t2);
    }

    __host__ __device__ inline float3 wavelength_to_xyz(float lambda)
    {
        return make_float3(cie_x(lambda), cie_y(lambda), cie_z(lambda));
    }

    __host__ __device__ inline float3 xyz_to_linear_srgb(const float3 &xyz)
    {
        return make_float3(
            3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
            -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
            0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
    }

    __host__ __device__ inline float3 linear_srgb_to_xyz(const float3 &rgb)
    {
        return make_float3(
            0.4124564f * rgb.x + 0.3575761f * rgb.y + 0.1804375f * rgb.z,
            0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f * rgb.z,
            0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z);
    }

    __host__ __device__ inline float linear_to_srgb_channel(float c)
    {
        if (c <= 0.0031308f)
        {
            return 12.92f * c;
        }
        return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
    }

    __host__ __device__ inline float3 linear_to_srgb(const float3 &linear)
    {
        return make_float3(
            linear_to_srgb_channel(linear.x),
            linear_to_srgb_channel(linear.y),
            linear_to_srgb_channel(linear.z));
    }

    __host__ __device__ inline float srgb_to_linear_channel(float c)
    {
        if (c <= 0.04045f)
        {
            return c / 12.92f;
        }
        return powf((c + 0.055f) / 1.055f, 2.4f);
    }

    __host__ __device__ inline float3 srgb_to_linear(const float3 &srgb)
    {
        return make_float3(
            srgb_to_linear_channel(srgb.x),
            srgb_to_linear_channel(srgb.y),
            srgb_to_linear_channel(srgb.z));
    }

    struct SpectralSample
    {
        float lambda[NUM_WAVELENGTHS];
        float pdf;

        __host__ __device__ SpectralSample() : pdf(1.0f)
        {
            for (int i = 0; i < NUM_WAVELENGTHS; i++)
            {
                lambda[i] = 550.0f;
            }
        }
    };

    __host__ __device__ inline SpectralSample sample_hero_wavelength(float u)
    {
        SpectralSample sample;
        float range = LAMBDA_MAX - LAMBDA_MIN;

        float hero = LAMBDA_MIN + u * range;
        sample.lambda[0] = hero;

        float stride = range / NUM_WAVELENGTHS;
        for (int i = 1; i < NUM_WAVELENGTHS; i++)
        {
            float lambda = hero + i * stride;
            if (lambda > LAMBDA_MAX)
            {
                lambda -= range;
            }
            sample.lambda[i] = lambda;
        }

        sample.pdf = 1.0f / range;

        return sample;
    }

    struct SpectralRadiance
    {
        float L[NUM_WAVELENGTHS];

        __host__ __device__ SpectralRadiance()
        {
            for (int i = 0; i < NUM_WAVELENGTHS; i++)
            {
                L[i] = 0.0f;
            }
        }

        __host__ __device__ SpectralRadiance(float val)
        {
            for (int i = 0; i < NUM_WAVELENGTHS; i++)
            {
                L[i] = val;
            }
        }

        __host__ __device__ SpectralRadiance &operator+=(const SpectralRadiance &other)
        {
            for (int i = 0; i < NUM_WAVELENGTHS; i++)
            {
                L[i] += other.L[i];
            }
            return *this;
        }

        __host__ __device__ SpectralRadiance operator*(float s) const
        {
            SpectralRadiance result;
            for (int i = 0; i < NUM_WAVELENGTHS; i++)
            {
                result.L[i] = L[i] * s;
            }
            return result;
        }

        __host__ __device__ SpectralRadiance operator*(const SpectralRadiance &other) const
        {
            SpectralRadiance result;
            for (int i = 0; i < NUM_WAVELENGTHS; i++)
            {
                result.L[i] = L[i] * other.L[i];
            }
            return result;
        }

        __host__ __device__ float max_component() const
        {
            float m = L[0];
            for (int i = 1; i < NUM_WAVELENGTHS; i++)
            {
                m = fmaxf(m, L[i]);
            }
            return m;
        }

        __host__ __device__ bool is_black() const
        {
            for (int i = 0; i < NUM_WAVELENGTHS; i++)
            {
                if (L[i] > 0.0f)
                    return false;
            }
            return true;
        }
    };

    __host__ __device__ inline float3 spectral_to_xyz(const SpectralRadiance &radiance, const SpectralSample &wavelengths)
    {
        float3 xyz = make_float3(0.0f);

        float weight = 1.0f / (NUM_WAVELENGTHS * wavelengths.pdf);

        for (int i = 0; i < NUM_WAVELENGTHS; i++)
        {
            float3 xyz_bar = wavelength_to_xyz(wavelengths.lambda[i]);
            xyz += radiance.L[i] * xyz_bar * weight;
        }

        return xyz;
    }

    __host__ __device__ inline float3 spectral_to_rgb(const SpectralRadiance &radiance, const SpectralSample &wavelengths)
    {
        float3 xyz = spectral_to_xyz(radiance, wavelengths);
        return xyz_to_linear_srgb(xyz);
    }

    __host__ __device__ inline float rgb_to_spectrum_coefficient(float lambda, int channel)
    {

        const float centers[3] = {610.0f, 550.0f, 465.0f};
        const float widths[3] = {80.0f, 50.0f, 50.0f};

        float diff = lambda - centers[channel];
        float width = widths[channel];
        return expf(-0.5f * diff * diff / (width * width));
    }

    __host__ __device__ inline SpectralRadiance rgb_to_spectrum(const float3 &rgb, const SpectralSample &wavelengths)
    {
        SpectralRadiance result;

        for (int i = 0; i < NUM_WAVELENGTHS; i++)
        {
            float lambda = wavelengths.lambda[i];
            result.L[i] = rgb.x * rgb_to_spectrum_coefficient(lambda, 0) +
                          rgb.y * rgb_to_spectrum_coefficient(lambda, 1) +
                          rgb.z * rgb_to_spectrum_coefficient(lambda, 2);
            result.L[i] = fmaxf(0.0f, result.L[i]);
        }

        return result;
    }

    __host__ __device__ inline float blackbody_spectral_radiance(float lambda_nm, float temperature_K)
    {
        const float h = 6.62607015e-34f;
        const float c = 299792458.0f;
        const float k = 1.380649e-23f;

        float lambda_m = lambda_nm * 1e-9f;
        float c1 = 2.0f * h * c * c;
        float c2 = h * c / k;

        float exp_term = expf(c2 / (lambda_m * temperature_K));
        return c1 / (lambda_m * lambda_m * lambda_m * lambda_m * lambda_m * (exp_term - 1.0f));
    }

    __host__ __device__ inline float blackbody_normalized(float lambda_nm, float temperature_K)
    {

        float lambda_peak = 2.8977719e6f / temperature_K;
        float peak_radiance = blackbody_spectral_radiance(lambda_peak, temperature_K);
        return blackbody_spectral_radiance(lambda_nm, temperature_K) / peak_radiance;
    }

    __host__ __device__ inline float d65_illuminant(float lambda_nm)
    {

        return blackbody_normalized(lambda_nm, 6504.0f);
    }

}
