#pragma once

#include "../../core/math/vector.cuh"

namespace lumina {

// =============================================================================
// Sellmeier Equation for Wavelength-Dependent Refractive Index
// =============================================================================

// Sellmeier equation: n^2 - 1 = sum_i( B_i * lambda^2 / (lambda^2 - C_i) )
// Where lambda is in micrometers, B_i and C_i are material-specific coefficients

struct SellmeierCoeffs {
    float B1, B2, B3;  // Sellmeier B coefficients
    float C1, C2, C3;  // Sellmeier C coefficients (in um^2)

    __host__ __device__ SellmeierCoeffs()
        : B1(0), B2(0), B3(0), C1(0), C2(0), C3(0) {}

    __host__ __device__ SellmeierCoeffs(float b1, float b2, float b3, float c1, float c2, float c3)
        : B1(b1), B2(b2), B3(b3), C1(c1), C2(c2), C3(c3) {}

    // Compute refractive index at wavelength (in nanometers)
    __host__ __device__ float ior(float wavelength_nm) const {
        float lambda_um = wavelength_nm * 0.001f;  // Convert nm to um
        float lambda2 = lambda_um * lambda_um;

        float n2_minus_1 = B1 * lambda2 / (lambda2 - C1) +
                          B2 * lambda2 / (lambda2 - C2) +
                          B3 * lambda2 / (lambda2 - C3);

        return sqrtf(1.0f + n2_minus_1);
    }

    // Compute group velocity dispersion (derivative of IOR)
    __host__ __device__ float dispersion(float wavelength_nm) const {
        // Numerical derivative
        float delta = 1.0f;  // 1 nm
        return (ior(wavelength_nm + delta) - ior(wavelength_nm - delta)) / (2.0f * delta);
    }

    // Abbe number (measure of dispersion)
    __host__ __device__ float abbe_number() const {
        float nD = ior(589.3f);   // Sodium D-line
        float nF = ior(486.1f);   // Hydrogen F-line (blue)
        float nC = ior(656.3f);   // Hydrogen C-line (red)
        return (nD - 1.0f) / (nF - nC);
    }

    // ==========================================================================
    // Preset Materials
    // ==========================================================================

    // BK7 optical glass (common crown glass)
    __host__ __device__ static SellmeierCoeffs bk7_glass() {
        return SellmeierCoeffs(
            1.03961212f, 0.231792344f, 1.01046945f,
            0.00600069867f, 0.0200179144f, 103.560653f
        );
    }

    // Fused silica (SiO2)
    __host__ __device__ static SellmeierCoeffs fused_silica() {
        return SellmeierCoeffs(
            0.6961663f, 0.4079426f, 0.8974794f,
            0.0684043f * 0.0684043f, 0.1162414f * 0.1162414f, 9.896161f * 9.896161f
        );
    }

    // SF11 glass (dense flint glass, high dispersion)
    __host__ __device__ static SellmeierCoeffs sf11_glass() {
        return SellmeierCoeffs(
            1.73759695f, 0.313747346f, 1.89878101f,
            0.013188707f, 0.0623068142f, 155.23629f
        );
    }

    // Diamond
    __host__ __device__ static SellmeierCoeffs diamond() {
        return SellmeierCoeffs(
            0.3306f, 4.3356f, 0.0f,
            0.1750f * 0.1750f, 0.1060f * 0.1060f, 0.0f
        );
    }

    // Sapphire (Al2O3)
    __host__ __device__ static SellmeierCoeffs sapphire() {
        return SellmeierCoeffs(
            1.4313493f, 0.65054713f, 5.3414021f,
            0.0726631f * 0.0726631f, 0.1193242f * 0.1193242f, 18.028251f * 18.028251f
        );
    }

    // Water at 20C
    __host__ __device__ static SellmeierCoeffs water() {
        return SellmeierCoeffs(
            0.5684027565f, 0.1726177391f, 0.02086189578f,
            0.005101829712f, 0.01821153936f, 0.02620722293f
        );
    }

    // Calcite (CaCO3) - ordinary ray
    __host__ __device__ static SellmeierCoeffs calcite_ordinary() {
        return SellmeierCoeffs(
            0.73358749f, 0.96464345f, 1.82831454f,
            0.0142382647f, 0.0120470032f, 120.0f
        );
    }
};

// =============================================================================
// Cauchy Equation (simpler approximation)
// =============================================================================

// Cauchy equation: n = A + B/lambda^2 + C/lambda^4
// Where lambda is in micrometers

struct CauchyCoeffs {
    float A, B, C;

    __host__ __device__ CauchyCoeffs() : A(1.5f), B(0.0f), C(0.0f) {}
    __host__ __device__ CauchyCoeffs(float a, float b, float c) : A(a), B(b), C(c) {}

    __host__ __device__ float ior(float wavelength_nm) const {
        float lambda_um = wavelength_nm * 0.001f;
        float lambda2 = lambda_um * lambda_um;
        float lambda4 = lambda2 * lambda2;
        return A + B / lambda2 + C / lambda4;
    }

    // Convert to approximate Sellmeier (for consistency)
    __host__ __device__ SellmeierCoeffs to_sellmeier() const {
        // This is an approximation
        return SellmeierCoeffs(
            A * A - 1.0f, 0.0f, 0.0f,
            B / (A * A - 1.0f), 0.0f, 0.0f
        );
    }
};

// =============================================================================
// Complex Refractive Index for Metals
// =============================================================================

struct ComplexIOR {
    float n;  // Real part (refractive index)
    float k;  // Imaginary part (extinction coefficient)

    __host__ __device__ ComplexIOR() : n(1.0f), k(0.0f) {}
    __host__ __device__ ComplexIOR(float n_val, float k_val) : n(n_val), k(k_val) {}

    // Fresnel reflectance at normal incidence for conductor
    __host__ __device__ float F0() const {
        float num = (n - 1.0f) * (n - 1.0f) + k * k;
        float denom = (n + 1.0f) * (n + 1.0f) + k * k;
        return num / denom;
    }

    // Full Fresnel equation for conductors
    __host__ __device__ float fresnel(float cos_theta) const {
        float cos2 = cos_theta * cos_theta;
        float sin2 = 1.0f - cos2;

        float eta2 = n * n - k * k;
        float etak2 = 4.0f * n * n * k * k;

        float t0 = eta2 - sin2;
        float a2plusb2 = sqrtf(t0 * t0 + etak2);
        float t1 = a2plusb2 + cos2;
        float a = sqrtf(0.5f * (a2plusb2 + t0));
        float t2 = 2.0f * a * cos_theta;

        float Rs = (t1 - t2) / (t1 + t2);

        float t3 = cos2 * a2plusb2 + sin2 * sin2;
        float t4 = t2 * sin2;

        float Rp = Rs * (t3 - t4) / (t3 + t4);

        return 0.5f * (Rs + Rp);
    }
};

// Spectral complex IOR data for common metals (at specific wavelengths)
struct MetalSpectralIOR {
    // Wavelength in nm
    float wavelengths[8];
    ComplexIOR ior[8];
    int count;

    __host__ __device__ MetalSpectralIOR() : count(0) {}

    // Interpolate IOR at arbitrary wavelength
    __host__ __device__ ComplexIOR interpolate(float wavelength_nm) const {
        if (count == 0) return ComplexIOR(1.0f, 0.0f);
        if (count == 1) return ior[0];

        // Find bracketing wavelengths
        if (wavelength_nm <= wavelengths[0]) return ior[0];
        if (wavelength_nm >= wavelengths[count - 1]) return ior[count - 1];

        for (int i = 0; i < count - 1; i++) {
            if (wavelength_nm >= wavelengths[i] && wavelength_nm <= wavelengths[i + 1]) {
                float t = (wavelength_nm - wavelengths[i]) / (wavelengths[i + 1] - wavelengths[i]);
                return ComplexIOR(
                    ior[i].n + t * (ior[i + 1].n - ior[i].n),
                    ior[i].k + t * (ior[i + 1].k - ior[i].k)
                );
            }
        }

        return ior[0];
    }

    // Gold
    __host__ static MetalSpectralIOR gold() {
        MetalSpectralIOR m;
        m.count = 6;
        m.wavelengths[0] = 400; m.ior[0] = ComplexIOR(1.658f, 1.956f);
        m.wavelengths[1] = 500; m.ior[1] = ComplexIOR(0.846f, 1.902f);
        m.wavelengths[2] = 550; m.ior[2] = ComplexIOR(0.370f, 2.610f);
        m.wavelengths[3] = 600; m.ior[3] = ComplexIOR(0.166f, 3.150f);
        m.wavelengths[4] = 650; m.ior[4] = ComplexIOR(0.160f, 3.800f);
        m.wavelengths[5] = 700; m.ior[5] = ComplexIOR(0.164f, 4.422f);
        return m;
    }

    // Silver
    __host__ static MetalSpectralIOR silver() {
        MetalSpectralIOR m;
        m.count = 6;
        m.wavelengths[0] = 400; m.ior[0] = ComplexIOR(0.173f, 1.950f);
        m.wavelengths[1] = 500; m.ior[1] = ComplexIOR(0.050f, 2.910f);
        m.wavelengths[2] = 550; m.ior[2] = ComplexIOR(0.040f, 3.429f);
        m.wavelengths[3] = 600; m.ior[3] = ComplexIOR(0.040f, 3.943f);
        m.wavelengths[4] = 650; m.ior[4] = ComplexIOR(0.040f, 4.435f);
        m.wavelengths[5] = 700; m.ior[5] = ComplexIOR(0.040f, 4.917f);
        return m;
    }

    // Copper
    __host__ static MetalSpectralIOR copper() {
        MetalSpectralIOR m;
        m.count = 6;
        m.wavelengths[0] = 400; m.ior[0] = ComplexIOR(1.130f, 2.570f);
        m.wavelengths[1] = 500; m.ior[1] = ComplexIOR(0.988f, 2.440f);
        m.wavelengths[2] = 550; m.ior[2] = ComplexIOR(0.893f, 2.420f);
        m.wavelengths[3] = 600; m.ior[3] = ComplexIOR(0.214f, 3.670f);
        m.wavelengths[4] = 650; m.ior[4] = ComplexIOR(0.214f, 3.860f);
        m.wavelengths[5] = 700; m.ior[5] = ComplexIOR(0.213f, 4.050f);
        return m;
    }

    // Aluminum
    __host__ static MetalSpectralIOR aluminum() {
        MetalSpectralIOR m;
        m.count = 6;
        m.wavelengths[0] = 400; m.ior[0] = ComplexIOR(0.49f, 4.86f);
        m.wavelengths[1] = 500; m.ior[1] = ComplexIOR(0.62f, 5.90f);
        m.wavelengths[2] = 550; m.ior[2] = ComplexIOR(0.77f, 6.42f);
        m.wavelengths[3] = 600; m.ior[3] = ComplexIOR(0.97f, 6.92f);
        m.wavelengths[4] = 650; m.ior[4] = ComplexIOR(1.24f, 7.44f);
        m.wavelengths[5] = 700; m.ior[5] = ComplexIOR(1.55f, 7.98f);
        return m;
    }
};

} // namespace lumina
