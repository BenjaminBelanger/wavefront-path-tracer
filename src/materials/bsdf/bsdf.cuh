#pragma once

#include "../../core/math/vector.cuh"
#include "../../core/math/sampling.cuh"
#include "../../core/math/spectral.cuh"
#include "../../core/random/pcg.cuh"

namespace wpt {





struct BSDFSample {
    float3 wi;          
    float pdf;          
    float3 f;           
    bool is_specular;   
    bool is_transmission; 

    __host__ __device__ BSDFSample()
        : wi(make_float3(0.0f)), pdf(0.0f), f(make_float3(0.0f))
        , is_specular(false), is_transmission(false) {}

    __host__ __device__ bool is_valid() const {
        return pdf > 0.0f && (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f);
    }
};


struct SpectralBSDFSample {
    float3 wi;
    float pdf;
    SpectralRadiance f;
    bool is_specular;
    bool is_transmission;

    __host__ __device__ SpectralBSDFSample()
        : wi(make_float3(0.0f)), pdf(0.0f), is_specular(false), is_transmission(false) {}

    __host__ __device__ bool is_valid() const {
        return pdf > 0.0f && !f.is_black();
    }
};





enum class MaterialType : uint32_t {
    Lambert = 0,
    Metal,
    Dielectric,
    Plastic,
    Emission,
    COUNT
};





struct Material {
    MaterialType type;

    
    float3 albedo;

    
    float roughness;      
    float anisotropy;     

    
    float ior;

    
    float3 eta;           
    float3 k;             

    
    float3 emission;
    float emission_strength;

    
    int albedo_tex;
    int roughness_tex;
    int normal_tex;

    
    int spectral_data_idx;

    __host__ __device__ Material()
        : type(MaterialType::Lambert)
        , albedo(make_float3(0.8f))
        , roughness(0.5f)
        , anisotropy(0.0f)
        , ior(1.5f)
        , eta(make_float3(1.0f))
        , k(make_float3(0.0f))
        , emission(make_float3(0.0f))
        , emission_strength(0.0f)
        , albedo_tex(-1)
        , roughness_tex(-1)
        , normal_tex(-1)
        , spectral_data_idx(-1)
    {}

    __host__ __device__ bool is_emissive() const {
        return emission_strength > 0.0f &&
               (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f);
    }

    __host__ __device__ float3 get_emission() const {
        return emission * emission_strength;
    }

    
    __host__ static Material diffuse(const float3& color) {
        Material m;
        m.type = MaterialType::Lambert;
        m.albedo = color;
        return m;
    }

    __host__ static Material metal(const float3& color, float rough) {
        Material m;
        m.type = MaterialType::Metal;
        m.albedo = color;
        m.roughness = rough;
        
        m.eta = color;
        m.k = make_float3(1.0f);
        return m;
    }

    __host__ static Material glass(float ior_val, float rough = 0.0f) {
        Material m;
        m.type = MaterialType::Dielectric;
        m.albedo = make_float3(1.0f);
        m.roughness = rough;
        m.ior = ior_val;
        return m;
    }

    __host__ static Material emissive(const float3& color, float strength) {
        Material m;
        m.type = MaterialType::Emission;
        m.albedo = make_float3(0.0f);
        m.emission = color;
        m.emission_strength = strength;
        return m;
    }
};





struct ShadingContext {
    float3 position;     
    float3 normal;       
    float3 geometric_normal;  
    float3 wo;           
    float2 uv;           

    Frame frame;         

    __host__ __device__ ShadingContext() {}

    __host__ __device__ void build_frame() {
        frame = Frame(normal);
    }

    __host__ __device__ float3 to_local(const float3& w) const {
        return frame.to_local(w);
    }

    __host__ __device__ float3 to_world(const float3& w) const {
        return frame.to_world(w);
    }

    
    __host__ __device__ void ensure_correct_orientation() {
        if (dot(wo, geometric_normal) < 0.0f) {
            geometric_normal = -geometric_normal;
        }
        if (dot(wo, normal) < 0.0f) {
            normal = -normal;
            frame = Frame(normal);
        }
    }
};

} 
