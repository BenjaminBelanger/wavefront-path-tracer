#pragma once

#include "../../core/math/vector.cuh"
#include "../../core/math/sampling.cuh"
#include "../../core/memory/device_buffer.cuh"
#include "../../core/random/pcg.cuh"

namespace wpt {








struct LightSample {
    int light_idx;          
    float3 point_on_light;  
    float3 light_normal;    
    float3 emission;        
    float pdf;              

    __host__ __device__ LightSample()
        : light_idx(-1)
        , point_on_light(make_float3(0.0f))
        , light_normal(make_float3(0.0f, 1.0f, 0.0f))
        , emission(make_float3(0.0f))
        , pdf(0.0f)
    {}

    __host__ __device__ bool is_valid() const {
        return light_idx >= 0 && pdf > 0.0f;
    }
};


struct Reservoir {
    LightSample y;          
    float w_sum;            
    int M;                  
    float W;                

    __host__ __device__ Reservoir()
        : w_sum(0.0f), M(0), W(0.0f) {}

    
    __host__ __device__ void reset() {
        y = LightSample();
        w_sum = 0.0f;
        M = 0;
        W = 0.0f;
    }

    
    
    __device__ bool update(const LightSample& x, float w, PCG32& rng) {
        w_sum += w;
        M += 1;

        if (rng.next_float() < w / w_sum) {
            y = x;
            return true;
        }
        return false;
    }

    
    __device__ void merge(const Reservoir& other, float p_hat, PCG32& rng) {
        if (other.M == 0) return;

        float w = p_hat * other.W * other.M;
        update(other.y, w, rng);
    }

    
    __device__ void finalize(float p_hat) {
        if (p_hat > 0.0f && M > 0) {
            W = w_sum / (M * p_hat);
        } else {
            W = 0.0f;
        }
    }

    
    __host__ __device__ bool is_valid() const {
        return M > 0 && y.is_valid();
    }
};





struct ReservoirSoA {
    
    DeviceBuffer<int> light_idx;
    DeviceBuffer<float> point_x, point_y, point_z;
    DeviceBuffer<float> normal_x, normal_y, normal_z;
    DeviceBuffer<float> emission_x, emission_y, emission_z;
    DeviceBuffer<float> pdf;

    
    DeviceBuffer<float> w_sum;
    DeviceBuffer<int> M;
    DeviceBuffer<float> W;

    void resize(size_t count) {
        light_idx.resize(count);
        point_x.resize(count);
        point_y.resize(count);
        point_z.resize(count);
        normal_x.resize(count);
        normal_y.resize(count);
        normal_z.resize(count);
        emission_x.resize(count);
        emission_y.resize(count);
        emission_z.resize(count);
        pdf.resize(count);
        w_sum.resize(count);
        M.resize(count);
        W.resize(count);
    }

    void clear() {
        cudaMemset(light_idx.data(), 0xFF, light_idx.size() * sizeof(int));  
        cudaMemset(w_sum.data(), 0, w_sum.size() * sizeof(float));
        cudaMemset(M.data(), 0, M.size() * sizeof(int));
        cudaMemset(W.data(), 0, W.size() * sizeof(float));
    }

    size_t size() const { return light_idx.size(); }
};

struct ReservoirView {
    int* __restrict__ light_idx;
    float* __restrict__ point_x;
    float* __restrict__ point_y;
    float* __restrict__ point_z;
    float* __restrict__ normal_x;
    float* __restrict__ normal_y;
    float* __restrict__ normal_z;
    float* __restrict__ emission_x;
    float* __restrict__ emission_y;
    float* __restrict__ emission_z;
    float* __restrict__ pdf;
    float* __restrict__ w_sum;
    int* __restrict__ M;
    float* __restrict__ W;

    __device__ Reservoir get(int idx) const {
        Reservoir r;
        r.y.light_idx = light_idx[idx];
        r.y.point_on_light = make_float3(point_x[idx], point_y[idx], point_z[idx]);
        r.y.light_normal = make_float3(normal_x[idx], normal_y[idx], normal_z[idx]);
        r.y.emission = make_float3(emission_x[idx], emission_y[idx], emission_z[idx]);
        r.y.pdf = pdf[idx];
        r.w_sum = w_sum[idx];
        r.M = M[idx];
        r.W = W[idx];
        return r;
    }

    __device__ void set(int idx, const Reservoir& r) {
        light_idx[idx] = r.y.light_idx;
        point_x[idx] = r.y.point_on_light.x;
        point_y[idx] = r.y.point_on_light.y;
        point_z[idx] = r.y.point_on_light.z;
        normal_x[idx] = r.y.light_normal.x;
        normal_y[idx] = r.y.light_normal.y;
        normal_z[idx] = r.y.light_normal.z;
        emission_x[idx] = r.y.emission.x;
        emission_y[idx] = r.y.emission.y;
        emission_z[idx] = r.y.emission.z;
        pdf[idx] = r.y.pdf;
        w_sum[idx] = r.w_sum;
        M[idx] = r.M;
        W[idx] = r.W;
    }

    __device__ void reset(int idx) {
        light_idx[idx] = -1;
        w_sum[idx] = 0.0f;
        M[idx] = 0;
        W[idx] = 0.0f;
    }
};

inline ReservoirView make_view(ReservoirSoA& soa) {
    ReservoirView v;
    v.light_idx = soa.light_idx.data();
    v.point_x = soa.point_x.data();
    v.point_y = soa.point_y.data();
    v.point_z = soa.point_z.data();
    v.normal_x = soa.normal_x.data();
    v.normal_y = soa.normal_y.data();
    v.normal_z = soa.normal_z.data();
    v.emission_x = soa.emission_x.data();
    v.emission_y = soa.emission_y.data();
    v.emission_z = soa.emission_z.data();
    v.pdf = soa.pdf.data();
    v.w_sum = soa.w_sum.data();
    v.M = soa.M.data();
    v.W = soa.W.data();
    return v;
}






__device__ inline float compute_target_pdf(
    const LightSample& sample,
    const float3& shading_point,
    const float3& shading_normal,
    const float3& albedo
) {
    if (!sample.is_valid()) return 0.0f;

    
    float3 to_light = sample.point_on_light - shading_point;
    float dist_sq = length_squared(to_light);
    float dist = sqrtf(dist_sq);
    float3 wi = to_light / dist;

    
    float cos_theta_i = dot(shading_normal, wi);
    float cos_theta_o = dot(sample.light_normal, -wi);

    if (cos_theta_i <= 0.0f || cos_theta_o <= 0.0f) {
        return 0.0f;
    }

    
    
    
    float3 L = sample.emission;
    float3 brdf = albedo * INV_PI;
    float G = cos_theta_i * cos_theta_o / dist_sq;

    float3 contrib = L * brdf * G;
    return luminance(contrib);
}





struct AliasEntry {
    float prob;         
    int alias;          
    float pdf_original; 
};

struct AliasTable {
    DeviceBuffer<AliasEntry> entries;
    int count;

    void build(const float* weights, int n);
    int count_entries() const { return count; }
};

struct AliasTableView {
    const AliasEntry* __restrict__ entries;
    int count;

    
    __device__ int sample(float u1, float u2, float& pdf) const {
        int idx = min_int(int(u1 * count), count - 1);
        const AliasEntry& entry = entries[idx];

        int selected;
        if (u2 < entry.prob) {
            selected = idx;
        } else {
            selected = entry.alias;
        }

        pdf = entries[selected].pdf_original;
        return selected;
    }

    __device__ float get_pdf(int idx) const {
        return entries[idx].pdf_original;
    }
};

inline AliasTableView make_view(const AliasTable& table) {
    return AliasTableView{table.entries.data(), table.count};
}





struct PathSample {
    
    float3 reconnection_pos;
    float3 reconnection_normal;

    
    float3 Lo;              

    
    int path_length;
    bool is_valid;

    __host__ __device__ PathSample()
        : reconnection_pos(make_float3(0.0f))
        , reconnection_normal(make_float3(0.0f, 1.0f, 0.0f))
        , Lo(make_float3(0.0f))
        , path_length(0)
        , is_valid(false)
    {}
};

struct PathReservoir {
    PathSample y;
    float w_sum;
    int M;
    float W;

    __host__ __device__ PathReservoir()
        : w_sum(0.0f), M(0), W(0.0f) {}

    __device__ void reset() {
        y = PathSample();
        w_sum = 0.0f;
        M = 0;
        W = 0.0f;
    }

    __device__ bool update(const PathSample& x, float w, PCG32& rng) {
        w_sum += w;
        M += 1;

        if (rng.next_float() < w / w_sum) {
            y = x;
            return true;
        }
        return false;
    }

    __device__ void finalize(float p_hat) {
        if (p_hat > 0.0f && M > 0) {
            W = w_sum / (M * p_hat);
        } else {
            W = 0.0f;
        }
    }
};

} 
