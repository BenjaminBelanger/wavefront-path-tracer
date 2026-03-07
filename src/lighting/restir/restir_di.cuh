#pragma once

#include "reservoir.cuh"
#include "../../core/memory/device_buffer.cuh"

namespace lumina
{

    struct BVHNode;
    struct TrianglePrecomputed;
    struct Light;

    class ReSTIRDI
    {
    public:
        struct Settings
        {
            int num_initial_candidates = 32;
            int num_spatial_samples = 5;
            float spatial_radius = 30.0f;
            float depth_threshold = 0.1f;
            float normal_threshold = 0.9f;
            int max_temporal_M = 20;
            bool enable_temporal = true;
            bool enable_spatial = true;
        };

        void initialize(int width, int height);

        void render(
            const float3 *positions,
            const float3 *normals,
            const float3 *albedos,
            const float *depths,
            const float2 *motion_vectors,
            const Light *lights,
            int num_lights,
            const AliasTable &light_alias,
            const BVHNode *bvh_nodes,
            const TrianglePrecomputed *precomputed,
            PCG32 *rngs);

        void reset();

        float3 *output() { return output_.data(); }
        Settings &settings() { return settings_; }

    private:
        int width_ = 0;
        int height_ = 0;
        int frame_count_ = 0;

        Settings settings_;

        ReservoirSoA current_reservoirs_;
        ReservoirSoA prev_reservoirs_;
        DeviceBuffer<float> prev_depths_;
        DeviceBuffer<float3> output_;
    };

}
