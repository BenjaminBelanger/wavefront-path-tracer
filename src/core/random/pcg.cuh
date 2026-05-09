#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace wpt
{

    struct PCGState
    {
        uint64_t state;
        uint64_t inc;
    };

    class PCG32
    {
    public:
        __host__ __device__ PCG32() : state_{0, 0} {}

        __host__ __device__ PCG32(uint64_t seed, uint64_t seq = 1)
        {
            init(seed, seq);
        }

        __host__ __device__ void init(uint64_t seed, uint64_t seq = 1)
        {
            state_.state = 0;
            state_.inc = (seq << 1) | 1;
            next_uint();
            state_.state += seed;
            next_uint();
        }

        __host__ __device__ void init_from_pixel(int x, int y, int frame, int sample_idx = 0)
        {

            uint64_t pixel_hash = static_cast<uint64_t>(x) + static_cast<uint64_t>(y) * 65537;
            uint64_t seed = pixel_hash ^ (static_cast<uint64_t>(frame) * 0x9E3779B97F4A7C15ULL);
            uint64_t seq = pixel_hash ^ (static_cast<uint64_t>(sample_idx) << 32);
            init(seed, seq);
        }

        __host__ __device__ uint32_t next_uint()
        {
            uint64_t oldstate = state_.state;
            state_.state = oldstate * 6364136223846793005ULL + state_.inc;

            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18) ^ oldstate) >> 27);
            uint32_t rot = static_cast<uint32_t>(oldstate >> 59);

            return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
        }

        __host__ __device__ float next_float()
        {
            return static_cast<float>(next_uint()) * (1.0f / 4294967296.0f);
        }

        __host__ __device__ float next_float_closed()
        {
            return static_cast<float>(next_uint()) * (1.0f / 4294967295.0f);
        }

        __host__ __device__ float next_float_open()
        {
            return (static_cast<float>(next_uint()) + 0.5f) * (1.0f / 4294967296.0f);
        }

        __host__ __device__ float2 next_float2()
        {
            float u1 = next_float();
            float u2 = next_float();
            return make_float2(u1, u2);
        }

        __host__ __device__ float next_float_range(float min_val, float max_val)
        {
            return min_val + next_float() * (max_val - min_val);
        }

        __host__ __device__ uint32_t next_uint(uint32_t max_val)
        {

            uint32_t threshold = (0xFFFFFFFFu - max_val + 1) % max_val;
            uint32_t r;
            do
            {
                r = next_uint();
            } while (r < threshold);
            return r % max_val;
        }

        __host__ __device__ int next_int_range(int min_val, int max_val)
        {
            return min_val + static_cast<int>(next_uint(static_cast<uint32_t>(max_val - min_val + 1)));
        }

        __host__ __device__ void advance(int64_t delta)
        {
            uint64_t cur_mult = 6364136223846793005ULL;
            uint64_t cur_plus = state_.inc;
            uint64_t acc_mult = 1;
            uint64_t acc_plus = 0;

            uint64_t d = static_cast<uint64_t>(delta);
            while (d > 0)
            {
                if (d & 1)
                {
                    acc_mult *= cur_mult;
                    acc_plus = acc_plus * cur_mult + cur_plus;
                }
                cur_plus = (cur_mult + 1) * cur_plus;
                cur_mult *= cur_mult;
                d >>= 1;
            }
            state_.state = acc_mult * state_.state + acc_plus;
        }

        PCGState state_;
    };

    __device__ inline PCG32 &get_thread_rng(PCG32 *rng_states, int thread_id)
    {
        return rng_states[thread_id];
    }

    inline void launch_init_rng_states(PCG32 *states, int count, int frame, uint64_t base_seed, cudaStream_t stream = 0)
    {

        std::vector<PCG32> host_states(count);
        for (int i = 0; i < count; i++)
        {
            host_states[i].init(base_seed + i, static_cast<uint64_t>(frame) * count + i);
        }
        cudaMemcpyAsync(states, host_states.data(), count * sizeof(PCG32), cudaMemcpyHostToDevice, stream);
    }

    __device__ inline float2 stratified_sample_2d(PCG32 &rng, int sample_idx, int total_samples)
    {

        int grid_size = static_cast<int>(sqrtf(static_cast<float>(total_samples)));
        if (grid_size * grid_size < total_samples)
            grid_size++;

        int x_stratum = sample_idx % grid_size;
        int y_stratum = sample_idx / grid_size;

        float stratum_size = 1.0f / grid_size;
        float u = (x_stratum + rng.next_float()) * stratum_size;
        float v = (y_stratum + rng.next_float()) * stratum_size;

        return make_float2(u, v);
    }

    __device__ inline float halton_sequence(int index, int base)
    {
        float result = 0.0f;
        float f = 1.0f / static_cast<float>(base);
        int i = index;
        while (i > 0)
        {
            result += f * static_cast<float>(i % base);
            i = i / base;
            f = f / static_cast<float>(base);
        }
        return result;
    }

    __device__ inline float2 halton_2d(int index)
    {
        return make_float2(halton_sequence(index, 2), halton_sequence(index, 3));
    }

    __device__ inline float cranley_patterson_rotation(float sample, float offset)
    {
        float result = sample + offset;
        return result - floorf(result);
    }

    __device__ inline float2 cranley_patterson_rotation_2d(float2 sample, float2 offset)
    {
        return make_float2(
            cranley_patterson_rotation(sample.x, offset.x),
            cranley_patterson_rotation(sample.y, offset.y));
    }

}
