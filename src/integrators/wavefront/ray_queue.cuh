#pragma once

#include <cstdio>
#include "../../core/memory/device_buffer.cuh"

namespace wpt
{

    struct MaterialQueue
    {
        DeviceBuffer<int> indices;
        DeviceBuffer<unsigned int> count;

        void resize(size_t max_count)
        {
            indices.resize(max_count);
            count.resize(1);
        }

        void clear()
        {
            CUDA_CHECK(cudaMemset(count.data(), 0, sizeof(unsigned int)));
        }

        unsigned int get_count() const
        {
            unsigned int h_count;
            CUDA_CHECK(cudaMemcpy(&h_count, count.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
            return h_count;
        }
    };

    struct MaterialQueueView
    {
        int *__restrict__ indices;
        unsigned int *__restrict__ count;

        __device__ void push(int path_idx)
        {
            unsigned int slot = atomicAdd(count, 1);
            indices[slot] = path_idx;
        }
    };

    inline MaterialQueueView make_view(MaterialQueue &queue)
    {
        return MaterialQueueView{queue.indices.data(), queue.count.data()};
    }

    enum class QueueType
    {
        RayGenerate,
        Intersect,
        ShadeMiss,
        ShadeHit,
        ShadowRay,
        Accumulate,
        COUNT
    };

    class WorkQueues
    {
    public:
        static constexpr int NUM_MATERIAL_QUEUES = 8;

        WorkQueues() = default;

        void resize(size_t max_paths)
        {
            max_paths_ = max_paths;

            active_paths_.resize(max_paths);
            next_paths_.resize(max_paths);

            active_count_.resize(1);
            next_count_.resize(1);

            for (int i = 0; i < NUM_MATERIAL_QUEUES; i++)
            {
                material_queues_[i].resize(max_paths);
            }

            shadow_queue_.resize(max_paths);
        }

        void reset()
        {
            CUDA_CHECK(cudaMemset(active_count_.data(), 0, sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(next_count_.data(), 0, sizeof(unsigned int)));

            for (int i = 0; i < NUM_MATERIAL_QUEUES; i++)
            {
                material_queues_[i].clear();
            }

            shadow_queue_.clear();
        }

        void swap_queues()
        {
            std::swap(active_paths_, next_paths_);
            std::swap(active_count_, next_count_);
            CUDA_CHECK(cudaMemset(next_count_.data(), 0, sizeof(unsigned int)));
        }

        unsigned int get_active_count() const
        {
            unsigned int h_count;
            CUDA_CHECK(cudaMemcpy(&h_count, active_count_.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
            return h_count;
        }

        void set_active_count(unsigned int count)
        {
            CUDA_CHECK(cudaMemcpy(active_count_.data(), &count, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

        int *active_paths() { return active_paths_.data(); }
        int *next_paths() { return next_paths_.data(); }
        unsigned int *active_count_ptr() { return active_count_.data(); }
        unsigned int *next_count_ptr() { return next_count_.data(); }

        MaterialQueue &material_queue(int idx) { return material_queues_[idx]; }
        MaterialQueue &shadow_queue() { return shadow_queue_; }

        size_t max_paths() const { return max_paths_; }

    private:
        size_t max_paths_ = 0;

        DeviceBuffer<int> active_paths_;
        DeviceBuffer<int> next_paths_;
        DeviceBuffer<unsigned int> active_count_;
        DeviceBuffer<unsigned int> next_count_;

        MaterialQueue material_queues_[NUM_MATERIAL_QUEUES];
        MaterialQueue shadow_queue_;
    };

    struct WorkQueuesView
    {
        int *__restrict__ active_paths;
        int *__restrict__ next_paths;
        unsigned int *__restrict__ active_count;
        unsigned int *__restrict__ next_count;

        MaterialQueueView material_queues[WorkQueues::NUM_MATERIAL_QUEUES];
        MaterialQueueView shadow_queue;

        __device__ void push_active(int path_idx)
        {
            unsigned int slot = atomicAdd(active_count, 1);
            active_paths[slot] = path_idx;
        }

        __device__ void push_next(int path_idx)
        {
            unsigned int slot = atomicAdd(next_count, 1);
            next_paths[slot] = path_idx;
        }

        __device__ void push_material(int material_type, int path_idx)
        {
            int queue_idx = material_type % WorkQueues::NUM_MATERIAL_QUEUES;
            material_queues[queue_idx].push(path_idx);
        }

        __device__ void push_shadow(int path_idx)
        {
            shadow_queue.push(path_idx);
        }
    };

    inline WorkQueuesView make_view(WorkQueues &queues)
    {
        WorkQueuesView view;
        view.active_paths = queues.active_paths();
        view.next_paths = queues.next_paths();
        view.active_count = queues.active_count_ptr();
        view.next_count = queues.next_count_ptr();

        for (int i = 0; i < WorkQueues::NUM_MATERIAL_QUEUES; i++)
        {
            view.material_queues[i] = make_view(queues.material_queue(i));
        }
        view.shadow_queue = make_view(queues.shadow_queue());

        return view;
    }

    static __global__ void compact_paths_kernel(
        const uint32_t *__restrict__ flags,
        const int *__restrict__ input_indices,
        int *__restrict__ output_indices,
        unsigned int *__restrict__ output_count,
        int input_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= input_count)
            return;

        int path_idx = input_indices[idx];

        if (flags[path_idx] & PATH_ACTIVE)
        {
            unsigned int slot = atomicAdd(output_count, 1);
            output_indices[slot] = path_idx;
        }
    }

    static __global__ void sort_by_material_kernel(
        const int *__restrict__ material_ids,
        const int *__restrict__ input_indices,
        MaterialQueueView *__restrict__ queues,
        int input_count,
        int num_queues)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= input_count)
            return;

        int path_idx = input_indices[idx];
        int mat_id = material_ids[path_idx];

        int queue_idx = mat_id % num_queues;
        queues[queue_idx].push(path_idx);
    }

}
