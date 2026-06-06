#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace wpt
{

    template <typename T>
    class DeviceBuffer
    {
    public:
        DeviceBuffer() : data_(nullptr), size_(0), capacity_(0) {}

        explicit DeviceBuffer(size_t count) : data_(nullptr), size_(0), capacity_(0)
        {
            resize(count);
        }

        ~DeviceBuffer()
        {
            free();
        }

        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        DeviceBuffer(DeviceBuffer &&other) noexcept
            : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
        {
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }

        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
        {
            if (this != &other)
            {
                free();
                data_ = other.data_;
                size_ = other.size_;
                capacity_ = other.capacity_;
                other.data_ = nullptr;
                other.size_ = 0;
                other.capacity_ = 0;
            }
            return *this;
        }

        void resize(size_t count)
        {
            if (count > capacity_)
            {
                free();
                size_t bytes = count * sizeof(T);
                cudaError_t err = cudaMalloc(&data_, bytes);
                assert(err == cudaSuccess && "Failed to allocate device memory");
                capacity_ = count;
            }
            size_ = count;
        }

        void resize_and_clear(size_t count)
        {
            resize(count);
            cudaMemset(data_, 0, size_ * sizeof(T));
        }

        void free()
        {
            if (data_)
            {
                cudaFree(data_);
                data_ = nullptr;
                size_ = 0;
                capacity_ = 0;
            }
        }

        void upload(const T *host_data, size_t count)
        {
            resize(count);
            cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        }

        void upload(const T *host_data, size_t count, cudaStream_t stream)
        {
            resize(count);
            cudaMemcpyAsync(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice, stream);
        }

        void download(T *host_data) const
        {
            cudaMemcpy(host_data, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        }

        void download(T *host_data, cudaStream_t stream) const
        {
            cudaMemcpyAsync(host_data, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream);
        }

        void copy_from(const DeviceBuffer<T> &other)
        {
            resize(other.size_);
            cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        }

        void copy_from(const DeviceBuffer<T> &other, cudaStream_t stream)
        {
            resize(other.size_);
            cudaMemcpyAsync(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }

        T *data() { return data_; }
        const T *data() const { return data_; }
        T *get() { return data_; }
        const T *get() const { return data_; }
        size_t size() const { return size_; }
        size_t capacity() const { return capacity_; }
        size_t bytes() const { return size_ * sizeof(T); }
        bool empty() const { return size_ == 0; }

        operator T *() { return data_; }
        operator const T *() const { return data_; }

    private:
        T *data_;
        size_t size_;
        size_t capacity_;
    };

    template <typename T>
    class PinnedBuffer
    {
    public:
        PinnedBuffer() : data_(nullptr), size_(0), capacity_(0) {}

        explicit PinnedBuffer(size_t count) : data_(nullptr), size_(0), capacity_(0)
        {
            resize(count);
        }

        ~PinnedBuffer()
        {
            free();
        }

        PinnedBuffer(const PinnedBuffer &) = delete;
        PinnedBuffer &operator=(const PinnedBuffer &) = delete;

        PinnedBuffer(PinnedBuffer &&other) noexcept
            : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
        {
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }

        PinnedBuffer &operator=(PinnedBuffer &&other) noexcept
        {
            if (this != &other)
            {
                free();
                data_ = other.data_;
                size_ = other.size_;
                capacity_ = other.capacity_;
                other.data_ = nullptr;
                other.size_ = 0;
                other.capacity_ = 0;
            }
            return *this;
        }

        void resize(size_t count)
        {
            if (count > capacity_)
            {
                free();
                size_t bytes = count * sizeof(T);
                cudaError_t err = cudaMallocHost(&data_, bytes);
                assert(err == cudaSuccess && "Failed to allocate pinned memory");
                capacity_ = count;
            }
            size_ = count;
        }

        void free()
        {
            if (data_)
            {
                cudaFreeHost(data_);
                data_ = nullptr;
                size_ = 0;
                capacity_ = 0;
            }
        }

        T *data() { return data_; }
        const T *data() const { return data_; }
        size_t size() const { return size_; }
        size_t capacity() const { return capacity_; }
        bool empty() const { return size_ == 0; }

        T &operator[](size_t i) { return data_[i]; }
        const T &operator[](size_t i) const { return data_[i]; }

        operator T *() { return data_; }
        operator const T *() const { return data_; }

    private:
        T *data_;
        size_t size_;
        size_t capacity_;
    };

    template <typename T>
    class ManagedBuffer
    {
    public:
        ManagedBuffer() : data_(nullptr), size_(0), capacity_(0) {}

        explicit ManagedBuffer(size_t count) : data_(nullptr), size_(0), capacity_(0)
        {
            resize(count);
        }

        ~ManagedBuffer()
        {
            free();
        }

        ManagedBuffer(const ManagedBuffer &) = delete;
        ManagedBuffer &operator=(const ManagedBuffer &) = delete;

        ManagedBuffer(ManagedBuffer &&other) noexcept
            : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
        {
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }

        ManagedBuffer &operator=(ManagedBuffer &&other) noexcept
        {
            if (this != &other)
            {
                free();
                data_ = other.data_;
                size_ = other.size_;
                capacity_ = other.capacity_;
                other.data_ = nullptr;
                other.size_ = 0;
                other.capacity_ = 0;
            }
            return *this;
        }

        void resize(size_t count)
        {
            if (count > capacity_)
            {
                free();
                size_t bytes = count * sizeof(T);
                cudaError_t err = cudaMallocManaged(&data_, bytes);
                assert(err == cudaSuccess && "Failed to allocate managed memory");
                capacity_ = count;
            }
            size_ = count;
        }

        void free()
        {
            if (data_)
            {
                cudaFree(data_);
                data_ = nullptr;
                size_ = 0;
                capacity_ = 0;
            }
        }

        void prefetch_to_device(int device = 0, cudaStream_t stream = 0)
        {
            cudaMemPrefetchAsync(data_, size_ * sizeof(T), device, stream);
        }

        void prefetch_to_host(cudaStream_t stream = 0)
        {
            cudaMemPrefetchAsync(data_, size_ * sizeof(T), cudaCpuDeviceId, stream);
        }

        T *data() { return data_; }
        const T *data() const { return data_; }
        size_t size() const { return size_; }
        size_t capacity() const { return capacity_; }
        bool empty() const { return size_ == 0; }

        T &operator[](size_t i) { return data_[i]; }
        const T &operator[](size_t i) const { return data_[i]; }

        operator T *() { return data_; }
        operator const T *() const { return data_; }

    private:
        T *data_;
        size_t size_;
        size_t capacity_;
    };

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_LAST()                                                    \
    do                                                                       \
    {                                                                        \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

    inline dim3 compute_grid_size(size_t total_threads, int block_size = 256)
    {
        return dim3((static_cast<unsigned int>(total_threads) + block_size - 1) / block_size);
    }

    inline dim3 compute_grid_size_2d(int width, int height, int block_x = 16, int block_y = 16)
    {
        return dim3((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);
    }

}
