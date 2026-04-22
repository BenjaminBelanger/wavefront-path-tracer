#include "camera.cuh"
#include "scene.cuh"
#include "../integrators/wavefront/path_state.cuh"
#include "../integrators/wavefront/ray_queue.cuh"
#include "../core/memory/device_buffer.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// OpenGL interop is handled separately in window.cpp
// Forward declare the GL types we need
typedef unsigned int GLuint;
typedef unsigned int GLenum;
#define GL_TEXTURE_2D 0x0DE1

// Stub for cudaGraphicsResource
struct cudaGraphicsResource;

namespace wpt {

// Forward declarations of kernels
__global__ void generate_rays_kernel(
    PathStateView paths,
    const Camera camera,
    int width, int height,
    int frame_number,
    int samples_per_pixel,
    int current_sample
);

__global__ void intersect_kernel(
    PathStateView paths,
    HitInfoView hits,
    const BVHNode* bvh_nodes,
    const Triangle* triangles,
    const int* active_paths,
    int active_count
);

__global__ void shade_miss_kernel(
    PathStateView paths,
    const HitInfoView hits,
    const int* active_paths,
    int active_count
);

__global__ void shade_surface_kernel(
    PathStateView paths,
    const HitInfoView hits,
    const Material* materials,
    const int* active_paths,
    unsigned int* next_count,
    int* next_paths,
    int active_count,
    int max_depth
);

__global__ void accumulate_kernel(
    const PathStateView paths,
    float4* accumulation_buffer,
    int* sample_count,
    int width, int height,
    int num_paths
);

__global__ void tonemap_kernel(
    const float4* accumulation_buffer,
    uchar4* display_buffer,
    int width, int height,
    float exposure
);

// =============================================================================
// Renderer Class
// =============================================================================

class Renderer {
public:
    struct Settings {
        int max_depth = 8;
        int samples_per_pixel = 1;  // Samples per frame
        float exposure = 1.0f;
        bool progressive = true;
        bool use_spectral = false;
    };

    Renderer() = default;
    ~Renderer() { cleanup(); }

    void initialize(int width, int height) {
        width_ = width;
        height_ = height;
        num_pixels_ = width * height;

        // Allocate path state
        path_state_.resize(num_pixels_);
        hit_info_.resize(num_pixels_);

        // Allocate work queues
        work_queues_.resize(num_pixels_);

        // Allocate framebuffer
        accumulation_buffer_.resize(num_pixels_);
        sample_count_.resize(num_pixels_);
        display_buffer_.resize(num_pixels_);

        reset_accumulation();

        printf("Wavefront Path Tracer Renderer initialized: %dx%d (%d pixels)\n", width, height, num_pixels_);
    }

    void cleanup() {
        // GL interop cleanup disabled
    }

    void reset_accumulation() {
        CUDA_CHECK(cudaMemset(accumulation_buffer_.data(), 0, num_pixels_ * sizeof(float4)));
        CUDA_CHECK(cudaMemset(sample_count_.data(), 0, num_pixels_ * sizeof(int)));
        frame_number_ = 0;
        total_samples_ = 0;
    }

    void set_scene(Scene* scene) {
        scene_ = scene;
        reset_accumulation();
    }

    void set_camera(const Camera& camera) {
        if (camera_.position.x != camera.position.x ||
            camera_.position.y != camera.position.y ||
            camera_.position.z != camera.position.z ||
            camera_.target.x != camera.target.x ||
            camera_.target.y != camera.target.y ||
            camera_.target.z != camera.target.z) {
            camera_ = camera;
            reset_accumulation();
        }
    }

    void render_frame() {
        if (!scene_) return;

        for (int sample = 0; sample < settings_.samples_per_pixel; sample++) {
            render_sample(sample);
        }

        frame_number_++;
        total_samples_ += settings_.samples_per_pixel;
    }

    void render_sample(int sample_index) {
        PathStateView paths = make_view(path_state_);
        HitInfoView hits = make_view(hit_info_);

        // Generate primary rays
        dim3 block_2d(16, 16);
        dim3 grid_2d((width_ + 15) / 16, (height_ + 15) / 16);

        generate_rays_kernel<<<grid_2d, block_2d>>>(
            paths, camera_, width_, height_,
            frame_number_, settings_.samples_per_pixel, sample_index
        );
        CUDA_CHECK_LAST();

        // Initialize active paths (all paths active initially)
        work_queues_.reset();

        // Create initial active list (all pixels)
        DeviceBuffer<int> initial_active(num_pixels_);
        thrust_sequence(initial_active.data(), num_pixels_);
        CUDA_CHECK(cudaMemcpy(work_queues_.active_paths(), initial_active.data(),
                              num_pixels_ * sizeof(int), cudaMemcpyDeviceToDevice));
        work_queues_.set_active_count(num_pixels_);

        // Path tracing loop
        int depth = 0;
        while (depth < settings_.max_depth) {
            unsigned int active_count = work_queues_.get_active_count();
            if (active_count == 0) break;

            int block_size = 256;
            int grid_size = (active_count + block_size - 1) / block_size;

            // Intersection
            intersect_kernel<<<grid_size, block_size>>>(
                paths, hits,
                scene_->bvh_nodes(),
                scene_->triangles(),
                work_queues_.active_paths(),
                active_count
            );
            CUDA_CHECK_LAST();

            // Shade misses
            shade_miss_kernel<<<grid_size, block_size>>>(
                paths, hits,
                work_queues_.active_paths(),
                active_count
            );
            CUDA_CHECK_LAST();

            // Reset next queue
            CUDA_CHECK(cudaMemset(work_queues_.next_count_ptr(), 0, sizeof(unsigned int)));

            // Shade surfaces
            shade_surface_kernel<<<grid_size, block_size>>>(
                paths, hits,
                scene_->materials(),
                work_queues_.active_paths(),
                work_queues_.next_count_ptr(),
                work_queues_.next_paths(),
                active_count,
                settings_.max_depth
            );
            CUDA_CHECK_LAST();

            // Swap queues
            work_queues_.swap_queues();
            depth++;
        }

        // Accumulate results
        int block_size = 256;
        int grid_size = (num_pixels_ + block_size - 1) / block_size;

        accumulate_kernel<<<grid_size, block_size>>>(
            paths,
            accumulation_buffer_.data(),
            sample_count_.data(),
            width_, height_,
            num_pixels_
        );
        CUDA_CHECK_LAST();
    }

    void tonemap_to_display() {
        dim3 block_2d(16, 16);
        dim3 grid_2d((width_ + 15) / 16, (height_ + 15) / 16);

        tonemap_kernel<<<grid_2d, block_2d>>>(
            accumulation_buffer_.data(),
            display_buffer_.data(),
            width_, height_,
            settings_.exposure
        );
        CUDA_CHECK_LAST();
    }

    // Register OpenGL texture for direct rendering
    // Note: GL interop disabled for initial build - use download_display() instead
    void register_gl_texture(unsigned int texture_id) {
        (void)texture_id;
        // GL interop requires OpenGL context - disabled for now
    }

    // Copy display buffer to OpenGL texture
    void copy_to_gl_texture() {
        // GL interop disabled - use download_display() instead
    }

    // Download display buffer to host
    void download_display(uchar4* host_buffer) {
        display_buffer_.download(host_buffer);
    }

    // Accessors
    int width() const { return width_; }
    int height() const { return height_; }
    int frame_number() const { return frame_number_; }
    int total_samples() const { return total_samples_; }

    Settings& settings() { return settings_; }
    const Settings& settings() const { return settings_; }

private:
    // Helper to initialize sequence 0, 1, 2, ...
    static void thrust_sequence(int* data, int count) {
        std::vector<int> seq(count);
        for (int i = 0; i < count; i++) seq[i] = i;
        CUDA_CHECK(cudaMemcpy(data, seq.data(), count * sizeof(int), cudaMemcpyHostToDevice));
    }

    int width_ = 0;
    int height_ = 0;
    int num_pixels_ = 0;

    int frame_number_ = 0;
    int total_samples_ = 0;

    Settings settings_;
    Camera camera_;
    Scene* scene_ = nullptr;

    // Path tracing state
    PathStateSoA path_state_;
    HitInfoSoA hit_info_;
    WorkQueues work_queues_;

    // Framebuffer
    DeviceBuffer<float4> accumulation_buffer_;
    DeviceBuffer<int> sample_count_;
    DeviceBuffer<uchar4> display_buffer_;

    // OpenGL interop (disabled for initial build)
    void* gl_texture_resource_ = nullptr;
};

} // namespace wpt
