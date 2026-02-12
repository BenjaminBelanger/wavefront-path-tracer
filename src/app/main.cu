#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

// GLFW_INCLUDE_NONE prevents GLFW from including platform OpenGL headers
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Lumina includes
#include "camera.cuh"
#include "scene.cuh"
#include "../core/memory/device_buffer.cuh"
#include "../integrators/wavefront/path_state.cuh"
#include "../integrators/wavefront/ray_queue.cuh"
#include "../geometry/bvh/bvh.cuh"
#include "../materials/bsdf/lambert.cuh"
#include "../materials/bsdf/ggx.cuh"

namespace lumina {

// External wrapper function declarations from kernels.cu
void launch_generate_rays(
    PathStateView paths,
    const Camera& camera,
    int width, int height,
    int frame_number,
    int samples_per_pixel,
    int current_sample
);

void launch_intersect(
    PathStateView paths,
    HitInfoView hits,
    const BVHNode* bvh_nodes,
    const Triangle* triangles,
    const int* active_paths,
    int active_count
);

void launch_shade_miss(
    PathStateView paths,
    const HitInfoView& hits,
    const int* active_paths,
    int active_count
);

void launch_shade_surface(
    PathStateView paths,
    const HitInfoView& hits,
    const Material* materials,
    const int* active_paths,
    unsigned int* next_count,
    int* next_paths,
    int active_count,
    int max_depth
);

void launch_accumulate(
    const PathStateView& paths,
    float4* accumulation_buffer,
    int* sample_count,
    int width, int height,
    int num_paths
);

void launch_tonemap(
    const float4* accumulation_buffer,
    uchar4* display_buffer,
    int width, int height,
    float exposure
);

} // namespace lumina

using namespace lumina;

// =============================================================================
// Interactive Renderer
// =============================================================================

class InteractiveRenderer {
public:
    InteractiveRenderer(int width, int height)
        : width_(width), height_(height), num_pixels_(width * height),
          frame_number_(0), total_samples_(0), exposure_(0.06f), max_depth_(8) {

        // Allocate path state
        path_state_.resize(num_pixels_);
        hit_info_.resize(num_pixels_);

        // Allocate work queues
        work_queues_.resize(num_pixels_);

        // Allocate framebuffer
        accumulation_buffer_.resize(num_pixels_);
        sample_count_.resize(num_pixels_);
        display_buffer_.resize(num_pixels_);
        host_display_buffer_.resize(num_pixels_);

        // Allocate initial active path buffer
        initial_active_.resize(num_pixels_);

        reset_accumulation();

        printf("Renderer initialized: %dx%d (%d pixels)\n", width, height, num_pixels_);
    }

    void reset_accumulation() {
        CUDA_CHECK(cudaMemset(accumulation_buffer_.data(), 0, num_pixels_ * sizeof(float4)));
        CUDA_CHECK(cudaMemset(sample_count_.data(), 0, num_pixels_ * sizeof(int)));
        frame_number_ = 0;
        total_samples_ = 0;
    }

    void render_frame(const Camera& camera, const Scene& scene) {
        PathStateView paths = make_view(path_state_);
        HitInfoView hits = make_view(hit_info_);

        // Generate primary rays
        launch_generate_rays(paths, camera, width_, height_, frame_number_, 1, 0);
        CUDA_CHECK_LAST();

        // Initialize active paths
        work_queues_.reset();

        // Initialize sequence on GPU
        init_sequence();
        CUDA_CHECK(cudaMemcpy(work_queues_.active_paths(), initial_active_.data(),
                              num_pixels_ * sizeof(int), cudaMemcpyDeviceToDevice));
        work_queues_.set_active_count(num_pixels_);

        // Path tracing loop
        int depth = 0;
        while (depth < max_depth_) {
            unsigned int active_count = work_queues_.get_active_count();
            if (active_count == 0) break;

            // Intersection
            launch_intersect(paths, hits, scene.bvh_nodes(), scene.triangles(),
                           work_queues_.active_paths(), active_count);
            CUDA_CHECK_LAST();

            // Shade misses
            launch_shade_miss(paths, hits, work_queues_.active_paths(), active_count);
            CUDA_CHECK_LAST();

            // Reset next queue
            CUDA_CHECK(cudaMemset(work_queues_.next_count_ptr(), 0, sizeof(unsigned int)));

            // Shade surfaces
            launch_shade_surface(paths, hits, scene.materials(),
                               work_queues_.active_paths(),
                               work_queues_.next_count_ptr(),
                               work_queues_.next_paths(),
                               active_count, max_depth_);
            CUDA_CHECK_LAST();

            // Swap queues
            work_queues_.swap_queues();
            depth++;
        }

        // Accumulate results
        launch_accumulate(paths, accumulation_buffer_.data(), sample_count_.data(),
                         width_, height_, num_pixels_);
        CUDA_CHECK_LAST();

        frame_number_++;
        total_samples_++;
    }

    void tonemap() {
        launch_tonemap(accumulation_buffer_.data(), display_buffer_.data(),
                      width_, height_, exposure_);
        CUDA_CHECK_LAST();
    }

    void download_display(uchar4* host_buffer) {
        CUDA_CHECK(cudaMemcpy(host_buffer, display_buffer_.data(),
                              num_pixels_ * sizeof(uchar4), cudaMemcpyDeviceToHost));
    }

    int total_samples() const { return total_samples_; }
    float& exposure() { return exposure_; }

private:
    void init_sequence() {
        std::vector<int> seq(num_pixels_);
        for (int i = 0; i < num_pixels_; i++) seq[i] = i;
        CUDA_CHECK(cudaMemcpy(initial_active_.data(), seq.data(),
                              num_pixels_ * sizeof(int), cudaMemcpyHostToDevice));
    }

    int width_, height_, num_pixels_;
    int frame_number_;
    int total_samples_;
    float exposure_;
    int max_depth_;

    PathStateSoA path_state_;
    HitInfoSoA hit_info_;
    WorkQueues work_queues_;

    DeviceBuffer<float4> accumulation_buffer_;
    DeviceBuffer<int> sample_count_;
    DeviceBuffer<uchar4> display_buffer_;
    DeviceBuffer<int> initial_active_;
    std::vector<uchar4> host_display_buffer_;
};

// =============================================================================
// Create Demo Scene
// =============================================================================

Scene create_demo_scene() {
    Scene scene;

    // Materials
    int white_diffuse = scene.add_material(Material::diffuse(::make_float3(0.73f, 0.73f, 0.73f)));
    int red_diffuse = scene.add_material(Material::diffuse(::make_float3(0.65f, 0.05f, 0.05f)));
    int green_diffuse = scene.add_material(Material::diffuse(::make_float3(0.12f, 0.45f, 0.15f)));
    int light_mat = scene.add_material(Material::emissive(::make_float3(1.0f, 0.95f, 0.8f), 16.0f));
    int mirror_mat = scene.add_material(Material::metal(::make_float3(0.95f, 0.95f, 0.95f), 0.02f));
    int chrome_mat = scene.add_material(Material::metal(::make_float3(0.92f, 0.94f, 0.98f), 0.05f));
    int gold_mat = scene.add_material(Material::metal(::make_float3(1.0f, 0.85f, 0.57f), 0.1f));
    int coral_diffuse = scene.add_material(Material::diffuse(::make_float3(0.86f, 0.42f, 0.31f)));
    int obsidian_metal = scene.add_material(Material::metal(::make_float3(0.14f, 0.16f, 0.2f), 0.35f));
    int velvet_violet = scene.add_material(Material::diffuse(::make_float3(0.44f, 0.24f, 0.64f)));
    int mango_diffuse = scene.add_material(Material::diffuse(::make_float3(0.94f, 0.62f, 0.2f)));
    int rose_gold_metal = scene.add_material(Material::metal(::make_float3(0.96f, 0.67f, 0.56f), 0.16f));
    int pearl_metal = scene.add_material(Material::metal(::make_float3(0.92f, 0.9f, 0.98f), 0.12f));
    int neon_pink_light = scene.add_material(Material::emissive(::make_float3(1.0f, 0.2f, 0.7f), 3.5f));
    int electric_blue_light = scene.add_material(Material::emissive(::make_float3(0.2f, 0.75f, 1.0f), 3.0f));

    // Cornell box dimensions
    float box_size = 7.0f;

    // Floor
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, 0.0f),
        ::make_float3(box_size, 0.0f, 0.0f),
        ::make_float3(box_size, 0.0f, box_size),
        white_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, 0.0f),
        ::make_float3(box_size, 0.0f, box_size),
        ::make_float3(0.0f, 0.0f, box_size),
        white_diffuse
    ));

    // Ceiling
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, box_size, 0.0f),
        ::make_float3(0.0f, box_size, box_size),
        ::make_float3(box_size, box_size, box_size),
        white_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, box_size, 0.0f),
        ::make_float3(box_size, box_size, box_size),
        ::make_float3(box_size, box_size, 0.0f),
        white_diffuse
    ));

    // Back wall
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, box_size),
        ::make_float3(box_size, 0.0f, box_size),
        ::make_float3(box_size, box_size, box_size),
        white_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, box_size),
        ::make_float3(box_size, box_size, box_size),
        ::make_float3(0.0f, box_size, box_size),
        white_diffuse
    ));

    // Left wall (red)
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, 0.0f),
        ::make_float3(0.0f, 0.0f, box_size),
        ::make_float3(0.0f, box_size, box_size),
        red_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, 0.0f),
        ::make_float3(0.0f, box_size, box_size),
        ::make_float3(0.0f, box_size, 0.0f),
        red_diffuse
    ));

    // Right wall (green)
    scene.add_triangle(Triangle(
        ::make_float3(box_size, 0.0f, 0.0f),
        ::make_float3(box_size, box_size, 0.0f),
        ::make_float3(box_size, box_size, box_size),
        green_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(box_size, 0.0f, 0.0f),
        ::make_float3(box_size, box_size, box_size),
        ::make_float3(box_size, 0.0f, box_size),
        green_diffuse
    ));

    // Light on ceiling
    float light_size = 2.0f;
    float light_y = box_size - 0.01f;
    float light_center = box_size / 2.0f;
    scene.add_triangle(Triangle(
        ::make_float3(light_center - light_size/2, light_y, light_center - light_size/2),
        ::make_float3(light_center + light_size/2, light_y, light_center - light_size/2),
        ::make_float3(light_center + light_size/2, light_y, light_center + light_size/2),
        light_mat
    ));
    scene.add_triangle(Triangle(
        ::make_float3(light_center - light_size/2, light_y, light_center - light_size/2),
        ::make_float3(light_center + light_size/2, light_y, light_center + light_size/2),
        ::make_float3(light_center - light_size/2, light_y, light_center + light_size/2),
        light_mat
    ));

    // Neon accent panels on the back wall
    scene.add_triangle(Triangle(
        ::make_float3(1.0f, 2.0f, box_size - 0.01f),
        ::make_float3(2.1f, 2.0f, box_size - 0.01f),
        ::make_float3(2.1f, 3.5f, box_size - 0.01f),
        neon_pink_light
    ));
    scene.add_triangle(Triangle(
        ::make_float3(1.0f, 2.0f, box_size - 0.01f),
        ::make_float3(2.1f, 3.5f, box_size - 0.01f),
        ::make_float3(1.0f, 3.5f, box_size - 0.01f),
        neon_pink_light
    ));
    scene.add_triangle(Triangle(
        ::make_float3(4.9f, 1.7f, box_size - 0.01f),
        ::make_float3(6.1f, 1.7f, box_size - 0.01f),
        ::make_float3(6.1f, 2.8f, box_size - 0.01f),
        electric_blue_light
    ));
    scene.add_triangle(Triangle(
        ::make_float3(4.9f, 1.7f, box_size - 0.01f),
        ::make_float3(6.1f, 2.8f, box_size - 0.01f),
        ::make_float3(4.9f, 2.8f, box_size - 0.01f),
        electric_blue_light
    ));

    // Helper to add a box
    auto add_box = [&](float3 center, float3 size, float angle, int mat_id) {
        float c = cosf(angle);
        float s = sinf(angle);
        float hx = size.x * 0.5f;
        float hy = size.y * 0.5f;
        float hz = size.z * 0.5f;

        auto rotate = [c, s, center](float3 p) {
            float x = p.x - center.x;
            float z = p.z - center.z;
            return ::make_float3(
                center.x + c * x - s * z,
                p.y,
                center.z + s * x + c * z
            );
        };

        float3 corners[8] = {
            rotate(::make_float3(center.x - hx, center.y - hy, center.z - hz)),
            rotate(::make_float3(center.x + hx, center.y - hy, center.z - hz)),
            rotate(::make_float3(center.x + hx, center.y - hy, center.z + hz)),
            rotate(::make_float3(center.x - hx, center.y - hy, center.z + hz)),
            rotate(::make_float3(center.x - hx, center.y + hy, center.z - hz)),
            rotate(::make_float3(center.x + hx, center.y + hy, center.z - hz)),
            rotate(::make_float3(center.x + hx, center.y + hy, center.z + hz)),
            rotate(::make_float3(center.x - hx, center.y + hy, center.z + hz)),
        };

        // Front
        scene.add_triangle(Triangle(corners[0], corners[1], corners[5], mat_id));
        scene.add_triangle(Triangle(corners[0], corners[5], corners[4], mat_id));
        // Back
        scene.add_triangle(Triangle(corners[2], corners[3], corners[7], mat_id));
        scene.add_triangle(Triangle(corners[2], corners[7], corners[6], mat_id));
        // Left
        scene.add_triangle(Triangle(corners[3], corners[0], corners[4], mat_id));
        scene.add_triangle(Triangle(corners[3], corners[4], corners[7], mat_id));
        // Right
        scene.add_triangle(Triangle(corners[1], corners[2], corners[6], mat_id));
        scene.add_triangle(Triangle(corners[1], corners[6], corners[5], mat_id));
        // Top
        scene.add_triangle(Triangle(corners[4], corners[5], corners[6], mat_id));
        scene.add_triangle(Triangle(corners[4], corners[6], corners[7], mat_id));
    };

    // Helper to add a sphere (tessellated)
    auto add_sphere = [&](float3 center, float radius, int mat_id, int segments = 16) {
        auto add_smooth_sphere_triangle = [&](const float3& a, const float3& b, const float3& c) {
            Triangle tri(a, b, c, mat_id);
            tri.n0 = normalize(a - center);
            tri.n1 = normalize(b - center);
            tri.n2 = normalize(c - center);
            scene.add_triangle(tri);
        };

        for (int i = 0; i < segments; i++) {
            for (int j = 0; j < segments * 2; j++) {
                float theta0 = PI * float(i) / float(segments);
                float theta1 = PI * float(i + 1) / float(segments);
                float phi0 = TWO_PI * float(j) / float(segments * 2);
                float phi1 = TWO_PI * float(j + 1) / float(segments * 2);

                float3 p00 = center + radius * ::make_float3(
                    sinf(theta0) * cosf(phi0),
                    cosf(theta0),
                    sinf(theta0) * sinf(phi0)
                );
                float3 p01 = center + radius * ::make_float3(
                    sinf(theta0) * cosf(phi1),
                    cosf(theta0),
                    sinf(theta0) * sinf(phi1)
                );
                float3 p10 = center + radius * ::make_float3(
                    sinf(theta1) * cosf(phi0),
                    cosf(theta1),
                    sinf(theta1) * sinf(phi0)
                );
                float3 p11 = center + radius * ::make_float3(
                    sinf(theta1) * cosf(phi1),
                    cosf(theta1),
                    sinf(theta1) * sinf(phi1)
                );

                if (i > 0) {
                    add_smooth_sphere_triangle(p00, p10, p01);
                }
                if (i < segments - 1) {
                    add_smooth_sphere_triangle(p10, p11, p01);
                }
            }
        }
    };

    // Helper to add a 4-sided pyramid on the floor (open bottom).
    auto add_pyramid = [&](float3 base_center, float base_size, float height, float angle, int mat_id) {
        float c = cosf(angle);
        float s = sinf(angle);
        float h = base_size * 0.5f;

        auto rotate = [c, s, base_center](float3 p) {
            float x = p.x - base_center.x;
            float z = p.z - base_center.z;
            return ::make_float3(
                base_center.x + c * x - s * z,
                p.y,
                base_center.z + s * x + c * z
            );
        };

        float3 b0 = rotate(::make_float3(base_center.x - h, base_center.y, base_center.z - h));
        float3 b1 = rotate(::make_float3(base_center.x + h, base_center.y, base_center.z - h));
        float3 b2 = rotate(::make_float3(base_center.x + h, base_center.y, base_center.z + h));
        float3 b3 = rotate(::make_float3(base_center.x - h, base_center.y, base_center.z + h));
        float3 apex = ::make_float3(base_center.x, base_center.y + height, base_center.z);

        scene.add_triangle(Triangle(b0, b1, apex, mat_id));
        scene.add_triangle(Triangle(b1, b2, apex, mat_id));
        scene.add_triangle(Triangle(b2, b3, apex, mat_id));
        scene.add_triangle(Triangle(b3, b0, apex, mat_id));
    };

    // Helper to add a torus with smooth normals.
    auto add_torus = [&](float3 center, float major_radius, float minor_radius, float y_angle,
                         int mat_id, int seg_major = 28, int seg_minor = 14) {
        float c = cosf(y_angle);
        float s = sinf(y_angle);

        auto rotate_y = [c, s](const float3& p) {
            return ::make_float3(
                c * p.x - s * p.z,
                p.y,
                s * p.x + c * p.z
            );
        };

        auto add_smooth_torus_triangle = [&](const float3& p0, const float3& p1, const float3& p2,
                                             const float3& n0, const float3& n1, const float3& n2) {
            Triangle tri(p0, p1, p2, mat_id);
            tri.n0 = normalize(n0);
            tri.n1 = normalize(n1);
            tri.n2 = normalize(n2);
            scene.add_triangle(tri);
        };

        for (int i = 0; i < seg_major; ++i) {
            int i1 = (i + 1) % seg_major;
            float u0 = TWO_PI * float(i) / float(seg_major);
            float u1 = TWO_PI * float(i1) / float(seg_major);

            for (int j = 0; j < seg_minor; ++j) {
                int j1 = (j + 1) % seg_minor;
                float v0 = TWO_PI * float(j) / float(seg_minor);
                float v1 = TWO_PI * float(j1) / float(seg_minor);

                auto torus_pos = [&](float u, float v) {
                    float ring = major_radius + minor_radius * cosf(v);
                    return ::make_float3(
                        ring * cosf(u),
                        minor_radius * sinf(v),
                        ring * sinf(u)
                    );
                };

                auto torus_normal = [&](float u, float v) {
                    return normalize(::make_float3(
                        cosf(u) * cosf(v),
                        sinf(v),
                        sinf(u) * cosf(v)
                    ));
                };

                float3 lp00 = torus_pos(u0, v0);
                float3 lp10 = torus_pos(u1, v0);
                float3 lp01 = torus_pos(u0, v1);
                float3 lp11 = torus_pos(u1, v1);

                float3 ln00 = torus_normal(u0, v0);
                float3 ln10 = torus_normal(u1, v0);
                float3 ln01 = torus_normal(u0, v1);
                float3 ln11 = torus_normal(u1, v1);

                float3 p00 = center + rotate_y(lp00);
                float3 p10 = center + rotate_y(lp10);
                float3 p01 = center + rotate_y(lp01);
                float3 p11 = center + rotate_y(lp11);

                float3 n00 = rotate_y(ln00);
                float3 n10 = rotate_y(ln10);
                float3 n01 = rotate_y(ln01);
                float3 n11 = rotate_y(ln11);

                add_smooth_torus_triangle(p00, p10, p01, n00, n10, n01);
                add_smooth_torus_triangle(p10, p11, p01, n10, n11, n01);
            }
        }
    };

    // Helper to add an octahedron.
    auto add_octahedron = [&](float3 center, float radius, float angle, int mat_id) {
        float c = cosf(angle);
        float s = sinf(angle);

        auto rotate_y = [c, s, center](float3 p) {
            float x = p.x - center.x;
            float z = p.z - center.z;
            return ::make_float3(
                center.x + c * x - s * z,
                p.y,
                center.z + s * x + c * z
            );
        };

        float3 top = ::make_float3(center.x, center.y + radius, center.z);
        float3 bottom = ::make_float3(center.x, center.y - radius, center.z);

        float3 m0 = rotate_y(::make_float3(center.x + radius, center.y, center.z));
        float3 m1 = rotate_y(::make_float3(center.x, center.y, center.z + radius));
        float3 m2 = rotate_y(::make_float3(center.x - radius, center.y, center.z));
        float3 m3 = rotate_y(::make_float3(center.x, center.y, center.z - radius));

        scene.add_triangle(Triangle(top, m0, m1, mat_id));
        scene.add_triangle(Triangle(top, m1, m2, mat_id));
        scene.add_triangle(Triangle(top, m2, m3, mat_id));
        scene.add_triangle(Triangle(top, m3, m0, mat_id));

        scene.add_triangle(Triangle(bottom, m1, m0, mat_id));
        scene.add_triangle(Triangle(bottom, m2, m1, mat_id));
        scene.add_triangle(Triangle(bottom, m3, m2, mat_id));
        scene.add_triangle(Triangle(bottom, m0, m3, mat_id));
    };

    // Main hero objects
    add_sphere(::make_float3(1.45f, 1.0f, 4.8f), 1.0f, mirror_mat, 24);
    add_sphere(::make_float3(3.45f, 0.9f, 2.35f), 0.9f, chrome_mat, 24);
    add_sphere(::make_float3(5.35f, 0.75f, 4.65f), 0.75f, gold_mat, 24);
    add_sphere(::make_float3(1.9f, 0.42f, 1.65f), 0.42f, velvet_violet, 16);

    // New playful shapes
    add_pyramid(::make_float3(5.45f, 0.0f, 1.95f), 1.0f, 1.35f, 0.42f, coral_diffuse);
    add_octahedron(::make_float3(2.55f, 1.1f, 3.25f), 0.62f, 0.5f, mango_diffuse);
    add_torus(::make_float3(4.05f, 1.65f, 3.8f), 0.78f, 0.24f, 0.35f, pearl_metal, 24, 12);

    // Stacked blocks in the back-right corner
    add_box(::make_float3(5.95f, 0.35f, 5.55f), ::make_float3(1.0f, 0.7f, 1.0f), 0.35f, obsidian_metal);
    add_box(::make_float3(5.95f, 0.92f, 5.55f), ::make_float3(0.68f, 0.44f, 0.68f), -0.2f, rose_gold_metal);

    // Keep global scene scaling as a single tuning knob.
    const float scene_scale = 1.0f;
    scene.scale_geometry(scene_scale, ::make_float3(box_size * 0.5f, box_size * 0.5f, box_size * 0.5f));

    // Build BVH
    scene.build();

    printf("Scene created: %d triangles, %d materials\n",
           scene.num_triangles(), scene.num_materials());

    return scene;
}

// =============================================================================
// Interactive Window
// =============================================================================

class RenderWindow {
public:
    RenderWindow(int width, int height, const char* title)
        : width_(width), height_(height), window_(nullptr),
          shader_program_(0), vao_(0), vbo_(0), texture_(0),
          camera_changed_(true), renderer_(nullptr) {

        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window_) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window_);

        if (!gladLoadGLLoader((void*(*)(const char*))glfwGetProcAddress)) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            glfwDestroyWindow(window_);
            glfwTerminate();
            window_ = nullptr;
            return;
        }

        glfwSwapInterval(0);  // Disable vsync for max performance

        glfwSetWindowUserPointer(window_, this);
        glfwSetMouseButtonCallback(window_, mouse_button_callback);
        glfwSetCursorPosCallback(window_, cursor_pos_callback);
        glfwSetScrollCallback(window_, scroll_callback);
        glfwSetKeyCallback(window_, key_callback);

        init_gl_resources();
        pixel_buffer_.resize(width * height * 4);

        // Initialize camera controller
        controller_.camera.look_at(
            ::make_float3(3.5f, 3.1f, -7.2f),
            ::make_float3(3.5f, 2.2f, 3.5f),
            ::make_float3(0.0f, 1.0f, 0.0f)
        );
        controller_.camera.fov = PI / 4.0f;
        controller_.camera.aspect_ratio = float(width) / float(height);
        controller_.camera.update();
    }

    ~RenderWindow() {
        if (texture_) glDeleteTextures(1, &texture_);
        if (vbo_) glDeleteBuffers(1, &vbo_);
        if (vao_) glDeleteVertexArrays(1, &vao_);
        if (shader_program_) glDeleteProgram(shader_program_);
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

    bool is_valid() const { return window_ != nullptr; }
    bool should_close() const { return window_ && glfwWindowShouldClose(window_); }

    void run(InteractiveRenderer& renderer, Scene& scene) {
        if (!window_) return;
        renderer_ = &renderer;

        std::cout << "\nStarting interactive path tracer..." << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  Left mouse drag: Orbit camera" << std::endl;
        std::cout << "  Shift + drag: Pan camera" << std::endl;
        std::cout << "  Scroll: Zoom in/out" << std::endl;
        std::cout << "  +/-: Adjust exposure" << std::endl;
        std::cout << "  R: Reset accumulation" << std::endl;
        std::cout << "  ESC: Quit" << std::endl;
        std::cout << std::endl;

        double last_time = glfwGetTime();
        int frame_count = 0;
        double fps_time = 0.0;

        while (!glfwWindowShouldClose(window_)) {
            double current_time = glfwGetTime();
            double delta = current_time - last_time;
            last_time = current_time;

            frame_count++;
            fps_time += delta;
            if (fps_time >= 1.0) {
                char title[256];
                snprintf(title, sizeof(title),
                         "Lumina Path Tracer - %.1f FPS - %d samples - Exposure: %.2f",
                         frame_count / fps_time, renderer.total_samples(), renderer.exposure());
                glfwSetWindowTitle(window_, title);
                frame_count = 0;
                fps_time = 0.0;
            }

            glfwPollEvents();

            if (camera_changed_) {
                renderer.reset_accumulation();
                camera_changed_ = false;
            }

            // Render a frame
            renderer.render_frame(controller_.camera, scene);

            // Tonemap and display
            renderer.tonemap();

            // Download and display
            renderer.download_display(reinterpret_cast<uchar4*>(pixel_buffer_.data()));
            display_frame();

            glfwSwapBuffers(window_);
        }
    }

    CameraController& camera_controller() { return controller_; }
    void set_camera_changed() { camera_changed_ = true; }

private:
    void init_gl_resources() {
        const char* vs_src = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        )";

        const char* fs_src = R"(
            #version 330 core
            in vec2 TexCoord;
            out vec4 FragColor;
            uniform sampler2D screenTexture;
            void main() {
                FragColor = texture(screenTexture, TexCoord);
            }
        )";

        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vs_src, nullptr);
        glCompileShader(vs);

        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fs_src, nullptr);
        glCompileShader(fs);

        shader_program_ = glCreateProgram();
        glAttachShader(shader_program_, vs);
        glAttachShader(shader_program_, fs);
        glLinkProgram(shader_program_);

        glDeleteShader(vs);
        glDeleteShader(fs);

        float vertices[] = {
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);

        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void display_frame() {
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                        GL_RGBA, GL_UNSIGNED_BYTE, pixel_buffer_.data());

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader_program_);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);

        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        RenderWindow* self = static_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        self->controller_.on_mouse_button(button, action == GLFW_PRESS,
                                          static_cast<float>(x), static_cast<float>(y));
    }

    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
        RenderWindow* self = static_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
        bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        bool ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        if (self->controller_.on_mouse_move(static_cast<float>(xpos),
                                            static_cast<float>(ypos), shift, ctrl)) {
            self->camera_changed_ = true;
        }
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        RenderWindow* self = static_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
        if (self->controller_.on_scroll(static_cast<float>(yoffset))) {
            self->camera_changed_ = true;
        }
    }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

        RenderWindow* self = static_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, true);
                break;
            case GLFW_KEY_R:
                self->camera_changed_ = true;
                break;
            case GLFW_KEY_EQUAL:
            case GLFW_KEY_KP_ADD:
                if (self->renderer_) {
                    self->renderer_->exposure() *= 1.1f;
                }
                break;
            case GLFW_KEY_MINUS:
            case GLFW_KEY_KP_SUBTRACT:
                if (self->renderer_) {
                    self->renderer_->exposure() *= 0.9f;
                }
                break;
        }
    }

    int width_, height_;
    GLFWwindow* window_;
    GLuint shader_program_;
    GLuint vao_;
    GLuint vbo_;
    GLuint texture_;
    std::vector<unsigned char> pixel_buffer_;
    CameraController controller_;
    bool camera_changed_;
    InteractiveRenderer* renderer_;
};

// =============================================================================
// Main
// =============================================================================

void print_cuda_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return;
    }

    std::cout << "=== Lumina Ray Tracer ===" << std::endl;
    std::cout << "CUDA Devices: " << device_count << std::endl;

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  SM count: " << prop.multiProcessorCount << std::endl;
    }
    std::cout << "\n=========================" << std::endl;
}

int main(int argc, char** argv) {
    print_cuda_info();

    int width = 1920;
    int height = 1080;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: lumina [options]" << std::endl;
            std::cout << "  --width <n>   Window width (default: 1920)" << std::endl;
            std::cout << "  --height <n>  Window height (default: 1080)" << std::endl;
            return 0;
        }
    }

    std::cout << "\nResolution: " << width << "x" << height << std::endl;

    // Create scene
    std::cout << "\nBuilding scene..." << std::endl;
    Scene scene = create_demo_scene();

    // Create renderer
    std::cout << "Initializing renderer..." << std::endl;
    InteractiveRenderer renderer(width, height);

    // Create window and run
    RenderWindow window(width, height, "Lumina Path Tracer");

    if (!window.is_valid()) {
        std::cerr << "Failed to create window!" << std::endl;
        return 1;
    }

    window.run(renderer, scene);

    std::cout << "\nLumina ray tracer exited successfully." << std::endl;
    return 0;
}
