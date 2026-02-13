#pragma once

#include <iostream>
#include <vector>
#include <cstdio>

#include <cuda_runtime.h>

// GLFW_INCLUDE_NONE prevents GLFW from including platform OpenGL headers
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "camera.cuh"
#include "scene.cuh"
#include "../core/memory/device_buffer.cuh"
#include "../integrators/wavefront/path_state.cuh"
#include "../integrators/wavefront/ray_queue.cuh"
#include "../geometry/bvh/bvh.cuh"

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

        // Allocate initial active path buffer
        initial_active_.resize(num_pixels_);
        init_sequence();

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

        CUDA_CHECK(cudaMemcpy(work_queues_.active_paths(), initial_active_.data(),
                              num_pixels_ * sizeof(int), cudaMemcpyDeviceToDevice));
        work_queues_.set_active_count(num_pixels_);

        // Path tracing loop
        int depth = 0;
        while (depth < max_depth_) {
            unsigned int active_count = work_queues_.get_active_count();
            if (active_count == 0) break;
            int active_count_i = static_cast<int>(active_count);

            // Intersection
            launch_intersect(paths, hits, scene.bvh_nodes(), scene.triangles(),
                           work_queues_.active_paths(), active_count_i);
            CUDA_CHECK_LAST();

            // Shade misses
            launch_shade_miss(paths, hits, work_queues_.active_paths(), active_count_i);
            CUDA_CHECK_LAST();

            // Reset next queue
            CUDA_CHECK(cudaMemset(work_queues_.next_count_ptr(), 0, sizeof(unsigned int)));

            // Shade surfaces
            launch_shade_surface(paths, hits, scene.materials(),
                               work_queues_.active_paths(),
                               work_queues_.next_count_ptr(),
                               work_queues_.next_paths(),
                               active_count_i, max_depth_);
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
};

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
        (void)mods;
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
        (void)xoffset;
        RenderWindow* self = static_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
        if (self->controller_.on_scroll(static_cast<float>(yoffset))) {
            self->camera_changed_ = true;
        }
    }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        (void)scancode;
        (void)mods;
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

} // namespace lumina
