#include "interactive_runtime.cuh"
#include "camera_animation.cuh"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Prevent the system OpenGL headers (pulled in by cuda_gl_interop.h) from
// redeclaring the GL entry points that glad already defines as function
// pointers. On Linux GL/gl.h also pulls glext, so guard those too.
#ifndef __gl_h_
#define __gl_h_
#endif
#ifndef __GL_H__
#define __GL_H__
#endif
#ifndef __gl_glext_h_
#define __gl_glext_h_
#endif
#ifndef __glext_h_
#define __glext_h_
#endif
#include <cuda_gl_interop.h>

#ifndef GL_PIXEL_UNPACK_BUFFER
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#endif
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif

namespace
{

#define CUDA_RUNTIME_CHECK(call)                                                 \
    do                                                                           \
    {                                                                            \
        cudaError_t err__ = call;                                                \
        if (err__ != cudaSuccess)                                                \
        {                                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err__) << std::endl;                 \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

}

namespace wpt
{

    RenderWindow::RenderWindow(int width, int height, const char *title)
        : width_(width), height_(height), window_(nullptr), shader_program_(0), vao_(0), vbo_(0), texture_(0), pbo_(0), cuda_pbo_resource_(nullptr), camera_changed_(true), screenshot_requested_(false), renderer_(nullptr)
    {
        if (!glfwInit())
        {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window_)
        {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window_);

        if (!gladLoadGLLoader((void *(*)(const char *))glfwGetProcAddress))
        {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            glfwDestroyWindow(window_);
            glfwTerminate();
            window_ = nullptr;
            return;
        }

        glfwSwapInterval(0);

        glfwSetWindowUserPointer(window_, this);
        glfwSetMouseButtonCallback(window_, mouse_button_callback);
        glfwSetCursorPosCallback(window_, cursor_pos_callback);
        glfwSetScrollCallback(window_, scroll_callback);
        glfwSetKeyCallback(window_, key_callback);

        init_gl_resources();

        controller_.camera.look_at(
            ::make_float3(3.5f, 3.1f, -7.2f),
            ::make_float3(3.5f, 2.2f, 3.5f),
            ::make_float3(0.0f, 1.0f, 0.0f));
        controller_.camera.fov = PI / 4.0f;
        controller_.camera.aspect_ratio = float(width) / float(height);
        controller_.camera.update();
    }

    RenderWindow::~RenderWindow()
    {
        if (window_)
            glfwMakeContextCurrent(window_);
        if (cuda_pbo_resource_)
            CUDA_RUNTIME_CHECK(cudaGraphicsUnregisterResource(cuda_pbo_resource_));
        if (pbo_)
            glDeleteBuffers(1, &pbo_);
        if (texture_)
            glDeleteTextures(1, &texture_);
        if (vbo_)
            glDeleteBuffers(1, &vbo_);
        if (vao_)
            glDeleteVertexArrays(1, &vao_);
        if (shader_program_)
            glDeleteProgram(shader_program_);
        if (window_)
            glfwDestroyWindow(window_);
        glfwTerminate();
    }

    bool RenderWindow::is_valid() const { return window_ != nullptr; }

    bool RenderWindow::should_close() const { return window_ && glfwWindowShouldClose(window_); }

    void RenderWindow::run(InteractiveRenderer &renderer, Scene &scene)
    {
        if (!window_)
            return;
        renderer_ = &renderer;

        std::cout << "\nStarting interactive path tracer..." << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  Left mouse drag: Orbit camera" << std::endl;
        std::cout << "  Shift + drag: Pan camera" << std::endl;
        std::cout << "  Scroll: Zoom in/out" << std::endl;
        std::cout << "  +/-: Adjust exposure" << std::endl;
        std::cout << "  R: Reset accumulation" << std::endl;
        std::cout << "  P: Save screenshot (PNG)" << std::endl;
        std::cout << "  ESC: Quit" << std::endl;
        std::cout << std::endl;

        double last_time = glfwGetTime();
        int frame_count = 0;
        double fps_time = 0.0;

        while (!glfwWindowShouldClose(window_))
        {
            double current_time = glfwGetTime();
            double delta = current_time - last_time;
            last_time = current_time;

            frame_count++;
            fps_time += delta;
            if (fps_time >= 1.0)
            {
                char title[256];
                snprintf(title, sizeof(title),
                         "Wavefront Path Tracer - %.1f FPS - %d samples - Exposure: %.2f",
                         frame_count / fps_time, renderer.total_samples(), renderer.exposure());
                glfwSetWindowTitle(window_, title);
                frame_count = 0;
                fps_time = 0.0;
            }

            glfwPollEvents();

            if (camera_changed_)
            {
                renderer.reset_accumulation();
                camera_changed_ = false;
            }

            renderer.render_frame(controller_.camera, scene);

            uchar4 *mapped_buffer = nullptr;
            size_t mapped_size = 0;
            CUDA_RUNTIME_CHECK(cudaGraphicsMapResources(1, &cuda_pbo_resource_, 0));
            CUDA_RUNTIME_CHECK(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void **>(&mapped_buffer), &mapped_size, cuda_pbo_resource_));
            if (mapped_size < static_cast<size_t>(width_) * height_ * sizeof(uchar4))
            {
                std::cerr << "Mapped PBO is smaller than expected" << std::endl;
                CUDA_RUNTIME_CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0));
                break;
            }
            renderer.tonemap_to_buffer(mapped_buffer);

            if (screenshot_requested_)
            {
                save_screenshot(mapped_buffer);
                screenshot_requested_ = false;
            }

            CUDA_RUNTIME_CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0));

            display_frame();

            glfwSwapBuffers(window_);
        }
    }

    void RenderWindow::frame_scene(float3 center, float radius)
    {
        float distance = controller_.camera.frame_distance(radius);

        float3 cam_pos = center - ::make_float3(0.0f, 0.0f, distance);

        controller_.camera.look_at(
            cam_pos,
            center,
            ::make_float3(0.0f, 1.0f, 0.0f));
        controller_.camera.update();
        camera_changed_ = true;
    }

    CameraController &RenderWindow::camera_controller() { return controller_; }

    void RenderWindow::set_camera_changed() { camera_changed_ = true; }

    void RenderWindow::init_gl_resources()
    {
        const char *vs_src = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )";

        const char *fs_src = R"(
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
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);

        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenBuffers(1, &pbo_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(width_) * height_ * sizeof(uchar4),
                     nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        CUDA_RUNTIME_CHECK(cudaGraphicsGLRegisterBuffer(
            &cuda_pbo_resource_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard));
    }

    void RenderWindow::display_frame()
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader_program_);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);

        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

    void RenderWindow::save_screenshot(const uchar4 *device_buffer)
    {
        const size_t row_bytes = static_cast<size_t>(width_) * 4;
        const size_t total_bytes = row_bytes * static_cast<size_t>(height_);

        std::vector<unsigned char> host(total_bytes);
        CUDA_RUNTIME_CHECK(cudaMemcpy(host.data(), device_buffer, total_bytes,
                                      cudaMemcpyDeviceToHost));

        // The displayed texture samples buffer row 0 at the bottom of the image,
        // so flip vertically to produce a conventional top-down PNG.
        std::vector<unsigned char> flipped(total_bytes);
        for (int y = 0; y < height_; ++y)
        {
            std::memcpy(&flipped[static_cast<size_t>(y) * row_bytes],
                        &host[static_cast<size_t>(height_ - 1 - y) * row_bytes],
                        row_bytes);
        }

        std::error_code ec;
        std::filesystem::create_directories("screenshots", ec);

        std::time_t now = std::time(nullptr);
        std::tm tm_buf{};
#if defined(_WIN32)
        localtime_s(&tm_buf, &now);
#else
        localtime_r(&now, &tm_buf);
#endif
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_buf);

        std::string path = std::string("screenshots/screenshot_") + stamp + ".png";

        if (stbi_write_png(path.c_str(), width_, height_, 4, flipped.data(),
                           static_cast<int>(row_bytes)))
        {
            std::cout << "Saved screenshot: " << path << std::endl;
        }
        else
        {
            std::cerr << "Failed to save screenshot: " << path << std::endl;
        }
    }

    void RenderWindow::mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
    {
        (void)mods;
        RenderWindow *self = static_cast<RenderWindow *>(glfwGetWindowUserPointer(window));
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        self->controller_.on_mouse_button(button, action == GLFW_PRESS,
                                          static_cast<float>(x), static_cast<float>(y));
    }

    void RenderWindow::cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
    {
        RenderWindow *self = static_cast<RenderWindow *>(glfwGetWindowUserPointer(window));
        bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        bool ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        if (self->controller_.on_mouse_move(static_cast<float>(xpos),
                                            static_cast<float>(ypos), shift, ctrl))
        {
            self->camera_changed_ = true;
        }
    }

    void RenderWindow::scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
    {
        (void)xoffset;
        RenderWindow *self = static_cast<RenderWindow *>(glfwGetWindowUserPointer(window));
        if (self->controller_.on_scroll(static_cast<float>(yoffset)))
        {
            self->camera_changed_ = true;
        }
    }

    void RenderWindow::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        (void)scancode;
        (void)mods;
        if (action != GLFW_PRESS && action != GLFW_REPEAT)
            return;

        RenderWindow *self = static_cast<RenderWindow *>(glfwGetWindowUserPointer(window));

        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
        case GLFW_KEY_R:
            self->camera_changed_ = true;
            break;
        case GLFW_KEY_P:
            if (action == GLFW_PRESS)
                self->screenshot_requested_ = true;
            break;
        case GLFW_KEY_EQUAL:
        case GLFW_KEY_KP_ADD:
            if (self->renderer_)
            {
                self->renderer_->exposure() *= 1.1f;
            }
            break;
        case GLFW_KEY_MINUS:
        case GLFW_KEY_KP_SUBTRACT:
            if (self->renderer_)
            {
                self->renderer_->exposure() *= 0.9f;
            }
            break;
        }
    }

    namespace
    {
        // Tonemaps the renderer's accumulated result into `device_buffer`, copies
        // it to the host, flips it to a conventional top-down orientation and
        // writes it as a PNG. Returns true on success.
        bool tonemap_and_write_png(InteractiveRenderer &renderer, uchar4 *device_buffer,
                                   int width, int height, const std::string &output_path)
        {
            renderer.tonemap_to_buffer(device_buffer);
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

            const size_t row_bytes = static_cast<size_t>(width) * 4;
            const size_t total_bytes = row_bytes * static_cast<size_t>(height);

            std::vector<unsigned char> host(total_bytes);
            CUDA_RUNTIME_CHECK(cudaMemcpy(host.data(), device_buffer, total_bytes,
                                          cudaMemcpyDeviceToHost));

            // Row 0 is the bottom of the image; flip to a conventional top-down PNG.
            std::vector<unsigned char> flipped(total_bytes);
            for (int y = 0; y < height; ++y)
            {
                std::memcpy(&flipped[static_cast<size_t>(y) * row_bytes],
                            &host[static_cast<size_t>(height - 1 - y) * row_bytes],
                            row_bytes);
            }

            std::filesystem::path out_path(output_path);
            if (out_path.has_parent_path())
            {
                std::error_code ec;
                std::filesystem::create_directories(out_path.parent_path(), ec);
            }

            return stbi_write_png(output_path.c_str(), width, height, 4, flipped.data(),
                                  static_cast<int>(row_bytes)) != 0;
        }
    }

    int run_headless(InteractiveRenderer &renderer, Scene &scene, const Camera &camera,
                     int width, int height, int frames, const std::string &output_path)
    {
        if (frames < 1)
            frames = 1;

        const size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);

        uchar4 *device_buffer = nullptr;
        CUDA_RUNTIME_CHECK(cudaMalloc(&device_buffer, num_pixels * sizeof(uchar4)));

        renderer.reset_accumulation();

        std::cout << "\nHeadless render: " << width << "x" << height << ", "
                  << frames << " samples/pixel" << std::endl;

        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < frames; ++i)
        {
            renderer.render_frame(camera, scene);
            if ((i + 1) % 16 == 0 || (i + 1) == frames)
            {
                CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
                std::cout << "\r  sample " << (i + 1) << "/" << frames << std::flush;
            }
        }
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "\n  " << ms << " ms total, " << (ms / frames)
                  << " ms/sample, " << (frames * 1000.0 / ms) << " samples/s" << std::endl;

        const bool ok = tonemap_and_write_png(renderer, device_buffer, width, height, output_path);
        CUDA_RUNTIME_CHECK(cudaFree(device_buffer));

        if (ok)
        {
            std::cout << "Saved render: " << output_path << std::endl;
            return 0;
        }

        std::cerr << "Failed to save render: " << output_path << std::endl;
        return 1;
    }

    int run_animation(InteractiveRenderer &renderer, Scene &scene, const Camera &base_camera,
                      int width, int height, int spp, const AnimationConfig &config,
                      const std::string &output_dir)
    {
        const int frames = (config.frames > 0) ? config.frames : 1;
        const int samples = (spp > 0) ? spp : 1;

        std::filesystem::path dir(output_dir);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);

        const size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);

        uchar4 *device_buffer = nullptr;
        CUDA_RUNTIME_CHECK(cudaMalloc(&device_buffer, num_pixels * sizeof(uchar4)));

        std::cout << "\nHeadless animation: " << animation_preset_name(config.preset) << ", "
                  << frames << " frames @ " << width << "x" << height << ", "
                  << samples << " spp/frame" << std::endl;

        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int f = 0; f < frames; ++f)
        {
            const Camera camera = animate_camera(base_camera, config, f);

            renderer.reset_accumulation();
            for (int s = 0; s < samples; ++s)
            {
                renderer.render_frame(camera, scene);
            }
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

            char name[32];
            std::snprintf(name, sizeof(name), "frame%04d.png", f);
            const std::string frame_path = (dir / name).string();

            if (!tonemap_and_write_png(renderer, device_buffer, width, height, frame_path))
            {
                std::cerr << "\nFailed to save frame: " << frame_path << std::endl;
                CUDA_RUNTIME_CHECK(cudaFree(device_buffer));
                return 1;
            }

            std::cout << "\r  frame " << (f + 1) << "/" << frames << std::flush;
        }
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();

        CUDA_RUNTIME_CHECK(cudaFree(device_buffer));

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "\n  " << ms << " ms total, " << (ms / frames)
                  << " ms/frame" << std::endl;
        std::cout << "Saved " << frames << " frames to: " << dir.string() << std::endl;
        return 0;
    }

}
