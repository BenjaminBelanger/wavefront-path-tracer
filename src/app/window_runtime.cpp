#include "interactive_runtime.cuh"

#include <cstdio>
#include <iostream>

// GLFW_INCLUDE_NONE prevents GLFW from including platform OpenGL headers.
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace wpt {

RenderWindow::RenderWindow(int width, int height, const char* title)
    : width_(width)
    , height_(height)
    , window_(nullptr)
    , shader_program_(0)
    , vao_(0)
    , vbo_(0)
    , texture_(0)
    , camera_changed_(true)
    , renderer_(nullptr) {
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

    glfwSwapInterval(0);  // Disable vsync for max performance.

    glfwSetWindowUserPointer(window_, this);
    glfwSetMouseButtonCallback(window_, mouse_button_callback);
    glfwSetCursorPosCallback(window_, cursor_pos_callback);
    glfwSetScrollCallback(window_, scroll_callback);
    glfwSetKeyCallback(window_, key_callback);

    init_gl_resources();
    pixel_buffer_.resize(width * height * 4);

    // Initialize camera controller.
    controller_.camera.look_at(
        ::make_float3(3.5f, 3.1f, -7.2f),
        ::make_float3(3.5f, 2.2f, 3.5f),
        ::make_float3(0.0f, 1.0f, 0.0f)
    );
    controller_.camera.fov = PI / 4.0f;
    controller_.camera.aspect_ratio = float(width) / float(height);
    controller_.camera.update();
}

RenderWindow::~RenderWindow() {
    if (texture_) glDeleteTextures(1, &texture_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (shader_program_) glDeleteProgram(shader_program_);
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

bool RenderWindow::is_valid() const { return window_ != nullptr; }

bool RenderWindow::should_close() const { return window_ && glfwWindowShouldClose(window_); }

void RenderWindow::run(InteractiveRenderer& renderer, Scene& scene) {
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
                     "Wavefront Path Tracer - %.1f FPS - %d samples - Exposure: %.2f",
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

        // Render a frame.
        renderer.render_frame(controller_.camera, scene);

        // Tonemap and display.
        renderer.tonemap();

        // Download and display.
        renderer.download_display(reinterpret_cast<uchar4*>(pixel_buffer_.data()));
        display_frame();

        glfwSwapBuffers(window_);
    }
}

CameraController& RenderWindow::camera_controller() { return controller_; }

void RenderWindow::set_camera_changed() { camera_changed_ = true; }

void RenderWindow::init_gl_resources() {
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

void RenderWindow::display_frame() {
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

void RenderWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    (void)mods;
    RenderWindow* self = static_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    self->controller_.on_mouse_button(button, action == GLFW_PRESS,
                                      static_cast<float>(x), static_cast<float>(y));
}

void RenderWindow::cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
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

void RenderWindow::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)xoffset;
    RenderWindow* self = static_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
    if (self->controller_.on_scroll(static_cast<float>(yoffset))) {
        self->camera_changed_ = true;
    }
}

void RenderWindow::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
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

} // namespace wpt
