// Window implementation - OpenGL display for Lumina renderer

// GLFW_INCLUDE_NONE prevents GLFW from including platform OpenGL headers
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <chrono>
#include <string>
#include <functional>
#include <cstring>

namespace lumina {

// =============================================================================
// Shader sources for fullscreen quad rendering
// =============================================================================

static const char* VERTEX_SHADER_SRC = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

static const char* FRAGMENT_SHADER_SRC = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform float exposure;

void main() {
    vec3 color = texture(screenTexture, TexCoord).rgb;

    // Apply exposure
    color *= exposure;

    // Reinhard tone mapping
    color = color / (color + vec3(1.0));

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
)";

// =============================================================================
// Window Class
// =============================================================================

class Window {
public:
    using ResizeCallback = std::function<void(int, int)>;
    using KeyCallback = std::function<void(int, int, int, int)>;
    using MouseButtonCallback = std::function<void(int, int, int)>;
    using CursorCallback = std::function<void(double, double)>;
    using ScrollCallback = std::function<void(double, double)>;

    Window(int width, int height, const char* title)
        : width_(width), height_(height), window_(nullptr),
          shader_program_(0), vao_(0), vbo_(0), texture_(0),
          exposure_(1.0f) {

        // Initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return;
        }

        // Request OpenGL 3.3 Core
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        // Create window
        window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window_) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window_);

        // Load OpenGL functions
        if (!gladLoadGLLoader((void*(*)(const char*))glfwGetProcAddress)) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            glfwDestroyWindow(window_);
            glfwTerminate();
            window_ = nullptr;
            return;
        }

        // Print OpenGL info
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
        std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;

        // Set up callbacks
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
        glfwSetKeyCallback(window_, key_callback);
        glfwSetMouseButtonCallback(window_, mouse_button_callback);
        glfwSetCursorPosCallback(window_, cursor_pos_callback);
        glfwSetScrollCallback(window_, scroll_callback);

        // Enable vsync
        glfwSwapInterval(1);

        // Initialize OpenGL resources
        init_gl_resources();
    }

    ~Window() {
        if (texture_) glDeleteTextures(1, &texture_);
        if (vbo_) glDeleteBuffers(1, &vbo_);
        if (vao_) glDeleteVertexArrays(1, &vao_);
        if (shader_program_) glDeleteProgram(shader_program_);

        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

    bool is_valid() const { return window_ != nullptr; }
    bool should_close() const { return window_ ? glfwWindowShouldClose(window_) : true; }
    void set_should_close(bool value) { if (window_) glfwSetWindowShouldClose(window_, value); }

    void poll_events() { glfwPollEvents(); }

    void swap_buffers() {
        if (window_) glfwSwapBuffers(window_);
    }

    void set_title(const std::string& title) {
        if (window_) glfwSetWindowTitle(window_, title.c_str());
    }

    int width() const { return width_; }
    int height() const { return height_; }

    float exposure() const { return exposure_; }
    void set_exposure(float exp) { exposure_ = exp; }
    void adjust_exposure(float delta) {
        exposure_ *= (delta > 0) ? 1.1f : 0.9f;
        exposure_ = (exposure_ < 0.1f) ? 0.1f : (exposure_ > 10.0f) ? 10.0f : exposure_;
    }

    // Upload rendered image to texture and display
    void display(const float* rgb_buffer, int buf_width, int buf_height) {
        if (!window_) return;

        // Update texture
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, buf_width, buf_height,
                        GL_RGB, GL_FLOAT, rgb_buffer);

        // Draw fullscreen quad
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shader_program_);
        glUniform1i(glGetUniformLocation(shader_program_, "screenTexture"), 0);
        glUniform1f(glGetUniformLocation(shader_program_, "exposure"), exposure_);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);

        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

    // Display from RGBA8 buffer (tonemapped already)
    void display_rgba8(const unsigned char* rgba_buffer, int buf_width, int buf_height) {
        if (!window_) return;

        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, buf_width, buf_height,
                        GL_RGBA, GL_UNSIGNED_BYTE, rgba_buffer);

        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shader_program_);
        glUniform1i(glGetUniformLocation(shader_program_, "screenTexture"), 0);
        glUniform1f(glGetUniformLocation(shader_program_, "exposure"), 1.0f);  // No additional exposure

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);

        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

    void resize_texture(int new_width, int new_height) {
        width_ = new_width;
        height_ = new_height;

        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, new_width, new_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }

    // Callback setters
    void set_resize_callback(ResizeCallback cb) { resize_callback_ = cb; }
    void set_key_callback(KeyCallback cb) { key_callback_ = cb; }
    void set_mouse_button_callback(MouseButtonCallback cb) { mouse_button_callback_ = cb; }
    void set_cursor_callback(CursorCallback cb) { cursor_callback_ = cb; }
    void set_scroll_callback(ScrollCallback cb) { scroll_callback_ = cb; }

    // Input state
    bool is_key_pressed(int key) const {
        return window_ && glfwGetKey(window_, key) == GLFW_PRESS;
    }

    bool is_mouse_button_pressed(int button) const {
        return window_ && glfwGetMouseButton(window_, button) == GLFW_PRESS;
    }

    void get_cursor_pos(double& x, double& y) const {
        if (window_) glfwGetCursorPos(window_, &x, &y);
    }

private:
    void init_gl_resources() {
        // Compile shaders
        GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, VERTEX_SHADER_SRC);
        GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);

        // Link program
        shader_program_ = glCreateProgram();
        glAttachShader(shader_program_, vertex_shader);
        glAttachShader(shader_program_, fragment_shader);
        glLinkProgram(shader_program_);

        GLint success;
        glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
        if (!success) {
            char info_log[512];
            glGetProgramInfoLog(shader_program_, 512, nullptr, info_log);
            std::cerr << "Shader program linking failed: " << info_log << std::endl;
        }

        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        // Fullscreen quad vertices (position + texcoord)
        float quad_vertices[] = {
            // positions   // texcoords
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
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);

        // Create texture for rendered image
        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    GLuint compile_shader(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char info_log[512];
            glGetShaderInfoLog(shader, 512, nullptr, info_log);
            std::cerr << "Shader compilation failed: " << info_log << std::endl;
        }
        return shader;
    }

    // Static GLFW callbacks
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        glViewport(0, 0, width, height);
        self->width_ = width;
        self->height_ = height;
        if (self->resize_callback_) self->resize_callback_(width, height);
    }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self->key_callback_) self->key_callback_(key, scancode, action, mods);
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self->mouse_button_callback_) self->mouse_button_callback_(button, action, mods);
    }

    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
        Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self->cursor_callback_) self->cursor_callback_(xpos, ypos);
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self->scroll_callback_) self->scroll_callback_(xoffset, yoffset);
    }

    int width_;
    int height_;
    GLFWwindow* window_;
    GLuint shader_program_;
    GLuint vao_;
    GLuint vbo_;
    GLuint texture_;
    float exposure_;

    ResizeCallback resize_callback_;
    KeyCallback key_callback_;
    MouseButtonCallback mouse_button_callback_;
    CursorCallback cursor_callback_;
    ScrollCallback scroll_callback_;
};

// =============================================================================
// Performance Timer
// =============================================================================

class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time_).count();
    }

    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// =============================================================================
// Frame Rate Counter
// =============================================================================

class FrameRateCounter {
public:
    FrameRateCounter() : frame_count_(0), fps_(0.0), last_time_(0.0) {}

    void update() {
        double current_time = glfwGetTime();
        frame_count_++;

        if (current_time - last_time_ >= 1.0) {
            fps_ = frame_count_ / (current_time - last_time_);
            frame_count_ = 0;
            last_time_ = current_time;
        }
    }

    double fps() const { return fps_; }

private:
    int frame_count_;
    double fps_;
    double last_time_;
};

} // namespace lumina
