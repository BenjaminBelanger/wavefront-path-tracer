#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "camera.cuh"

struct GLFWwindow;

namespace wpt {

class Scene;

class InteractiveRenderer {
public:
    InteractiveRenderer(int width, int height);
    ~InteractiveRenderer();
    InteractiveRenderer(const InteractiveRenderer&) = delete;
    InteractiveRenderer& operator=(const InteractiveRenderer&) = delete;

    void reset_accumulation();
    void render_frame(const Camera& camera, const Scene& scene);
    void tonemap();
    void download_display(uchar4* host_buffer);

    int total_samples() const;
    float& exposure();

private:
    struct Impl;
    Impl* impl_;
};

class RenderWindow {
public:
    RenderWindow(int width, int height, const char* title);
    ~RenderWindow();

    bool is_valid() const;
    bool should_close() const;

    void run(InteractiveRenderer& renderer, Scene& scene);

    CameraController& camera_controller();
    void set_camera_changed();

private:
    void init_gl_resources();
    void display_frame();

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

    int width_;
    int height_;
    GLFWwindow* window_;
    unsigned int shader_program_;
    unsigned int vao_;
    unsigned int vbo_;
    unsigned int texture_;
    std::vector<unsigned char> pixel_buffer_;
    CameraController controller_;
    bool camera_changed_;
    InteractiveRenderer* renderer_;
};

} // namespace wpt
