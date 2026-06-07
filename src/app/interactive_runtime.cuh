#pragma once

#include <string>

#include <cuda_runtime.h>

#include "camera.cuh"
#include "camera_animation.cuh"

struct GLFWwindow;
struct cudaGraphicsResource;

namespace wpt
{

    class Scene;

    class InteractiveRenderer
    {
    public:
        InteractiveRenderer(int width, int height);
        ~InteractiveRenderer();
        InteractiveRenderer(const InteractiveRenderer &) = delete;
        InteractiveRenderer &operator=(const InteractiveRenderer &) = delete;

        void reset_accumulation();
        void render_frame(const Camera &camera, const Scene &scene);
        void tonemap_to_buffer(uchar4 *device_buffer);

        int total_samples() const;
        float &exposure();

    private:
        struct Impl;
        Impl *impl_;
    };

    class RenderWindow
    {
    public:
        RenderWindow(int width, int height, const char *title);
        ~RenderWindow();

        bool is_valid() const;
        bool should_close() const;

        void run(InteractiveRenderer &renderer, Scene &scene);
        void frame_scene(float3 center, float radius);

        CameraController &camera_controller();
        void set_camera_changed();

    private:
        void init_gl_resources();
        void display_frame();
        void save_screenshot(const uchar4 *device_buffer);

        static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
        static void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);
        static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
        static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);

        int width_;
        int height_;
        GLFWwindow *window_;
        unsigned int shader_program_;
        unsigned int vao_;
        unsigned int vbo_;
        unsigned int texture_;
        unsigned int pbo_;
        cudaGraphicsResource *cuda_pbo_resource_;
        CameraController controller_;
        bool camera_changed_;
        bool screenshot_requested_;
        InteractiveRenderer *renderer_;
    };

    // Headless rendering: accumulates `frames` samples into a plain CUDA buffer
    // and writes the tonemapped result to a PNG. Requires no window, GL context
    // or display, so it works over a bare SSH session on a cloud GPU.
    int run_headless(InteractiveRenderer &renderer, Scene &scene, const Camera &camera,
                     int width, int height, int frames, const std::string &output_path);

    // Headless camera animation: renders a sequence of frames along a built-in
    // camera path preset (orbit/dolly), accumulating `spp` samples per frame and
    // writing frameNNNN.png into `output_dir`. Requires no window or display.
    int run_animation(InteractiveRenderer &renderer, Scene &scene, const Camera &base_camera,
                      int width, int height, int spp, const AnimationConfig &config,
                      const std::string &output_dir);

}
