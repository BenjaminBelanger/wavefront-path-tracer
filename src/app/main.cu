#include <iostream>
#include <string>
#include <cstdlib>

#include <cuda_runtime.h>

#include "demo_scene.cuh"
#include "interactive_runtime.cuh"
#include "../io/obj_loader.cuh"

using namespace wpt;

// On NVIDIA Optimus laptops the OpenGL context is created on the integrated GPU
// by default, which breaks CUDA-OpenGL interop (CUDA runs on the discrete GPU).
// Exporting these symbols forces the system to select the high-performance NVIDIA
// GPU for this process, keeping OpenGL and CUDA on the same device.
#if defined(_WIN32)
extern "C"
{
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

void print_cuda_info()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0)
    {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return;
    }

    std::cout << "=== Wavefront Path Tracer ===" << std::endl;
    std::cout << "CUDA Devices: " << device_count << std::endl;

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  SM count: " << prop.multiProcessorCount << std::endl;
    }
    std::cout << "\n=========================" << std::endl;
}

static bool ffmpeg_available()
{
#if defined(_WIN32)
    return std::system("ffmpeg -version >NUL 2>&1") == 0;
#else
    return std::system("ffmpeg -version >/dev/null 2>&1") == 0;
#endif
}

static std::string build_ffmpeg_command(const std::string &dir, int fps, const std::string &format)
{
    const std::string input = dir + "/frame%04d.png";
    std::string cmd = "ffmpeg -y -framerate " + std::to_string(fps) + " -i \"" + input + "\" ";
    if (format == "gif")
    {
        cmd += "-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" \"" +
               dir + "/animation.gif\"";
    }
    else
    {
        cmd += "-c:v libx264 -pix_fmt yuv420p -crf 18 -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" \"" +
               dir + "/animation.mp4\"";
    }
    return cmd;
}

// Map a --material preset name to a Material. Lets bare OBJ test meshes (no MTL)
// be rendered as gold, jade, chrome, etc. for hero shots. Returns false if unknown.
static bool material_from_preset(const std::string &name, Material &out)
{
    if (name == "gold")
        out = Material::metal(::make_float3(1.0f, 0.78f, 0.34f), 0.08f);
    else if (name == "chrome")
        out = Material::metal(::make_float3(0.95f, 0.96f, 0.98f), 0.02f);
    else if (name == "copper")
        out = Material::metal(::make_float3(0.95f, 0.64f, 0.54f), 0.12f);
    else if (name == "silver")
        out = Material::metal(::make_float3(0.97f, 0.96f, 0.91f), 0.05f);
    else if (name == "obsidian")
        out = Material::metal(::make_float3(0.05f, 0.05f, 0.07f), 0.1f);
    else if (name == "jade")
        out = Material::plastic(::make_float3(0.18f, 0.55f, 0.4f), 0.22f, 1.5f);
    else if (name == "pearl")
        out = Material::metal(::make_float3(0.92f, 0.9f, 0.98f), 0.12f);
    else if (name == "marble" || name == "white")
        out = Material::diffuse(::make_float3(0.9f, 0.9f, 0.88f));
    else if (name == "glass")
        out = Material::glass(1.5f, 0.0f);
    else if (name == "plastic")
        out = Material::plastic(::make_float3(0.8f, 0.2f, 0.2f), 0.2f, 1.5f);
    else
        return false;
    return true;
}

int main(int argc, char **argv)
{
    print_cuda_info();

    int width = 1920;
    int height = 1080;
    std::string scene_path;
    std::string hdri_path;
    float hdri_intensity = 2.0f;
    float scene_scale = 1.0f;
    bool add_floor = false;
    std::string material_preset;
    bool cornell = false;
    bool headless = false;
    int frames = 256;
    std::string output_path = "render.png";
    float cam_yaw = 0.0f;
    float cam_pitch = 0.0f;
    float cam_zoom = 1.0f;
    float model_yaw = 0.0f;
    float model_pitch = 0.0f;
    float model_roll = 0.0f;

    bool animate = false;
    AnimationConfig anim_config;
    std::string output_dir = "animation";
    bool make_video = false;
    std::string video_format = "mp4";
    int fps = 30;
    bool spp_set = false;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc)
        {
            width = std::stoi(argv[++i]);
        }
        else if (arg == "--height" && i + 1 < argc)
        {
            height = std::stoi(argv[++i]);
        }
        else if (arg == "--scene" && i + 1 < argc)
        {
            scene_path = argv[++i];
        }
        else if (arg == "--scale" && i + 1 < argc)
        {
            scene_scale = std::stof(argv[++i]);
        }
        else if (arg == "--hdri" && i + 1 < argc)
        {
            hdri_path = argv[++i];
        }
        else if (arg == "--hdri-intensity" && i + 1 < argc)
        {
            hdri_intensity = std::stof(argv[++i]);
        }
        else if (arg == "--floor")
        {
            add_floor = true;
        }
        else if (arg == "--material" && i + 1 < argc)
        {
            material_preset = argv[++i];
        }
        else if (arg == "--cornell")
        {
            cornell = true;
        }
        else if (arg == "--headless")
        {
            headless = true;
        }
        else if ((arg == "--frames" || arg == "--spp") && i + 1 < argc)
        {
            frames = std::stoi(argv[++i]);
            spp_set = true;
        }
        else if (arg == "--animate" && i + 1 < argc)
        {
            std::string preset_name = argv[++i];
            if (!parse_animation_preset(preset_name, anim_config.preset))
            {
                std::cerr << "Unknown animation preset '" << preset_name
                          << "'. Valid presets: orbit, dolly." << std::endl;
                return 1;
            }
            animate = true;
        }
        else if (arg == "--anim-frames" && i + 1 < argc)
        {
            anim_config.frames = std::stoi(argv[++i]);
        }
        else if (arg == "--fps" && i + 1 < argc)
        {
            fps = std::stoi(argv[++i]);
        }
        else if (arg == "--orbit-degrees" && i + 1 < argc)
        {
            anim_config.orbit_degrees = std::stof(argv[++i]);
        }
        else if (arg == "--dolly-start" && i + 1 < argc)
        {
            anim_config.dolly_start = std::stof(argv[++i]);
        }
        else if (arg == "--dolly-end" && i + 1 < argc)
        {
            anim_config.dolly_end = std::stof(argv[++i]);
        }
        else if (arg == "--output-dir" && i + 1 < argc)
        {
            output_dir = argv[++i];
        }
        else if (arg == "--video")
        {
            make_video = true;
        }
        else if (arg == "--video-format" && i + 1 < argc)
        {
            video_format = argv[++i];
            if (video_format != "mp4" && video_format != "gif")
            {
                std::cerr << "Unknown --video-format '" << video_format
                          << "'. Valid formats: mp4, gif." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc)
        {
            output_path = argv[++i];
        }
        else if (arg == "--cam-yaw" && i + 1 < argc)
        {
            cam_yaw = std::stof(argv[++i]);
        }
        else if (arg == "--cam-pitch" && i + 1 < argc)
        {
            cam_pitch = std::stof(argv[++i]);
        }
        else if (arg == "--cam-zoom" && i + 1 < argc)
        {
            cam_zoom = std::stof(argv[++i]);
        }
        else if (arg == "--model-yaw" && i + 1 < argc)
        {
            model_yaw = std::stof(argv[++i]);
        }
        else if (arg == "--model-pitch" && i + 1 < argc)
        {
            model_pitch = std::stof(argv[++i]);
        }
        else if (arg == "--model-roll" && i + 1 < argc)
        {
            model_roll = std::stof(argv[++i]);
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: wavefront-path-tracer [options]" << std::endl;
            std::cout << "  --width <n>       Render width (default: 1920)" << std::endl;
            std::cout << "  --height <n>      Render height (default: 1080)" << std::endl;
            std::cout << "  --scene <path>    Load OBJ file" << std::endl;
            std::cout << "  --scale <float>   Scale factor for OBJ (default: 1.0)" << std::endl;
            std::cout << "  --hdri <path>     HDR environment map (.hdr)" << std::endl;
            std::cout << "  --hdri-intensity <float>  Env map intensity (default: 2.0)" << std::endl;
            std::cout << "  --floor           Add a neutral ground plane under a loaded OBJ (contact shadows)" << std::endl;
            std::cout << "  --material <name> Override a loaded OBJ's material: gold, chrome, copper, silver, obsidian, jade, pearl, marble, glass, plastic" << std::endl;
            std::cout << "  --cornell         Render the built-in classic Cornell box (ignored if --scene is given)" << std::endl;
            std::cout << "  --headless        Render to PNG without a window (no display needed)" << std::endl;
            std::cout << "  --frames <n>      Samples per pixel in headless/animation mode (alias --spp, default: 256; 64 for animation)" << std::endl;
            std::cout << "  --output <path>   Output PNG path in headless mode (alias -o, default: render.png)" << std::endl;
            std::cout << "  --cam-yaw <deg>   Orbit the auto-framed camera horizontally (default: 0 = straight-on)" << std::endl;
            std::cout << "  --cam-pitch <deg> Orbit the auto-framed camera vertically (default: 0; e.g. --cam-yaw 45 --cam-pitch 25 for a 3/4 corner view)" << std::endl;
            std::cout << "  --cam-zoom <mult> Scale the auto-framed distance (default: 1.0; <1 zooms in, >1 zooms out)" << std::endl;
            std::cout << "  --model-yaw <deg>   Rotate a loaded OBJ about the vertical (Y) axis, e.g. --model-yaw 180 to face the camera" << std::endl;
            std::cout << "  --model-pitch <deg> Rotate a loaded OBJ about the X axis (default: 0)" << std::endl;
            std::cout << "  --model-roll <deg>  Rotate a loaded OBJ about the Z axis (default: 0)" << std::endl;
            std::cout << "\n Headless camera animation (implies --headless):" << std::endl;
            std::cout << "  --animate <preset>  Render a camera-path video. Presets: orbit, dolly" << std::endl;
            std::cout << "  --anim-frames <n>   Number of frames in the animation (default: 120)" << std::endl;
            std::cout << "  --fps <n>           Frames per second for video encoding (default: 30)" << std::endl;
            std::cout << "  --orbit-degrees <d> Total yaw sweep for the orbit preset (default: 360)" << std::endl;
            std::cout << "  --dolly-start <m>   Dolly distance multiplier at start (default: 1.0)" << std::endl;
            std::cout << "  --dolly-end <m>     Dolly distance multiplier at end (default: 0.45)" << std::endl;
            std::cout << "  --output-dir <dir>  Directory for the PNG frame sequence (default: animation)" << std::endl;
            std::cout << "  --video             Encode the frame sequence to a video via ffmpeg" << std::endl;
            std::cout << "  --video-format <f>  Video container: mp4 or gif (default: mp4)" << std::endl;
            return 0;
        }
    }

    if (animate)
        headless = true;

    std::cout << "\nResolution: " << width << "x" << height << std::endl;

    std::cout << "\nBuilding scene..." << std::endl;
    Scene scene;
    if (!scene_path.empty())
    {
        ObjLoadOptions opts;
        opts.scale = scene_scale;
        opts.rotation = ::make_float3(
            model_pitch * (PI / 180.0f),
            model_yaw * (PI / 180.0f),
            model_roll * (PI / 180.0f));
        if (!load_obj(scene, scene_path, opts))
        {
            std::cerr << "Failed to load scene, falling back to demo scene." << std::endl;
            scene = create_demo_scene();
        }
        else
        {
            if (!material_preset.empty())
            {
                Material mat;
                if (material_from_preset(material_preset, mat))
                    scene.set_all_triangle_material(mat);
                else
                    std::cerr << "Unknown --material '" << material_preset
                              << "'. Valid: gold, chrome, copper, silver, obsidian, jade, pearl, marble, glass, plastic." << std::endl;
            }
            if (!hdri_path.empty())
            {
                scene.set_environment_map(hdri_path);
                scene.set_environment_intensity(hdri_intensity);
            }
            else
                scene.add_default_lighting();
            if (add_floor)
                scene.add_ground_plane();
            scene.build();
        }
    }
    else if (cornell)
    {
        scene = Scene::create_cornell_box();
        if (!hdri_path.empty())
        {
            scene.set_environment_map(hdri_path);
            scene.set_environment_intensity(hdri_intensity);
        }
    }
    else
    {
        scene = create_demo_scene();
        if (!hdri_path.empty())
        {
            scene.set_environment_map(hdri_path);
            scene.set_environment_intensity(hdri_intensity);
        }
    }

    if (headless)
    {
        Camera camera;
        camera.fov = PI / 4.0f;
        camera.aspect_ratio = float(width) / float(height);

        bool framed = false;
        AABB bounds = scene.framing_bounds();
        if (bounds.is_valid())
        {
            float3 center = bounds.center();
            float radius = length(bounds.extent()) * 0.5f;
            float distance = camera.frame_distance(radius) * fmaxf(cam_zoom, 0.01f);

            float yaw = cam_yaw * (PI / 180.0f);
            float pitch = clamp(cam_pitch * (PI / 180.0f), -PI * 0.49f, PI * 0.49f);
            float3 dir = ::make_float3(
                sinf(yaw) * cosf(pitch),
                sinf(pitch),
                -cosf(yaw) * cosf(pitch));

            float3 cam_pos = center + dir * distance;
            camera.look_at(cam_pos, center, ::make_float3(0.0f, 1.0f, 0.0f));
            framed = true;
        }
        if (!framed)
        {
            camera.look_at(
                ::make_float3(3.5f, 3.1f, -7.2f),
                ::make_float3(3.5f, 2.2f, 3.5f),
                ::make_float3(0.0f, 1.0f, 0.0f));
        }
        camera.update();

        std::cout << "Initializing renderer..." << std::endl;
        InteractiveRenderer renderer(width, height);

        if (animate)
        {
            int spp = spp_set ? frames : 64;
            int rc = run_animation(renderer, scene, camera, width, height, spp, anim_config, output_dir);
            if (rc == 0 && make_video)
            {
                std::string cmd = build_ffmpeg_command(output_dir, fps, video_format);
                if (ffmpeg_available())
                {
                    std::cout << "\nEncoding video:\n  " << cmd << std::endl;
                    int vrc = std::system(cmd.c_str());
                    if (vrc != 0)
                        std::cerr << "ffmpeg exited with code " << vrc
                                  << "; the PNG frame sequence is still available." << std::endl;
                }
                else
                {
                    std::cout << "\nffmpeg was not found on PATH; the frame sequence was written but not encoded.\n"
                              << "To create the video, install ffmpeg and run:\n  " << cmd << std::endl;
                }
            }
            std::cout << "\nWavefront Path Tracer exited successfully." << std::endl;
            return rc;
        }

        int rc = run_headless(renderer, scene, camera, width, height, frames, output_path);
        std::cout << "\nWavefront Path Tracer exited successfully." << std::endl;
        return rc;
    }

    RenderWindow window(width, height, "Wavefront Path Tracer");

    if (!window.is_valid())
    {
        std::cerr << "Failed to create window!" << std::endl;
        return 1;
    }

    if (!scene_path.empty())
    {
        AABB bounds = scene.framing_bounds();
        if (bounds.is_valid())
        {
            float3 center = bounds.center();
            float radius = length(bounds.extent()) * 0.5f;
            window.frame_scene(center, radius);
        }
    }

    std::cout << "Initializing renderer..." << std::endl;
    InteractiveRenderer renderer(width, height);

    window.run(renderer, scene);

    std::cout << "\nWavefront Path Tracer exited successfully." << std::endl;
    return 0;
}
