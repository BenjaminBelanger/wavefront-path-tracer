#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "demo_scene.cuh"
#include "interactive_runtime.cuh"
#include "../io/obj_loader.cuh"

using namespace lumina;

void print_cuda_info()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0)
    {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return;
    }

    std::cout << "=== Lumina Ray Tracer ===" << std::endl;
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

int main(int argc, char **argv)
{
    print_cuda_info();

    int width = 1920;
    int height = 1080;
    std::string scene_path;
    float scene_scale = 1.0f;

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
        else if (arg == "--help")
        {
            std::cout << "Usage: lumina [options]" << std::endl;
            std::cout << "  --width <n>       Window width (default: 1920)" << std::endl;
            std::cout << "  --height <n>      Window height (default: 1080)" << std::endl;
            std::cout << "  --scene <path>    Load OBJ file" << std::endl;
            std::cout << "  --scale <float>   Scale factor for OBJ (default: 1.0)" << std::endl;
            return 0;
        }
    }

    std::cout << "\nResolution: " << width << "x" << height << std::endl;

    std::cout << "\nBuilding scene..." << std::endl;
    Scene scene;
    if (!scene_path.empty())
    {
        ObjLoadOptions opts;
        opts.scale = scene_scale;
        if (!load_obj(scene, scene_path, opts))
        {
            std::cerr << "Failed to load scene, falling back to demo scene." << std::endl;
            scene = create_demo_scene();
        }
        else
        {
            scene.build();
        }
    }
    else
    {
        scene = create_demo_scene();
    }

    RenderWindow window(width, height, "Lumina Path Tracer");

    if (!window.is_valid())
    {
        std::cerr << "Failed to create window!" << std::endl;
        return 1;
    }

    if (!scene_path.empty())
    {
        AABB bounds = scene.world_bounds();
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

    std::cout << "\nLumina ray tracer exited successfully." << std::endl;
    return 0;
}
