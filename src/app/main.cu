#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "demo_scene.cuh"
#include "interactive_runtime.cuh"

using namespace wpt;

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

    std::cout << "=== Wavefront Path Tracer ===" << std::endl;
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
            std::cout << "Usage: wavefront-path-tracer [options]" << std::endl;
            std::cout << "  --width <n>   Window width (default: 1920)" << std::endl;
            std::cout << "  --height <n>  Window height (default: 1080)" << std::endl;
            return 0;
        }
    }

    std::cout << "\nResolution: " << width << "x" << height << std::endl;

    // Create scene
    std::cout << "\nBuilding scene..." << std::endl;
    Scene scene = create_demo_scene();

    // Create window first so CUDA/GL interop registers against an active GL context.
    RenderWindow window(width, height, "Wavefront Path Tracer");

    if (!window.is_valid()) {
        std::cerr << "Failed to create window!" << std::endl;
        return 1;
    }

    // Create renderer
    std::cout << "Initializing renderer..." << std::endl;
    InteractiveRenderer renderer(width, height);

    window.run(renderer, scene);

    std::cout << "\nWavefront Path Tracer exited successfully." << std::endl;
    return 0;
}
