# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Lumina** is a GPU-accelerated wavefront path tracer written in CUDA/C++17. It renders interactively via a GLFW/OpenGL window with CUDA-OpenGL interop (PBO). The project targets Windows with Visual Studio 2022.

## Build & Run

```powershell
# Configure, build, and run (all-in-one)
powershell -File run.ps1

# Or manually:
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target lumina
./build/bin/Release/lumina.exe [--width 1920] [--height 1080]
```

Requires: CUDA Toolkit, OpenGL. Dependencies (GLFW 3.3.9, GLM 1.0.1) are auto-fetched via CMake FetchContent. GLAD is vendored in `external/glad/`.

Target CUDA architectures: 75, 86, 89 (Turing/Ampere/Ada Lovelace).

## Architecture

### Render Loop (wavefront path tracing)

The renderer (`InteractiveRenderer` in `src/app/renderer_runtime.cu`) runs a wavefront loop per frame:

1. **Generate rays** - one ray per pixel with jittered sampling + optional DOF
2. **Bounce loop** (up to `max_depth=8`):
   - **Intersect** - BVH traversal on GPU against triangle primitives
   - **Shade miss** - gradient sky environment
   - **Shade surface** - BSDF sampling, Russian roulette at depth > 3, emissive hit detection
   - **Swap work queues** - active paths compacted via atomic counter
3. **Accumulate** - progressive running average into accumulation buffer
4. **Tonemap** - ACES tonemapping + sRGB conversion into display PBO

Kernel launch functions are declared in `renderer_runtime.cu` and defined in `src/integrators/wavefront/kernels.cu`.

### SoA (Structure of Arrays) Pattern

All per-path and per-hit data uses SoA layout for GPU coalesced memory access:
- `PathStateSoA` / `PathStateView` - ray state, throughput, radiance, RNG, spectral data
- `HitInfoSoA` / `HitInfoView` - intersection results
- `ReservoirSoA` / `ReservoirView` - ReSTIR reservoirs

Each SoA struct owns `DeviceBuffer`s; the corresponding `View` struct holds raw device pointers for use in kernels. Convert with `make_view()`.

### Key Modules

| Directory | Purpose |
|---|---|
| `src/app/` | Entry point, window (GLFW+OpenGL), camera controller, scene definition, demo scenes |
| `src/core/math/` | Vector/matrix ops, sampling utilities, spectral rendering helpers |
| `src/core/memory/` | `DeviceBuffer`, `PinnedBuffer`, `ManagedBuffer` RAII wrappers + `CUDA_CHECK` macros |
| `src/core/random/` | PCG32 RNG for device code |
| `src/geometry/bvh/` | BVH node structure, SAH-based CPU builder (`bvh_builder.cu`), GPU traversal |
| `src/geometry/primitives/` | Triangle, Sphere, AABB |
| `src/integrators/wavefront/` | Wavefront kernels, path state, work queues with material sorting |
| `src/lighting/restir/` | ReSTIR DI (reservoir-based importance sampling for direct illumination) |
| `src/materials/bsdf/` | Material system: Lambert, GGX microfacet, dielectric; unified `sample_bsdf()` dispatch |
| `src/materials/spectral/` | Sellmeier equation for wavelength-dependent IOR |

### CUDA-OpenGL Interop

`RenderWindow` (in `window_runtime.cpp`, compiled as C++) creates a PBO registered with CUDA. Each frame, CUDA writes tonemapped `uchar4` pixels into the mapped PBO, which is then blitted to a fullscreen quad via an OpenGL texture. This file is `.cpp` (not `.cu`) due to OpenGL/GLAD header conflicts with NVCC.

### Scene Construction

Scenes are built on the CPU: add triangles/spheres, materials, and lights to a `Scene` object, then call `scene.build()` which constructs the BVH (SAH on CPU) and uploads all data to GPU buffers. See `Scene::create_cornell_box()` and `create_demo_scene()` for examples.

## Conventions

- All code is in the `lumina` namespace
- CUDA device/host annotations: `__host__ __device__` on shared math/structs
- Header-only `.cuh` files for types/inline functions; `.cu` files for kernel implementations
- Error checking via `CUDA_CHECK(call)` and `CUDA_CHECK_LAST()` macros (defined in `device_buffer.cuh`)
- Include paths are relative from `src/` (e.g., `#include "../../core/math/vector.cuh"`)
- `clean.ps1` strips all C-style comments from the codebase - code has no comments by design
- `LUMINA_DEBUG` is defined in Debug builds
