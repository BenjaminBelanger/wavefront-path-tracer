# Wavefront Path Tracer

A GPU-accelerated wavefront path tracer built from scratch in CUDA C++17. Renders physically-based scenes interactively at 1920x1080 with real-time camera control via CUDA-OpenGL interop.

<!-- If you have a screenshot, uncomment and add the path: -->
<!-- ![Wavefront Path Tracer render](assets/screenshot.png) -->

## Highlights

- **Wavefront path tracing** -- separates ray generation, intersection, shading, and accumulation into distinct GPU kernels for maximum occupancy and minimal warp divergence
- **Structure-of-Arrays (SoA) memory layout** -- all per-path and per-hit data is stored in SoA form (`PathStateSoA`, `HitInfoSoA`, `ReservoirSoA`) with `__restrict__` pointer views for coalesced global memory access
- **Physically-based material system** -- Lambertian diffuse, GGX microfacet conductor (VNDF importance sampling), rough/smooth dielectric with proper Fresnel transmission, and emissive materials
- **Spectral rendering** -- hero wavelength sampling across 380--780nm with CIE XYZ color matching functions and sRGB conversion; wavelength-dependent IOR via the Sellmeier equation (BK7, fused silica, SF11, diamond, sapphire, water)
- **ReSTIR DI** -- reservoir-based spatiotemporal importance sampling for direct illumination with alias table light selection
- **SAH-accelerated BVH** -- CPU-built surface area heuristic BVH with stackful GPU traversal and dedicated shadow ray occlusion test
- **CUDA-OpenGL interop** -- zero-copy display via pixel buffer object (PBO); CUDA writes tonemapped pixels directly into the mapped OpenGL buffer each frame
- **ACES tonemapping + sRGB gamma** -- HDR accumulation buffer with progressive running average, ACES filmic curve, and proper linear-to-sRGB conversion

## Architecture

```
  Frame N
  =======
  generate_rays_kernel        1 ray per pixel, jittered + optional DOF
       |
       v
  +-----------  bounce loop (up to 8 bounces)  -----------+
  |                                                        |
  |  intersect_kernel         BVH traversal per active ray |
  |       |                                                |
  |  shade_miss_kernel        gradient sky environment      |
  |       |                                                |
  |  shade_surface_kernel     BSDF sampling, Russian       |
  |       |                   roulette (depth > 3),        |
  |       |                   emissive hit detection        |
  |       |                                                |
  |  swap work queues         compact via atomicAdd         |
  +--------------------------------------------------------+
       |
  accumulate_kernel           progressive running average
       |
  tonemap_kernel              ACES filmic + sRGB output
       |
  OpenGL PBO blit             display via fullscreen quad
```

Active path compaction between bounces uses a pair of work queues swapped each iteration. Surviving paths are appended to the next queue via `atomicAdd`, naturally eliminating terminated paths without a separate stream compaction pass.

## Build & Run

**Requirements:** CUDA Toolkit (12.x recommended), OpenGL, CMake 3.24+, Visual Studio 2022

GLFW 3.3.9 and GLM 1.0.1 are fetched automatically via CMake FetchContent. GLAD is vendored.

```powershell
# One-step build and run
powershell -File run.ps1

# Manual build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target wavefront-path-tracer
./build/bin/Release/wavefront-path-tracer.exe [--width 1920] [--height 1080]
```

Target CUDA architectures: SM 75 (Turing), SM 86 (Ampere), SM 89 (Ada Lovelace).

## Controls

| Input | Action |
|---|---|
| Left mouse drag | Orbit camera |
| Shift + drag | Pan camera |
| Scroll wheel | Zoom in/out |
| `+` / `-` | Adjust exposure |
| `R` | Reset accumulation |
| `Esc` | Quit |

## Project Structure

```
src/
  app/                      Entry point, window, camera, scene, demo scenes
  core/
    math/                   Vector/matrix ops, sampling, spectral rendering
    memory/                 DeviceBuffer RAII wrappers, CUDA_CHECK macros
    random/                 PCG32 RNG for device code
  geometry/
    bvh/                    BVH node layout, SAH builder, GPU traversal
    primitives/             Triangle, Sphere, AABB
  integrators/
    wavefront/              Wavefront kernels, path state (SoA), work queues
  lighting/
    restir/                 ReSTIR DI reservoirs, alias table sampling
  materials/
    bsdf/                   Lambert, GGX conductor, dielectric, mirror, glass
    spectral/               Sellmeier equation, Cauchy dispersion, complex IOR
external/
  glad/                     Vendored OpenGL loader
```

~8,200 lines of CUDA/C++. No comments by design -- code is kept self-documenting.

## Technical Details

### Wavefront vs. Megakernel

Traditional GPU path tracers use a single megakernel per bounce. This renderer uses the wavefront architecture ([Laine et al. 2013](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus)) which splits each bounce into separate kernels. This eliminates register pressure from monolithic shaders and allows each kernel to run at optimal occupancy.

### SoA for Coalesced Access

All path state is stored in Structure-of-Arrays layout rather than AoS. When a warp of 32 threads accesses `ray_origin_x[thread_id]`, the loads coalesce into a single memory transaction. The `View` structs hold raw `__restrict__` device pointers passed to kernels, while `SoA` structs own the underlying `DeviceBuffer` allocations.

### Material System

| BSDF | Model | Sampling |
|---|---|---|
| Lambertian | Cosine-weighted diffuse | Cosine hemisphere sampling |
| GGX Conductor | Microfacet with Schlick Fresnel | VNDF (visible normal distribution function) |
| Dielectric | Microfacet with Fresnel transmission | VNDF + refraction via Snell's law |
| Mirror | Perfect specular reflection | Delta distribution |
| Glass | Smooth Fresnel reflection/refraction | Stochastic reflection vs. refraction |

GGX sampling uses the [Heitz 2018](https://jcgt.org/published/0007/04/01/) VNDF method for importance sampling the visible microfacet normals, which gives zero-variance weighting in the specular limit.

### Spectral Rendering

Rather than tracing fixed RGB, Wavefront Path Tracer samples a hero wavelength uniformly in [380, 780] nm and stratifies 3 additional wavelengths at equal spectral spacing. Each wavelength carries independent throughput, enabling physically correct dispersion through dielectrics. The Sellmeier dispersion model provides wavelength-dependent IOR for real glass types (BK7, SF11, fused silica, diamond, sapphire). Final spectral radiance is converted to CIE XYZ via the standard color matching functions, then to linear sRGB.

### ReSTIR Direct Illumination

Implements reservoir-based importance sampling ([Bitterli et al. 2020](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct)) for direct lighting. An alias table provides O(1) light selection, and per-pixel reservoirs accumulate weighted light samples with streaming updates.
