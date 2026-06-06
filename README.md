# Wavefront Path Tracer

A GPU-accelerated wavefront path tracer built from scratch in CUDA C++17. Renders physically-based scenes interactively at 1920x1080 with real-time camera control via CUDA-OpenGL interop. Supports OBJ mesh loading, HDRI environment maps, texture-mapped materials, and a physically-based shading system, with some experimental spectral and ReSTIR code also present in the repository.

<!-- If you have a screenshot, uncomment and add the path: -->
<!-- ![Wavefront Path Tracer render](assets/screenshot.png) -->

## Highlights

- **Wavefront path tracing** -- separates ray generation, intersection, shading, and accumulation into distinct GPU kernels for maximum occupancy and minimal warp divergence
- **Structure-of-Arrays (SoA) memory layout** -- all per-path and per-hit data is stored in SoA form (`PathStateSoA`, `HitInfoSoA`, `ReservoirSoA`) with `__restrict__` pointer views for coalesced global memory access
- **Physically-based material system** -- Lambertian, Oren-Nayar rough diffuse, GGX metal, rough/smooth dielectric, plastic, thin-film, and emissive materials
- **Texture-mapped materials** -- per-material albedo, roughness, and normal textures are sampled in the main shading path, and the OBJ loader wires common material texture fields when present
- **OBJ mesh loading** -- import arbitrary triangle meshes with per-vertex normals, UV coordinates, optional normal recalculation, and configurable scale/transform
- **HDRI environment mapping** -- HDR radiance environment maps with configurable intensity for image-based lighting
- **SAH-accelerated BVH** -- CPU-built surface area heuristic BVH with stackful GPU traversal, watertight triangle intersection, and dedicated shadow ray occlusion test
- **CUDA-OpenGL interop** -- zero-copy display via pixel buffer object (PBO); CUDA writes tonemapped pixels directly into the mapped OpenGL buffer each frame
- **ACES tonemapping + sRGB gamma** -- HDR accumulation buffer with progressive running average, ACES filmic curve, interactive exposure control, and proper linear-to-sRGB conversion

## Architecture

```
  Frame N
  =======
  generate_rays_kernel        1 ray per pixel, jittered + optional DOF
       |
       v
  +-----------  bounce loop (up to 8 bounces)   -----------+
  |                                                        |
  |  intersect_kernel         BVH traversal per active ray |
  |       |                                                |
  |  shade_miss_kernel        HDRI environment / gradient  |
  |       |                   sky fallback                 |
  |       |                                                |
  |  shade_surface_kernel     BSDF sampling, Russian       |
  |       |                   roulette (depth > 3),        |
  |       |                   emissive hit detection       |
  |       |                                                |
  |  swap work queues         compact via atomicAdd        |
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
./build/bin/Release/wavefront-path-tracer.exe [options]
```

Target CUDA architectures: SM 75 (Turing), SM 86 (Ampere), SM 89 (Ada Lovelace).

### Command-Line Options

| Option | Description | Default |
|---|---|---|
| `--width <n>` | Window / render width | 1920 |
| `--height <n>` | Window / render height | 1080 |
| `--scene <path>` | Load an OBJ mesh file | (built-in demo) |
| `--scale <float>` | Scale factor for loaded OBJ | 1.0 |
| `--hdri <path>` | HDR environment map (`.hdr`) | (gradient sky) |
| `--hdri-intensity <float>` | Environment map intensity | 2.0 |
| `--help` | Show usage | |

```powershell
# Example: load a mesh with an HDRI environment
./build/bin/Release/wavefront-path-tracer.exe --scene models/dragon.obj --scale 0.5 --hdri envmaps/studio.hdr
```

## Controls

| Input | Action |
|---|---|
| Left mouse drag | Orbit camera |
| Shift + drag | Pan camera |
| Ctrl + drag | Zoom |
| Scroll wheel | Zoom in/out |
| `+` / `-` | Adjust exposure |
| `R` | Reset accumulation |
| `P` | Save screenshot (PNG) |
| `Esc` | Quit |

The camera automatically frames loaded scenes based on world bounds.

Pressing `P` writes the current frame to `screenshots/screenshot_YYYYMMDD_HHMMSS.png` (the directory is created automatically next to the working directory).

## Project Structure

```
src/
  app/                      Entry point, window, camera, scene, demo scenes
  core/
    math/                   Vector/matrix ops, sampling, spectral rendering
    memory/                 DeviceBuffer RAII wrappers, CUDA_CHECK macros
    random/                 PCG32 RNG for device code
    texture/                Texture manager, CUDA texture objects, HDR/LDR loading
  geometry/
    bvh/                    BVH node layout, SAH builder, GPU traversal
    primitives/             Triangle, sphere helpers, AABB
  integrators/
    wavefront/              Wavefront kernels, path state (SoA), active/next work queues
  lighting/
    restir/                 Experimental ReSTIR DI prototype and reservoir helpers
  materials/
    bsdf/                   Lambert, Oren-Nayar, GGX conductor, dielectric,
                            plastic, thin film, emission
    spectral/               Sellmeier equation, Cauchy dispersion, complex IOR,
                            blackbody radiation, metal spectral data utilities
external/
  glad/                     Vendored OpenGL loader
```

## Technical Details

### Wavefront vs. Megakernel

Traditional GPU path tracers use a single megakernel per bounce. This renderer uses the wavefront architecture ([Laine et al. 2013](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus)) which splits each bounce into separate kernels. This eliminates register pressure from monolithic shaders and allows each kernel to run at optimal occupancy.

### SoA for Coalesced Access

All path state is stored in Structure-of-Arrays layout rather than AoS. When a warp of 32 threads accesses `ray_origin_x[thread_id]`, the loads coalesce into a single memory transaction. The `View` structs hold raw `__restrict__` device pointers passed to kernels, while `SoA` structs own the underlying `DeviceBuffer` allocations.

### Material System

| BSDF | Model | Sampling |
|---|---|---|
| Lambert | Cosine-weighted diffuse | Cosine hemisphere sampling |
| Oren-Nayar | Rough diffuse with angle-dependent reflectance | Cosine hemisphere sampling |
| Metal (GGX Conductor) | Microfacet metal; perfect mirror at zero roughness | VNDF (visible normal distribution function) |
| Dielectric | Smooth or rough glass with Fresnel reflection/transmission | VNDF + refraction via Snell's law |
| Plastic | Diffuse substrate with specular GGX coat | Cosine hemisphere + VNDF |
| Thin Film | Thin-film-inspired RGB interference model | Stochastic specular or diffuse branch |
| Emission | Emissive surface (area light) | N/A (light source) |

GGX sampling uses the [Heitz 2018](https://jcgt.org/published/0007/04/01/) VNDF method for importance sampling the visible microfacet normals, which gives zero-variance weighting in the specular limit.

### Texture Mapping

Per-material texture support includes albedo maps, roughness maps, and normal maps. Textures are loaded via the HDR/LDR image loaders and bound as CUDA texture objects for hardware-accelerated bilinear filtering. Color textures are decoded as sRGB while roughness and normal textures stay in linear space. A texture cache automatically deduplicates repeated loads.

### Spectral Utilities

The codebase includes hero-wavelength sampling helpers, CIE XYZ/sRGB conversion utilities, Sellmeier dispersion models for several dielectrics, and blackbody helper functions. The main interactive render loop on this branch still accumulates RGB radiance, so these spectral pieces are better thought of as supporting utilities and experimental groundwork than the default rendering path.

### Environment Mapping

HDRI environment maps (`.hdr` radiance format) provide image-based lighting. When no HDRI is loaded, a procedural gradient sky is used as fallback. For OBJ scenes without explicit lights, a default three-point lighting setup (key, fill, rim) is generated automatically.

### Experimental ReSTIR DI

The repository contains a reservoir-based direct-lighting prototype inspired by [Bitterli et al. 2020](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct), including alias-table sampling plus temporal and spatial resampling passes. It is not currently wired into the main interactive renderer loop described above.

### Geometry

The active renderer traces triangle geometry through the BVH. Triangles use watertight intersection with precomputed edge data and per-vertex normal/UV interpolation. Procedural shape generators (box, pyramid, torus, octahedron, UV sphere) are available for scene construction, and arbitrary meshes can be imported via OBJ loading with configurable scale and optional smooth normal recalculation.
