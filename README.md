# Wavefront Path Tracer

A GPU-accelerated wavefront path tracer built from scratch in CUDA C++17. Renders physically-based scenes interactively at 1920x1080 with real-time camera control via CUDA-OpenGL interop. Supports OBJ mesh loading, HDRI environment maps, spectral rendering, and a full physically-based material system.

<!-- If you have a screenshot, uncomment and add the path: -->
<!-- ![Wavefront Path Tracer render](assets/screenshot.png) -->

## Highlights

- **Wavefront path tracing** -- separates ray generation, intersection, shading, and accumulation into distinct GPU kernels for maximum occupancy and minimal warp divergence
- **Structure-of-Arrays (SoA) memory layout** -- all per-path and per-hit data is stored in SoA form (`PathStateSoA`, `HitInfoSoA`, `ReservoirSoA`) with `__restrict__` pointer views for coalesced global memory access
- **Physically-based material system** -- Lambertian, Oren-Nayar rough diffuse, GGX microfacet conductor (VNDF importance sampling), rough/smooth dielectric with Fresnel transmission, thin film interference, and emissive materials with spectral complex-IOR metals (gold, silver, copper, aluminum)
- **Spectral rendering** -- hero wavelength sampling across 380--780nm with CIE XYZ color matching functions and sRGB conversion; wavelength-dependent IOR via the Sellmeier equation (BK7, fused silica, SF11, diamond, sapphire, water); blackbody radiation from temperature
- **OBJ mesh loading** -- import arbitrary triangle meshes with per-vertex normals, UV coordinates, optional normal recalculation, and configurable scale/transform
- **HDRI environment mapping** -- HDR radiance environment maps with configurable intensity for image-based lighting
- **Texture system** -- per-material albedo, roughness, and normal map textures via CUDA texture objects with automatic deduplication
- **ReSTIR DI** -- reservoir-based spatiotemporal importance sampling for direct illumination with alias table light selection
- **SAH-accelerated BVH** -- CPU-built surface area heuristic BVH with stackful GPU traversal, watertight triangle intersection, and dedicated shadow ray occlusion test
- **Material-sorted work queues** -- 8 material-type queues reduce warp divergence by grouping paths with the same BSDF before shading
- **CUDA-OpenGL interop** -- zero-copy display via pixel buffer object (PBO); CUDA writes tonemapped pixels directly into the mapped OpenGL buffer each frame
- **ACES tonemapping + sRGB gamma** -- HDR accumulation buffer with progressive running average, ACES filmic curve, interactive exposure control, and proper linear-to-sRGB conversion

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
  |  shade_miss_kernel        HDRI environment / gradient   |
  |       |                   sky fallback                  |
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

Active path compaction between bounces uses a pair of work queues swapped each iteration. Surviving paths are appended to the next queue via `atomicAdd`, naturally eliminating terminated paths without a separate stream compaction pass. Paths are sorted into material-type queues so that warps shade coherent BSDFs together.

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
| `Esc` | Quit |

The camera automatically frames loaded scenes based on world bounds.

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
    primitives/             Triangle, Sphere, Ellipsoid, AABB
  integrators/
    wavefront/              Wavefront kernels, path state (SoA), work queues
  lighting/
    restir/                 ReSTIR DI reservoirs, alias table sampling
  materials/
    bsdf/                   Lambert, Oren-Nayar, GGX conductor, dielectric,
                            mirror, glass, thin film
    spectral/               Sellmeier equation, Cauchy dispersion, complex IOR,
                            blackbody radiation, metal spectral data
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
| Lambertian | Cosine-weighted diffuse | Cosine hemisphere sampling |
| Oren-Nayar | Rough diffuse with angle-dependent reflectance | Cosine hemisphere sampling |
| GGX Conductor | Microfacet with spectral complex-IOR Fresnel | VNDF (visible normal distribution function) |
| Dielectric | Microfacet with Fresnel transmission | VNDF + refraction via Snell's law |
| Rough Dielectric | GGX-based rough glass with transmission | VNDF + stochastic refraction |
| Mirror | Perfect specular reflection | Delta distribution |
| Glass | Smooth Fresnel reflection/refraction | Stochastic reflection vs. refraction |
| Thin Film | Coherent thin film interference colors | Spectral interference superposition |

GGX sampling uses the [Heitz 2018](https://jcgt.org/published/0007/04/01/) VNDF method for importance sampling the visible microfacet normals, which gives zero-variance weighting in the specular limit. Conductor materials support spectral complex IOR with built-in data for gold, silver, copper, and aluminum.

### Texture Mapping

Per-material texture support includes albedo maps, roughness maps, and normal maps. Textures are loaded via the HDR/LDR image loaders and bound as CUDA texture objects for hardware-accelerated bilinear filtering. A texture cache automatically deduplicates repeated loads.

### Spectral Rendering

Rather than tracing fixed RGB, Wavefront Path Tracer samples a hero wavelength uniformly in [380, 780] nm and stratifies 3 additional wavelengths at equal spectral spacing. Each wavelength carries independent throughput, enabling physically correct dispersion through dielectrics. The Sellmeier dispersion model provides wavelength-dependent IOR for real glass types (BK7, SF11, fused silica, diamond, sapphire). Final spectral radiance is converted to CIE XYZ via the standard color matching functions, then to linear sRGB. Blackbody radiation is available for emissive materials driven by color temperature.

### Environment Mapping

HDRI environment maps (`.hdr` radiance format) provide image-based lighting. When no HDRI is loaded, a procedural gradient sky is used as fallback. For OBJ scenes without explicit lights, a default three-point lighting setup (key, fill, rim) is generated automatically.

### ReSTIR Direct Illumination

Implements reservoir-based importance sampling ([Bitterli et al. 2020](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct)) for direct lighting. An alias table provides O(1) light selection, and per-pixel reservoirs accumulate weighted light samples with streaming updates.

### Geometry

Supported primitives include triangles, spheres, and ellipsoids. Triangles use watertight intersection with precomputed edge data and per-vertex normal/UV interpolation. Procedural shape generators (box, pyramid, torus, octahedron, UV sphere) are available for scene construction. Arbitrary meshes can be imported via OBJ loading with configurable scale and optional smooth normal recalculation.
