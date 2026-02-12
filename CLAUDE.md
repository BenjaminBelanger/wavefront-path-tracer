# Lumina - Spectral ReSTIR Ray Tracer

## Project Overview

Lumina is a GPU-accelerated spectral ray tracer implementing cutting-edge rendering techniques:

- **Hero Wavelength Spectral Rendering** - Full spectral light transport with dispersion
- **ReSTIR DI** - Reservoir-based importance sampling for many lights
- **Wavefront Path Tracing** - Production-quality GPU architecture

## Build Instructions

```bash
# Configure with CMake
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

# Build
cmake --build . --config Release
```

### Requirements
- CUDA Toolkit 12.0+ (tested with 13.1)
- Visual Studio 2022 (MSVC v143)
- CMake 3.24+
- OpenGL 4.5+ capable GPU

## Architecture

```
src/
├── core/           # Math, memory, random number generation
├── geometry/       # BVH, primitives (triangle, sphere)
├── materials/      # BSDFs (Lambert, GGX, dielectric)
├── lighting/       # ReSTIR implementation
├── integrators/    # Wavefront path tracer
└── app/            # Renderer, camera, window
```

## Key Files

| File | Purpose |
|------|---------|
| `core/math/spectral.cuh` | Hero wavelength sampling, XYZ conversion |
| `core/math/vector.cuh` | GPU-friendly vector operations |
| `geometry/bvh/bvh.cuh` | BVH traversal with warp-level optimization |
| `materials/bsdf/ggx.cuh` | GGX microfacet BRDF with VNDF sampling |
| `materials/spectral/sellmeier.cuh` | Wavelength-dependent IOR |
| `lighting/restir/reservoir.cuh` | ReSTIR reservoir data structure |
| `integrators/wavefront/path_state.cuh` | SoA path state for coalescing |

## CUDA Optimization Patterns Used

1. **SoA Layout** - All path state uses Structure of Arrays for coalesced memory access
2. **Warp-Cooperative BVH** - Traversal designed for warp coherence
3. **Material Sorting** - Rays sorted by material for coherent BSDF evaluation
4. **Persistent Threads** - Wavefront architecture keeps all SMs busy

## Current Status

**Phase 1: Foundation** - Complete
- [x] Math library (vector, matrix, sampling)
- [x] GPU memory management (RAII buffers)
- [x] PCG random number generator
- [x] BVH construction and traversal
- [x] Triangle/sphere intersection
- [x] Lambert BSDF
- [x] Wavefront architecture

**Phase 2: Spectral** - In Progress
- [x] Hero wavelength sampling
- [x] XYZ/sRGB conversion
- [x] Sellmeier IOR equations
- [ ] Spectral BSDFs integration

**Phase 3: ReSTIR** - In Progress
- [x] Reservoir data structure
- [x] Initial candidate generation
- [x] Temporal resampling
- [x] Spatial resampling
- [ ] Full integration with path tracer

## Running

```bash
./bin/lumina --width 1920 --height 1080
```

Controls:
- Left drag: Orbit camera
- Shift+drag: Pan
- Scroll: Zoom
- R: Reset accumulation
- +/-: Adjust exposure
- ESC: Quit
