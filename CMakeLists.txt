cmake_minimum_required(VERSION 3.24)

# Enable CUDA language
project(WavefrontPathTracer LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - target modern GPUs (Ampere and newer for best features)
# Can be overridden with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find OpenGL
find_package(OpenGL REQUIRED)

# GLFW - try to find system installation, otherwise use FetchContent
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.9
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

# GLM for CPU-side math (GLFW interop)
find_package(glm QUIET)
if(NOT glm_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 1.0.1
    )
    FetchContent_MakeAvailable(glm)
endif()

# Source files
set(CORE_SOURCES
    src/core/math/vector.cuh
    src/core/math/matrix.cuh
    src/core/math/sampling.cuh
    src/core/math/spectral.cuh
    src/core/memory/device_buffer.cuh
    src/core/memory/soa_array.cuh
    src/core/random/pcg.cuh
)

set(GEOMETRY_SOURCES
    src/geometry/primitives/triangle.cuh
    src/geometry/primitives/sphere.cuh
    src/geometry/primitives/aabb.cuh
    src/geometry/bvh/bvh.cuh
    src/geometry/bvh/lbvh_builder.cu
)

set(MATERIAL_SOURCES
    src/materials/bsdf/bsdf.cuh
    src/materials/bsdf/lambert.cuh
    src/materials/bsdf/ggx.cuh
    src/materials/bsdf/dielectric.cuh
    src/materials/spectral/sellmeier.cuh
)

set(LIGHTING_SOURCES
    src/lighting/restir/reservoir.cuh
    src/lighting/restir/restir_di.cu
)

set(INTEGRATOR_SOURCES
    src/integrators/wavefront/path_state.cuh
    src/integrators/wavefront/ray_queue.cuh
    src/integrators/wavefront/kernels.cu
)

set(APP_SOURCES
    src/app/camera.cuh
    src/app/scene.cuh
    src/app/renderer.cu
    src/app/window.cpp
    src/app/main.cu
)

# GLAD OpenGL loader
set(GLAD_SOURCES
    external/glad/src/glad.c
)

# Header-only external libraries and GLAD
set(EXTERNAL_INCLUDES
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/external/glad/include
)

# Create executable
add_executable(wavefront-path-tracer
    ${CORE_SOURCES}
    ${GEOMETRY_SOURCES}
    ${MATERIAL_SOURCES}
    ${LIGHTING_SOURCES}
    ${INTEGRATOR_SOURCES}
    ${APP_SOURCES}
    ${GLAD_SOURCES}
)

# Include directories
target_include_directories(wavefront-path-tracer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${EXTERNAL_INCLUDES}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(wavefront-path-tracer PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
    glfw
    glm::glm
)

# CUDA-specific compile options
target_compile_options(wavefront-path-tracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler=/W3
        -lineinfo
    >
)

# Enable separable compilation for device code
set_target_properties(wavefront-path-tracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(wavefront-path-tracer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
    )
    target_compile_definitions(wavefront-path-tracer PRIVATE WAVEFRONT_PATH_TRACER_DEBUG)
endif()

# Release optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(wavefront-path-tracer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif()

# Copy assets to build directory
add_custom_command(TARGET wavefront-path-tracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:wavefront-path-tracer>/assets
)

# Print configuration summary
message(STATUS "=== Wavefront Path Tracer Configuration ===")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "============================")
