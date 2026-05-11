cmake_minimum_required(VERSION 3.24)

project(WavefrontPathTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(CUDAToolkit REQUIRED)

find_package(OpenGL REQUIRED)

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

include(FetchContent)
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG v2.0.0rc13
)
FetchContent_Populate(tinyobjloader)

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

set(IO_SOURCES
    src/io/obj_loader.cuh
    src/io/obj_loader.cu
)

set(APP_SOURCES
    src/app/camera.cuh
    src/app/scene.cuh
    src/app/interactive_runtime.cuh
    src/app/renderer_runtime.cu
    src/app/window_runtime.cpp
    src/app/demo_scene.cuh
    src/app/demo_scene.cu
    src/app/scene_shapes.cuh
    src/app/scene_shapes.cu
    src/app/main.cu
)

set(GLAD_SOURCES
    external/glad/src/glad.c
)

set(EXTERNAL_INCLUDES
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/external/glad/include
    ${tinyobjloader_SOURCE_DIR}
)

add_executable(wavefront-path-tracer
    ${CORE_SOURCES}
    ${GEOMETRY_SOURCES}
    ${MATERIAL_SOURCES}
    ${LIGHTING_SOURCES}
    ${INTEGRATOR_SOURCES}
    ${IO_SOURCES}
    ${APP_SOURCES}
    ${GLAD_SOURCES}
)

target_include_directories(wavefront-path-tracer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${EXTERNAL_INCLUDES}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(wavefront-path-tracer PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
    glfw
    glm::glm
)

target_compile_options(wavefront-path-tracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler=/W3
        -lineinfo
    >
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -g>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
)
target_compile_definitions(wavefront-path-tracer PRIVATE
    $<$<CONFIG:Debug>:WAVEFRONT_PATH_TRACER_DEBUG>
)

set_target_properties(wavefront-path-tracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_custom_command(TARGET wavefront-path-tracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:wavefront-path-tracer>/assets
)

message(STATUS "=== Wavefront Path Tracer Configuration ===")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "============================")
