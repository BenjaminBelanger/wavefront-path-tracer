#pragma once

#include "scene.cuh"

namespace wpt::scene_shapes {

void add_box(Scene& scene, const float3& center, const float3& size, float angle, int mat_id);
void add_uv_sphere(Scene& scene, const float3& center, float radius, int mat_id, int segments = 16);
void add_pyramid(Scene& scene, const float3& base_center, float base_size, float height, float angle, int mat_id);
void add_torus(
    Scene& scene,
    const float3& center,
    float major_radius,
    float minor_radius,
    float y_angle,
    int mat_id,
    int seg_major = 28,
    int seg_minor = 14
);
void add_octahedron(Scene& scene, const float3& center, float radius, float angle, int mat_id);

} // namespace wpt::scene_shapes
