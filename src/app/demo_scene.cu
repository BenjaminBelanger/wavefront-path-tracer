#include "demo_scene.cuh"

#include <cmath>
#include <cstdio>

namespace lumina {

Scene create_demo_scene() {
    Scene scene;

    // Materials
    int white_diffuse = scene.add_material(Material::diffuse(::make_float3(0.73f, 0.73f, 0.73f)));
    int red_diffuse = scene.add_material(Material::diffuse(::make_float3(0.65f, 0.05f, 0.05f)));
    int green_diffuse = scene.add_material(Material::diffuse(::make_float3(0.12f, 0.45f, 0.15f)));
    int light_mat = scene.add_material(Material::emissive(::make_float3(1.0f, 0.95f, 0.8f), 16.0f));
    int mirror_mat = scene.add_material(Material::metal(::make_float3(0.95f, 0.95f, 0.95f), 0.02f));
    int chrome_mat = scene.add_material(Material::metal(::make_float3(0.92f, 0.94f, 0.98f), 0.05f));
    int gold_mat = scene.add_material(Material::metal(::make_float3(1.0f, 0.85f, 0.57f), 0.1f));
    int coral_diffuse = scene.add_material(Material::diffuse(::make_float3(0.86f, 0.42f, 0.31f)));
    int obsidian_metal = scene.add_material(Material::metal(::make_float3(0.14f, 0.16f, 0.2f), 0.35f));
    int velvet_violet = scene.add_material(Material::diffuse(::make_float3(0.44f, 0.24f, 0.64f)));
    int mango_diffuse = scene.add_material(Material::diffuse(::make_float3(0.94f, 0.62f, 0.2f)));
    int rose_gold_metal = scene.add_material(Material::metal(::make_float3(0.96f, 0.67f, 0.56f), 0.16f));
    int pearl_metal = scene.add_material(Material::metal(::make_float3(0.92f, 0.9f, 0.98f), 0.12f));
    int neon_pink_light = scene.add_material(Material::emissive(::make_float3(1.0f, 0.2f, 0.7f), 3.5f));
    int electric_blue_light = scene.add_material(Material::emissive(::make_float3(0.2f, 0.75f, 1.0f), 3.0f));

    // Cornell box dimensions
    float box_size = 7.0f;

    // Floor
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, 0.0f),
        ::make_float3(box_size, 0.0f, 0.0f),
        ::make_float3(box_size, 0.0f, box_size),
        white_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, 0.0f),
        ::make_float3(box_size, 0.0f, box_size),
        ::make_float3(0.0f, 0.0f, box_size),
        white_diffuse
    ));

    // Ceiling
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, box_size, 0.0f),
        ::make_float3(0.0f, box_size, box_size),
        ::make_float3(box_size, box_size, box_size),
        white_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, box_size, 0.0f),
        ::make_float3(box_size, box_size, box_size),
        ::make_float3(box_size, box_size, 0.0f),
        white_diffuse
    ));

    // Back wall
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, box_size),
        ::make_float3(box_size, 0.0f, box_size),
        ::make_float3(box_size, box_size, box_size),
        white_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, box_size),
        ::make_float3(box_size, box_size, box_size),
        ::make_float3(0.0f, box_size, box_size),
        white_diffuse
    ));

    // Left wall (red)
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, 0.0f),
        ::make_float3(0.0f, 0.0f, box_size),
        ::make_float3(0.0f, box_size, box_size),
        red_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(0.0f, 0.0f, 0.0f),
        ::make_float3(0.0f, box_size, box_size),
        ::make_float3(0.0f, box_size, 0.0f),
        red_diffuse
    ));

    // Right wall (green)
    scene.add_triangle(Triangle(
        ::make_float3(box_size, 0.0f, 0.0f),
        ::make_float3(box_size, box_size, 0.0f),
        ::make_float3(box_size, box_size, box_size),
        green_diffuse
    ));
    scene.add_triangle(Triangle(
        ::make_float3(box_size, 0.0f, 0.0f),
        ::make_float3(box_size, box_size, box_size),
        ::make_float3(box_size, 0.0f, box_size),
        green_diffuse
    ));

    // Light on ceiling
    float light_size = 2.0f;
    float light_y = box_size - 0.01f;
    float light_center = box_size / 2.0f;
    scene.add_triangle(Triangle(
        ::make_float3(light_center - light_size/2, light_y, light_center - light_size/2),
        ::make_float3(light_center + light_size/2, light_y, light_center - light_size/2),
        ::make_float3(light_center + light_size/2, light_y, light_center + light_size/2),
        light_mat
    ));
    scene.add_triangle(Triangle(
        ::make_float3(light_center - light_size/2, light_y, light_center - light_size/2),
        ::make_float3(light_center + light_size/2, light_y, light_center + light_size/2),
        ::make_float3(light_center - light_size/2, light_y, light_center + light_size/2),
        light_mat
    ));

    // Neon accent panels on the back wall
    scene.add_triangle(Triangle(
        ::make_float3(1.0f, 2.0f, box_size - 0.01f),
        ::make_float3(2.1f, 2.0f, box_size - 0.01f),
        ::make_float3(2.1f, 3.5f, box_size - 0.01f),
        neon_pink_light
    ));
    scene.add_triangle(Triangle(
        ::make_float3(1.0f, 2.0f, box_size - 0.01f),
        ::make_float3(2.1f, 3.5f, box_size - 0.01f),
        ::make_float3(1.0f, 3.5f, box_size - 0.01f),
        neon_pink_light
    ));
    scene.add_triangle(Triangle(
        ::make_float3(4.9f, 1.7f, box_size - 0.01f),
        ::make_float3(6.1f, 1.7f, box_size - 0.01f),
        ::make_float3(6.1f, 2.8f, box_size - 0.01f),
        electric_blue_light
    ));
    scene.add_triangle(Triangle(
        ::make_float3(4.9f, 1.7f, box_size - 0.01f),
        ::make_float3(6.1f, 2.8f, box_size - 0.01f),
        ::make_float3(4.9f, 2.8f, box_size - 0.01f),
        electric_blue_light
    ));

    // Helper to add a box
    auto add_box = [&](float3 center, float3 size, float angle, int mat_id) {
        float c = cosf(angle);
        float s = sinf(angle);
        float hx = size.x * 0.5f;
        float hy = size.y * 0.5f;
        float hz = size.z * 0.5f;

        auto rotate = [c, s, center](float3 p) {
            float x = p.x - center.x;
            float z = p.z - center.z;
            return ::make_float3(
                center.x + c * x - s * z,
                p.y,
                center.z + s * x + c * z
            );
        };

        float3 corners[8] = {
            rotate(::make_float3(center.x - hx, center.y - hy, center.z - hz)),
            rotate(::make_float3(center.x + hx, center.y - hy, center.z - hz)),
            rotate(::make_float3(center.x + hx, center.y - hy, center.z + hz)),
            rotate(::make_float3(center.x - hx, center.y - hy, center.z + hz)),
            rotate(::make_float3(center.x - hx, center.y + hy, center.z - hz)),
            rotate(::make_float3(center.x + hx, center.y + hy, center.z - hz)),
            rotate(::make_float3(center.x + hx, center.y + hy, center.z + hz)),
            rotate(::make_float3(center.x - hx, center.y + hy, center.z + hz)),
        };

        // Front
        scene.add_triangle(Triangle(corners[0], corners[1], corners[5], mat_id));
        scene.add_triangle(Triangle(corners[0], corners[5], corners[4], mat_id));
        // Back
        scene.add_triangle(Triangle(corners[2], corners[3], corners[7], mat_id));
        scene.add_triangle(Triangle(corners[2], corners[7], corners[6], mat_id));
        // Left
        scene.add_triangle(Triangle(corners[3], corners[0], corners[4], mat_id));
        scene.add_triangle(Triangle(corners[3], corners[4], corners[7], mat_id));
        // Right
        scene.add_triangle(Triangle(corners[1], corners[2], corners[6], mat_id));
        scene.add_triangle(Triangle(corners[1], corners[6], corners[5], mat_id));
        // Top
        scene.add_triangle(Triangle(corners[4], corners[5], corners[6], mat_id));
        scene.add_triangle(Triangle(corners[4], corners[6], corners[7], mat_id));
    };

    // Helper to add a sphere (tessellated)
    auto add_sphere = [&](float3 center, float radius, int mat_id, int segments = 16) {
        auto add_smooth_sphere_triangle = [&](const float3& a, const float3& b, const float3& c) {
            Triangle tri(a, b, c, mat_id);
            tri.n0 = normalize(a - center);
            tri.n1 = normalize(b - center);
            tri.n2 = normalize(c - center);
            scene.add_triangle(tri);
        };

        for (int i = 0; i < segments; i++) {
            for (int j = 0; j < segments * 2; j++) {
                float theta0 = PI * float(i) / float(segments);
                float theta1 = PI * float(i + 1) / float(segments);
                float phi0 = TWO_PI * float(j) / float(segments * 2);
                float phi1 = TWO_PI * float(j + 1) / float(segments * 2);

                float3 p00 = center + radius * ::make_float3(
                    sinf(theta0) * cosf(phi0),
                    cosf(theta0),
                    sinf(theta0) * sinf(phi0)
                );
                float3 p01 = center + radius * ::make_float3(
                    sinf(theta0) * cosf(phi1),
                    cosf(theta0),
                    sinf(theta0) * sinf(phi1)
                );
                float3 p10 = center + radius * ::make_float3(
                    sinf(theta1) * cosf(phi0),
                    cosf(theta1),
                    sinf(theta1) * sinf(phi0)
                );
                float3 p11 = center + radius * ::make_float3(
                    sinf(theta1) * cosf(phi1),
                    cosf(theta1),
                    sinf(theta1) * sinf(phi1)
                );

                if (i > 0) {
                    add_smooth_sphere_triangle(p00, p10, p01);
                }
                if (i < segments - 1) {
                    add_smooth_sphere_triangle(p10, p11, p01);
                }
            }
        }
    };

    // Helper to add a 4-sided pyramid on the floor (open bottom).
    auto add_pyramid = [&](float3 base_center, float base_size, float height, float angle, int mat_id) {
        float c = cosf(angle);
        float s = sinf(angle);
        float h = base_size * 0.5f;

        auto rotate = [c, s, base_center](float3 p) {
            float x = p.x - base_center.x;
            float z = p.z - base_center.z;
            return ::make_float3(
                base_center.x + c * x - s * z,
                p.y,
                base_center.z + s * x + c * z
            );
        };

        float3 b0 = rotate(::make_float3(base_center.x - h, base_center.y, base_center.z - h));
        float3 b1 = rotate(::make_float3(base_center.x + h, base_center.y, base_center.z - h));
        float3 b2 = rotate(::make_float3(base_center.x + h, base_center.y, base_center.z + h));
        float3 b3 = rotate(::make_float3(base_center.x - h, base_center.y, base_center.z + h));
        float3 apex = ::make_float3(base_center.x, base_center.y + height, base_center.z);

        scene.add_triangle(Triangle(b0, b1, apex, mat_id));
        scene.add_triangle(Triangle(b1, b2, apex, mat_id));
        scene.add_triangle(Triangle(b2, b3, apex, mat_id));
        scene.add_triangle(Triangle(b3, b0, apex, mat_id));
    };

    // Helper to add a torus with smooth normals.
    auto add_torus = [&](float3 center, float major_radius, float minor_radius, float y_angle,
                         int mat_id, int seg_major = 28, int seg_minor = 14) {
        float c = cosf(y_angle);
        float s = sinf(y_angle);

        auto rotate_y = [c, s](const float3& p) {
            return ::make_float3(
                c * p.x - s * p.z,
                p.y,
                s * p.x + c * p.z
            );
        };

        auto add_smooth_torus_triangle = [&](const float3& p0, const float3& p1, const float3& p2,
                                             const float3& n0, const float3& n1, const float3& n2) {
            Triangle tri(p0, p1, p2, mat_id);
            tri.n0 = normalize(n0);
            tri.n1 = normalize(n1);
            tri.n2 = normalize(n2);
            scene.add_triangle(tri);
        };

        for (int i = 0; i < seg_major; ++i) {
            int i1 = (i + 1) % seg_major;
            float u0 = TWO_PI * float(i) / float(seg_major);
            float u1 = TWO_PI * float(i1) / float(seg_major);

            for (int j = 0; j < seg_minor; ++j) {
                int j1 = (j + 1) % seg_minor;
                float v0 = TWO_PI * float(j) / float(seg_minor);
                float v1 = TWO_PI * float(j1) / float(seg_minor);

                auto torus_pos = [&](float u, float v) {
                    float ring = major_radius + minor_radius * cosf(v);
                    return ::make_float3(
                        ring * cosf(u),
                        minor_radius * sinf(v),
                        ring * sinf(u)
                    );
                };

                auto torus_normal = [&](float u, float v) {
                    return normalize(::make_float3(
                        cosf(u) * cosf(v),
                        sinf(v),
                        sinf(u) * cosf(v)
                    ));
                };

                float3 lp00 = torus_pos(u0, v0);
                float3 lp10 = torus_pos(u1, v0);
                float3 lp01 = torus_pos(u0, v1);
                float3 lp11 = torus_pos(u1, v1);

                float3 ln00 = torus_normal(u0, v0);
                float3 ln10 = torus_normal(u1, v0);
                float3 ln01 = torus_normal(u0, v1);
                float3 ln11 = torus_normal(u1, v1);

                float3 p00 = center + rotate_y(lp00);
                float3 p10 = center + rotate_y(lp10);
                float3 p01 = center + rotate_y(lp01);
                float3 p11 = center + rotate_y(lp11);

                float3 n00 = rotate_y(ln00);
                float3 n10 = rotate_y(ln10);
                float3 n01 = rotate_y(ln01);
                float3 n11 = rotate_y(ln11);

                add_smooth_torus_triangle(p00, p10, p01, n00, n10, n01);
                add_smooth_torus_triangle(p10, p11, p01, n10, n11, n01);
            }
        }
    };

    // Helper to add an octahedron.
    auto add_octahedron = [&](float3 center, float radius, float angle, int mat_id) {
        float c = cosf(angle);
        float s = sinf(angle);

        auto rotate_y = [c, s, center](float3 p) {
            float x = p.x - center.x;
            float z = p.z - center.z;
            return ::make_float3(
                center.x + c * x - s * z,
                p.y,
                center.z + s * x + c * z
            );
        };

        float3 top = ::make_float3(center.x, center.y + radius, center.z);
        float3 bottom = ::make_float3(center.x, center.y - radius, center.z);

        float3 m0 = rotate_y(::make_float3(center.x + radius, center.y, center.z));
        float3 m1 = rotate_y(::make_float3(center.x, center.y, center.z + radius));
        float3 m2 = rotate_y(::make_float3(center.x - radius, center.y, center.z));
        float3 m3 = rotate_y(::make_float3(center.x, center.y, center.z - radius));

        scene.add_triangle(Triangle(top, m0, m1, mat_id));
        scene.add_triangle(Triangle(top, m1, m2, mat_id));
        scene.add_triangle(Triangle(top, m2, m3, mat_id));
        scene.add_triangle(Triangle(top, m3, m0, mat_id));

        scene.add_triangle(Triangle(bottom, m1, m0, mat_id));
        scene.add_triangle(Triangle(bottom, m2, m1, mat_id));
        scene.add_triangle(Triangle(bottom, m3, m2, mat_id));
        scene.add_triangle(Triangle(bottom, m0, m3, mat_id));
    };

    // Main hero objects
    add_sphere(::make_float3(1.45f, 1.0f, 4.8f), 1.0f, mirror_mat, 24);
    add_sphere(::make_float3(3.45f, 0.9f, 2.35f), 0.9f, chrome_mat, 24);
    add_sphere(::make_float3(5.35f, 0.75f, 4.65f), 0.75f, gold_mat, 24);
    add_sphere(::make_float3(1.9f, 0.42f, 1.65f), 0.42f, velvet_violet, 16);

    // New playful shapes
    add_pyramid(::make_float3(5.45f, 0.0f, 1.95f), 1.0f, 1.35f, 0.42f, coral_diffuse);
    add_octahedron(::make_float3(2.55f, 1.1f, 3.25f), 0.62f, 0.5f, mango_diffuse);
    add_torus(::make_float3(4.05f, 1.65f, 3.8f), 0.78f, 0.24f, 0.35f, pearl_metal, 24, 12);

    // Accent cubes (separated and more visible from the default camera)
    add_box(::make_float3(6.25f, 0.45f, 3.6f), ::make_float3(0.9f, 0.9f, 0.9f), 0.12f, obsidian_metal);
    add_box(::make_float3(4.25f, 0.36f, 4.35f), ::make_float3(0.72f, 0.72f, 0.72f), -0.5f, rose_gold_metal);

    // Keep global scene scaling as a single tuning knob.
    const float scene_scale = 1.0f;
    scene.scale_geometry(scene_scale, ::make_float3(box_size * 0.5f, box_size * 0.5f, box_size * 0.5f));

    // Build BVH
    scene.build();

    printf("Scene created: %d triangles, %d materials\n",
           scene.num_triangles(), scene.num_materials());

    return scene;
}

} // namespace lumina
