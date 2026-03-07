#include "demo_scene.cuh"
#include "scene_shapes.cuh"

#include <cstdio>

namespace lumina
{

    Scene create_demo_scene()
    {
        Scene scene;

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

        float box_size = 7.0f;

        scene.add_triangle(Triangle(
            ::make_float3(0.0f, 0.0f, 0.0f),
            ::make_float3(box_size, 0.0f, 0.0f),
            ::make_float3(box_size, 0.0f, box_size),
            white_diffuse));
        scene.add_triangle(Triangle(
            ::make_float3(0.0f, 0.0f, 0.0f),
            ::make_float3(box_size, 0.0f, box_size),
            ::make_float3(0.0f, 0.0f, box_size),
            white_diffuse));

        scene.add_triangle(Triangle(
            ::make_float3(0.0f, box_size, 0.0f),
            ::make_float3(0.0f, box_size, box_size),
            ::make_float3(box_size, box_size, box_size),
            white_diffuse));
        scene.add_triangle(Triangle(
            ::make_float3(0.0f, box_size, 0.0f),
            ::make_float3(box_size, box_size, box_size),
            ::make_float3(box_size, box_size, 0.0f),
            white_diffuse));

        scene.add_triangle(Triangle(
            ::make_float3(0.0f, 0.0f, box_size),
            ::make_float3(box_size, 0.0f, box_size),
            ::make_float3(box_size, box_size, box_size),
            white_diffuse));
        scene.add_triangle(Triangle(
            ::make_float3(0.0f, 0.0f, box_size),
            ::make_float3(box_size, box_size, box_size),
            ::make_float3(0.0f, box_size, box_size),
            white_diffuse));

        scene.add_triangle(Triangle(
            ::make_float3(0.0f, 0.0f, 0.0f),
            ::make_float3(0.0f, 0.0f, box_size),
            ::make_float3(0.0f, box_size, box_size),
            red_diffuse));
        scene.add_triangle(Triangle(
            ::make_float3(0.0f, 0.0f, 0.0f),
            ::make_float3(0.0f, box_size, box_size),
            ::make_float3(0.0f, box_size, 0.0f),
            red_diffuse));

        scene.add_triangle(Triangle(
            ::make_float3(box_size, 0.0f, 0.0f),
            ::make_float3(box_size, box_size, 0.0f),
            ::make_float3(box_size, box_size, box_size),
            green_diffuse));
        scene.add_triangle(Triangle(
            ::make_float3(box_size, 0.0f, 0.0f),
            ::make_float3(box_size, box_size, box_size),
            ::make_float3(box_size, 0.0f, box_size),
            green_diffuse));

        float light_size = 2.0f;
        float light_y = box_size - 0.01f;
        float light_center = box_size / 2.0f;
        scene.add_triangle(Triangle(
            ::make_float3(light_center - light_size / 2, light_y, light_center - light_size / 2),
            ::make_float3(light_center + light_size / 2, light_y, light_center - light_size / 2),
            ::make_float3(light_center + light_size / 2, light_y, light_center + light_size / 2),
            light_mat));
        scene.add_triangle(Triangle(
            ::make_float3(light_center - light_size / 2, light_y, light_center - light_size / 2),
            ::make_float3(light_center + light_size / 2, light_y, light_center + light_size / 2),
            ::make_float3(light_center - light_size / 2, light_y, light_center + light_size / 2),
            light_mat));

        scene.add_triangle(Triangle(
            ::make_float3(1.0f, 2.0f, box_size - 0.01f),
            ::make_float3(2.1f, 2.0f, box_size - 0.01f),
            ::make_float3(2.1f, 3.5f, box_size - 0.01f),
            neon_pink_light));
        scene.add_triangle(Triangle(
            ::make_float3(1.0f, 2.0f, box_size - 0.01f),
            ::make_float3(2.1f, 3.5f, box_size - 0.01f),
            ::make_float3(1.0f, 3.5f, box_size - 0.01f),
            neon_pink_light));
        scene.add_triangle(Triangle(
            ::make_float3(4.9f, 1.7f, box_size - 0.01f),
            ::make_float3(6.1f, 1.7f, box_size - 0.01f),
            ::make_float3(6.1f, 2.8f, box_size - 0.01f),
            electric_blue_light));
        scene.add_triangle(Triangle(
            ::make_float3(4.9f, 1.7f, box_size - 0.01f),
            ::make_float3(6.1f, 2.8f, box_size - 0.01f),
            ::make_float3(4.9f, 2.8f, box_size - 0.01f),
            electric_blue_light));

        scene_shapes::add_uv_sphere(scene, ::make_float3(1.45f, 1.0f, 4.8f), 1.0f, mirror_mat, 24);
        scene_shapes::add_uv_sphere(scene, ::make_float3(3.45f, 0.9f, 2.35f), 0.9f, chrome_mat, 24);
        scene_shapes::add_uv_sphere(scene, ::make_float3(5.35f, 0.75f, 4.65f), 0.75f, gold_mat, 24);
        scene_shapes::add_uv_sphere(scene, ::make_float3(1.9f, 0.42f, 1.65f), 0.42f, velvet_violet, 16);

        scene_shapes::add_pyramid(scene, ::make_float3(5.45f, 0.0f, 1.95f), 1.0f, 1.35f, 0.42f, coral_diffuse);
        scene_shapes::add_octahedron(scene, ::make_float3(2.55f, 1.1f, 3.25f), 0.62f, 0.5f, mango_diffuse);
        scene_shapes::add_torus(scene, ::make_float3(4.05f, 1.65f, 3.8f), 0.78f, 0.24f, 0.35f, pearl_metal, 24, 12);

        scene_shapes::add_box(scene, ::make_float3(6.25f, 0.45f, 3.6f), ::make_float3(0.9f, 0.9f, 0.9f), 0.12f, obsidian_metal);
        scene_shapes::add_box(scene, ::make_float3(4.25f, 0.36f, 4.35f), ::make_float3(0.72f, 0.72f, 0.72f), -0.5f, rose_gold_metal);

        const float scene_scale = 1.0f;
        scene.scale_geometry(scene_scale, ::make_float3(box_size * 0.5f, box_size * 0.5f, box_size * 0.5f));

        scene.build();

        printf("Scene created: %d triangles, %d materials\n",
               scene.num_triangles(), scene.num_materials());

        return scene;
    }

}
