#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "obj_loader.cuh"
#include "../core/math/matrix.cuh"
#include "../core/math/spectral.cuh"

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <cmath>

namespace lumina
{

    static bool is_non_zero(const float col[3])
    {
        return col[0] > 0.0f || col[1] > 0.0f || col[2] > 0.0f;
    }

    static float3 linear_color(const float col[3])
    {
        return srgb_to_linear(make_float3(col[0], col[1], col[2]));
    }

    static float specular_luminance(const tinyobj::material_t &mat)
    {
        return 0.2126f * mat.specular[0] + 0.7152f * mat.specular[1] + 0.0722f * mat.specular[2];
    }

    static float diffuse_luminance(const tinyobj::material_t &mat)
    {
        return 0.2126f * mat.diffuse[0] + 0.7152f * mat.diffuse[1] + 0.0722f * mat.diffuse[2];
    }

    static float shininess_to_roughness(float shininess)
    {
        return std::clamp(1.0f - sqrtf(shininess / 1000.0f), 0.02f, 1.0f);
    }

    static int load_material_texture(
        Scene &scene,
        const std::string &texname_in,
        const std::string &mtl_basedir,
        bool srgb)
    {
        if (texname_in.empty())
            return -1;

        std::string texname = texname_in;
        std::replace(texname.begin(), texname.end(), '\\', '/');

        bool is_absolute = (texname.size() >= 2 && texname[1] == ':') || (!texname.empty() && texname[0] == '/');
        std::string tex_path = is_absolute ? texname : mtl_basedir + texname;
        int tex_idx = scene.texture_manager().load_texture(tex_path, srgb);
        if (tex_idx >= 0 || !is_absolute)
            return tex_idx;

        size_t slash = texname.find_last_of('/');
        if (slash == std::string::npos)
            return tex_idx;

        std::string fallback = mtl_basedir + texname.substr(slash + 1);
        return scene.texture_manager().load_texture(fallback, srgb);
    }

    static void load_surface_textures(Material &m, Scene &scene, const tinyobj::material_t &mat, const std::string &mtl_basedir)
    {
        m.albedo_tex = load_material_texture(scene, mat.diffuse_texname, mtl_basedir, true);

        if (!mat.roughness_texname.empty())
            m.roughness_tex = load_material_texture(scene, mat.roughness_texname, mtl_basedir, false);

        if (!mat.normal_texname.empty())
            m.normal_tex = load_material_texture(scene, mat.normal_texname, mtl_basedir, false);
    }

    static int map_material(Scene &scene, const tinyobj::material_t &mat, const std::string &mtl_basedir)
    {
        Material m;

        bool has_pbr = mat.metallic > 0.0f || mat.roughness > 0.0f;

        if (is_non_zero(mat.emission))
        {
            float3 lin = linear_color(mat.emission);
            float magnitude = fmaxf(fmaxf(lin.x, lin.y), lin.z);
            float3 color = lin * (1.0f / magnitude);
            m = Material::emissive(color, magnitude);
        }
        else if (has_pbr)
        {
            float3 color = is_non_zero(mat.diffuse) ? linear_color(mat.diffuse) : make_float3(0.8f);
            float roughness = std::clamp(mat.roughness, 0.02f, 1.0f);

            if (mat.metallic >= 0.5f)
            {
                m = Material::metal(color, roughness);
            }
            else if (roughness < 1.0f)
            {
                m = Material::plastic(color, roughness, mat.ior > 0.0f ? mat.ior : 1.5f);
            }
            else
            {
                m = Material::diffuse(color);
            }
        }
        else if (mat.dissolve < 1.0f || mat.illum == 4 || mat.illum == 6 || mat.illum == 7)
        {
            float ior = mat.ior > 0.0f ? mat.ior : 1.5f;
            float roughness = mat.roughness > 0.0f ? mat.roughness : 0.0f;
            m = Material::glass(ior, roughness);
            if (is_non_zero(mat.transmittance))
            {
                m.albedo = linear_color(mat.transmittance);
            }
            else if (is_non_zero(mat.diffuse))
            {
                m.albedo = linear_color(mat.diffuse);
            }
        }
        else if (mat.illum == 3 || mat.illum == 5)
        {
            float roughness = mat.shininess > 0.0f ? shininess_to_roughness(mat.shininess) : 0.02f;
            float3 color = is_non_zero(mat.specular) ? linear_color(mat.specular) : make_float3(0.9f);
            m = Material::metal(color, roughness);
        }
        else if (is_non_zero(mat.specular))
        {
            float spec_lum = specular_luminance(mat);
            float diff_lum = diffuse_luminance(mat);

            if (spec_lum > 0.5f && diff_lum < 0.05f)
            {
                float roughness = mat.shininess > 0.0f ? shininess_to_roughness(mat.shininess) : 0.1f;
                m = Material::metal(linear_color(mat.specular), roughness);
            }
            else
            {
                float3 color = is_non_zero(mat.diffuse) ? linear_color(mat.diffuse) : make_float3(0.8f);
                float roughness = mat.shininess > 0.0f ? shininess_to_roughness(mat.shininess) : 0.5f;
                m = Material::plastic(color, roughness, mat.ior > 0.0f ? mat.ior : 1.5f);
            }
        }
        else
        {
            float3 color = is_non_zero(mat.diffuse) ? linear_color(mat.diffuse) : make_float3(0.8f);
            m = Material::diffuse(color);
        }

        load_surface_textures(m, scene, mat, mtl_basedir);

        return scene.add_material(m);
    }

    bool load_obj(Scene &scene, const std::string &filepath, const ObjLoadOptions &opts)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        std::string mtl_basedir;
        size_t last_slash = filepath.find_last_of("/\\");
        if (last_slash != std::string::npos)
        {
            mtl_basedir = filepath.substr(0, last_slash + 1);
        }

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                    filepath.c_str(), mtl_basedir.c_str(), true);

        if (!warn.empty())
        {
            std::cout << "OBJ Warning: " << warn << std::endl;
        }
        if (!err.empty())
        {
            std::cerr << "OBJ Error: " << err << std::endl;
        }
        if (!ret)
        {
            std::cerr << "Failed to load OBJ: " << filepath << std::endl;
            return false;
        }

        std::vector<int> material_map;
        material_map.reserve(materials.size());
        for (const auto &mat : materials)
        {
            material_map.push_back(map_material(scene, mat, mtl_basedir));
        }
        int default_material_id = scene.add_material(Material::diffuse(make_float3(0.8f)));

        bool has_normals = !attrib.normals.empty() && !opts.recalculate_normals;
        bool has_texcoords = !attrib.texcoords.empty();

        std::vector<float3> smooth_normals;
        if (!has_normals)
        {
            size_t vertex_count = attrib.vertices.size() / 3;
            smooth_normals.resize(vertex_count, make_float3(0.0f));

            for (const auto &shape : shapes)
            {
                size_t index_offset = 0;
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
                {
                    int fv = shape.mesh.num_face_vertices[f];
                    if (fv == 3)
                    {
                        int idx0 = shape.mesh.indices[index_offset + 0].vertex_index;
                        int idx1 = shape.mesh.indices[index_offset + 1].vertex_index;
                        int idx2 = shape.mesh.indices[index_offset + 2].vertex_index;

                        float3 p0 = make_float3(
                            attrib.vertices[3 * idx0 + 0],
                            attrib.vertices[3 * idx0 + 1],
                            attrib.vertices[3 * idx0 + 2]);
                        float3 p1 = make_float3(
                            attrib.vertices[3 * idx1 + 0],
                            attrib.vertices[3 * idx1 + 1],
                            attrib.vertices[3 * idx1 + 2]);
                        float3 p2 = make_float3(
                            attrib.vertices[3 * idx2 + 0],
                            attrib.vertices[3 * idx2 + 1],
                            attrib.vertices[3 * idx2 + 2]);

                        float3 face_normal = cross(p1 - p0, p2 - p0);
                        smooth_normals[idx0] = smooth_normals[idx0] + face_normal;
                        smooth_normals[idx1] = smooth_normals[idx1] + face_normal;
                        smooth_normals[idx2] = smooth_normals[idx2] + face_normal;
                    }
                    index_offset += fv;
                }
            }

            for (auto &n : smooth_normals)
            {
                float len = length(n);
                if (len > 0.0f)
                {
                    n = n * (1.0f / len);
                }
                else
                {
                    n = make_float3(0.0f, 1.0f, 0.0f);
                }
            }
        }

        bool needs_transform = opts.scale != 1.0f ||
                               opts.translation.x != 0.0f || opts.translation.y != 0.0f || opts.translation.z != 0.0f ||
                               opts.rotation.x != 0.0f || opts.rotation.y != 0.0f || opts.rotation.z != 0.0f;

        Matrix4x4 transform;
        Matrix4x4 normal_matrix;
        if (needs_transform)
        {
            transform = Matrix4x4::translate(opts.translation) *
                        Matrix4x4::rotate_z(opts.rotation.z) *
                        Matrix4x4::rotate_y(opts.rotation.y) *
                        Matrix4x4::rotate_x(opts.rotation.x) *
                        Matrix4x4::scale(opts.scale);
            normal_matrix = transpose(inverse(transform));
        }

        int total_triangles = 0;
        for (const auto &shape : shapes)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                int fv = shape.mesh.num_face_vertices[f];
                if (fv != 3)
                {
                    index_offset += fv;
                    continue;
                }

                Triangle tri;

                for (int i = 0; i < 3; i++)
                {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + i];

                    float3 pos = make_float3(
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]);

                    float3 normal;
                    if (has_normals && idx.normal_index >= 0)
                    {
                        normal = make_float3(
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]);
                    }
                    else
                    {
                        normal = smooth_normals[idx.vertex_index];
                    }

                    float2 uv = make_float2(0.0f, 0.0f);
                    if (has_texcoords && idx.texcoord_index >= 0)
                    {
                        uv = make_float2(
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            attrib.texcoords[2 * idx.texcoord_index + 1]);
                    }

                    if (needs_transform)
                    {
                        pos = transform_point(transform, pos);
                        normal = transform_normal(normal_matrix, normal);
                    }

                    if (i == 0)
                    {
                        tri.v0 = pos;
                        tri.n0 = normal;
                        tri.uv0 = uv;
                    }
                    else if (i == 1)
                    {
                        tri.v1 = pos;
                        tri.n1 = normal;
                        tri.uv1 = uv;
                    }
                    else
                    {
                        tri.v2 = pos;
                        tri.n2 = normal;
                        tri.uv2 = uv;
                    }
                }

                int mat_idx = shape.mesh.material_ids[f];
                tri.material_id = (mat_idx >= 0 && mat_idx < static_cast<int>(material_map.size()))
                                      ? material_map[mat_idx]
                                      : default_material_id;

                scene.add_triangle(tri);
                total_triangles++;

                index_offset += fv;
            }
        }

        std::cout << "Loaded OBJ: " << filepath << std::endl;
        std::cout << "  Triangles: " << total_triangles << std::endl;
        std::cout << "  Materials: " << materials.size() << std::endl;

        return true;
    }

}
