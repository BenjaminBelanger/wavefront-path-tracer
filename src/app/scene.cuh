#pragma once

#include "../core/math/vector.cuh"
#include "../core/memory/device_buffer.cuh"
#include "../geometry/bvh/bvh.cuh"
#include "../geometry/primitives/triangle.cuh"
#include "../geometry/primitives/sphere.cuh"
#include "../materials/bsdf/bsdf.cuh"
#include <vector>

namespace wpt {





enum class LightType : uint32_t {
    Point = 0,
    Directional,
    Area,       
    Environment,
    Sphere,
    COUNT
};

struct Light {
    LightType type;
    float3 position;      
    float3 direction;     
    float3 emission;      
    float radius;         
    int triangle_offset;  
    int triangle_count;   
    float total_area;     

    __host__ __device__ Light()
        : type(LightType::Point)
        , position(make_float3(0.0f))
        , direction(make_float3(0.0f, -1.0f, 0.0f))
        , emission(make_float3(1.0f))
        , radius(0.1f)
        , triangle_offset(0)
        , triangle_count(0)
        , total_area(0.0f)
    {}

    __host__ static Light point(const float3& pos, const float3& color, float intensity) {
        Light l;
        l.type = LightType::Point;
        l.position = pos;
        l.emission = color * intensity;
        return l;
    }

    __host__ static Light directional(const float3& dir, const float3& color, float intensity) {
        Light l;
        l.type = LightType::Directional;
        l.direction = normalize(dir);
        l.emission = color * intensity;
        return l;
    }

    __host__ static Light sphere(const float3& pos, float r, const float3& color, float intensity) {
        Light l;
        l.type = LightType::Sphere;
        l.position = pos;
        l.radius = r;
        l.emission = color * intensity;
        return l;
    }
};





class Scene {
public:
    Scene() = default;

    
    void add_triangle(const Triangle& tri) {
        triangles_.push_back(tri);
    }

    void add_mesh(const std::vector<Triangle>& mesh, int material_id) {
        for (const auto& tri : mesh) {
            Triangle t = tri;
            t.material_id = material_id;
            triangles_.push_back(t);
        }
    }

    void add_sphere(const Sphere& sphere) {
        spheres_.push_back(sphere);
    }

    
    void scale_geometry(float scale, const float3& pivot) {
        if (scale == 1.0f) return;

        for (auto& tri : triangles_) {
            tri.v0 = pivot + (tri.v0 - pivot) * scale;
            tri.v1 = pivot + (tri.v1 - pivot) * scale;
            tri.v2 = pivot + (tri.v2 - pivot) * scale;
        }

        for (auto& sphere : spheres_) {
            sphere.center = pivot + (sphere.center - pivot) * scale;
            sphere.radius *= scale;
        }
    }

    
    int add_material(const Material& mat) {
        int id = static_cast<int>(materials_.size());
        materials_.push_back(mat);
        return id;
    }

    
    void add_light(const Light& light) {
        lights_.push_back(light);
    }

    
    void build() {
        if (!triangles_.empty()) {
            bvh_.build(triangles_.data(), static_cast<int>(triangles_.size()));
        }

        
        if (!materials_.empty()) {
            materials_gpu_.upload(materials_.data(), materials_.size());
        }
        if (!lights_.empty()) {
            lights_gpu_.upload(lights_.data(), lights_.size());
        }
        if (!spheres_.empty()) {
            spheres_gpu_.upload(spheres_.data(), spheres_.size());
        }
    }

    
    const BVHNode* bvh_nodes() const { return bvh_.nodes(); }
    const Triangle* triangles() const { return bvh_.primitives(); }
    const Material* materials() const { return materials_gpu_.data(); }
    const Light* lights() const { return lights_gpu_.data(); }
    const Sphere* spheres() const { return spheres_gpu_.data(); }

    int num_triangles() const { return static_cast<int>(triangles_.size()); }
    int num_materials() const { return static_cast<int>(materials_.size()); }
    int num_lights() const { return static_cast<int>(lights_.size()); }
    int num_spheres() const { return static_cast<int>(spheres_.size()); }

    AABB world_bounds() const { return bvh_.world_bounds(); }

    
    static Scene create_cornell_box() {
        Scene scene;

        
        int white_id = scene.add_material(Material::diffuse(make_float3(0.73f)));
        int red_id = scene.add_material(Material::diffuse(make_float3(0.65f, 0.05f, 0.05f)));
        int green_id = scene.add_material(Material::diffuse(make_float3(0.12f, 0.45f, 0.15f)));
        int light_id = scene.add_material(Material::emissive(make_float3(1.0f), 15.0f));

        
        scene.add_triangle(Triangle(
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(5.0f, 0.0f, 0.0f),
            make_float3(5.0f, 0.0f, 5.0f),
            white_id
        ));
        scene.add_triangle(Triangle(
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(5.0f, 0.0f, 5.0f),
            make_float3(0.0f, 0.0f, 5.0f),
            white_id
        ));

        
        scene.add_triangle(Triangle(
            make_float3(0.0f, 5.0f, 0.0f),
            make_float3(0.0f, 5.0f, 5.0f),
            make_float3(5.0f, 5.0f, 5.0f),
            white_id
        ));
        scene.add_triangle(Triangle(
            make_float3(0.0f, 5.0f, 0.0f),
            make_float3(5.0f, 5.0f, 5.0f),
            make_float3(5.0f, 5.0f, 0.0f),
            white_id
        ));

        
        scene.add_triangle(Triangle(
            make_float3(0.0f, 0.0f, 5.0f),
            make_float3(5.0f, 0.0f, 5.0f),
            make_float3(5.0f, 5.0f, 5.0f),
            white_id
        ));
        scene.add_triangle(Triangle(
            make_float3(0.0f, 0.0f, 5.0f),
            make_float3(5.0f, 5.0f, 5.0f),
            make_float3(0.0f, 5.0f, 5.0f),
            white_id
        ));

        
        scene.add_triangle(Triangle(
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 5.0f),
            make_float3(0.0f, 5.0f, 5.0f),
            red_id
        ));
        scene.add_triangle(Triangle(
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 5.0f, 5.0f),
            make_float3(0.0f, 5.0f, 0.0f),
            red_id
        ));

        
        scene.add_triangle(Triangle(
            make_float3(5.0f, 0.0f, 0.0f),
            make_float3(5.0f, 5.0f, 0.0f),
            make_float3(5.0f, 5.0f, 5.0f),
            green_id
        ));
        scene.add_triangle(Triangle(
            make_float3(5.0f, 0.0f, 0.0f),
            make_float3(5.0f, 5.0f, 5.0f),
            make_float3(5.0f, 0.0f, 5.0f),
            green_id
        ));

        
        float light_size = 1.3f;
        float light_y = 4.99f;
        float light_center = 2.5f;
        scene.add_triangle(Triangle(
            make_float3(light_center - light_size/2, light_y, light_center - light_size/2),
            make_float3(light_center + light_size/2, light_y, light_center - light_size/2),
            make_float3(light_center + light_size/2, light_y, light_center + light_size/2),
            light_id
        ));
        scene.add_triangle(Triangle(
            make_float3(light_center - light_size/2, light_y, light_center - light_size/2),
            make_float3(light_center + light_size/2, light_y, light_center + light_size/2),
            make_float3(light_center - light_size/2, light_y, light_center + light_size/2),
            light_id
        ));

        
        auto add_box = [&](float3 center, float3 size, float angle, int mat_id) {
            float c = cosf(angle);
            float s = sinf(angle);
            float hx = size.x * 0.5f;
            float hy = size.y * 0.5f;
            float hz = size.z * 0.5f;

            auto rotate = [c, s, center](float3 p) {
                float x = p.x - center.x;
                float z = p.z - center.z;
                return make_float3(
                    center.x + c * x - s * z,
                    p.y,
                    center.z + s * x + c * z
                );
            };

            
            float3 corners[8] = {
                rotate(make_float3(center.x - hx, center.y - hy, center.z - hz)),
                rotate(make_float3(center.x + hx, center.y - hy, center.z - hz)),
                rotate(make_float3(center.x + hx, center.y - hy, center.z + hz)),
                rotate(make_float3(center.x - hx, center.y - hy, center.z + hz)),
                rotate(make_float3(center.x - hx, center.y + hy, center.z - hz)),
                rotate(make_float3(center.x + hx, center.y + hy, center.z - hz)),
                rotate(make_float3(center.x + hx, center.y + hy, center.z + hz)),
                rotate(make_float3(center.x - hx, center.y + hy, center.z + hz)),
            };

            
            scene.add_triangle(Triangle(corners[0], corners[1], corners[5], mat_id));
            scene.add_triangle(Triangle(corners[0], corners[5], corners[4], mat_id));
            
            scene.add_triangle(Triangle(corners[2], corners[3], corners[7], mat_id));
            scene.add_triangle(Triangle(corners[2], corners[7], corners[6], mat_id));
            
            scene.add_triangle(Triangle(corners[3], corners[0], corners[4], mat_id));
            scene.add_triangle(Triangle(corners[3], corners[4], corners[7], mat_id));
            
            scene.add_triangle(Triangle(corners[1], corners[2], corners[6], mat_id));
            scene.add_triangle(Triangle(corners[1], corners[6], corners[5], mat_id));
            
            scene.add_triangle(Triangle(corners[4], corners[5], corners[6], mat_id));
            scene.add_triangle(Triangle(corners[4], corners[6], corners[7], mat_id));
        };

        
        add_box(make_float3(1.85f, 0.8f, 1.69f), make_float3(1.65f, 1.65f, 1.65f), -0.29f, white_id);

        
        add_box(make_float3(3.68f, 1.65f, 3.51f), make_float3(1.65f, 3.3f, 1.65f), 0.26f, white_id);

        scene.build();
        return scene;
    }

private:
    
    std::vector<Triangle> triangles_;
    std::vector<Sphere> spheres_;
    std::vector<Material> materials_;
    std::vector<Light> lights_;

    
    BVH bvh_;
    DeviceBuffer<Material> materials_gpu_;
    DeviceBuffer<Light> lights_gpu_;
    DeviceBuffer<Sphere> spheres_gpu_;
};

} 
