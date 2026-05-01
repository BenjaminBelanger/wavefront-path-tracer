#pragma once

#include <vector>

#include "../../core/math/vector.cuh"
#include "../../core/memory/device_buffer.cuh"
#include "../primitives/aabb.cuh"
#include "../primitives/triangle.cuh"

namespace lumina
{

    struct BVHNode
    {
        float3 bounds_min;
        uint32_t left_or_first;
        float3 bounds_max;
        uint32_t prim_count;

        __host__ __device__ bool is_leaf() const
        {
            return prim_count > 0;
        }

        __host__ __device__ uint32_t left_child() const
        {
            return left_or_first;
        }

        __host__ __device__ uint32_t right_child() const
        {
            return left_or_first + 1;
        }

        __host__ __device__ uint32_t first_prim() const
        {
            return left_or_first;
        }

        __host__ __device__ AABB bounds() const
        {
            return AABB(bounds_min, bounds_max);
        }

        __host__ __device__ void set_bounds(const AABB &box)
        {
            bounds_min = box.min_bound;
            bounds_max = box.max_bound;
        }
    };

    class BVH
    {
    public:
        BVH() : num_nodes_(0), num_primitives_(0) {}

        void build(const Triangle *triangles, int count);

        const BVHNode *nodes() const { return nodes_.data(); }
        const Triangle *primitives() const { return primitives_.data(); }
        const int *primitive_indices() const { return prim_indices_.data(); }

        const Triangle *primitives_host() const { return primitives_host_.data(); }

        BVHNode *nodes_ptr() { return nodes_.data(); }
        Triangle *primitives_ptr() { return primitives_.data(); }
        int *primitive_indices_ptr() { return prim_indices_.data(); }

        const TrianglePrecomputed *precomputed() const { return precomputed_.data(); }

        int num_nodes() const { return num_nodes_; }
        int num_primitives() const { return num_primitives_; }

        AABB world_bounds() const { return world_bounds_; }

    private:
        DeviceBuffer<BVHNode> nodes_;
        DeviceBuffer<Triangle> primitives_;
        DeviceBuffer<TrianglePrecomputed> precomputed_;
        DeviceBuffer<int> prim_indices_;

        std::vector<Triangle> primitives_host_;

        int num_nodes_;
        int num_primitives_;
        AABB world_bounds_;
    };

    struct BVHTraversalState
    {
        static constexpr int MAX_STACK_SIZE = 64;
        int stack[MAX_STACK_SIZE];
        int stack_ptr;

        __device__ void init()
        {
            stack_ptr = 0;
            push(0);
        }

        __device__ void push(int node_idx)
        {
            stack[stack_ptr++] = node_idx;
        }

        __device__ int pop()
        {
            return stack[--stack_ptr];
        }

        __device__ bool empty() const
        {
            return stack_ptr == 0;
        }
    };

    __device__ inline bool traverse_bvh(
        const BVHNode *__restrict__ nodes,
        const TrianglePrecomputed *__restrict__ primitives,
        const Ray &ray,
        float &t_hit,
        float &u_hit,
        float &v_hit,
        int &prim_id,
        int &mat_id)
    {
        BVHTraversalState state;
        state.init();

        bool hit = false;
        float closest_t = ray.t_max;

        float3 inv_dir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

        while (!state.empty())
        {
            int node_idx = state.pop();
            const BVHNode &node = nodes[node_idx];

            if (node.is_leaf())
            {
                for (uint32_t i = 0; i < node.prim_count; i++)
                {
                    const TrianglePrecomputed &tri = primitives[node.first_prim() + i];
                    float t, u, v;
                    if (tri.intersect_watertight(ray, t, u, v) && t < closest_t)
                    {
                        closest_t = t;
                        t_hit = t;
                        u_hit = u;
                        v_hit = v;
                        prim_id = node.first_prim() + i;
                        mat_id = tri.material_id;
                        hit = true;
                    }
                }
            }
            else
            {
                const BVHNode &left_node = nodes[node.left_child()];
                const BVHNode &right_node = nodes[node.right_child()];
                float t_left, t_right;
                bool hit_left = left_node.bounds().intersect_fast(ray.origin, inv_dir, ray.t_min, closest_t, t_left);
                bool hit_right = right_node.bounds().intersect_fast(ray.origin, inv_dir, ray.t_min, closest_t, t_right);
                if (hit_left && hit_right)
                {
                    if (t_left < t_right)
                    {
                        state.push(node.right_child());
                        state.push(node.left_child());
                    }
                    else
                    {
                        state.push(node.left_child());
                        state.push(node.right_child());
                    }
                }
                else if (hit_left)
                {
                    state.push(node.left_child());
                }
                else if (hit_right)
                {
                    state.push(node.right_child());
                }
            }
        }

        return hit;
    }

    __device__ inline bool traverse_bvh_shadow(
        const BVHNode *__restrict__ nodes,
        const TrianglePrecomputed *__restrict__ primitives,
        const Ray &ray)
    {
        BVHTraversalState state;
        state.init();

        float3 inv_dir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

        while (!state.empty())
        {
            int node_idx = state.pop();
            const BVHNode &node = nodes[node_idx];

            if (node.is_leaf())
            {
                for (uint32_t i = 0; i < node.prim_count; i++)
                {
                    const TrianglePrecomputed &tri = primitives[node.first_prim() + i];
                    float t, u, v;
                    if (tri.intersect_watertight(ray, t, u, v))
                    {
                        return true;
                    }
                }
            }
            else
            {
                const BVHNode &left_node = nodes[node.left_child()];
                const BVHNode &right_node = nodes[node.right_child()];
                if (left_node.bounds().intersect_fast(ray.origin, inv_dir, ray.t_min, ray.t_max))
                    state.push(node.left_child());
                if (right_node.bounds().intersect_fast(ray.origin, inv_dir, ray.t_min, ray.t_max))
                    state.push(node.right_child());
            }
        }

        return false;
    }

}
