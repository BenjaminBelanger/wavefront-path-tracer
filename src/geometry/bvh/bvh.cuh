#pragma once

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
                    if (tri.intersect(ray, t, u, v) && t < closest_t)
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
                    if (tri.intersect(ray, t, u, v))
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

    __host__ __device__ inline uint32_t expand_bits(uint32_t v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    __host__ __device__ inline uint32_t morton_code_3d(float3 p)
    {
        p = clamp(p * 1024.0f, 0.0f, 1023.0f);
        uint32_t x = expand_bits(static_cast<uint32_t>(p.x));
        uint32_t y = expand_bits(static_cast<uint32_t>(p.y));
        uint32_t z = expand_bits(static_cast<uint32_t>(p.z));
        return (z << 2) | (y << 1) | x;
    }

    __device__ inline int clz_device(uint32_t x)
    {
        return __clz(x);
    }

    __host__ inline int clz_host(uint32_t x)
    {
        if (x == 0)
            return 32;
        int n = 0;
        if ((x & 0xFFFF0000) == 0)
        {
            n += 16;
            x <<= 16;
        }
        if ((x & 0xFF000000) == 0)
        {
            n += 8;
            x <<= 8;
        }
        if ((x & 0xF0000000) == 0)
        {
            n += 4;
            x <<= 4;
        }
        if ((x & 0xC0000000) == 0)
        {
            n += 2;
            x <<= 2;
        }
        if ((x & 0x80000000) == 0)
        {
            n += 1;
        }
        return n;
    }

    __device__ inline int find_split(uint32_t *morton_codes, int first, int last)
    {
        uint32_t first_code = morton_codes[first];
        uint32_t last_code = morton_codes[last];

        if (first_code == last_code)
        {
            return (first + last) >> 1;
        }

        int common_prefix = __clz(first_code ^ last_code);

        int split = first;
        int step = last - first;

        do
        {
            step = (step + 1) >> 1;
            int new_split = split + step;

            if (new_split < last)
            {
                uint32_t split_code = morton_codes[new_split];
                int split_prefix = __clz(first_code ^ split_code);
                if (split_prefix > common_prefix)
                {
                    split = new_split;
                }
            }
        } while (step > 1);

        return split;
    }

}
