#include "bvh.cuh"
#include "../../core/memory/device_buffer.cuh"
#include <algorithm>
#include <vector>

namespace lumina
{

    struct BVHBuildEntry
    {
        int node_idx;
        int start;
        int end;
    };

    void BVH::build(const Triangle *triangles, int count)
    {
        if (count == 0)
            return;

        num_primitives_ = count;

        world_bounds_ = AABB::empty();
        for (int i = 0; i < count; i++)
        {
            world_bounds_.expand(triangles[i].bounds());
        }

        std::vector<Triangle> sorted_prims(triangles, triangles + count);

        std::vector<BVHNode> nodes;
        nodes.reserve(2 * count);

        std::vector<BVHBuildEntry> stack;
        nodes.push_back(BVHNode());
        stack.push_back({0, 0, count});

        while (!stack.empty())
        {
            BVHBuildEntry entry = stack.back();
            stack.pop_back();

            BVHNode &node = nodes[entry.node_idx];

            AABB bounds = AABB::empty();
            for (int i = entry.start; i < entry.end; i++)
            {
                bounds.expand(sorted_prims[i].bounds());
            }
            node.set_bounds(bounds);

            int prim_count = entry.end - entry.start;

            if (prim_count <= 4)
            {
                node.left_or_first = entry.start;
                node.prim_count = prim_count;
                continue;
            }

            int best_axis = bounds.largest_axis();
            int best_split = entry.start + prim_count / 2;

            float best_cost = INFINITY_F;
            float inv_parent_area = 1.0f / bounds.surface_area();

            for (int axis = 0; axis < 3; axis++)
            {

                std::vector<std::pair<float, int>> axis_sorted(prim_count);
                for (int i = 0; i < prim_count; i++)
                {
                    int idx = entry.start + i;
                    float centroid_val = (axis == 0) ? sorted_prims[idx].centroid().x : (axis == 1) ? sorted_prims[idx].centroid().y
                                                                                                    : sorted_prims[idx].centroid().z;
                    axis_sorted[i] = {centroid_val, idx};
                }
                std::sort(axis_sorted.begin(), axis_sorted.end());

                std::vector<AABB> right_bounds_arr(prim_count);
                right_bounds_arr[prim_count - 1] = sorted_prims[axis_sorted[prim_count - 1].second].bounds();
                for (int i = prim_count - 2; i >= 0; i--)
                {
                    right_bounds_arr[i] = union_aabb(right_bounds_arr[i + 1],
                                                     sorted_prims[axis_sorted[i].second].bounds());
                }

                AABB left_bounds = AABB::empty();
                for (int i = 0; i < prim_count - 1; i++)
                {
                    left_bounds.expand(sorted_prims[axis_sorted[i].second].bounds());

                    float cost = 1.0f + (left_bounds.surface_area() * (i + 1) +
                                           right_bounds_arr[i + 1].surface_area() * (prim_count - i - 1)) *
                                              inv_parent_area;

                    if (cost < best_cost)
                    {
                        best_cost = cost;
                        best_axis = axis;
                        best_split = entry.start + i + 1;
                    }
                }
            }

            if (best_cost >= prim_count && prim_count <= 8)
            {
                node.left_or_first = entry.start;
                node.prim_count = prim_count;
                continue;
            }

            std::nth_element(sorted_prims.begin() + entry.start,
                             sorted_prims.begin() + best_split,
                             sorted_prims.begin() + entry.end,
                             [best_axis](const Triangle &a, const Triangle &b)
                             {
                                 float ca = (best_axis == 0) ? a.centroid().x : (best_axis == 1) ? a.centroid().y
                                                                                                 : a.centroid().z;
                                 float cb = (best_axis == 0) ? b.centroid().x : (best_axis == 1) ? b.centroid().y
                                                                                                 : b.centroid().z;
                                 return ca < cb;
                             });

            int left_child = static_cast<int>(nodes.size());
            nodes.push_back(BVHNode());
            int right_child = static_cast<int>(nodes.size());
            nodes.push_back(BVHNode());

            node.left_or_first = left_child;
            node.prim_count = 0;

            stack.push_back({right_child, best_split, entry.end});
            stack.push_back({left_child, entry.start, best_split});
        }

        num_nodes_ = static_cast<int>(nodes.size());

        nodes_.upload(nodes.data(), nodes.size());
        primitives_.upload(sorted_prims.data(), sorted_prims.size());

        std::vector<TrianglePrecomputed> precomputed(sorted_prims.size());
        for (size_t i = 0; i < sorted_prims.size(); i++)
        {
            precomputed[i].from_triangle(sorted_prims[i]);
        }
        precomputed_.upload(precomputed.data(), precomputed.size());

        primitives_host_ = std::move(sorted_prims);
    }

}
