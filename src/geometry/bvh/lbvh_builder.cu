#include "bvh.cuh"
#include "../../core/memory/device_buffer.cuh"
#include <algorithm>
#include <vector>

namespace lumina {

// =============================================================================
// LBVH Builder Kernels
// =============================================================================

// Compute Morton codes for all primitives
__global__ void compute_morton_codes_kernel(
    const Triangle* __restrict__ primitives,
    uint32_t* __restrict__ morton_codes,
    int* __restrict__ indices,
    AABB scene_bounds,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Compute centroid and normalize to [0,1]^3
    float3 centroid = primitives[idx].centroid();
    float3 extent = scene_bounds.extent();
    float3 offset = scene_bounds.min_bound;

    float3 normalized = make_float3(0.5f);
    if (extent.x > 0) normalized.x = (centroid.x - offset.x) / extent.x;
    if (extent.y > 0) normalized.y = (centroid.y - offset.y) / extent.y;
    if (extent.z > 0) normalized.z = (centroid.z - offset.z) / extent.z;

    morton_codes[idx] = morton_code_3d(normalized);
    indices[idx] = idx;
}

// Build internal nodes using parallel Karras algorithm
__global__ void build_tree_kernel(
    const uint32_t* __restrict__ morton_codes,
    BVHNode* __restrict__ nodes,
    int num_leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves - 1) return;

    // Determine direction of the range
    int d = (idx == 0 || (__clz(morton_codes[idx] ^ morton_codes[idx + 1]) >
                          __clz(morton_codes[idx] ^ morton_codes[idx - 1]))) ? 1 : -1;

    // Compute upper bound for the length of the range
    int delta_min = (idx == 0 && d == -1) ? -1 :
                    (idx == num_leaves - 1 && d == 1) ? -1 :
                    __clz(morton_codes[idx] ^ morton_codes[idx - d]);

    int l_max = 2;
    while (idx + l_max * d >= 0 &&
           idx + l_max * d < num_leaves &&
           __clz(morton_codes[idx] ^ morton_codes[idx + l_max * d]) > delta_min) {
        l_max *= 2;
    }

    // Find the other end using binary search
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (idx + (l + t) * d >= 0 &&
            idx + (l + t) * d < num_leaves &&
            __clz(morton_codes[idx] ^ morton_codes[idx + (l + t) * d]) > delta_min) {
            l += t;
        }
    }

    int j = idx + l * d;

    // Find the split position using binary search
    int delta_node = __clz(morton_codes[idx] ^ morton_codes[j]);
    int s = 0;
    int t = l;

    do {
        t = (t + 1) / 2;
        if (idx + (s + t) * d >= 0 &&
            idx + (s + t) * d < num_leaves &&
            __clz(morton_codes[idx] ^ morton_codes[idx + (s + t) * d]) > delta_node) {
            s += t;
        }
    } while (t > 1);

    int split = idx + s * d + (d < 0 ? d : 0);

    // Output child pointers
    int left, right;

    // Left child
    int min_ij = (idx < j) ? idx : j;
    if (min_ij == split) {
        left = num_leaves - 1 + split;  // Leaf
    } else {
        left = split;  // Internal
    }

    // Right child
    int max_ij = (idx > j) ? idx : j;
    if (max_ij == split + 1) {
        right = num_leaves - 1 + split + 1;  // Leaf
    } else {
        right = split + 1;  // Internal
    }

    // Store in node
    nodes[idx].left_or_first = left;
    nodes[idx].prim_count = 0;  // Mark as internal

    // Store right child info (we need to track it separately for bound computation)
    // Use a convention: right child = left_or_first + 1 for consecutive allocation
}

// Initialize leaf nodes
__global__ void init_leaves_kernel(
    const Triangle* __restrict__ primitives,
    const int* __restrict__ sorted_indices,
    BVHNode* __restrict__ nodes,
    int num_leaves,
    int internal_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    int leaf_idx = internal_offset + idx;
    int prim_idx = sorted_indices[idx];

    const Triangle& tri = primitives[prim_idx];
    AABB bounds = tri.bounds();

    nodes[leaf_idx].bounds_min = bounds.min_bound;
    nodes[leaf_idx].bounds_max = bounds.max_bound;
    nodes[leaf_idx].left_or_first = idx;  // Index into sorted primitives
    nodes[leaf_idx].prim_count = 1;       // Single primitive per leaf
}

// Compute bounding boxes bottom-up using atomic operations
__global__ void compute_bounds_kernel(
    BVHNode* __restrict__ nodes,
    int* __restrict__ node_counters,
    int num_leaves,
    int num_internal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    int leaf_idx = num_internal + idx;

    // Walk up the tree
    int current = leaf_idx;
    while (current > 0) {
        // Find parent (this is a simplification - actual implementation needs parent pointers)
        int parent = (current - 1) / 2;

        // Atomic increment to ensure both children have been processed
        int old_count = atomicAdd(&node_counters[parent], 1);

        if (old_count == 0) {
            // First child to arrive, wait for sibling
            return;
        }

        // Second child - both children ready, compute bounds
        int left = parent * 2 + 1;
        int right = parent * 2 + 2;

        AABB left_bounds(nodes[left].bounds_min, nodes[left].bounds_max);
        AABB right_bounds(nodes[right].bounds_min, nodes[right].bounds_max);
        AABB combined = union_aabb(left_bounds, right_bounds);

        nodes[parent].bounds_min = combined.min_bound;
        nodes[parent].bounds_max = combined.max_bound;
        nodes[parent].left_or_first = left;
        nodes[parent].prim_count = 0;

        current = parent;
    }
}

// =============================================================================
// BVH Build Implementation (CPU fallback with simpler algorithm)
// =============================================================================

struct BVHBuildEntry {
    int node_idx;
    int start;
    int end;
};

void BVH::build(const Triangle* triangles, int count) {
    if (count == 0) return;

    num_primitives_ = count;

    // Compute scene bounds
    world_bounds_ = AABB::empty();
    for (int i = 0; i < count; i++) {
        world_bounds_.expand(triangles[i].bounds());
    }

    // Copy primitives to device
    primitives_.upload(triangles, count);

    // Create primitive indices and compute centroids/morton codes on CPU
    std::vector<int> indices(count);
    std::vector<uint32_t> morton_codes(count);
    std::vector<float3> centroids(count);

    float3 extent = world_bounds_.extent();
    float3 offset = world_bounds_.min_bound;

    for (int i = 0; i < count; i++) {
        indices[i] = i;
        centroids[i] = triangles[i].centroid();

        float3 normalized = make_float3(0.5f);
        if (extent.x > 0) normalized.x = (centroids[i].x - offset.x) / extent.x;
        if (extent.y > 0) normalized.y = (centroids[i].y - offset.y) / extent.y;
        if (extent.z > 0) normalized.z = (centroids[i].z - offset.z) / extent.z;

        morton_codes[i] = morton_code_3d(normalized);
    }

    // Sort by Morton code
    std::vector<std::pair<uint32_t, int>> sorted(count);
    for (int i = 0; i < count; i++) {
        sorted[i] = {morton_codes[i], i};
    }
    std::sort(sorted.begin(), sorted.end());

    // Reorder primitives
    std::vector<Triangle> sorted_prims(count);
    for (int i = 0; i < count; i++) {
        sorted_prims[i] = triangles[sorted[i].second];
    }
    primitives_.upload(sorted_prims.data(), count);

    // Build BVH using recursive SAH-based splitting
    std::vector<BVHNode> nodes;
    nodes.reserve(2 * count);

    std::vector<BVHBuildEntry> stack;
    nodes.push_back(BVHNode());  // Root node
    stack.push_back({0, 0, count});

    while (!stack.empty()) {
        BVHBuildEntry entry = stack.back();
        stack.pop_back();

        BVHNode& node = nodes[entry.node_idx];

        // Compute bounds for this node
        AABB bounds = AABB::empty();
        for (int i = entry.start; i < entry.end; i++) {
            bounds.expand(sorted_prims[i].bounds());
        }
        node.set_bounds(bounds);

        int prim_count = entry.end - entry.start;

        // Make leaf if few primitives
        if (prim_count <= 4) {
            node.left_or_first = entry.start;
            node.prim_count = prim_count;
            continue;
        }

        // Find best split using SAH
        int best_axis = bounds.largest_axis();
        int best_split = entry.start + prim_count / 2;

        float best_cost = INFINITY_F;
        float inv_parent_area = 1.0f / bounds.surface_area();

        for (int axis = 0; axis < 3; axis++) {
            // Sort centroids along axis
            std::vector<std::pair<float, int>> axis_sorted(prim_count);
            for (int i = 0; i < prim_count; i++) {
                int idx = entry.start + i;
                float centroid_val = (axis == 0) ? sorted_prims[idx].centroid().x :
                                     (axis == 1) ? sorted_prims[idx].centroid().y :
                                                   sorted_prims[idx].centroid().z;
                axis_sorted[i] = {centroid_val, idx};
            }
            std::sort(axis_sorted.begin(), axis_sorted.end());

            // Sweep to find best split
            AABB left_bounds = AABB::empty();
            for (int i = 0; i < prim_count - 1; i++) {
                left_bounds.expand(sorted_prims[axis_sorted[i].second].bounds());

                AABB right_bounds = AABB::empty();
                for (int j = i + 1; j < prim_count; j++) {
                    right_bounds.expand(sorted_prims[axis_sorted[j].second].bounds());
                }

                float cost = 0.125f + (left_bounds.surface_area() * (i + 1) +
                                       right_bounds.surface_area() * (prim_count - i - 1)) * inv_parent_area;

                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = entry.start + i + 1;
                }
            }
        }

        // If no good split found, make leaf
        if (best_cost >= prim_count) {
            node.left_or_first = entry.start;
            node.prim_count = prim_count;
            continue;
        }

        // Sort primitives by best axis for partition
        std::sort(sorted_prims.begin() + entry.start, sorted_prims.begin() + entry.end,
            [best_axis](const Triangle& a, const Triangle& b) {
                float ca = (best_axis == 0) ? a.centroid().x :
                           (best_axis == 1) ? a.centroid().y : a.centroid().z;
                float cb = (best_axis == 0) ? b.centroid().x :
                           (best_axis == 1) ? b.centroid().y : b.centroid().z;
                return ca < cb;
            });

        // Internal node: reserve two consecutive child slots so traversal
        // rule right_child = left_child + 1 remains valid.
        int left_child = static_cast<int>(nodes.size());
        nodes.push_back(BVHNode());
        int right_child = static_cast<int>(nodes.size());
        nodes.push_back(BVHNode());

        node.left_or_first = left_child;
        node.prim_count = 0;

        // Push right first so left subtree is processed next.
        stack.push_back({right_child, best_split, entry.end});
        stack.push_back({left_child, entry.start, best_split});
    }

    num_nodes_ = static_cast<int>(nodes.size());

    // Upload nodes and re-upload sorted primitives
    nodes_.upload(nodes.data(), nodes.size());
    primitives_.upload(sorted_prims.data(), sorted_prims.size());
}

} // namespace lumina
