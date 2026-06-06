#pragma once

#include <vector>

#include "../core/math/vector.cuh"
#include "../core/math/sampling.cuh"
#include "../core/memory/device_buffer.cuh"
#include "../geometry/primitives/triangle.cuh"
#include "../materials/bsdf/bsdf.cuh"

namespace wpt
{

    struct LightTriangle
    {
        int prim_id;
        int material_id;
        float area;
        float cdf;
    };

    struct LightTableView
    {
        const LightTriangle *__restrict__ entries;
        int count;
        float total_area;

        __device__ bool empty() const { return count <= 0; }

        __device__ int sample_index(float u) const
        {
            int lo = 0;
            int hi = count - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (entries[mid].cdf < u)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            return lo;
        }

        __device__ float pdf_area() const
        {
            return (total_area > 0.0f) ? (1.0f / total_area) : 0.0f;
        }
    };

    class LightTable
    {
    public:
        LightTable() : count_(0), total_area_(0.0f) {}

        void build(const Triangle *triangles, int num_triangles, const Material *materials, int num_materials)
        {
            std::vector<LightTriangle> entries;
            entries.reserve(64);

            float sum_area = 0.0f;
            for (int i = 0; i < num_triangles; i++)
            {
                int mat_id = triangles[i].material_id;
                if (mat_id < 0 || mat_id >= num_materials)
                    continue;
                if (!materials[mat_id].is_emissive())
                    continue;

                float area = triangles[i].area();
                if (area <= 0.0f)
                    continue;

                LightTriangle e;
                e.prim_id = i;
                e.material_id = mat_id;
                e.area = area;
                e.cdf = 0.0f;
                entries.push_back(e);
                sum_area += area;
            }

            count_ = static_cast<int>(entries.size());
            total_area_ = sum_area;

            if (count_ > 0 && sum_area > 0.0f)
            {
                float running = 0.0f;
                for (int i = 0; i < count_; i++)
                {
                    running += entries[i].area;
                    entries[i].cdf = running / sum_area;
                }
                entries[count_ - 1].cdf = 1.0f;
                buffer_.upload(entries.data(), entries.size());
            }
        }

        LightTableView view() const
        {
            LightTableView v;
            v.entries = buffer_.data();
            v.count = count_;
            v.total_area = total_area_;
            return v;
        }

        int count() const { return count_; }
        float total_area() const { return total_area_; }

    private:
        DeviceBuffer<LightTriangle> buffer_;
        int count_;
        float total_area_;
    };

    __device__ inline float light_pdf_solid_angle(float pdf_area, float dist_sq, float cos_light)
    {
        if (cos_light <= 0.0f)
            return 0.0f;
        return pdf_area * dist_sq / cos_light;
    }

    __device__ inline float3 sample_triangle_point(
        const Triangle &tri,
        float u1, float u2,
        float3 &normal_out)
    {
        float2 bary = sample_triangle_uniform(u1, u2);
        float u = bary.x;
        float v = bary.y;
        float3 p = tri.interpolate_position(u, v);
        normal_out = tri.geometric_normal();
        return p;
    }

}
