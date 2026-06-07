#include "interactive_runtime.cuh"

#include <cstdio>
#include <vector>

#include "scene.cuh"
#include "../core/memory/device_buffer.cuh"
#include "../integrators/wavefront/path_state.cuh"
#include "../integrators/wavefront/ray_queue.cuh"
#include "../lighting/light_sampling.cuh"

namespace wpt
{

    void launch_generate_rays(
        PathStateView paths,
        const Camera &camera,
        int width, int height,
        int frame_number,
        int samples_per_pixel,
        int current_sample);

    void launch_intersect(
        PathStateView paths,
        HitInfoView hits,
        const BVHNode *bvh_nodes,
        const TrianglePrecomputed *precomputed,
        const Triangle *triangles,
        const int *active_paths,
        const unsigned int *active_count_ptr,
        int max_threads);

    void launch_shade_miss(
        PathStateView paths,
        const HitInfoView &hits,
        const int *active_paths,
        const unsigned int *active_count_ptr,
        int max_threads,
        cudaTextureObject_t env_map,
        float env_intensity);

    void launch_shade_surface(
        PathStateView paths,
        const HitInfoView &hits,
        const Material *materials,
        const cudaTextureObject_t *textures,
        const Triangle *triangles_array,
        LightTableView light_table,
        ShadowRayView shadow_queue,
        const int *active_paths,
        unsigned int *next_count,
        int *next_paths,
        const unsigned int *active_count_ptr,
        int max_threads,
        int max_depth);

    void launch_trace_shadow(
        PathStateView paths,
        const BVHNode *bvh_nodes,
        const TrianglePrecomputed *precomputed,
        ShadowRayView shadow_queue,
        const unsigned int *shadow_count_ptr,
        int max_threads);

    void launch_accumulate(
        const PathStateView &paths,
        float4 *accumulation_buffer,
        int *sample_count,
        int width, int height,
        int num_paths);

    void launch_tonemap(
        const float4 *accumulation_buffer,
        uchar4 *display_buffer,
        int width, int height,
        float exposure);

    struct InteractiveRenderer::Impl
    {
        int width;
        int height;
        int num_pixels;
        int frame_number;
        int total_samples;
        float exposure;
        int max_depth;

        PathStateSoA path_state;
        HitInfoSoA hit_info;
        WorkQueues work_queues;
        ShadowRaySoA shadow_rays;

        DeviceBuffer<float4> accumulation_buffer;
        DeviceBuffer<int> sample_count;
        DeviceBuffer<int> initial_active;

        cudaTextureObject_t env_map;
        float env_intensity;

        Impl(int width_in, int height_in)
            : width(width_in), height(height_in), num_pixels(width_in * height_in), frame_number(0), total_samples(0), exposure(0.46f), max_depth(8), env_map(0), env_intensity(1.0f)
        {

            path_state.resize(num_pixels);
            hit_info.resize(num_pixels);

            work_queues.resize(num_pixels);
            shadow_rays.resize(num_pixels);

            accumulation_buffer.resize(num_pixels);
            sample_count.resize(num_pixels);

            initial_active.resize(num_pixels);
            init_sequence();

            reset_accumulation();

            printf("Renderer initialized: %dx%d (%d pixels)\n", width, height, num_pixels);
        }

        void init_sequence()
        {
            std::vector<int> seq(num_pixels);
            for (int i = 0; i < num_pixels; i++)
                seq[i] = i;
            CUDA_CHECK(cudaMemcpy(initial_active.data(), seq.data(),
                                  num_pixels * sizeof(int), cudaMemcpyHostToDevice));
        }

        void reset_accumulation()
        {
            CUDA_CHECK(cudaMemset(accumulation_buffer.data(), 0, num_pixels * sizeof(float4)));
            CUDA_CHECK(cudaMemset(sample_count.data(), 0, num_pixels * sizeof(int)));
            frame_number = 0;
            total_samples = 0;
        }
    };

    InteractiveRenderer::InteractiveRenderer(int width, int height)
        : impl_(new Impl(width, height)) {}

    InteractiveRenderer::~InteractiveRenderer()
    {
        delete impl_;
    }

    void InteractiveRenderer::reset_accumulation()
    {
        impl_->reset_accumulation();
    }

    void InteractiveRenderer::render_frame(const Camera &camera, const Scene &scene)
    {
        impl_->env_map = scene.environment_map();
        impl_->env_intensity = scene.environment_intensity();

        PathStateView paths = make_view(impl_->path_state);
        HitInfoView hits = make_view(impl_->hit_info);

        launch_generate_rays(paths, camera, impl_->width, impl_->height, impl_->frame_number, 1, 0);
        CUDA_CHECK_LAST();

        impl_->work_queues.reset();

        CUDA_CHECK(cudaMemcpy(impl_->work_queues.active_paths(), impl_->initial_active.data(),
                              impl_->num_pixels * sizeof(int), cudaMemcpyDeviceToDevice));
        impl_->work_queues.set_active_count(impl_->num_pixels);

        ShadowRayView shadow_view = make_view(impl_->shadow_rays);
        LightTableView light_table = scene.light_table();

        for (int depth = 0; depth < impl_->max_depth; depth++)
        {
            launch_intersect(paths, hits, scene.bvh_nodes(), scene.precomputed_triangles(),
                             scene.triangles(), impl_->work_queues.active_paths(),
                             impl_->work_queues.active_count_ptr(), impl_->num_pixels);
            CUDA_CHECK_LAST();

            launch_shade_miss(paths, hits, impl_->work_queues.active_paths(),
                              impl_->work_queues.active_count_ptr(), impl_->num_pixels,
                              impl_->env_map, impl_->env_intensity);
            CUDA_CHECK_LAST();

            CUDA_CHECK(cudaMemset(impl_->work_queues.next_count_ptr(), 0, sizeof(unsigned int)));
            impl_->shadow_rays.clear();

            launch_shade_surface(paths, hits, scene.materials(),
                                 scene.textures(),
                                 scene.triangles(),
                                 light_table,
                                 shadow_view,
                                 impl_->work_queues.active_paths(),
                                 impl_->work_queues.next_count_ptr(),
                                 impl_->work_queues.next_paths(),
                                 impl_->work_queues.active_count_ptr(), impl_->num_pixels,
                                 impl_->max_depth);
            CUDA_CHECK_LAST();

            launch_trace_shadow(paths, scene.bvh_nodes(), scene.precomputed_triangles(),
                                shadow_view, impl_->shadow_rays.count.data(), impl_->num_pixels);
            CUDA_CHECK_LAST();

            impl_->work_queues.swap_queues();
        }

        launch_accumulate(paths, impl_->accumulation_buffer.data(), impl_->sample_count.data(),
                          impl_->width, impl_->height, impl_->num_pixels);
        CUDA_CHECK_LAST();

        impl_->frame_number++;
        impl_->total_samples++;
    }

    void InteractiveRenderer::tonemap_to_buffer(uchar4 *device_buffer)
    {
        launch_tonemap(impl_->accumulation_buffer.data(), device_buffer,
                       impl_->width, impl_->height, impl_->exposure);
        CUDA_CHECK_LAST();
    }

    int InteractiveRenderer::total_samples() const
    {
        return impl_->total_samples;
    }

    float &InteractiveRenderer::exposure()
    {
        return impl_->exposure;
    }

}
