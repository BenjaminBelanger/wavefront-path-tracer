#pragma once

#include "../core/math/vector.cuh"
#include "../core/math/matrix.cuh"
#include "../core/math/sampling.cuh"

namespace wpt
{

    struct Camera
    {
        float3 position;
        float3 target;
        float3 up;

        float fov;
        float aspect_ratio;
        float near_plane;
        float far_plane;

        float aperture;
        float focus_distance;

        float3 forward;
        float3 right;
        float3 up_vec;
        float3 lower_left;
        float3 horizontal;
        float3 vertical;

        __host__ __device__ Camera()
            : position(make_float3(0.0f, 0.0f, 5.0f)), target(make_float3(0.0f)), up(make_float3(0.0f, 1.0f, 0.0f)), fov(PI / 4.0f), aspect_ratio(16.0f / 9.0f), near_plane(0.1f), far_plane(1000.0f), aperture(0.0f), focus_distance(5.0f)
        {
            update();
        }

        __host__ __device__ void update()
        {
            forward = normalize(target - position);
            right = normalize(cross(forward, up));
            up_vec = cross(right, forward);

            float half_height = tanf(fov * 0.5f);
            float half_width = aspect_ratio * half_height;

            float fd = (aperture > 0.0f) ? focus_distance : 1.0f;

            lower_left = position + fd * forward - fd * half_width * right - fd * half_height * up_vec;
            horizontal = 2.0f * fd * half_width * right;
            vertical = 2.0f * fd * half_height * up_vec;
        }

        __host__ __device__ Ray generate_ray(float u, float v) const
        {
            float3 direction = lower_left + u * horizontal + v * vertical - position;
            return Ray(position, normalize(direction));
        }

        __host__ __device__ Ray generate_ray_dof(float u, float v, float lens_u, float lens_v) const
        {
            if (aperture <= 0.0f)
            {
                return generate_ray(u, v);
            }

            float2 lens_sample = sample_disk_concentric(lens_u, lens_v);
            float3 lens_offset = aperture * (lens_sample.x * right + lens_sample.y * up_vec);

            float3 focus_point = lower_left + u * horizontal + v * vertical;

            float3 new_origin = position + lens_offset;
            float3 direction = normalize(focus_point - new_origin);

            return Ray(new_origin, direction);
        }

        __host__ void look_at(const float3 &pos, const float3 &tgt, const float3 &up_dir)
        {
            position = pos;
            target = tgt;
            up = up_dir;
            focus_distance = length(tgt - pos);
            update();
        }

        // Distance along the view axis required to fully fit a bounding sphere of
        // the given radius within the frame. Considers both the vertical and
        // horizontal field of view (so the tighter axis never clips) and uses the
        // sphere-correct sin() fit rather than a flat-disk tan() fit. `margin`
        // (>= 1) leaves breathing room so the scene never touches the frame edges.
        __host__ __device__ float frame_distance(float radius, float margin = 1.05f) const
        {
            float r = fmaxf(radius, 1e-4f);
            float half_v = fov * 0.5f;
            float half_h = atanf(aspect_ratio * tanf(half_v));
            float min_half = fmaxf(fminf(half_v, half_h), 1e-3f);
            return r / sinf(min_half) * margin;
        }

        __host__ void orbit(float delta_yaw, float delta_pitch)
        {
            float3 offset = position - target;
            float distance = length(offset);

            float theta = atan2f(offset.x, offset.z);
            float phi = asinf(clamp(offset.y / distance, -1.0f, 1.0f));

            theta += delta_yaw;
            phi = clamp(phi + delta_pitch, -PI * 0.49f, PI * 0.49f);

            position = target + distance * make_float3(
                                               cosf(phi) * sinf(theta),
                                               sinf(phi),
                                               cosf(phi) * cosf(theta));

            update();
        }

        __host__ void zoom(float delta)
        {
            float3 offset = position - target;
            float distance = fmaxf(0.1f, length(offset) + delta);
            position = target + normalize(offset) * distance;
            update();
        }

        __host__ void pan(float delta_x, float delta_y)
        {
            float3 offset = delta_x * right + delta_y * up_vec;
            position = position + offset;
            target = target + offset;
            update();
        }

        __host__ Matrix4x4 view_matrix() const
        {
            return Matrix4x4::look_at(position, target, up);
        }

        __host__ Matrix4x4 projection_matrix() const
        {
            return Matrix4x4::perspective(fov, aspect_ratio, near_plane, far_plane);
        }
    };

    struct CameraController
    {
        Camera camera;

        bool is_dragging;
        float last_mouse_x;
        float last_mouse_y;

        float orbit_speed;
        float pan_speed;
        float zoom_speed;

        __host__ CameraController()
            : is_dragging(false), last_mouse_x(0.0f), last_mouse_y(0.0f), orbit_speed(0.005f), pan_speed(0.01f), zoom_speed(0.1f)
        {
        }

        __host__ void on_mouse_button(int button, bool pressed, float x, float y)
        {
            if (button == 0)
            {
                is_dragging = pressed;
                last_mouse_x = x;
                last_mouse_y = y;
            }
        }

        __host__ bool on_mouse_move(float x, float y, bool shift_held, bool ctrl_held)
        {
            if (!is_dragging)
                return false;

            float dx = x - last_mouse_x;
            float dy = y - last_mouse_y;
            last_mouse_x = x;
            last_mouse_y = y;

            if (shift_held)
            {

                camera.pan(-dx * pan_speed, dy * pan_speed);
            }
            else if (ctrl_held)
            {

                camera.zoom(dy * zoom_speed);
            }
            else
            {

                camera.orbit(-dx * orbit_speed, -dy * orbit_speed);
            }

            return true;
        }

        __host__ bool on_scroll(float delta)
        {
            camera.zoom(-delta * zoom_speed * 10.0f);
            return true;
        }
    };

    struct MotionVectorData
    {
        Matrix4x4 prev_view_proj;
        Matrix4x4 curr_view_proj;
        Matrix4x4 curr_view_proj_inv;

        __host__ void update(const Camera &camera, const Camera &prev_camera)
        {
            prev_view_proj = prev_camera.projection_matrix() * prev_camera.view_matrix();
            curr_view_proj = camera.projection_matrix() * camera.view_matrix();
            curr_view_proj_inv = inverse(curr_view_proj);
        }

        __device__ float2 compute_motion(const float3 &world_pos) const
        {

            float4 curr_clip = make_float4(
                curr_view_proj(0, 0) * world_pos.x + curr_view_proj(0, 1) * world_pos.y +
                    curr_view_proj(0, 2) * world_pos.z + curr_view_proj(0, 3),
                curr_view_proj(1, 0) * world_pos.x + curr_view_proj(1, 1) * world_pos.y +
                    curr_view_proj(1, 2) * world_pos.z + curr_view_proj(1, 3),
                curr_view_proj(2, 0) * world_pos.x + curr_view_proj(2, 1) * world_pos.y +
                    curr_view_proj(2, 2) * world_pos.z + curr_view_proj(2, 3),
                curr_view_proj(3, 0) * world_pos.x + curr_view_proj(3, 1) * world_pos.y +
                    curr_view_proj(3, 2) * world_pos.z + curr_view_proj(3, 3));
            float2 curr_ndc = make_float2(curr_clip.x / curr_clip.w, curr_clip.y / curr_clip.w);

            float4 prev_clip = make_float4(
                prev_view_proj(0, 0) * world_pos.x + prev_view_proj(0, 1) * world_pos.y +
                    prev_view_proj(0, 2) * world_pos.z + prev_view_proj(0, 3),
                prev_view_proj(1, 0) * world_pos.x + prev_view_proj(1, 1) * world_pos.y +
                    prev_view_proj(1, 2) * world_pos.z + prev_view_proj(1, 3),
                prev_view_proj(2, 0) * world_pos.x + prev_view_proj(2, 1) * world_pos.y +
                    prev_view_proj(2, 2) * world_pos.z + prev_view_proj(2, 3),
                prev_view_proj(3, 0) * world_pos.x + prev_view_proj(3, 1) * world_pos.y +
                    prev_view_proj(3, 2) * world_pos.z + prev_view_proj(3, 3));
            float2 prev_ndc = make_float2(prev_clip.x / prev_clip.w, prev_clip.y / prev_clip.w);

            return curr_ndc - prev_ndc;
        }
    };

}
