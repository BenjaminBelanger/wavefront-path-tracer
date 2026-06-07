#pragma once

#include <string>

#include "camera.cuh"

namespace wpt
{

    enum class AnimationPreset
    {
        Orbit,
        Dolly
    };

    // Parameters describing a programmatic, headless camera animation. The base
    // camera is the framed "hero" shot; each preset perturbs it over normalized
    // time t in [0, 1] across `frames` discrete frames.
    struct AnimationConfig
    {
        AnimationPreset preset = AnimationPreset::Orbit;
        int frames = 120;

        // Orbit: total yaw sweep (degrees) around the camera target.
        float orbit_degrees = 360.0f;

        // Dolly: distance-to-target multipliers at t = 0 and t = 1. Values < 1
        // move the camera closer to the target, > 1 pulls it away.
        float dolly_start = 1.0f;
        float dolly_end = 0.45f;
    };

    inline bool parse_animation_preset(const std::string &name, AnimationPreset &out)
    {
        if (name == "orbit")
        {
            out = AnimationPreset::Orbit;
            return true;
        }
        if (name == "dolly")
        {
            out = AnimationPreset::Dolly;
            return true;
        }
        return false;
    }

    inline const char *animation_preset_name(AnimationPreset preset)
    {
        switch (preset)
        {
        case AnimationPreset::Orbit:
            return "orbit";
        case AnimationPreset::Dolly:
            return "dolly";
        }
        return "unknown";
    }

    // Returns the camera for a given frame index by perturbing `base` according
    // to the animation preset. Host-only; reused by the headless animation loop.
    inline Camera animate_camera(const Camera &base, const AnimationConfig &config, int frame)
    {
        const int frames = (config.frames > 0) ? config.frames : 1;
        const float t = (frames > 1) ? static_cast<float>(frame) / static_cast<float>(frames - 1) : 0.0f;

        Camera cam = base;

        switch (config.preset)
        {
        case AnimationPreset::Orbit:
        {
            const float yaw = config.orbit_degrees * (PI / 180.0f) * t;
            cam.orbit(yaw, 0.0f);
            break;
        }
        case AnimationPreset::Dolly:
        {
            const float3 offset = base.position - base.target;
            const float distance = length(offset);
            const float3 dir = safe_normalize(offset);
            const float mult = config.dolly_start + (config.dolly_end - config.dolly_start) * t;
            cam.position = base.target + dir * (distance * fmaxf(mult, 0.01f));
            cam.update();
            break;
        }
        }

        return cam;
    }

}
