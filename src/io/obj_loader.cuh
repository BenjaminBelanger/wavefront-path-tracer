#pragma once

#include "../app/scene.cuh"
#include <string>

namespace lumina
{

    struct ObjLoadOptions
    {
        float3 translation = {0, 0, 0};
        float3 rotation = {0, 0, 0};
        float scale = 1.0f;
        bool recalculate_normals = false;
    };

    bool load_obj(Scene &scene, const std::string &filepath,
                  const ObjLoadOptions &opts = {});

}
