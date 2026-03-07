#pragma once

#include "../memory/device_buffer.cuh"
#include <vector>
#include <string>
#include <unordered_map>

namespace lumina
{

    class TextureManager
    {
    public:
        TextureManager() = default;
        ~TextureManager();

        TextureManager(const TextureManager &) = delete;
        TextureManager &operator=(const TextureManager &) = delete;
        TextureManager(TextureManager &&other) noexcept
            : textures_(std::move(other.textures_)),
              texture_cache_(std::move(other.texture_cache_)),
              textures_gpu_(std::move(other.textures_gpu_)) {}
        TextureManager &operator=(TextureManager &&other) noexcept
        {
            if (this != &other)
            {
                for (auto &tex : textures_)
                {
                    if (tex.tex_obj) cudaDestroyTextureObject(tex.tex_obj);
                    if (tex.array) cudaFreeArray(tex.array);
                }
                textures_ = std::move(other.textures_);
                texture_cache_ = std::move(other.texture_cache_);
                textures_gpu_ = std::move(other.textures_gpu_);
            }
            return *this;
        }

        int load_texture(const std::string &filepath);

        void upload();

        const cudaTextureObject_t *device_textures() const { return textures_gpu_.data(); }
        int num_textures() const { return static_cast<int>(textures_.size()); }

    private:
        struct TextureData
        {
            cudaArray_t array = nullptr;
            cudaTextureObject_t tex_obj = 0;
        };

        std::vector<TextureData> textures_;
        std::unordered_map<std::string, int> texture_cache_;
        DeviceBuffer<cudaTextureObject_t> textures_gpu_;
    };

}
