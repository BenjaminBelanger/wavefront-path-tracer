#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "texture_manager.cuh"
#include "../math/spectral.cuh"

#include <iostream>
#include <algorithm>

namespace wpt
{

    TextureManager::~TextureManager()
    {
        for (auto &tex : textures_)
        {
            if (tex.tex_obj)
            {
                cudaDestroyTextureObject(tex.tex_obj);
            }
            if (tex.array)
            {
                cudaFreeArray(tex.array);
            }
        }
    }

    int TextureManager::load_texture(const std::string &filepath, bool srgb)
    {
        std::string normalized = filepath;
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
        std::string cache_key = normalized;
        if (!srgb)
            cache_key += "#linear";

        auto it = texture_cache_.find(cache_key);
        if (it != texture_cache_.end())
        {
            return it->second;
        }

        stbi_set_flip_vertically_on_load(true);

        int width, height, channels;
        unsigned char *data = stbi_load(normalized.c_str(), &width, &height, &channels, 4);
        if (!data)
        {
            std::cerr << "Failed to load texture: " << normalized << std::endl;
            return -1;
        }

        std::vector<float4> linear_data(width * height);
        for (int i = 0; i < width * height; i++)
        {
            float r = data[4 * i + 0] / 255.0f;
            float g = data[4 * i + 1] / 255.0f;
            float b = data[4 * i + 2] / 255.0f;
            float a = data[4 * i + 3] / 255.0f;
            if (srgb)
            {
                r = srgb_to_linear_channel(r);
                g = srgb_to_linear_channel(g);
                b = srgb_to_linear_channel(b);
            }
            linear_data[i] = make_float4(r, g, b, a);
        }

        stbi_image_free(data);

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaArray_t cuda_array;
        CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, linear_data.data(),
                                       width * sizeof(float4), width * sizeof(float4), height,
                                       cudaMemcpyHostToDevice));

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        cudaTextureObject_t tex_obj = 0;
        CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

        int index = static_cast<int>(textures_.size());
        textures_.push_back({cuda_array, tex_obj});
        texture_cache_[cache_key] = index;

        std::cout << "  Loaded " << (srgb ? "sRGB" : "linear") << " texture [" << index << "]: " << filepath
                  << " (" << width << "x" << height << ")" << std::endl;

        return index;
    }

    int TextureManager::load_hdr(const std::string &filepath)
    {
        std::string normalized = filepath;
        std::replace(normalized.begin(), normalized.end(), '\\', '/');

        auto it = texture_cache_.find(normalized);
        if (it != texture_cache_.end())
            return it->second;

        stbi_set_flip_vertically_on_load(true);

        int width, height, channels;
        float *data = stbi_loadf(normalized.c_str(), &width, &height, &channels, 4);
        if (!data)
        {
            std::cerr << "Failed to load HDR: " << normalized << std::endl;
            return -1;
        }

        std::vector<float4> hdr_data(width * height);
        for (int i = 0; i < width * height; i++)
        {
            hdr_data[i] = make_float4(data[4 * i + 0], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]);
        }

        stbi_image_free(data);

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaArray_t cuda_array;
        CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, hdr_data.data(),
                                       width * sizeof(float4), width * sizeof(float4), height,
                                       cudaMemcpyHostToDevice));

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        cudaTextureObject_t tex_obj = 0;
        CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

        int index = static_cast<int>(textures_.size());
        textures_.push_back({cuda_array, tex_obj});
        texture_cache_[normalized] = index;

        std::cout << "  Loaded HDR [" << index << "]: " << normalized
                  << " (" << width << "x" << height << ")" << std::endl;

        return index;
    }

    void TextureManager::upload()
    {
        if (textures_.empty())
            return;

        std::vector<cudaTextureObject_t> handles(textures_.size());
        for (size_t i = 0; i < textures_.size(); i++)
        {
            handles[i] = textures_[i].tex_obj;
        }
        textures_gpu_.upload(handles.data(), handles.size());
    }

}
