#pragma once

#include "vector.cuh"

namespace lumina
{

    struct Matrix4x4
    {
        float m[16];

        __host__ __device__ Matrix4x4()
        {
            for (int i = 0; i < 16; i++)
                m[i] = 0.0f;
            m[0] = m[5] = m[10] = m[15] = 1.0f;
        }

        __host__ __device__ static Matrix4x4 identity()
        {
            return Matrix4x4();
        }

        __host__ __device__ float &operator()(int row, int col)
        {
            return m[col * 4 + row];
        }

        __host__ __device__ float operator()(int row, int col) const
        {
            return m[col * 4 + row];
        }

        __host__ __device__ static Matrix4x4 translate(const float3 &t)
        {
            Matrix4x4 result;
            result(0, 3) = t.x;
            result(1, 3) = t.y;
            result(2, 3) = t.z;
            return result;
        }

        __host__ __device__ static Matrix4x4 scale(const float3 &s)
        {
            Matrix4x4 result;
            result(0, 0) = s.x;
            result(1, 1) = s.y;
            result(2, 2) = s.z;
            return result;
        }

        __host__ __device__ static Matrix4x4 scale(float s)
        {
            return scale(make_float3(s, s, s));
        }

        __host__ __device__ static Matrix4x4 rotate_x(float radians)
        {
            Matrix4x4 result;
            float c = cosf(radians);
            float s = sinf(radians);
            result(1, 1) = c;
            result(1, 2) = -s;
            result(2, 1) = s;
            result(2, 2) = c;
            return result;
        }

        __host__ __device__ static Matrix4x4 rotate_y(float radians)
        {
            Matrix4x4 result;
            float c = cosf(radians);
            float s = sinf(radians);
            result(0, 0) = c;
            result(0, 2) = s;
            result(2, 0) = -s;
            result(2, 2) = c;
            return result;
        }

        __host__ __device__ static Matrix4x4 rotate_z(float radians)
        {
            Matrix4x4 result;
            float c = cosf(radians);
            float s = sinf(radians);
            result(0, 0) = c;
            result(0, 1) = -s;
            result(1, 0) = s;
            result(1, 1) = c;
            return result;
        }

        __host__ __device__ static Matrix4x4 rotate(const float3 &axis, float radians)
        {
            float3 a = normalize(axis);
            float c = cosf(radians);
            float s = sinf(radians);
            float t = 1.0f - c;

            Matrix4x4 result;
            result(0, 0) = t * a.x * a.x + c;
            result(0, 1) = t * a.x * a.y - s * a.z;
            result(0, 2) = t * a.x * a.z + s * a.y;
            result(1, 0) = t * a.x * a.y + s * a.z;
            result(1, 1) = t * a.y * a.y + c;
            result(1, 2) = t * a.y * a.z - s * a.x;
            result(2, 0) = t * a.x * a.z - s * a.y;
            result(2, 1) = t * a.y * a.z + s * a.x;
            result(2, 2) = t * a.z * a.z + c;
            return result;
        }

        __host__ __device__ static Matrix4x4 look_at(const float3 &eye, const float3 &target, const float3 &up)
        {
            float3 f = normalize(target - eye);
            float3 r = normalize(cross(f, up));
            float3 u = cross(r, f);

            Matrix4x4 result;
            result(0, 0) = r.x;
            result(0, 1) = r.y;
            result(0, 2) = r.z;
            result(0, 3) = -dot(r, eye);
            result(1, 0) = u.x;
            result(1, 1) = u.y;
            result(1, 2) = u.z;
            result(1, 3) = -dot(u, eye);
            result(2, 0) = -f.x;
            result(2, 1) = -f.y;
            result(2, 2) = -f.z;
            result(2, 3) = dot(f, eye);
            result(3, 0) = 0.0f;
            result(3, 1) = 0.0f;
            result(3, 2) = 0.0f;
            result(3, 3) = 1.0f;
            return result;
        }

        __host__ __device__ static Matrix4x4 perspective(float fov_y, float aspect, float near_plane, float far_plane)
        {
            float tan_half_fov = tanf(fov_y * 0.5f);
            Matrix4x4 result;
            for (int i = 0; i < 16; i++)
                result.m[i] = 0.0f;

            result(0, 0) = 1.0f / (aspect * tan_half_fov);
            result(1, 1) = 1.0f / tan_half_fov;
            result(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
            result(2, 3) = -2.0f * far_plane * near_plane / (far_plane - near_plane);
            result(3, 2) = -1.0f;
            return result;
        }
    };

    __host__ __device__ inline Matrix4x4 operator*(const Matrix4x4 &a, const Matrix4x4 &b)
    {
        Matrix4x4 result;
        for (int i = 0; i < 16; i++)
            result.m[i] = 0.0f;

        for (int col = 0; col < 4; col++)
        {
            for (int row = 0; row < 4; row++)
            {
                for (int k = 0; k < 4; k++)
                {
                    result(row, col) += a(row, k) * b(k, col);
                }
            }
        }
        return result;
    }

    __host__ __device__ inline float3 transform_point(const Matrix4x4 &m, const float3 &p)
    {
        float4 result = make_float4(
            m(0, 0) * p.x + m(0, 1) * p.y + m(0, 2) * p.z + m(0, 3),
            m(1, 0) * p.x + m(1, 1) * p.y + m(1, 2) * p.z + m(1, 3),
            m(2, 0) * p.x + m(2, 1) * p.y + m(2, 2) * p.z + m(2, 3),
            m(3, 0) * p.x + m(3, 1) * p.y + m(3, 2) * p.z + m(3, 3));
        if (result.w != 1.0f && result.w != 0.0f)
        {
            return make_float3(result.x / result.w, result.y / result.w, result.z / result.w);
        }
        return make_float3(result.x, result.y, result.z);
    }

    __host__ __device__ inline float3 transform_vector(const Matrix4x4 &m, const float3 &v)
    {
        return make_float3(
            m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z,
            m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z,
            m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z);
    }

    __host__ __device__ inline float3 transform_normal(const Matrix4x4 &m_inv_transpose, const float3 &n)
    {
        return normalize(transform_vector(m_inv_transpose, n));
    }

    __host__ inline Matrix4x4 inverse(const Matrix4x4 &m)
    {
        Matrix4x4 inv;
        float det;

        inv.m[0] = m.m[5] * m.m[10] * m.m[15] -
                   m.m[5] * m.m[11] * m.m[14] -
                   m.m[9] * m.m[6] * m.m[15] +
                   m.m[9] * m.m[7] * m.m[14] +
                   m.m[13] * m.m[6] * m.m[11] -
                   m.m[13] * m.m[7] * m.m[10];

        inv.m[4] = -m.m[4] * m.m[10] * m.m[15] +
                   m.m[4] * m.m[11] * m.m[14] +
                   m.m[8] * m.m[6] * m.m[15] -
                   m.m[8] * m.m[7] * m.m[14] -
                   m.m[12] * m.m[6] * m.m[11] +
                   m.m[12] * m.m[7] * m.m[10];

        inv.m[8] = m.m[4] * m.m[9] * m.m[15] -
                   m.m[4] * m.m[11] * m.m[13] -
                   m.m[8] * m.m[5] * m.m[15] +
                   m.m[8] * m.m[7] * m.m[13] +
                   m.m[12] * m.m[5] * m.m[11] -
                   m.m[12] * m.m[7] * m.m[9];

        inv.m[12] = -m.m[4] * m.m[9] * m.m[14] +
                    m.m[4] * m.m[10] * m.m[13] +
                    m.m[8] * m.m[5] * m.m[14] -
                    m.m[8] * m.m[6] * m.m[13] -
                    m.m[12] * m.m[5] * m.m[10] +
                    m.m[12] * m.m[6] * m.m[9];

        inv.m[1] = -m.m[1] * m.m[10] * m.m[15] +
                   m.m[1] * m.m[11] * m.m[14] +
                   m.m[9] * m.m[2] * m.m[15] -
                   m.m[9] * m.m[3] * m.m[14] -
                   m.m[13] * m.m[2] * m.m[11] +
                   m.m[13] * m.m[3] * m.m[10];

        inv.m[5] = m.m[0] * m.m[10] * m.m[15] -
                   m.m[0] * m.m[11] * m.m[14] -
                   m.m[8] * m.m[2] * m.m[15] +
                   m.m[8] * m.m[3] * m.m[14] +
                   m.m[12] * m.m[2] * m.m[11] -
                   m.m[12] * m.m[3] * m.m[10];

        inv.m[9] = -m.m[0] * m.m[9] * m.m[15] +
                   m.m[0] * m.m[11] * m.m[13] +
                   m.m[8] * m.m[1] * m.m[15] -
                   m.m[8] * m.m[3] * m.m[13] -
                   m.m[12] * m.m[1] * m.m[11] +
                   m.m[12] * m.m[3] * m.m[9];

        inv.m[13] = m.m[0] * m.m[9] * m.m[14] -
                    m.m[0] * m.m[10] * m.m[13] -
                    m.m[8] * m.m[1] * m.m[14] +
                    m.m[8] * m.m[2] * m.m[13] +
                    m.m[12] * m.m[1] * m.m[10] -
                    m.m[12] * m.m[2] * m.m[9];

        inv.m[2] = m.m[1] * m.m[6] * m.m[15] -
                   m.m[1] * m.m[7] * m.m[14] -
                   m.m[5] * m.m[2] * m.m[15] +
                   m.m[5] * m.m[3] * m.m[14] +
                   m.m[13] * m.m[2] * m.m[7] -
                   m.m[13] * m.m[3] * m.m[6];

        inv.m[6] = -m.m[0] * m.m[6] * m.m[15] +
                   m.m[0] * m.m[7] * m.m[14] +
                   m.m[4] * m.m[2] * m.m[15] -
                   m.m[4] * m.m[3] * m.m[14] -
                   m.m[12] * m.m[2] * m.m[7] +
                   m.m[12] * m.m[3] * m.m[6];

        inv.m[10] = m.m[0] * m.m[5] * m.m[15] -
                    m.m[0] * m.m[7] * m.m[13] -
                    m.m[4] * m.m[1] * m.m[15] +
                    m.m[4] * m.m[3] * m.m[13] +
                    m.m[12] * m.m[1] * m.m[7] -
                    m.m[12] * m.m[3] * m.m[5];

        inv.m[14] = -m.m[0] * m.m[5] * m.m[14] +
                    m.m[0] * m.m[6] * m.m[13] +
                    m.m[4] * m.m[1] * m.m[14] -
                    m.m[4] * m.m[2] * m.m[13] -
                    m.m[12] * m.m[1] * m.m[6] +
                    m.m[12] * m.m[2] * m.m[5];

        inv.m[3] = -m.m[1] * m.m[6] * m.m[11] +
                   m.m[1] * m.m[7] * m.m[10] +
                   m.m[5] * m.m[2] * m.m[11] -
                   m.m[5] * m.m[3] * m.m[10] -
                   m.m[9] * m.m[2] * m.m[7] +
                   m.m[9] * m.m[3] * m.m[6];

        inv.m[7] = m.m[0] * m.m[6] * m.m[11] -
                   m.m[0] * m.m[7] * m.m[10] -
                   m.m[4] * m.m[2] * m.m[11] +
                   m.m[4] * m.m[3] * m.m[10] +
                   m.m[8] * m.m[2] * m.m[7] -
                   m.m[8] * m.m[3] * m.m[6];

        inv.m[11] = -m.m[0] * m.m[5] * m.m[11] +
                    m.m[0] * m.m[7] * m.m[9] +
                    m.m[4] * m.m[1] * m.m[11] -
                    m.m[4] * m.m[3] * m.m[9] -
                    m.m[8] * m.m[1] * m.m[7] +
                    m.m[8] * m.m[3] * m.m[5];

        inv.m[15] = m.m[0] * m.m[5] * m.m[10] -
                    m.m[0] * m.m[6] * m.m[9] -
                    m.m[4] * m.m[1] * m.m[10] +
                    m.m[4] * m.m[2] * m.m[9] +
                    m.m[8] * m.m[1] * m.m[6] -
                    m.m[8] * m.m[2] * m.m[5];

        det = m.m[0] * inv.m[0] + m.m[1] * inv.m[4] + m.m[2] * inv.m[8] + m.m[3] * inv.m[12];

        if (det == 0.0f)
        {
            return Matrix4x4::identity();
        }

        det = 1.0f / det;

        for (int i = 0; i < 16; i++)
        {
            inv.m[i] *= det;
        }

        return inv;
    }

    __host__ inline Matrix4x4 transpose(const Matrix4x4 &m)
    {
        Matrix4x4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result(i, j) = m(j, i);
            }
        }
        return result;
    }

}
