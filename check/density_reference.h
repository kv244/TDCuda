// CPU reference of ParticleDensityTOP's kernels -- a line-for-line port used
// only by the tests, so the density pipeline's LOGIC (masking, trilinear
// splat, mass conservation, blur, normalisation, atlas layout) can be checked
// without a GPU. Atomics become plain adds (order differs only in fp rounding).
#pragma once
#include "../DensityAtlas.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace ref {

inline bool finitef(float v) { uint32_t u; std::memcpy(&u, &v, 4); return (u & 0x7f800000u) != 0x7f800000u; }

// --- port of ParticleCloudTOP::initKernel's seeding (same hashes) ---------
inline float hash01(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return (float)(x & 0x00FFFFFFu) / 16777216.0f;
}
inline float gaussian(uint32_t a, uint32_t b)
{
    float u1 = std::max(hash01(a), 1e-7f), u2 = hash01(b);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}
// RGBA32F position texture exactly as the TOP lays it out: side*side texels,
// first `count` live (alpha 1), the rest padding (alpha 0 + garbage xyz to
// prove the mask is what protects us, not the padding contents).
inline std::vector<float> makePositionTexture(int count, float radius, int& side)
{
    side = (int)std::ceil(std::sqrt((double)count));
    std::vector<float> tex((size_t)side * side * 4);
    for (int i = 0; i < side * side; ++i)
    {
        float* t = &tex[(size_t)i * 4];
        if (i >= count) { t[0] = NAN; t[1] = 1e30f; t[2] = -INFINITY; t[3] = 0.0f; continue; }
        uint32_t s = (uint32_t)i * 2654435761u + 12345u;
        float gx = gaussian(s + 1u, s + 2u), gy = gaussian(s + 3u, s + 4u), gz = gaussian(s + 5u, s + 6u);
        float len = std::sqrt(gx * gx + gy * gy + gz * gz) + 1e-6f;
        float shell = std::pow(std::fabs(gaussian(s + 7u, s + 8u)), 0.6f);
        t[0] = gx / len * shell * radius;
        t[1] = gy / len * shell * radius * 0.7f;
        t[2] = gz / len * shell * radius;
        t[3] = 1.0f;
    }
    return tex;
}

struct DensityResult {
    std::vector<float> rawGrid;     // after splat, before blur (for mass checks)
    std::vector<float> atlas;       // what the TOP outputs (row-major W*H)
    unsigned liveCount = 0;
    unsigned inBoxCount = 0;
};

// --- port of splatKernel + blurAxisKernel x3 + writeAtlasKernel -----------
inline DensityResult runDensityTOP(const std::vector<float>& tex, int texW, int texH,
                                   int res, float size, float smooth)
{
    DensityResult R;
    const float cell = daCell(size, res), bmin = daBMin(size), invCell = 1.0f / cell;
    const int   r = std::min(std::max((int)std::lround(smooth / cell), 0), 8);
    const size_t n = (size_t)res * res * res;
    std::vector<float> A(n, 0.0f), B(n, 0.0f);

    for (int i = 0; i < texW * texH; ++i)                         // splatKernel
    {
        const float* p = &tex[(size_t)i * 4];
        bool live = (p[3] > 0.5f) && finitef(p[0]) && finitef(p[1]) && finitef(p[2]);
        if (!live) continue;
        ++R.liveCount;
        float gx = (p[0] - bmin) * invCell, gy = (p[1] - bmin) * invCell, gz = (p[2] - bmin) * invCell;
        float hiLim = (float)(res - 1);
        if (!(gx >= 0 && gy >= 0 && gz >= 0 && gx < hiLim && gy < hiLim && gz < hiLim)) continue;
        ++R.inBoxCount;
        int x0 = (int)gx, y0 = (int)gy, z0 = (int)gz;
        float fx = gx - x0, fy = gy - y0, fz = gz - z0, ix = 1 - fx, iy = 1 - fy, iz = 1 - fz;
        size_t sy = res, sz = (size_t)res * res, b = x0 + sy * y0 + sz * z0;
        A[b] += ix*iy*iz;          A[b+1] += fx*iy*iz;
        A[b+sy] += ix*fy*iz;       A[b+1+sy] += fx*fy*iz;
        A[b+sz] += ix*iy*fz;       A[b+1+sz] += fx*iy*fz;
        A[b+sy+sz] += ix*fy*fz;    A[b+1+sy+sz] += fx*fy*fz;
    }
    R.rawGrid = A;

    auto blur = [&](const std::vector<float>& in, std::vector<float>& out, int axis) {
        for (size_t i = 0; i < n; ++i)                            // blurAxisKernel
        {
            int x = (int)(i % res), y = (int)((i / res) % res), z = (int)(i / ((size_t)res * res));
            int c = axis == 0 ? x : axis == 1 ? y : z;
            size_t stride = axis == 0 ? 1 : axis == 1 ? (size_t)res : (size_t)res * res;
            const float* line = in.data() + (i - (size_t)c * stride);
            int lo = std::max(c - r, 0), hi = std::min(c + r, res - 1);
            float s = 0; for (int k = lo; k <= hi; ++k) s += line[(size_t)k * stride];
            out[i] = s / (float)(2 * r + 1);
        }
    };
    const std::vector<float>* result = &A;
    if (r > 0) { blur(A, B, 0); blur(B, A, 1); blur(A, B, 2); result = &B; }

    const int tilesX = daTilesX(res), W = daAtlasW(res), H = daAtlasH(res);   // writeAtlasKernel
    R.atlas.assign((size_t)W * H, -1.0f);   // sentinel: every texel must be overwritten
    for (int i = 0; i < W * H; ++i)
    {
        int px = i % W, py = i / W, tx = px / res, x = px - tx * res, ty = py / res, y = py - ty * res;
        int z = ty * tilesX + tx;
        float v = 0.0f;
        if (z < res)
        {
            float live = (float)std::max(R.liveCount, 1u);
            v = (*result)[x + (size_t)res * (y + (size_t)res * z)] * ((float)res * res * res / live);
        }
        R.atlas[(size_t)i] = v;
    }
    return R;
}

} // namespace ref
