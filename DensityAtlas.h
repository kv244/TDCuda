// ===========================================================================
// DensityAtlas.h -- shared layout contract between ParticleDensityTOP (writer,
// CUDA) and ParticleSurfaceSOP (reader, C++).
// ---------------------------------------------------------------------------
// A TOP outputs a 2D image, but the density field is a 3D grid of res^3
// samples. We pack it as an ATLAS of Z-slices: each slice is a res x res tile,
// tiles are laid out left-to-right then bottom-to-top in a near-square grid of
// tilesX x tilesY tiles:
//
//      texel (px, py)  with  px = tx*res + x,  py = ty*res + y,
//      slice z = ty*tilesX + tx            (tiles with z >= res are padding = 0)
//
//      atlas size: W = res*tilesX,  H = res*tilesY
//
// A near-square atlas keeps W,H well under GPU texture limits up to large res
// (res 128 -> 12x11 tiles -> 1536x1408).
//
// Grid geometry is likewise fixed by contract: the box is a CUBE of edge
// `size`, centred on the origin (the sim is origin-centred), sampled at res
// points per axis:
//      bmin = -size/2,   cell = size / (res - 1),   point(i) = bmin + i*cell
//
// Both operators compute everything from (res, size) through these helpers, so
// as long as their Gridres / Boundssize parameters match (the component binds
// both to the same controls), they agree by construction.
// ===========================================================================
#pragma once

#ifdef __CUDACC__
  #define DA_HD __host__ __device__ __forceinline__
#else
  #define DA_HD inline
#endif

// smallest t with t*t >= res  (integer ceil(sqrt(res)), no float rounding risk)
DA_HD int daTilesX(int res)
{
    int t = 1;
    while (t * t < res) ++t;
    return t;
}

DA_HD int daTilesY(int res)
{
    int tx = daTilesX(res);
    return (res + tx - 1) / tx;
}

DA_HD int daAtlasW(int res) { return res * daTilesX(res); }
DA_HD int daAtlasH(int res) { return res * daTilesY(res); }

// Flat index of atlas texel holding grid point (x, y, z), row-major (py*W + px).
DA_HD int daTexelIndex(int res, int x, int y, int z)
{
    int tx = daTilesX(res);
    int px = (z % tx) * res + x;
    int py = (z / tx) * res + y;
    return py * daAtlasW(res) + px;
}

DA_HD float daBMin(float size)           { return -0.5f * size; }
DA_HD float daCell(float size, int res)  { return size / (float)(res - 1); }
