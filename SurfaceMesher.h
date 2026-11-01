// ===========================================================================
// SurfaceMesher.h -- isosurface extraction core for ParticleSurfaceSOP.
// ---------------------------------------------------------------------------
// Pure C++17, no TouchDesigner dependency (so it can be unit-tested on its
// own). Takes the density field produced by ParticleDensityTOP -- delivered as
// the DensityAtlas.h slice atlas -- and extracts the surface density == iso as
// a WELDED, consistently-wound triangle mesh with smooth gradient normals.
//
// ALGORITHM: marching tetrahedra over the Kuhn (Freudenthal) triangulation
// ---------------------------------------------------------------------------
// Each grid cube is split into the 6 tetrahedra that walk from corner 0 to
// corner 7 along every permutation of the axes (0 -> +a -> +b -> +c = 7). This
// split is the same rule in every cube, so neighbouring cubes cut their shared
// faces along the same diagonal: the tetrahedral mesh is conforming, and the
// extracted surface is watertight. Each tet is then triangulated with Paul
// Bourke's PolygoniseTri cases (0, 1 or 2 triangles) -- no 256-entry table.
//
// WELDING: rolling two-slab edge cache
// ---------------------------------------------------------------------------
// Surface vertices lie on tet edges. In the Kuhn split every tet edge runs from
// a grid point `lo` in one of 7 POSITIVE directions:
//      +x, +y, +z, +x+y, +x+z, +y+z, +x+y+z
// so (lo, direction) is a unique key for a shared vertex. While marching cube
// layer z, every edge's `lo` lies in layer z or z+1, so a cache of just two
// layers (res^2 * 7 slots each) suffices. Slots carry a stamp unique per
// (frame, layer), so nothing is ever cleared: a stale stamp simply misses.
// That is ~6 MB even at 192^3, no hashing, no per-frame memset.
//
// NORMALS: per-vertex, from the density gradient (central differences at the
// two edge endpoints, interpolated along the edge). Density rises INTO the
// blob, so the outward normal is -gradient.
//
// WINDING: each triangle is re-oriented so its geometric normal agrees with
// its vertices' gradient normals -> all faces are counter-clockwise when
// viewed from outside (right-hand rule along the outward normal).
// ===========================================================================
#pragma once

#include "DensityAtlas.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>

struct SurfaceMesh
{
    std::vector<float>   positions;   // x,y,z per vertex
    std::vector<float>   normals;     // x,y,z per vertex (unit length, outward)
    std::vector<int32_t> indices;     // 3 vertex indices per triangle

    int  numVertices()  const { return (int)(positions.size() / 3); }
    int  numTriangles() const { return (int)(indices.size()   / 3); }
    void clear() { positions.clear(); normals.clear(); indices.clear(); }
};

class SurfaceMesher
{
public:
    // -----------------------------------------------------------------------
    // Unpack a density atlas (row-major samples, as a TOP-to-CHOP delivers
    // them) into the internal res^3 grid. `flipY` handles a reader that lists
    // texture rows top-first instead of bottom-first. Non-finite samples are
    // treated as 0. Returns false, with error(), if the sample count can't
    // hold a res^3 atlas -- i.e. Gridres doesn't match the density TOP.
    // -----------------------------------------------------------------------
    bool loadAtlas(const float* samples, int64_t numSamples, int res, bool flipY)
    {
        myError[0] = 0;
        if (!samples || res < 2)
        {
            setError("No density samples.");
            return false;
        }
        const int64_t W = daAtlasW(res), H = daAtlasH(res);
        if (numSamples != W * H)
        {
            std::snprintf(myError, sizeof(myError),
                "Density has %lld samples but Gridres %d needs a %lldx%lld atlas "
                "(%lld). Gridres must match the Particle Density TOP.",
                (long long)numSamples, res, (long long)W, (long long)H,
                (long long)(W * H));
            return false;
        }

        resize(res);
        const int tx = daTilesX(res);
        for (int z = 0; z < res; ++z)
        {
            const int tileX = (z % tx) * res;
            const int tileY = (z / tx) * res;
            for (int y = 0; y < res; ++y)
            {
                int64_t py  = tileY + y;
                int64_t row = flipY ? (H - 1 - py) : py;
                const float* src = samples + row * W + tileX;
                float*       dst = &myD[(size_t)res * (y + (size_t)res * z)];
                for (int x = 0; x < res; ++x)
                {
                    float v = src[x];
                    dst[x] = isFiniteBits(v) ? v : 0.0f;
                }
            }
        }
        zeroBoundary();
        return true;
    }

    // Load a plain res^3 grid directly (x fastest). Used by the unit tests.
    void loadGrid(int res, const float* grid)
    {
        resize(res);
        std::memcpy(myD.data(), grid, myD.size() * sizeof(float));
        zeroBoundary();
    }

    // -----------------------------------------------------------------------
    // Extract the surface density == iso. `size` is the Boundssize of the
    // density TOP (the grid spans a cube of that edge, centred on the origin).
    // `flip` reverses normals AND winding (render inside-out).
    // -----------------------------------------------------------------------
    void extract(float size, float iso, bool flip, SurfaceMesh& out)
    {
        out.clear();
        if (myRes < 2) return;

        myOut  = &out;
        myIso  = iso;
        mySign = flip ? -1.0f : 1.0f;
        myCell = daCell(size, myRes);
        myBMin = daBMin(size);
        ++myFrame;

        const int    res = myRes;
        const size_t sy  = (size_t)res, sz = (size_t)res * res;
        const int    nc  = res - 1;                       // cubes per axis
        const int    nb  = (nc + kBrick - 1) / kBrick;     // bricks per axis

        // --- 1. brick classification ------------------------------------
        // A brick of up to kBrick^3 cubes spans grid points [b*kBrick, b*kBrick+kBrick]
        // (inclusive, clamped). If every one of those points is on the same side
        // of iso, no cube inside can produce a triangle: skip the whole brick.
        // The cloud fills a small part of the box, so this removes most of the
        // empty-space scan (which otherwise dominates the cost).
        myBrickActive.assign((size_t)nb * nb * nb, 0);
        for (int bz = 0; bz < nb; ++bz)
        for (int by = 0; by < nb; ++by)
        for (int bx = 0; bx < nb; ++bx)
        {
            const int x0 = bx * kBrick, x1 = std::min(x0 + kBrick, nc);   // points x0..x1
            const int y0 = by * kBrick, y1 = std::min(y0 + kBrick, nc);
            const int z0 = bz * kBrick, z1 = std::min(z0 + kBrick, nc);
            bool anyAbove = false, anyBelow = false;
            for (int z = z0; z <= z1 && !(anyAbove && anyBelow); ++z)
            for (int y = y0; y <= y1 && !(anyAbove && anyBelow); ++y)
            {
                const float* row = &myD[(size_t)x0 + sy * y + sz * z];
                for (int x = 0; x <= x1 - x0; ++x)
                {
                    if (row[x] >= iso) anyAbove = true; else anyBelow = true;
                }
            }
            myBrickActive[(size_t)bx + (size_t)nb * (by + (size_t)nb * bz)] = (anyAbove && anyBelow);
        }

        // --- 2. march the active bricks -----------------------------------
        // ORDER MATTERS: the two-slab edge cache is only valid if cube layers
        // are visited in increasing z. So z stays the OUTER loop and bricks are
        // skipped within each layer -- never "for each brick: for its z", which
        // would revisit earlier layers after their cache slots were recycled and
        // silently emit duplicate (unwelded) vertices along brick borders.
        for (int z = 0; z < nc; ++z)
        {
            const int bz = z / kBrick;
            for (int by = 0; by < nb; ++by)
            for (int bx = 0; bx < nb; ++bx)
            {
                if (!myBrickActive[(size_t)bx + (size_t)nb * (by + (size_t)nb * bz)]) continue;
                const int x0 = bx * kBrick, x1 = std::min(x0 + kBrick, nc);
                const int y0 = by * kBrick, y1 = std::min(y0 + kBrick, nc);

                for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x)
                {
                    // global indices of the 8 cube corners; corner c = (c&1, c>>1&1, c>>2&1)
                    const int g0 = (int)(x + sy * y + sz * z);
                    int   g[8];
                    float v[8];
                    int   above = 0;
                    for (int c = 0; c < 8; ++c)
                    {
                        g[c] = g0 + (int)((c & 1) + sy * ((c >> 1) & 1) + sz * ((c >> 2) & 1));
                        v[c] = myD[g[c]];
                        above += (v[c] >= iso);
                    }
                    if (above == 0 || above == 8) continue;   // cube-level early-out

                    for (int t = 0; t < 6; ++t)
                    {
                        const int* T = kTets[t];
                        const int   tg[4] = { g[T[0]], g[T[1]], g[T[2]], g[T[3]] };
                        const float tv[4] = { v[T[0]], v[T[1]], v[T[2]], v[T[3]] };
                        polygoniseTet(tg, tv);
                    }
                }
            }
        }
        myOut = nullptr;
    }

    const char*         error() const { return myError; }
    int                 res()   const { return myRes; }
    uint64_t            uncachedEdges() const { return myUncached; }   // must stay 0
    const std::vector<float>& grid() const { return myD; }

private:
    // The 6 Kuhn tetrahedra of a cube (each a monotone path corner 0 -> 7).
    static constexpr int kTets[6][4] = {
        {0, 1, 3, 7},   // x, y, z
        {0, 1, 5, 7},   // x, z, y
        {0, 2, 3, 7},   // y, x, z
        {0, 2, 6, 7},   // y, z, x
        {0, 4, 5, 7},   // z, x, y
        {0, 4, 6, 7},   // z, y, x
    };

    void setError(const char* msg) { std::snprintf(myError, sizeof(myError), "%s", msg); }

    // Force the outermost grid shell to 0 so the field is below any positive
    // iso on every box face. If the cloud spills past Boundssize, the surface
    // is then CAPPED one cell inside the box instead of left open: the mesh is
    // always closed (watertight), which matters for shading, booleans and any
    // downstream SOP that assumes a solid.
    void zeroBoundary()
    {
        const int r = myRes;
        auto Z = [&](int x, int y, int z) { myD[(size_t)x + (size_t)r * (y + (size_t)r * z)] = 0.0f; };
        for (int a = 0; a < r; ++a)
            for (int b = 0; b < r; ++b)
            {
                Z(0, a, b); Z(r - 1, a, b);     // x faces
                Z(a, 0, b); Z(a, r - 1, b);     // y faces
                Z(a, b, 0); Z(a, b, r - 1);     // z faces
            }
    }

    // Finite test on the bit pattern (exponent not all-ones): unaffected by
    // fast-math flags that let the compiler assume NaN/inf never occur.
    static bool isFiniteBits(float v)
    {
        uint32_t u;
        std::memcpy(&u, &v, sizeof u);
        return (u & 0x7f800000u) != 0x7f800000u;
    }

    void resize(int res)
    {
        if (res == myRes) return;
        myRes = res;
        myD.assign((size_t)res * res * res, 0.0f);
        const size_t slab = (size_t)res * res * 7;
        myStamp.assign(2 * slab, 0u);
        myVid.assign(2 * slab, -1);
        myFrame = 0;
        const int64_t r = res, r2 = r * r;
        const int64_t d[7] = { 1, r, r2, 1 + r, 1 + r2, r + r2, 1 + r + r2 };
        for (int k = 0; k < 7; ++k) myDelta[k] = d[k];
    }

    // --- grid helpers -----------------------------------------------------
    inline void coords(int i, int& x, int& y, int& z) const
    {
        const int r2 = myRes * myRes;
        z = i / r2;  int rem = i - z * r2;
        y = rem / myRes;  x = rem - y * myRes;
    }
    inline float at(int x, int y, int z) const
    {
        x = std::min(std::max(x, 0), myRes - 1);
        y = std::min(std::max(y, 0), myRes - 1);
        z = std::min(std::max(z, 0), myRes - 1);
        return myD[(size_t)x + (size_t)myRes * (y + (size_t)myRes * z)];
    }
    inline void gradient(int i, float g[3]) const
    {
        int x, y, z; coords(i, x, y, z);
        g[0] = at(x + 1, y, z) - at(x - 1, y, z);
        g[1] = at(x, y + 1, z) - at(x, y - 1, z);
        g[2] = at(x, y, z + 1) - at(x, y, z - 1);
    }

    // --- shared vertex on edge (ga, gb), created once per frame -----------
    int32_t edgeVertex(int ga, int gb)
    {
        const int     lo = std::min(ga, gb), hi = std::max(ga, gb);
        const int64_t delta = (int64_t)hi - lo;
        int dir = 0;
        while (dir < 7 && myDelta[dir] != delta) ++dir;   // always found for Kuhn edges

        const int64_t r2    = (int64_t)myRes * myRes;
        const int64_t lz    = lo / r2;
        // Defensive: an edge outside the 7 Kuhn directions can't be cached
        // (it would index past its slot). Never happens with kTets -- the unit
        // tests assert it -- but if it did we'd create an unwelded vertex
        // rather than corrupt the cache.
        const bool    cacheable = dir < 7;
        const size_t  slot  = cacheable
            ? ((size_t)(lz & 1) * (size_t)r2 + (size_t)(lo - lz * r2)) * 7 + dir : 0;
        const uint64_t want = myFrame * (uint64_t)(myRes + 1) + (uint64_t)lz + 1;
        if (cacheable && myStamp[slot] == want) return myVid[slot];
        if (!cacheable) ++myUncached;

        // interpolate the crossing point along lo -> hi
        const float vl = myD[lo], vh = myD[hi];
        float t = (std::fabs(vh - vl) > 1e-12f) ? (myIso - vl) / (vh - vl) : 0.5f;
        t = std::min(std::max(t, 0.0f), 1.0f);

        int xl, yl, zl, xh, yh, zh;
        coords(lo, xl, yl, zl);
        coords(hi, xh, yh, zh);
        // grid coords are < 2^24, so the int->float casts are exact
        const float px = myBMin + myCell * ((float)xl + t * (float)(xh - xl));
        const float py = myBMin + myCell * ((float)yl + t * (float)(yh - yl));
        const float pz = myBMin + myCell * ((float)zl + t * (float)(zh - zl));

        float gl[3], gh[3];
        gradient(lo, gl);
        gradient(hi, gh);
        float nx = -(gl[0] + t * (gh[0] - gl[0])) * mySign;
        float ny = -(gl[1] + t * (gh[1] - gl[1])) * mySign;
        float nz = -(gl[2] + t * (gh[2] - gl[2])) * mySign;
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-20f) { nx /= len; ny /= len; nz /= len; }
        else              { nx = 0.0f; ny = 1.0f; nz = 0.0f; }

        const int32_t id = myOut->numVertices();
        myOut->positions.insert(myOut->positions.end(), { px, py, pz });
        myOut->normals.insert(myOut->normals.end(),     { nx, ny, nz });

        if (cacheable)
        {
            myStamp[slot] = want;
            myVid[slot]   = id;
        }
        return id;
    }

    // --- emit one triangle, wound CCW around the outward normal -----------
    void addTri(int32_t a, int32_t b, int32_t c)
    {
        if (a == b || b == c || a == c) return;          // degenerate (vertices merged)
        const float* P = myOut->positions.data();
        const float* N = myOut->normals.data();
        const float e1[3] = { P[3*b] - P[3*a], P[3*b+1] - P[3*a+1], P[3*b+2] - P[3*a+2] };
        const float e2[3] = { P[3*c] - P[3*a], P[3*c+1] - P[3*a+1], P[3*c+2] - P[3*a+2] };
        const float f[3]  = { e1[1]*e2[2] - e1[2]*e2[1],
                              e1[2]*e2[0] - e1[0]*e2[2],
                              e1[0]*e2[1] - e1[1]*e2[0] };
        const float s = f[0] * (N[3*a]   + N[3*b]   + N[3*c])
                      + f[1] * (N[3*a+1] + N[3*b+1] + N[3*c+1])
                      + f[2] * (N[3*a+2] + N[3*b+2] + N[3*c+2]);
        if (s < 0.0f) std::swap(b, c);
        myOut->indices.insert(myOut->indices.end(), { a, b, c });
    }

    // --- Bourke PolygoniseTri over grid indices ----------------------------
    // idx bit i set  <=>  corner i is BELOW iso. A corner exactly at iso
    // counts as above, consistently with the cube early-out (v >= iso).
    void polygoniseTet(const int g[4], const float v[4])
    {
        int idx = 0;
        if (v[0] < myIso) idx |= 1;
        if (v[1] < myIso) idx |= 2;
        if (v[2] < myIso) idx |= 4;
        if (v[3] < myIso) idx |= 8;

        auto E = [&](int i, int j) { return edgeVertex(g[i], g[j]); };

        switch (idx)
        {
        case 0x00: case 0x0F: break;                                  // no crossing
        case 0x01: case 0x0E: addTri(E(0,1), E(0,2), E(0,3)); break;  // corner 0 alone
        case 0x02: case 0x0D: addTri(E(1,0), E(1,3), E(1,2)); break;  // corner 1 alone
        case 0x04: case 0x0B: addTri(E(2,0), E(2,1), E(2,3)); break;  // corner 2 alone
        case 0x08: case 0x07: addTri(E(3,0), E(3,2), E(3,1)); break;  // corner 3 alone
        case 0x03: case 0x0C:                                         // {0,1} | {2,3}
        {   // quad (0-2)(0-3)(1-3)(1-2), split on diagonal (0-2)-(1-3)
            int32_t a = E(0,3), b = E(0,2), c = E(1,3), d = E(1,2);
            addTri(a, b, c); addTri(c, b, d); break;
        }
        case 0x05: case 0x0A:                                         // {0,2} | {1,3}
        {   // quad (0-1)(0-3)(2-3)(2-1), split on diagonal (0-1)-(2-3)
            int32_t a = E(0,1), b = E(2,3), c = E(0,3), d = E(1,2);
            addTri(a, b, c); addTri(a, d, b); break;
        }
        case 0x06: case 0x09:                                         // {1,2} | {0,3}
        {   // quad (0-1)(1-3)(2-3)(0-2), split on diagonal (0-1)-(2-3)
            int32_t a = E(0,1), b = E(1,3), c = E(2,3), d = E(0,2);
            addTri(a, b, c); addTri(a, c, d); break;
        }
        }
    }

    // --- state (persistent across cooks: no per-frame allocation) ----------
    static constexpr int  kBrick  = 8;  // cubes per brick edge for empty-space skipping
    std::vector<uint8_t>  myBrickActive;
    int                   myRes   = 0;
    std::vector<float>    myD;          // res^3 densities, x fastest
    std::vector<uint64_t> myStamp;      // 2 slabs * res^2 * 7
    std::vector<int32_t>  myVid;
    uint64_t              myFrame = 0;
    uint64_t              myUncached = 0;
    int64_t               myDelta[7] = {};

    SurfaceMesh* myOut  = nullptr;
    float        myIso  = 0.0f, mySign = 1.0f, myCell = 1.0f, myBMin = 0.0f;
    char         myError[256] = {};
};
