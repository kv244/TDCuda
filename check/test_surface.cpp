// Unit / integration tests for the Option B surface pipeline (sandbox build).
//   g++ -std=c++17 -O2 -I.. test_surface.cpp -o test_surface && ./test_surface
// Exercises SurfaceMesher (the SOP's core, unmodified) and a CPU port of the
// ParticleDensityTOP kernels. Exit code = number of failed checks.
#include "../SurfaceMesher.h"
#include "density_reference.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <random>
#include <chrono>
#include <numeric>
#include <functional>

static int gFail = 0, gPass = 0;
#define CHECK(cond, ...) do { if (cond) { ++gPass; } else { ++gFail; \
    std::printf("  FAIL %s:%d  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// ---------------------------------------------------------------------------
// Mesh topology / geometry analysis
// ---------------------------------------------------------------------------
struct Topo {
    bool   closed = true;        // every undirected edge shared by exactly 2 tris
    bool   oriented = true;      // every directed edge used at most once
    int    components = 0;
    long   V = 0, E = 0, F = 0;
    double signedVolume = 0.0;
    double minNormalAgreement = 1.0;   // min dot(vertex normal, adjacent face normal)
    double meanNormalAgreement = 0.0;
    int    unreferenced = 0;
};

static Topo analyse(const SurfaceMesh& m)
{
    Topo T;
    const int nv = m.numVertices(), nt = m.numTriangles();
    T.V = nv; T.F = nt;
    std::map<std::pair<int,int>, int> undirected;
    std::set<std::pair<int,int>> directed;
    std::vector<int> parent(nv); std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int a) { while (parent[a] != a) a = parent[a] = parent[parent[a]]; return a; };
    std::vector<char> used(nv, 0);
    const float* P = m.positions.data(); const float* N = m.normals.data();
    double sumAgree = 0; long nAgree = 0;

    for (int t = 0; t < nt; ++t)
    {
        int v[3] = { m.indices[3*t], m.indices[3*t+1], m.indices[3*t+2] };
        for (int k = 0; k < 3; ++k)
        {
            int a = v[k], b = v[(k + 1) % 3];
            used[a] = 1;
            undirected[{std::min(a,b), std::max(a,b)}]++;
            if (!directed.insert({a, b}).second) T.oriented = false;
            int ra = find(a), rb = find(b); if (ra != rb) parent[ra] = rb;
        }
        const float* p0 = P + 3*v[0]; const float* p1 = P + 3*v[1]; const float* p2 = P + 3*v[2];
        double e1[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
        double e2[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
        double f[3]  = { e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0] };
        T.signedVolume += (p0[0]*(p1[1]*p2[2]-p1[2]*p2[1]) - p0[1]*(p1[0]*p2[2]-p1[2]*p2[0])
                         + p0[2]*(p1[0]*p2[1]-p1[1]*p2[0])) / 6.0;
        double fl = std::sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
        if (fl > 1e-10)
            for (int k = 0; k < 3; ++k)
            {
                const float* n = N + 3*v[k];
                double d = (f[0]*n[0] + f[1]*n[1] + f[2]*n[2]) / fl;
                T.minNormalAgreement = std::min(T.minNormalAgreement, d);
                sumAgree += d; ++nAgree;
            }
    }
    for (auto& kv : undirected) if (kv.second != 2) { T.closed = false; break; }
    T.E = (long)undirected.size();
    std::set<int> roots; for (int i = 0; i < nv; ++i) if (used[i]) roots.insert(find(i)); else ++T.unreferenced;
    T.components = (int)roots.size();
    T.meanNormalAgreement = nAgree ? sumAgree / nAgree : 0;
    return T;
}

// Gaussian-sum field sampled on the grid (x fastest), for analytic checks.
struct Blob { float cx, cy, cz, sigma, amp; };
static std::vector<float> fieldFromBlobs(int res, float size, const std::vector<Blob>& blobs)
{
    std::vector<float> g((size_t)res * res * res);
    const float c = daCell(size, res), b0 = daBMin(size);
    for (int z = 0; z < res; ++z) for (int y = 0; y < res; ++y) for (int x = 0; x < res; ++x)
    {
        float px = b0 + c*x, py = b0 + c*y, pz = b0 + c*z, s = 0;
        for (auto& B : blobs) { float dx=px-B.cx, dy=py-B.cy, dz=pz-B.cz;
            s += B.amp * std::exp(-(dx*dx+dy*dy+dz*dz) / (2*B.sigma*B.sigma)); }
        g[(size_t)x + (size_t)res * (y + (size_t)res * z)] = s;
    }
    return g;
}

static double ms(std::chrono::steady_clock::time_point a)
{ return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); }

// ---------------------------------------------------------------------------
int main()
{
    std::printf("== T1  Kuhn tetra edges all map to the 7 cached directions\n");
    {
        const int kT[6][4] = {{0,1,3,7},{0,1,5,7},{0,2,3,7},{0,2,6,7},{0,4,5,7},{0,4,6,7}};
        const int r = 11; const long d[7] = {1, r, r*r, 1+r, 1+r*r, r+r*r, 1+r+r*r};
        bool ok = true;
        for (auto& t : kT) for (int i = 0; i < 4; ++i) for (int j = i+1; j < 4; ++j)
        {
            auto off = [&](int c){ return (long)((c&1) + r*((c>>1)&1) + r*r*((c>>2)&1)); };
            long delta = std::labs(off(t[i]) - off(t[j]));
            ok &= std::count(d, d+7, delta) == 1;
        }
        CHECK(ok, "a tet edge falls outside the 7 positive Kuhn directions");
        // the 6 tets must exactly fill the cube: volumes sum to 1
        double vol = 0;
        for (auto& t : kT) {
            auto P = [&](int c){ return std::array<double,3>{double(c&1), double((c>>1)&1), double((c>>2)&1)}; };
            auto a=P(t[0]), b=P(t[1]), c=P(t[2]), e=P(t[3]);
            double u[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, v[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]}, w[3]={e[0]-a[0],e[1]-a[1],e[2]-a[2]};
            vol += std::fabs(u[0]*(v[1]*w[2]-v[2]*w[1]) - u[1]*(v[0]*w[2]-v[2]*w[0]) + u[2]*(v[0]*w[1]-v[1]*w[0])) / 6.0;
        }
        CHECK(std::fabs(vol - 1.0) < 1e-12, "tets cover volume %.6f, expected 1", vol);
    }

    std::printf("== T2  Analytic sphere: radius, normals, watertight, orientation, Euler, volume\n");
    for (int res : {32, 64, 96})
    {
        const float size = 20.f, sigma = 3.f, amp = 10.f, iso = 2.f;
        const double rIso = sigma * std::sqrt(2.0 * std::log(amp / iso));   // exact iso radius
        auto g = fieldFromBlobs(res, size, {{0.3f, -0.2f, 0.1f, sigma, amp}});
        SurfaceMesher M; M.loadGrid(res, g.data());
        SurfaceMesh mesh; M.extract(size, iso, false, mesh);
        Topo T = analyse(mesh);
        const double cell = daCell(size, res);
        double maxErr = 0, minNdot = 1;
        for (int i = 0; i < mesh.numVertices(); ++i) {
            double x = mesh.positions[3*i]-0.3, y = mesh.positions[3*i+1]+0.2, z = mesh.positions[3*i+2]-0.1;
            double r = std::sqrt(x*x+y*y+z*z);
            maxErr = std::max(maxErr, std::fabs(r - rIso));
            minNdot = std::min(minNdot, (x*mesh.normals[3*i] + y*mesh.normals[3*i+1] + z*mesh.normals[3*i+2]) / r);
        }
        double vExact = 4.0/3.0*M_PI*rIso*rIso*rIso;
        std::printf("   res %3d: V=%ld F=%ld  maxRadiusErr=%.4f (%.2f cells)  minNormalDot=%.4f  vol=%.2f/%.2f\n",
                    res, T.V, T.F, maxErr, maxErr / cell, minNdot, T.signedVolume, vExact);
        CHECK(T.closed,   "res %d: surface not closed", res);
        CHECK(T.oriented, "res %d: inconsistent winding", res);
        CHECK(T.components == 1, "res %d: %d components", res, T.components);
        CHECK(T.V - T.E + T.F == 2, "res %d: Euler %ld != 2", res, T.V - T.E + T.F);
        CHECK(maxErr < 0.25 * cell, "res %d: radius error %.3f cells", res, maxErr / cell);
        CHECK(minNdot > 0.99, "res %d: normal off-radial (%.4f)", res, minNdot);
        CHECK(T.signedVolume > 0 && std::fabs(T.signedVolume - vExact) / vExact < 0.02,
              "res %d: volume %.3f vs %.3f (sign = winding)", res, T.signedVolume, vExact);
        CHECK(T.minNormalAgreement > 0.0, "res %d: a face winds against its normals", res);
        CHECK(T.unreferenced == 0, "res %d: %d unreferenced vertices", res, T.unreferenced);
        CHECK(M.uncachedEdges() == 0, "res %d: %llu uncached edges", res, (unsigned long long)M.uncachedEdges());
    }

    std::printf("== T3  Flip Normals reverses normals AND winding (volume sign)\n");
    {
        auto g = fieldFromBlobs(48, 20.f, {{0, 0, 0, 3.f, 10.f}});
        SurfaceMesher M; M.loadGrid(48, g.data());
        SurfaceMesh a, b; M.extract(20.f, 2.f, false, a); M.extract(20.f, 2.f, true, b);
        Topo A = analyse(a), B = analyse(b);
        CHECK(A.signedVolume > 0 && B.signedVolume < 0 && std::fabs(A.signedVolume + B.signedVolume) < 1e-3,
              "volumes %.3f / %.3f", A.signedVolume, B.signedVolume);
        CHECK(B.closed && B.oriented && B.minNormalAgreement > 0, "flipped mesh not consistent");
    }

    std::printf("== T4  Multiple blobs + a blob clipped by the box (capped, still closed)\n");
    {
        const int res = 64; const float size = 24.f;
        auto g = fieldFromBlobs(res, size, {{-6,0,0,2.f,8.f},{6,0,0,2.f,8.f},{0,0,11.5f,2.5f,8.f}});
        SurfaceMesher M; M.loadGrid(res, g.data());
        SurfaceMesh mesh; M.extract(size, 1.5f, false, mesh);
        Topo T = analyse(mesh);
        std::printf("   components=%d  Euler=%ld  closed=%d oriented=%d\n",
                    T.components, T.V - T.E + T.F, T.closed, T.oriented);
        CHECK(T.components == 3, "expected 3 blobs, got %d", T.components);
        CHECK(T.closed && T.oriented, "clipped/multi mesh not watertight+oriented");
        CHECK(T.V - T.E + T.F == 2 * T.components, "Euler %ld != 2*components", T.V - T.E + T.F);
    }

    std::printf("== T5  Edge-cache stamps: reuse across cooks == fresh mesher\n");
    {
        const int res = 48; const float size = 20.f;
        auto g1 = fieldFromBlobs(res, size, {{0,0,0,3.f,10.f}});
        auto g2 = fieldFromBlobs(res, size, {{1,-1,0,2.5f,9.f},{-3,2,1,2.f,7.f}});
        SurfaceMesher A; SurfaceMesh ma;
        A.loadGrid(res, g1.data()); A.extract(size, 2.f, false, ma);   // cook 1
        A.loadGrid(res, g2.data()); A.extract(size, 1.7f, false, ma);  // cook 2 (stale stamps around)
        SurfaceMesher B; SurfaceMesh mb;
        B.loadGrid(res, g2.data()); B.extract(size, 1.7f, false, mb);
        CHECK(ma.positions == mb.positions && ma.indices == mb.indices && ma.normals == mb.normals,
              "stale edge-cache entries leaked between cooks");
    }

    std::printf("== T6  Atlas pack/unpack round trip, flipped rows, size mismatch\n");
    for (int res : {16, 30, 64, 101, 192})
    {
        const int W = daAtlasW(res), H = daAtlasH(res);
        std::vector<float> grid((size_t)res*res*res); std::mt19937 rng(res);
        std::uniform_real_distribution<float> U(0, 5);
        for (auto& v : grid) v = U(rng);
        // writer: exactly ParticleDensityTOP's writeAtlasKernel indexing
        std::vector<float> atlas((size_t)W*H, -7.f), atlasTopFirst((size_t)W*H);
        int tx = daTilesX(res);
        for (int i = 0; i < W*H; ++i) {
            int px = i % W, py = i / W, x = px % res, y = py % res, z = (py / res) * tx + px / res;
            atlas[i] = z < res ? grid[x + (size_t)res*(y + (size_t)res*z)] : 0.f;
        }
        for (int row = 0; row < H; ++row)   // a reader that lists rows top-first
            std::copy(atlas.begin() + (size_t)(H-1-row)*W, atlas.begin() + (size_t)(H-row)*W,
                      atlasTopFirst.begin() + (size_t)row*W);
        bool layoutOK = true;
        for (int z = 0; z < res; z += std::max(1, res/7)) for (int y = 0; y < res; y += 3) for (int x = 0; x < res; x += 5)
            layoutOK &= atlas[daTexelIndex(res, x, y, z)] == grid[x + (size_t)res*(y + (size_t)res*z)];
        CHECK(layoutOK, "res %d: daTexelIndex disagrees with the writer", res);
        CHECK(W <= 16384 && H <= 16384, "res %d: atlas %dx%d exceeds texture limits", res, W, H);

        SurfaceMesher M1, M2;
        CHECK(M1.loadAtlas(atlas.data(), (int64_t)atlas.size(), res, false), "res %d load: %s", res, M1.error());
        CHECK(M2.loadAtlas(atlasTopFirst.data(), (int64_t)atlas.size(), res, true), "res %d flip load", res);
        bool same = true;
        for (int z = 1; z < res-1; ++z) for (int y = 1; y < res-1; ++y) for (int x = 1; x < res-1; ++x) {
            size_t k = x + (size_t)res*(y + (size_t)res*z);
            same &= M1.grid()[k] == grid[k] && M2.grid()[k] == grid[k];
        }
        CHECK(same, "res %d: unpacked interior differs", res);
        SurfaceMesher bad;
        CHECK(!bad.loadAtlas(atlas.data(), (int64_t)atlas.size() - 1, res, false) && bad.error()[0],
              "res %d: wrong sample count accepted", res);
        std::printf("   res %3d -> atlas %4dx%-4d (%2dx%-2d tiles) %6.2f MB  ok\n",
                    res, W, H, daTilesX(res), daTilesY(res), W*H*4.0/1048576.0);
    }

    std::printf("== T7  Full Option-B pipeline on the TOP's real particle distribution\n");
    {
        const int count = 200000; int side = 0;
        auto tex = ref::makePositionTexture(count, 6.0f, side);
        // sprinkle hostile LIVE particles: NaN, inf, and far escapees (neg. gravity)
        tex[4*10+0] = NAN; tex[4*11+1] = INFINITY; tex[4*12+2] = 5e6f; tex[4*13+0] = -900.f;
        const int res = 64; const float size = 32.f, smooth = 0.6f;
        auto t0 = std::chrono::steady_clock::now();
        ref::DensityResult D = ref::runDensityTOP(tex, side, side, res, size, smooth);
        double tRef = ms(t0);
        double rawMass = 0; for (float v : D.rawGrid) rawMass += v;
        double meanNorm = 0; int n = 0;
        for (int z=0;z<res;++z) for (int y=0;y<res;++y) for (int x=0;x<res;++x) { meanNorm += D.atlas[daTexelIndex(res,x,y,z)]; ++n; }
        meanNorm /= n;
        std::printf("   texture %dx%d (%d padding texels with garbage)  live=%u  inBox=%u  (cpu ref %.0f ms)\n",
                    side, side, side*side - count, D.liveCount, D.inBoxCount, tRef);
        CHECK(D.liveCount == (unsigned)count - 2, "live=%u, expected %d (padding + NaN/inf masked)", D.liveCount, count - 2);
        CHECK(D.inBoxCount == D.liveCount - 2, "inBox=%u: escapees should be dropped", D.inBoxCount);
        CHECK(std::fabs(rawMass - D.inBoxCount) / D.inBoxCount < 1e-4,
              "trilinear splat not mass-conserving: %.2f vs %u", rawMass, D.inBoxCount);
        CHECK(std::fabs(meanNorm - (double)D.inBoxCount / D.liveCount) < 0.02,
              "normalised mean %.4f, expected ~%.4f", meanNorm, (double)D.inBoxCount / D.liveCount);
        bool allWritten = true; for (float v : D.atlas) allWritten &= (v >= 0.f);
        CHECK(allWritten, "some atlas texels were never written (padding tiles must be 0)");

        float peak = 0; for (float v : D.atlas) peak = std::max(peak, v);
        std::printf("   normalised density: mean %.3f  peak %.1f (x mean)\n", meanNorm, peak);

        SurfaceMesher M;
        CHECK(M.loadAtlas(D.atlas.data(), (int64_t)D.atlas.size(), res, false), "load: %s", M.error());
        for (float iso : {2.f, 4.f, 8.f, 16.f})
        {
            SurfaceMesh mesh;
            auto t1 = std::chrono::steady_clock::now();
            M.extract(size, iso, false, mesh);
            double tx = ms(t1);
            Topo T = analyse(mesh);
            std::printf("   iso %5.1f: %7ld verts %7ld tris  comps %3d  closed %d oriented %d  extract %.2f ms\n",
                        iso, T.V, T.F, T.components, T.closed, T.oriented, tx);
            CHECK(T.F > 0, "iso %.1f produced an empty mesh", iso);
            CHECK(T.closed && T.oriented && T.minNormalAgreement > 0.0,
                  "iso %.1f: mesh not watertight / consistently wound", iso);
            CHECK(T.V - T.E + T.F == 2 * T.components || T.components > 1,
                  "iso %.1f: single body has Euler %ld", iso, T.V - T.E + T.F);
        }
    }

    std::printf("== T8  SOP-side cost vs grid resolution (load + extract, iso 4, 200k particles)\n");
    {
        int side = 0; auto tex = ref::makePositionTexture(200000, 6.0f, side);
        for (int res : {48, 64, 96, 128, 160})
        {
            auto D = ref::runDensityTOP(tex, side, side, res, 32.f, 0.6f);
            SurfaceMesher M; SurfaceMesh mesh;
            M.loadAtlas(D.atlas.data(), (int64_t)D.atlas.size(), res, false);
            M.extract(32.f, 4.f, false, mesh);                       // warm the buffers
            double best = 1e9;
            for (int k = 0; k < 5; ++k) {
                auto t1 = std::chrono::steady_clock::now();
                M.loadAtlas(D.atlas.data(), (int64_t)D.atlas.size(), res, false);
                M.extract(32.f, 4.f, false, mesh);
                best = std::min(best, ms(t1));
            }
            std::printf("   res %3d: readback %5.2f MB  %7d verts %7d tris  CPU %7.2f ms/cook\n",
                        res, D.atlas.size() * 4.0 / 1048576.0, mesh.numVertices(), mesh.numTriangles(), best);
        }
    }

    std::printf("\n%d checks passed, %d failed\n", gPass, gFail);
    return gFail;
}
