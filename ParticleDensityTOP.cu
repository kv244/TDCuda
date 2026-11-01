// ===========================================================================
// ParticleDensityTOP.cu
// ---------------------------------------------------------------------------
// A TouchDesigner CPlusPlus TOP (CUDA execute mode) that converts the particle
// POSITION texture (from ParticleCloudTOP) into a smooth 3D DENSITY FIELD,
// entirely on the GPU, and outputs it as a single-channel float atlas of
// Z-slices (layout: DensityAtlas.h). ParticleSurfaceSOP then downloads only
// this grid -- ~1 MB at 64^3 regardless of particle count -- and marches it
// into a mesh. This is "Option B": the per-particle work stays where the
// particles already live, and the GPU->CPU transfer is bounded by the grid.
//
// PIPELINE (all on one CUDA stream, no host sync)
// ---------------------------------------------------------------------------
//   1. clear      : zero the grid + the live-particle counter
//   2. splat      : one thread per input texel. Dead/padding texels (alpha<0.5)
//                   and non-finite positions are skipped. Each live particle
//                   deposits weight 1 TRILINEARLY into the 8 surrounding grid
//                   points (atomicAdd). Particles outside the fixed box are
//                   ignored -- escaped particles (negative gravity) can't
//                   stretch the grid. Live count is reduced per block first,
//                   then one atomic per block (avoids 1M atomics on one word).
//   3. blur       : separable box blur, radius = Smooth (world) / cell, run
//                   along x, y, z (ping-pong buffers). Turns the point deposit
//                   into a continuous field; cost is res^3 * (2r+1) * 3 and
//                   does NOT depend on the particle count.
//   4. normalize  : scale by res^3 / liveCount so 1.0 == the mean density if
//   + write atlas   every live particle sat inside the box. The Isolevel on
//                   the SOP therefore means "N x average density" and stays
//                   stable when you change the particle count.
//
// WHY A FIXED, ORIGIN-CENTRED BOX
// ---------------------------------------------------------------------------
// Re-fitting bounds to the particles every frame (a) lets a single outlier blow
// the grid up so the cloud covers a few cells, and (b) slides the sampling
// lattice every frame so the surface shimmers. A fixed box avoids both.
//
//   CUDA major must match TouchDesigner's (12.x for 2025.x).  RTX 4070 = sm_89.
//
//   TD API usage verified against the 2025.x headers (Samples/CPlusPlus/
//   CudaTOP): input acquired via getCUDAArray(OP_CUDAAcquireInfo) BEFORE
//   beginCUDAOperations(), exactly as the CudaTOP sample does.
// ===========================================================================

#include "TOP_CPlusPlusBase.h"
#include "DensityAtlas.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace TD;

// Output format. 1 = single-channel 32-bit float (OP_PixelFormat::Mono32Float,
// 1/4 the readback). 0 = RGBA32Float with density replicated in r,g,b (the SOP
// reads channel 0 either way) -- only useful for debugging in a TOP viewer.
#ifndef DENSITY_MONO
#define DENSITY_MONO 1
#endif

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Finite check by bit pattern: immune to fast-math assumptions about NaN/inf.
__device__ __forceinline__ bool finitef(float v)
{
    return (__float_as_uint(v) & 0x7f800000u) != 0x7f800000u;
}

// 2. Trilinear atomic splat of every live particle into the grid.
__global__ void splatKernel(cudaTextureObject_t posTex, int texW, int texH,
                            float* grid, unsigned int* liveCount,
                            int res, float bmin, float invCell)
{
    __shared__ unsigned int sLive;
    if (threadIdx.x == 0) sLive = 0u;
    __syncthreads();

    // NOTE: no early return before the __syncthreads below -- every thread in
    // the block must reach it, so out-of-range threads just skip the work.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < texW * texH)
    {
        int tx = i % texW;
        int ty = i / texW;
        float4 p = tex2D<float4>(posTex, tx + 0.5f, ty + 0.5f);

        bool live = (p.w > 0.5f) && finitef(p.x) && finitef(p.y) && finitef(p.z);
        if (live)
        {
            atomicAdd(&sLive, 1u);

            float gx = (p.x - bmin) * invCell;
            float gy = (p.y - bmin) * invCell;
            float gz = (p.z - bmin) * invCell;
            float hiLim = (float)(res - 1);

            // trilinear needs the +1 neighbour inside the grid
            if (gx >= 0.0f && gy >= 0.0f && gz >= 0.0f &&
                gx < hiLim && gy < hiLim && gz < hiLim)
            {
                int x0 = (int)gx, y0 = (int)gy, z0 = (int)gz;
                float fx = gx - x0, fy = gy - y0, fz = gz - z0;
                float ix = 1.0f - fx, iy = 1.0f - fy, iz = 1.0f - fz;

                int sy = res, sz = res * res;
                int b  = x0 + sy * y0 + sz * z0;

                atomicAdd(&grid[b],                ix * iy * iz);
                atomicAdd(&grid[b + 1],            fx * iy * iz);
                atomicAdd(&grid[b + sy],           ix * fy * iz);
                atomicAdd(&grid[b + 1 + sy],       fx * fy * iz);
                atomicAdd(&grid[b + sz],           ix * iy * fz);
                atomicAdd(&grid[b + 1 + sz],       fx * iy * fz);
                atomicAdd(&grid[b + sy + sz],      ix * fy * fz);
                atomicAdd(&grid[b + 1 + sy + sz],  fx * fy * fz);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && sLive) atomicAdd(liveCount, sLive);
}

// 3. One separable box-blur pass along `axis` (0=x, 1=y, 2=z). Samples outside
// the grid count as zero (window always divided by 2r+1), so density falls off
// to zero at the box faces and the extracted surface closes there.
__global__ void blurAxisKernel(const float* __restrict__ in, float* __restrict__ out,
                               int res, int axis, int r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = res * res * res;
    if (i >= n) return;

    int x = i % res;
    int y = (i / res) % res;
    int z = i / (res * res);

    int c      = (axis == 0) ? x : (axis == 1) ? y : z;
    int stride = (axis == 0) ? 1 : (axis == 1) ? res : res * res;

    const float* line = in + (i - c * stride);    // start of this axis line
    int lo = max(c - r, 0);
    int hi = min(c + r, res - 1);
    float s = 0.0f;
    for (int k = lo; k <= hi; ++k)
        s += line[k * stride];

    out[i] = s / (float)(2 * r + 1);
}

// 4. Normalise and write every atlas texel (padding tiles get 0).
__global__ void writeAtlasKernel(const float* __restrict__ grid,
                                 const unsigned int* __restrict__ liveCount,
                                 cudaSurfaceObject_t surf,
                                 int res, int tilesX, int atlasW, int atlasH)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= atlasW * atlasH) return;

    int px = i % atlasW;
    int py = i / atlasW;
    int tx = px / res, x = px - tx * res;
    int ty = py / res, y = py - ty * res;
    int z  = ty * tilesX + tx;

    float v = 0.0f;
    if (z < res)
    {
        float live = (float)max(*liveCount, 1u);
        float norm = ((float)res * (float)res * (float)res) / live;  // 1.0 == mean density
        v = grid[x + res * (y + res * z)] * norm;
    }

#if DENSITY_MONO
    surf2Dwrite(v, surf, px * (int)sizeof(float), py);
#else
    surf2Dwrite(make_float4(v, v, v, 1.0f), surf, px * (int)sizeof(float4), py);
#endif
}

// ---------------------------------------------------------------------------
// TouchDesigner operator
// ---------------------------------------------------------------------------
class ParticleDensityTOP : public TOP_CPlusPlusBase
{
public:
    ParticleDensityTOP(const OP_NodeInfo*, TOP_Context* context)
        : myContext(context), myStream(0),
          myGridA(nullptr), myGridB(nullptr), myLive(nullptr), myRes(0)
    {
        cudaStreamCreate(&myStream);
        cudaMalloc(&myLive, sizeof(unsigned int));
        myError[0] = 0;
    }

    virtual ~ParticleDensityTOP()
    {
        freeGrids();
        if (myLive)   cudaFree(myLive);
        if (myStream) cudaStreamDestroy(myStream);
    }

    virtual void getGeneralInfo(TOP_GeneralInfo* g, const OP_Inputs*, void*) override
    {
        g->cookEveryFrameIfAsked = true;   // follows the position TOP each frame
    }

    virtual void setupParameters(OP_ParameterManager* m, void*) override
    {
        appendI(m, "Gridres",    "Grid Resolution", 64,   16,  192);
        appendF(m, "Boundssize", "Bounds Size",     32.0, 1.0, 200.0);
        appendF(m, "Smooth",     "Smoothing Radius", 0.6, 0.0, 5.0);
    }

    virtual void getErrorString(OP_String* error, void*) override
    {
        if (myError[0]) error->setString(myError);
    }

    virtual void execute(TOP_Output* output, const OP_Inputs* inputs, void*) override
    {
        myError[0] = 0;

        const OP_TOPInput* top = inputs->getNumInputs() > 0 ? inputs->getInputTOP(0) : nullptr;
        if (!top)
        {
            setError("Connect the particle position TOP to input 0.");
            return;
        }
        if (top->textureDesc.pixelFormat != OP_PixelFormat::RGBA32Float)
        {
            setError("Input must be the RGBA32Float position TOP (x,y,z,life).");
            return;
        }

        const int   res    = std::min(std::max(inputs->getParInt("Gridres"), 16), 192);
        float       size   = (float)inputs->getParDouble("Boundssize");
        if (size <= 0.0f) size = 1.0f;
        const float smooth = (float)inputs->getParDouble("Smooth");

        const float cell    = daCell(size, res);
        const float bmin    = daBMin(size);
        const int   r       = std::min(std::max((int)std::lround(smooth / cell), 0), 8);
        const int   tilesX  = daTilesX(res);
        const int   atlasW  = daAtlasW(res);
        const int   atlasH  = daAtlasH(res);
        const int   nGrid   = res * res * res;

        if (!ensureGrids(res))
        {
            setError("Out of GPU memory allocating the density grid.");
            return;
        }

        TOP_CUDAOutputInfo info;
        info.textureDesc.texDim      = OP_TexDim::e2D;
        info.textureDesc.width       = atlasW;
        info.textureDesc.height      = atlasH;
#if DENSITY_MONO
        info.textureDesc.pixelFormat = OP_PixelFormat::Mono32Float;
#else
        info.textureDesc.pixelFormat = OP_PixelFormat::RGBA32Float;
#endif
        info.stream                  = myStream;

        // Acquire the input and create the output BEFORE beginCUDAOperations()
        // (the TD API requires all 'inputs' calls first). Their cudaArray
        // members are only filled in once beginCUDAOperations() has run.
        OP_CUDAAcquireInfo acquire;
        acquire.stream = myStream;
        const OP_CUDAArrayInfo* inArr  = top->getCUDAArray(acquire, nullptr);
        const OP_CUDAArrayInfo* outArr = output->createCUDAArray(info, nullptr);
        if (!inArr || !outArr) return;

        if (!myContext->beginCUDAOperations(nullptr)) return;

        if (!inArr->cudaArray || !outArr->cudaArray)
        {
            myContext->endCUDAOperations(nullptr);
            return;
        }
        const int texW = (int)inArr->textureDesc.width;
        const int texH = (int)inArr->textureDesc.height;

        // read-only texture object over the input positions (point sampled)
        cudaResourceDesc inRes = {};
        inRes.resType         = cudaResourceTypeArray;
        inRes.res.array.array = inArr->cudaArray;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModePoint;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        cudaTextureObject_t posTex = 0;

        cudaResourceDesc outRes = {};
        outRes.resType         = cudaResourceTypeArray;
        outRes.res.array.array = outArr->cudaArray;
        cudaSurfaceObject_t outSurf = 0;

        if (cudaCreateTextureObject(&posTex, &inRes, &texDesc, nullptr) == cudaSuccess &&
            cudaCreateSurfaceObject(&outSurf, &outRes) == cudaSuccess)
        {
            const int block = 256;

            // 1. clear
            cudaMemsetAsync(myGridA, 0, (size_t)nGrid * sizeof(float), myStream);
            cudaMemsetAsync(myLive,  0, sizeof(unsigned int),          myStream);

            // 2. splat
            int nTex = texW * texH;
            splatKernel<<<(nTex + block - 1) / block, block, 0, myStream>>>(
                posTex, texW, texH, myGridA, myLive, res, bmin, 1.0f / cell);

            // 3. separable blur  A->B (x), B->A (y), A->B (z)  => result in B
            const float* result = myGridA;
            if (r > 0)
            {
                int gb = (nGrid + block - 1) / block;
                blurAxisKernel<<<gb, block, 0, myStream>>>(myGridA, myGridB, res, 0, r);
                blurAxisKernel<<<gb, block, 0, myStream>>>(myGridB, myGridA, res, 1, r);
                blurAxisKernel<<<gb, block, 0, myStream>>>(myGridA, myGridB, res, 2, r);
                result = myGridB;
            }

            // 4. normalise + write the atlas
            int nAtlas = atlasW * atlasH;
            writeAtlasKernel<<<(nAtlas + block - 1) / block, block, 0, myStream>>>(
                result, myLive, outSurf, res, tilesX, atlasW, atlasH);
        }

        if (outSurf) cudaDestroySurfaceObject(outSurf);
        if (posTex)  cudaDestroyTextureObject(posTex);

        myContext->endCUDAOperations(nullptr);
    }

private:
    void setError(const char* msg)
    {
        std::snprintf(myError, sizeof(myError), "%s", msg);
    }

    void freeGrids()
    {
        if (myGridA) cudaFree(myGridA);
        if (myGridB) cudaFree(myGridB);
        myGridA = myGridB = nullptr;
        myRes = 0;
    }

    // Persistent grids, reallocated only when the resolution changes.
    bool ensureGrids(int res)
    {
        if (res == myRes && myGridA && myGridB) return true;
        freeGrids();
        size_t bytes = (size_t)res * res * res * sizeof(float);
        if (cudaMalloc(&myGridA, bytes) != cudaSuccess) { freeGrids(); return false; }
        if (cudaMalloc(&myGridB, bytes) != cudaSuccess) { freeGrids(); return false; }
        myRes = res;
        return true;
    }

    static void appendI(OP_ParameterManager* m, const char* name, const char* label,
                        int def, int lo, int hi)
    {
        OP_NumericParameter p;
        p.name = name; p.label = label;
        p.defaultValues[0] = def;
        p.minSliders[0] = lo; p.maxSliders[0] = hi;
        p.minValues[0]  = lo; p.clampMins[0] = true;
        p.maxValues[0]  = hi; p.clampMaxes[0] = true;
        m->appendInt(p);
    }
    static void appendF(OP_ParameterManager* m, const char* name, const char* label,
                        double def, double lo, double hi)
    {
        OP_NumericParameter p;
        p.name = name; p.label = label;
        p.defaultValues[0] = def;
        p.minSliders[0] = lo; p.maxSliders[0] = hi;
        m->appendFloat(p);
    }

    TOP_Context*  myContext;
    cudaStream_t  myStream;
    float*        myGridA;
    float*        myGridB;
    unsigned int* myLive;
    int           myRes;
    char          myError[256];
};

// ---------------------------------------------------------------------------
// API version across header revisions (same SFINAE helper as ParticleCloudTOP)
// ---------------------------------------------------------------------------
template <typename T>
static auto tdSetAPIVersion(T* info, int)
    -> decltype(info->setAPIVersion(TOPCPlusPlusAPIVersion), void())
{
    // setAPIVersion() is [[nodiscard]] bool in 2025.x headers: it always stores
    // the version and returns whether TD declared it supported. TD itself
    // refuses to load an incompatible plugin, so there is nothing to recover
    // here -- discard the flag explicitly instead of tripping -Wunused-result.
    (void)info->setAPIVersion(TOPCPlusPlusAPIVersion);
}
template <typename T>
static void tdSetAPIVersion(T* info, long)
{
    info->apiVersion = TOPCPlusPlusAPIVersion;
}

// ---------------------------------------------------------------------------
// Plugin entry points
// ---------------------------------------------------------------------------
extern "C"
{
    DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo* info)
    {
        tdSetAPIVersion(info, 0);
        info->executeMode = TOP_ExecuteMode::CUDA;

        OP_CustomOPInfo& c = info->customOPInfo;
        c.opType->setString("Particledensity");
        c.opLabel->setString("Particle Density");
        c.authorName->setString("you");
        c.authorEmail->setString("you@example.com");
        c.minInputs = 1;            // the position TOP
        c.maxInputs = 1;
    }

    DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info,
                                                   TOP_Context* context)
    {
        return new ParticleDensityTOP(info, context);
    }

    DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
    {
        delete (ParticleDensityTOP*)instance;
    }
}
