// ===========================================================================
// ParticleCloudTOP.cu  --  STATEFUL version
// ---------------------------------------------------------------------------
// A TouchDesigner CPlusPlus TOP (CUDA execute mode) that runs a GPU-resident
// particle simulation with PERSISTENT position + velocity buffers and
// integrates a force field every frame (central gravity well + swirl +
// turbulence + damping). Positions are written to an RGBA32F output texture:
//   texel = (x, y, z, life)
// The render side reads that texture per-instance with texelFetch(TDInstanceID).
//
// This is the step up from the earlier "procedural positions per frame":
// particles now carry state (velocity) and evolve, like a cheap N-body/agent
// system. Forces here are a field (O(N)), not all-pairs gravity (O(N^2)) --
// all-pairs would need tiling/Barnes-Hut and is a further step.
//
//   CUDA major version must match TouchDesigner's bundled runtime (major 12
//   for 2025.x). Built here with 12.8; runs against TD's 12.8/12.9 via the
//   major-versioned cudart64_12.dll. RTX 4070 = Ada = sm_89.
// ===========================================================================

#include "TOP_CPlusPlusBase.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace TD;

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float hash01(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return (x & 0x00FFFFFFu) / 16777216.0f;
}
__device__ __forceinline__ float gaussian(uint32_t a, uint32_t b)
{
    float u1 = fmaxf(hash01(a), 1e-7f);
    float u2 = hash01(b);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// Seed positions as a soft gaussian blob; give a little tangential spin.
__global__ void initKernel(float4* pos, float4* vel, int count,
                           float radius, float swirl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint32_t s = (uint32_t)i * 2654435761u + 12345u;
    float gx = gaussian(s + 1u, s + 2u);
    float gy = gaussian(s + 3u, s + 4u);
    float gz = gaussian(s + 5u, s + 6u);
    float len = sqrtf(gx * gx + gy * gy + gz * gz) + 1e-6f;
    float shell = powf(fabsf(gaussian(s + 7u, s + 8u)), 0.6f);

    float x = gx / len * shell * radius;
    float y = gy / len * shell * radius * 0.7f;   // flatten a touch
    float z = gz / len * shell * radius;

    pos[i] = make_float4(x, y, z, 1.0f);
    // tangential kick around Y: cross((0,1,0),(x,y,z)) = (z,0,-x)
    vel[i] = make_float4(z * swirl * 0.1f, 0.0f, -x * swirl * 0.1f, 0.0f);
}

// Integrate one step and write the new position into the output surface.
__global__ void simKernel(float4* pos, float4* vel, cudaSurfaceObject_t surf,
                          int side, int count, float dt, float t,
                          float gravity, float swirl, float damping, float turb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float4 P = pos[i], V = vel[i];
    float px = P.x, py = P.y, pz = P.z;
    float vx = V.x, vy = V.y, vz = V.z;

    // central spring / gravity well
    float ax = -gravity * px;
    float ay = -gravity * py;
    float az = -gravity * pz;

    // swirl: normalized tangential around Y = (z,0,-x)/r
    float r = sqrtf(px * px + py * py + pz * pz) + 1e-3f;
    ax += swirl * ( pz) / r;
    az += swirl * (-px) / r;

    // turbulence (divergence-ish flow field)
    ax += turb * sinf(1.3f * py + t);
    ay += turb * sinf(1.1f * pz + t * 1.3f);
    az += turb * sinf(0.9f * px + t * 0.7f);

    // semi-implicit Euler + damping
    vx += ax * dt;  vy += ay * dt;  vz += az * dt;
    vx *= damping;  vy *= damping;  vz *= damping;
    px += vx * dt;  py += vy * dt;  pz += vz * dt;

    pos[i] = make_float4(px, py, pz, 1.0f);
    vel[i] = make_float4(vx, vy, vz, 0.0f);

    int x = i % side;
    int y = i / side;
    surf2Dwrite(make_float4(px, py, pz, 1.0f), surf, x * (int)sizeof(float4), y);
}

// ---------------------------------------------------------------------------
// TouchDesigner operator
// ---------------------------------------------------------------------------
class ParticleCloudTOP : public TOP_CPlusPlusBase
{
public:
    ParticleCloudTOP(const OP_NodeInfo*, TOP_Context* context)
        : myContext(context), myFrame(0), myStream(0),
          myPos(nullptr), myVel(nullptr), myAlloc(0), myNeedsInit(true)
    {
        cudaStreamCreate(&myStream);
    }

    virtual ~ParticleCloudTOP()
    {
        if (myStream) cudaStreamDestroy(myStream);
        if (myPos)    cudaFree(myPos);
        if (myVel)    cudaFree(myVel);
    }

    virtual void getGeneralInfo(TOP_GeneralInfo* g, const OP_Inputs*, void*) override
    {
        g->cookEveryFrameIfAsked = true;   // advance the sim whenever viewed/rendered
    }

    virtual void setupParameters(OP_ParameterManager* m, void*) override
    {
        appendF(m, "Count",      "Particle Count", 65536, 1024, 1048576, true);
        appendF(m, "Radius",     "Cloud Radius",   6.0,   0.1,  50.0,    false);
        appendF(m, "Speed",      "Speed",          1.0,   0.0,  5.0,     false);
        appendF(m, "Gravity",    "Gravity Well",   0.8,   0.0,  5.0,     false);
        appendF(m, "Swirl",      "Swirl",          1.4,   0.0,  8.0,     false);
        appendF(m, "Damping",    "Damping",        0.985, 0.90, 1.0,     false);
        appendF(m, "Turbulence", "Turbulence",     0.4,   0.0,  5.0,     false);
        {
            OP_NumericParameter p;
            p.name = "Reset"; p.label = "Reset";
            m->appendPulse(p);
        }
    }

    virtual void pulsePressed(const char* name, void*) override
    {
        if (name && strcmp(name, "Reset") == 0)
            myNeedsInit = true;
    }

    virtual void execute(TOP_Output* output, const OP_Inputs* inputs, void*) override
    {
        int    count   = inputs->getParInt("Count");
        if (count < 1) count = 1;
        double radius  = inputs->getParDouble("Radius");
        double speed   = inputs->getParDouble("Speed");
        double gravity = inputs->getParDouble("Gravity");
        double swirl   = inputs->getParDouble("Swirl");
        double damping = inputs->getParDouble("Damping");
        double turb    = inputs->getParDouble("Turbulence");

        int side = (int)ceil(sqrt((double)count));
        myFrame++;
        float dt = (float)((1.0 / 60.0) * speed);
        float t  = (float)(myFrame * (1.0 / 60.0));

        // (re)allocate persistent state when the count changes
        if (count != myAlloc)
        {
            if (myPos) cudaFree(myPos);
            if (myVel) cudaFree(myVel);
            cudaMalloc(&myPos, (size_t)count * sizeof(float4));
            cudaMalloc(&myVel, (size_t)count * sizeof(float4));
            myAlloc = count;
            myNeedsInit = true;
        }

        TOP_CUDAOutputInfo info;
        info.textureDesc.texDim      = OP_TexDim::e2D;
        info.textureDesc.width       = side;
        info.textureDesc.height      = side;
        info.textureDesc.pixelFormat = OP_PixelFormat::RGBA32Float;
        info.stream                  = myStream;

        const OP_CUDAArrayInfo* outArrayInfo = output->createCUDAArray(info, nullptr);
        if (!outArrayInfo) return;

        if (!myContext->beginCUDAOperations(nullptr)) return;

        int block = 256;
        int grid  = (count + block - 1) / block;

        if (myNeedsInit && myPos && myVel)
        {
            initKernel<<<grid, block, 0, myStream>>>(
                myPos, myVel, count, (float)radius, (float)swirl);
            myNeedsInit = false;
        }

        if (outArrayInfo->cudaArray && myPos && myVel)
        {
            cudaResourceDesc resDesc = {};
            resDesc.resType         = cudaResourceTypeArray;
            resDesc.res.array.array = outArrayInfo->cudaArray;

            cudaSurfaceObject_t surf = 0;
            if (cudaCreateSurfaceObject(&surf, &resDesc) == cudaSuccess)
            {
                simKernel<<<grid, block, 0, myStream>>>(
                    myPos, myVel, surf, side, count, dt, t,
                    (float)gravity, (float)swirl, (float)damping, (float)turb);
                cudaDestroySurfaceObject(surf);
            }
        }

        myContext->endCUDAOperations(nullptr);
    }

private:
    static void appendF(OP_ParameterManager* m, const char* name, const char* label,
                        double def, double lo, double hi, bool isInt)
    {
        OP_NumericParameter p;
        p.name = name; p.label = label;
        p.defaultValues[0] = def;
        p.minSliders[0] = lo; p.maxSliders[0] = hi;
        if (isInt) { p.minValues[0] = lo; p.clampMins[0] = true; m->appendInt(p); }
        else       { m->appendFloat(p); }
    }

    TOP_Context* myContext;
    uint64_t     myFrame;
    cudaStream_t myStream;
    float4*      myPos;
    float4*      myVel;
    int          myAlloc;
    bool         myNeedsInit;
};

// ---------------------------------------------------------------------------
// Set the plugin API version across TouchDesigner header revisions.
// Newer headers (2025.x) make TOP_PluginInfo::apiVersion private and expose
// setAPIVersion(); older public headers (CustomOperatorSamples) expose a
// public apiVersion member to assign directly. SFINAE picks whichever exists:
// the (…, int) overload is preferred and only compiles when setAPIVersion is
// available; otherwise the (…, long) overload assigns apiVersion directly.
// ---------------------------------------------------------------------------
template <typename T>
static auto tdSetAPIVersion(T* info, int)
    -> decltype(info->setAPIVersion(TOPCPlusPlusAPIVersion), void())
{
    info->setAPIVersion(TOPCPlusPlusAPIVersion);
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
        tdSetAPIVersion(info, 0);   // works on both old and new TD headers
        info->executeMode = TOP_ExecuteMode::CUDA;

        OP_CustomOPInfo& c = info->customOPInfo;
        c.opType->setString("Particlecloudcuda");
        c.opLabel->setString("Particle Cloud CUDA");
        c.authorName->setString("you");
        c.authorEmail->setString("you@example.com");
        c.minInputs = 0;
        c.maxInputs = 0;
    }

    DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info,
                                                   TOP_Context* context)
    {
        return new ParticleCloudTOP(info, context);
    }

    DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
    {
        delete (ParticleCloudTOP*)instance;
    }
}
