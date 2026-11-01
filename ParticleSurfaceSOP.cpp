// ===========================================================================
// ParticleSurfaceSOP.cpp
// ---------------------------------------------------------------------------
// A TouchDesigner CPlusPlus SOP that turns the particle cloud into a watertight
// MESH: a glowing liquid / metaball surface driven live by the CUDA sim. This
// is the CPU half of "Option B":
//
//   ParticleCloudTOP   (CUDA)  -> RGBA32F positions, alpha 0 marks padding
//        |  wired input
//   ParticleDensityTOP (CUDA)  -> splat + blur + normalise on the GPU, output a
//        |                        res^3 density field as a slice atlas
//        |  TOP to CHOP  (Download Type: Next Frame; channel r only)
//   ParticleSurfaceSOP (C++)   -> unpack atlas -> marching tetrahedra (welded,
//                                 gradient normals) -> SOP geometry
//
// Only the density grid crosses GPU->CPU (~1 MB at 64^3, ~8 MB at 128^3),
// independent of the particle count. The meshing core lives in SurfaceMesher.h
// (no TD dependency, unit-tested separately); this file is the TD glue.
//
// MATCHING PARAMETERS: Gridres and Boundssize here MUST equal the density
// TOP's. The atlas layout and grid geometry are both derived from those two
// numbers via DensityAtlas.h, so matching them is sufficient. The builder
// script binds both operators to the same component controls. If they drift,
// the sample count no longer fits and this SOP posts a warning instead of
// meshing garbage.
//
// TD API usage verified against the 2025.x SOP headers (Samples/CPlusPlus/
//   SimpleShapesSOP): SOP_Output batch calls, setNormals, addTriangles (count
//   = triangles), CHOP channelData, and CCW winding as the header recommends.
// ===========================================================================

#include "SOP_CPlusPlusBase.h"
#include "DensityAtlas.h"
#include "SurfaceMesher.h"

#include <chrono>
#include <cstdio>
#include <algorithm>

using namespace TD;

// 1 = hand TD the whole mesh with three batch calls (fast, default);
// 0 = per-point / per-triangle calls (kept for older SOP header revisions).
#ifndef SOP_BATCH_OUTPUT
#define SOP_BATCH_OUTPUT 1
#endif

namespace {

void appendI(OP_ParameterManager* m, const char* name, const char* label,
             int def, int lo, int hi)
{
    OP_NumericParameter p;
    p.name = name; p.label = label;
    p.defaultValues[0] = def;
    p.minSliders[0] = lo; p.maxSliders[0] = hi;
    p.minValues[0]  = lo; p.clampMins[0]  = true;
    p.maxValues[0]  = hi; p.clampMaxes[0] = true;
    m->appendInt(p);
}
void appendF(OP_ParameterManager* m, const char* name, const char* label,
             double def, double lo, double hi)
{
    OP_NumericParameter p;
    p.name = name; p.label = label;
    p.defaultValues[0] = def;
    p.minSliders[0] = lo; p.maxSliders[0] = hi;
    m->appendFloat(p);
}
void appendToggle(OP_ParameterManager* m, const char* name, const char* label, bool def)
{
    OP_NumericParameter p;
    p.name = name; p.label = label;
    p.defaultValues[0] = def ? 1.0 : 0.0;
    m->appendToggle(p);
}

} // namespace

class ParticleSurfaceSOP : public SOP_CPlusPlusBase
{
public:
    explicit ParticleSurfaceSOP(const OP_NodeInfo*) { myWarning[0] = 0; }
    virtual ~ParticleSurfaceSOP() {}

    virtual void getGeneralInfo(SOP_GeneralInfo* ginfo, const OP_Inputs*, void*) override
    {
        // Recook when something downstream asks and the density CHOP changed;
        // never burn main-thread time when nothing is displaying the surface.
        ginfo->cookEveryFrameIfAsked = true;
        ginfo->directToGPU           = false;               // CPU-side SOP_Output
        // The mesher emits triangles counter-clockwise around the outward
        // normal. The header defaults to LegacyCW; new plugins should be CCW.
        ginfo->winding               = SOP_Winding::CCW;
    }

    virtual void setupParameters(OP_ParameterManager* m, void*) override
    {
        {
            OP_StringParameter p;
            p.name = "Density"; p.label = "Density CHOP";
            m->appendCHOP(p);
        }
        appendI(m, "Gridres",    "Grid Resolution", 64,   16,  192);
        appendF(m, "Boundssize", "Bounds Size",     32.0, 1.0, 200.0);
        appendF(m, "Iso",        "Isolevel",        4.0,  0.0, 100.0);
        appendToggle(m, "Flipnormals", "Flip Normals",     false);
        appendToggle(m, "Flipatlasy",  "Flip Atlas Rows",  false);
    }

    virtual void getWarningString(OP_String* warning, void*) override
    {
        if (myWarning[0]) warning->setString(myWarning);
    }

    virtual void getInfoPopupString(OP_String* info, void*) override
    {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%d vertices, %d triangles, %.2f ms",
                      myMesh.numVertices(), myMesh.numTriangles(), myLastMs);
        info->setString(buf);
    }

    virtual void execute(SOP_Output* output, const OP_Inputs* inputs, void*) override
    {
        myWarning[0] = 0;
        const auto t0 = std::chrono::steady_clock::now();

        const OP_CHOPInput* cin = inputs->getParCHOP("Density");
        if (!cin || cin->numChannels < 1 || cin->numSamples < 1)
        {
            setWarning("Set Density CHOP to a TOP to CHOP of the Particle Density TOP.");
            return;
        }

        const int   res   = std::min(std::max(inputs->getParInt("Gridres"), 16), 192);
        const float size  = std::max((float)inputs->getParDouble("Boundssize"), 1e-3f);
        const float iso   = (float)inputs->getParDouble("Iso");
        const bool  flipN = inputs->getParInt("Flipnormals") != 0;
        const bool  flipY = inputs->getParInt("Flipatlasy")  != 0;
        if (!myMesher.loadAtlas(cin->channelData[0], cin->numSamples, res, flipY))
        {
            setWarning(myMesher.error());
            return;
        }

        myMesher.extract(size, iso, flipN, myMesh);
        emit(output);

        myLastMs = std::chrono::duration<float, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    }

    // Pure virtual in the 2025.x base class, so it must exist. TD only calls it
    // when getGeneralInfo() sets directToGPU = true; we fill CPU-side geometry
    // through execute() instead. (Writing the VBO directly here is the v2 path
    // for very high grid resolutions -- see SURFACE.md.)
    virtual void executeVBO(SOP_VBOOutput*, const OP_Inputs*, void*) override {}

private:
    // Hand the welded mesh to TD.
    void emit(SOP_Output* output)
    {
        const int nv = myMesh.numVertices();
        const int nt = myMesh.numTriangles();
        if (nv == 0 || nt == 0) return;

#if SOP_BATCH_OUTPUT
        // Position / Vector are three packed floats (CPlusPlus_Common.h itself
        // static_asserts sizeof == 12), the same layout as our xyz arrays, so
        // they can be handed over without a copy.
        static_assert(sizeof(Position) == 3 * sizeof(float), "Position layout");
        static_assert(sizeof(Vector)   == 3 * sizeof(float), "Vector layout");
        output->addPoints(reinterpret_cast<const Position*>(myMesh.positions.data()), nv);
        output->setNormals(reinterpret_cast<const Vector*>(myMesh.normals.data()), nv, 0);
        output->addTriangles(myMesh.indices.data(), nt);
#else
        const float* P = myMesh.positions.data();
        const float* N = myMesh.normals.data();
        for (int i = 0; i < nv; ++i)
        {
            output->addPoint(Position(P[3*i], P[3*i+1], P[3*i+2]));
            output->setNormal(Vector(N[3*i], N[3*i+1], N[3*i+2]), i);
        }
        const int32_t* I = myMesh.indices.data();
        for (int t = 0; t < nt; ++t)
            output->addTriangle(I[3*t], I[3*t+1], I[3*t+2]);
#endif
    }

    void setWarning(const char* msg) { std::snprintf(myWarning, sizeof(myWarning), "%s", msg); }

    SurfaceMesher myMesher;          // persistent grid + edge cache (no per-cook alloc)
    SurfaceMesh   myMesh;            // persistent output buffers (capacity reused)
    float         myLastMs = 0.0f;
    char          myWarning[320];
};

// ===========================================================================
// API version across header revisions (same SFINAE helper as the TOPs)
// ===========================================================================
template <typename T>
static auto tdSetAPIVersion(T* info, int)
    -> decltype(info->setAPIVersion(SOPCPlusPlusAPIVersion), void())
{
    // setAPIVersion() is [[nodiscard]] bool in 2025.x headers: it always stores
    // the version and returns whether TD declared it supported. TD itself
    // refuses to load an incompatible plugin, so there is nothing to recover
    // here -- discard the flag explicitly instead of tripping -Wunused-result.
    (void)info->setAPIVersion(SOPCPlusPlusAPIVersion);
}
template <typename T>
static void tdSetAPIVersion(T* info, long)
{
    info->apiVersion = SOPCPlusPlusAPIVersion;
}

// ===========================================================================
// Plugin entry points
// ===========================================================================
extern "C"
{
    DLLEXPORT void FillSOPPluginInfo(SOP_PluginInfo* info)
    {
        tdSetAPIVersion(info, 0);
        OP_CustomOPInfo& c = info->customOPInfo;
        c.opType->setString("Particlesurface");
        c.opLabel->setString("Particle Surface");
        c.authorName->setString("you");
        c.authorEmail->setString("you@example.com");
        c.minInputs = 0;            // density arrives through the CHOP parameter
        c.maxInputs = 0;
    }

    DLLEXPORT SOP_CPlusPlusBase* CreateSOPInstance(const OP_NodeInfo* info)
    {
        return new ParticleSurfaceSOP(info);
    }

    DLLEXPORT void DestroySOPInstance(SOP_CPlusPlusBase* instance)
    {
        delete (ParticleSurfaceSOP*)instance;
    }
}
