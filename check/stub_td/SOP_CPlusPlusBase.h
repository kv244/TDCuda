// STUB -- NOT the real TouchDesigner header (see CPlusPlus_Common.h note).
// Mirrors the 2025.x SOP API used by ParticleSurfaceSOP.cpp.
#pragma once
#include "CPlusPlus_Common.h"

namespace TD {

constexpr int32_t SOPCPlusPlusAPIVersion = 3;

struct Position { float x, y, z; Position(float a=0, float b=0, float c=0) : x(a), y(b), z(c) {} };
struct Vector   { float x, y, z; Vector(float a=0, float b=0, float c=0)   : x(a), y(b), z(c) {} };

enum class SOP_Winding : int32_t { LegacyCW = 0, CCW };

struct SOP_GeneralInfo {
    bool cookEveryFrame = false; bool cookEveryFrameIfAsked = false; bool directToGPU = false;
    SOP_Winding winding = SOP_Winding::LegacyCW;
};

struct SOP_VBOOutput {};

class SOP_PluginInfo {
public:
    [[nodiscard]] bool setAPIVersion(int32_t v) { apiVersion = v; return true; }
    OP_CustomOPInfo customOPInfo;
private:
    int32_t apiVersion = 0;
};

struct SOP_Output {
    int32_t addPoint(const Position&)                         { return 0; }
    bool    addPoints(const Position*, int32_t)               { return true; }
    bool    setNormal(const Vector&, int32_t)                 { return true; }
    bool    setNormals(const Vector*, int32_t, int32_t)       { return true; }
    bool    addTriangle(int32_t, int32_t, int32_t)            { return true; }
    bool    addTriangles(const int32_t*, int32_t)             { return true; }
};

class SOP_CPlusPlusBase {
public:
    virtual ~SOP_CPlusPlusBase() {}
    virtual void getGeneralInfo(SOP_GeneralInfo*, const OP_Inputs*, void*) {}
    virtual void execute(SOP_Output*, const OP_Inputs*, void*) = 0;
    virtual void executeVBO(SOP_VBOOutput*, const OP_Inputs*, void*) = 0;
    virtual void setupParameters(OP_ParameterManager*, void*) {}
    virtual void getWarningString(OP_String*, void*) {}
    virtual void getErrorString(OP_String*, void*) {}
    virtual void getInfoPopupString(OP_String*, void*) {}
};

} // namespace TD
