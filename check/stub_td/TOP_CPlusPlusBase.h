// STUB -- NOT the real TouchDesigner header (see CPlusPlus_Common.h note).
#pragma once
#include "CPlusPlus_Common.h"

namespace TD {

constexpr int32_t TOPCPlusPlusAPIVersion = 11;

enum class TOP_ExecuteMode { CPUMem, OpenGL, CUDA };                 

struct TOP_GeneralInfo { bool cookEveryFrame = false; bool cookEveryFrameIfAsked = false; }; 

// As in the real 2025.x headers: apiVersion is private and setAPIVersion() is
// [[nodiscard]] bool, so the SFINAE helper takes the same path as the real build.
class TOP_PluginInfo {                                               
public:
    [[nodiscard]] bool setAPIVersion(int32_t v) { apiVersion = v; return true; }
    TOP_ExecuteMode executeMode = TOP_ExecuteMode::CPUMem;
    OP_CustomOPInfo customOPInfo;
private:
    int32_t apiVersion = 0;
};

struct TOP_CUDAOutputInfo {                                          
    OP_TextureDesc textureDesc; uint32_t colorBufferIndex = 0;
    CUstream_st* stream = nullptr;
};

struct TOP_Output {                                                  
    const OP_CUDAArrayInfo* createCUDAArray(const TOP_CUDAOutputInfo&, void*) { return nullptr; }
};

struct TOP_Context {                                                 
    bool beginCUDAOperations(void*) { return true; }
    void endCUDAOperations(void*) {}
};

class TOP_CPlusPlusBase {
public:
    virtual ~TOP_CPlusPlusBase() {}
    virtual void getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void*) {}
    virtual void execute(TOP_Output*, const OP_Inputs*, void*) = 0;
    virtual void setupParameters(OP_ParameterManager*, void*) {}
    virtual void pulsePressed(const char*, void*) {}
    virtual void getErrorString(OP_String*, void*) {}                
    virtual void getWarningString(OP_String*, void*) {}              
};

} // namespace TD
