// STUB -- NOT the real TouchDesigner header (that one is (c) Derivative and is
// not redistributed). Minimal stand-in so CI can compile the operators on
// Linux. Every member below was checked against the real 2025.x headers
// (TouchDesigner 2025.33230, Samples/CPlusPlus/{CudaTOP,SimpleShapesSOP}):
// same names, signatures and return types, incl. [[nodiscard]] and pure
// virtuals, so a stub build fails where the real build would.
#pragma once
#include <cstdint>
#include <cstddef>

#ifndef DLLEXPORT
#define DLLEXPORT
#endif

struct cudaArray;   // CUDA runtime type, global namespace (matches cuda_runtime)

namespace TD {

struct OP_String { void setString(const char*) {} };                

struct OP_NodeInfo {};

struct OP_NumericParameter {                                         
    const char* name = nullptr; const char* label = nullptr; const char* page = nullptr;
    double defaultValues[4] = {}; double minValues[4] = {}; double maxValues[4] = {};
    bool   clampMins[4] = {};  bool clampMaxes[4] = {};              
    double minSliders[4] = {}; double maxSliders[4] = {};
};
struct OP_StringParameter {                                         
    const char* name = nullptr; const char* label = nullptr; const char* page = nullptr;
    const char* defaultValue = nullptr;
};

enum class OP_ParAppendResult { Success };

struct OP_ParameterManager {
    OP_ParAppendResult appendInt(const OP_NumericParameter&, int = 1)    { return {}; }
    OP_ParAppendResult appendFloat(const OP_NumericParameter&, int = 1)  { return {}; }
    OP_ParAppendResult appendPulse(const OP_NumericParameter&)           { return {}; }
    OP_ParAppendResult appendToggle(const OP_NumericParameter&)          { return {}; }
    OP_ParAppendResult appendCHOP(const OP_StringParameter&)             { return {}; }
};

enum class OP_TexDim { e2D, e2DArray, e3D, eCube };                    
enum class OP_PixelFormat { Invalid, RGBA8Fixed, Mono32Float, RGBA32Float }; 

struct OP_TextureDesc {                                              
    OP_TexDim texDim = OP_TexDim::e2D; uint32_t width = 0, height = 0, depth = 1;
    OP_PixelFormat pixelFormat = OP_PixelFormat::Invalid;
};

struct OP_CUDAArrayInfo {                                            
    ::cudaArray* cudaArray = nullptr;
    OP_TextureDesc textureDesc;                                     
};

struct CUstream_st;
struct OP_CUDAAcquireInfo { CUstream_st* stream = nullptr; };

struct OP_TOPInput {
    const OP_CUDAArrayInfo* getCUDAArray(const OP_CUDAAcquireInfo&, void*) const { return nullptr; }
    OP_TextureDesc textureDesc;
};

struct OP_CHOPInput {                                               
    int32_t numChannels = 0; int32_t numSamples = 0; double sampleRate = 60;
    const float** channelData = nullptr;
};

struct OP_Inputs {
    int    getParInt(const char*, int = 0) const    { return 0; }  
    double getParDouble(const char*, int = 0) const { return 0; }  
    int32_t             getNumInputs() const          { return 0; }
    const OP_TOPInput*  getInputTOP(int) const       { return nullptr; }
    const OP_CHOPInput* getParCHOP(const char*) const { return nullptr; }
};

struct OP_CustomOPInfo {                                            
    OP_String* opType = nullptr; OP_String* opLabel = nullptr; OP_String* opIcon = nullptr;
    OP_String* authorName = nullptr; OP_String* authorEmail = nullptr;
    int32_t minInputs = 0, maxInputs = 0;
};

} // namespace TD
