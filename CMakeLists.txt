cmake_minimum_required(VERSION 3.18)
project(ParticleCloudTOP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Point this at the folder that contains TOP_CPlusPlusBase.h ------------
# It ships inside TouchDesigner, e.g. under
#   <TD install>/Samples/CPlusPlus/...          (copy the API headers next to
# this file, or set the cache var below at configure time):
#   cmake -B build -A x64 -DTD_INCLUDE="C:/path/to/TD/CPlusPlus/headers"
set(TD_INCLUDE "${CMAKE_SOURCE_DIR}" CACHE PATH
    "Folder containing TOP_CPlusPlusBase.h and CPlusPlus_Common.h")

# The SOP base class (SOP_CPlusPlusBase.h) ships with a SOP sample, not with
# CudaTOP. Default: the SimpleShapesSOP sample next to the CudaTOP folder.
set(TD_SOP_INCLUDE "${TD_INCLUDE}/../SimpleShapesSOP" CACHE PATH
    "Folder containing SOP_CPlusPlusBase.h and CPlusPlus_Common.h")

# --- CUDA compile options (shared by both CUDA TOPs) -----------------------
# --use_fast_math: maps the kernels' sin/cos/log/sqrt/pow to fast SFU
#   intrinsics + flushes denormals. The output is purely visual, so the slight
#   precision loss is irrelevant and it's a free win on the 4070 (Ada).
# -O3 (device): ensure full device-side optimization regardless of config.
# -allow-unsupported-compiler / version-mismatch: VS 2022 (toolset v143) +
#   some CUDA toolkits need the host-compiler check relaxed. Harmless on
#   matching pairs; required if CUDA < 12.4 with VS2022.
set(TDCUDA_CUDA_FLAGS
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-allow-unsupported-compiler;-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>)

# --- ParticleCloudTOP: the CUDA particle simulation ------------------------
add_library(ParticleCloudTOP SHARED ParticleCloudTOP.cu)
target_include_directories(ParticleCloudTOP PRIVATE ${TD_INCLUDE})
set_target_properties(ParticleCloudTOP PROPERTIES
    CUDA_ARCHITECTURES "89"     # RTX 4070 (Ada). Add "86;89" etc. for others.
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""                   # produce "ParticleCloudTOP.dll", no "lib" prefix
    SUFFIX ".dll")
target_compile_options(ParticleCloudTOP PRIVATE ${TDCUDA_CUDA_FLAGS})

# --- ParticleDensityTOP: GPU splat + blur -> density slice atlas -----------
# Reads the position TOP as its input; see SURFACE.md / DensityAtlas.h.
add_library(ParticleDensityTOP SHARED ParticleDensityTOP.cu)
target_include_directories(ParticleDensityTOP PRIVATE ${TD_INCLUDE} ${CMAKE_SOURCE_DIR})
set_target_properties(ParticleDensityTOP PROPERTIES
    CUDA_ARCHITECTURES "89"
    PREFIX ""
    SUFFIX ".dll")
target_compile_options(ParticleDensityTOP PRIVATE ${TDCUDA_CUDA_FLAGS})

# --- ParticleSurfaceSOP: marching tetrahedra -> welded mesh (plain C++) ----
# Only built when the SOP API headers are found, so a configure that points
# solely at TOP headers still builds the two TOPs instead of failing outright.
if(EXISTS "${TD_SOP_INCLUDE}/SOP_CPlusPlusBase.h")
    add_library(ParticleSurfaceSOP SHARED ParticleSurfaceSOP.cpp)
    target_include_directories(ParticleSurfaceSOP PRIVATE ${TD_SOP_INCLUDE} ${CMAKE_SOURCE_DIR})
    set_target_properties(ParticleSurfaceSOP PROPERTIES
        PREFIX ""
        SUFFIX ".dll")
else()
    message(WARNING "SOP_CPlusPlusBase.h not found in TD_SOP_INCLUDE='${TD_SOP_INCLUDE}' "
                    "-- skipping ParticleSurfaceSOP. Pass -DTD_SOP_INCLUDE=<SOP sample dir>.")
endif()

# Build:
#   cmake -B build -A x64 -DTD_INCLUDE="..." [-DTD_SOP_INCLUDE="..."]
#   cmake --build build --config Release
# Result: build/Release/{ParticleCloudTOP,ParticleDensityTOP,ParticleSurfaceSOP}.dll
