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

add_library(ParticleCloudTOP SHARED ParticleCloudTOP.cu)

target_include_directories(ParticleCloudTOP PRIVATE ${TD_INCLUDE})

set_target_properties(ParticleCloudTOP PROPERTIES
    CUDA_ARCHITECTURES "89"     # RTX 4070 (Ada). Add "86;89" etc. for others.
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""                   # produce "ParticleCloudTOP.dll", no "lib" prefix
    SUFFIX ".dll")

# --- CUDA compile options --------------------------------------------------
# --use_fast_math: maps the kernel's sin/cos/log/sqrt/pow to fast SFU
#   intrinsics + flushes denormals. The cloud is purely visual, so the slight
#   precision loss is irrelevant and it's a free win on the 4070 (Ada).
# -O3 (device): ensure full device-side optimization regardless of config.
# -allow-unsupported-compiler / version-mismatch: VS 2022 (toolset v143) +
#   some CUDA toolkits need the host-compiler check relaxed. Harmless on
#   matching pairs; required if CUDA < 12.4 with VS2022.
target_compile_options(ParticleCloudTOP PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-allow-unsupported-compiler;-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>)

# Build:
#   cmake -B build -A x64 -DTD_INCLUDE="..."
#   cmake --build build --config Release
# Result: build/Release/ParticleCloudTOP.dll
