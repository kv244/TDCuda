// CHECK-ONLY shim: clang lacks nvcc's surf2Dwrite intrinsic. Same signature as
// CUDA's surface_indirect_functions.h so call sites type-check identically.
// Implemented with the real PTX store so the generated PTX is meaningful.
#pragma once
#include <cuda_runtime.h>
__device__ __forceinline__ void surf2Dwrite(float v, cudaSurfaceObject_t s, int x, int y,
                                            cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)
{ asm volatile("sust.b.2d.b32.trap [%0, {%1, %2}], {%3};" :: "l"(s), "r"(x), "r"(y), "f"(v) : "memory"); }
__device__ __forceinline__ void surf2Dwrite(float4 v, cudaSurfaceObject_t s, int x, int y,
                                            cudaSurfaceBoundaryMode = cudaBoundaryModeTrap)
{ asm volatile("sust.b.2d.v4.b32.trap [%0, {%1, %2}], {%3, %4, %5, %6};"
               :: "l"(s), "r"(x), "r"(y), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory"); }
