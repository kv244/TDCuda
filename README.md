# TDCuda — GPU Particle Cloud for TouchDesigner

![Compile](https://github.com/kv244/TDCuda/actions/workflows/compile.yml/badge.svg)

A GPU-resident 3D particle cloud for [TouchDesigner](https://derivative.ca),
driven by a custom **CUDA** operator. The simulation runs as a CUDA kernel
with persistent position **and velocity** buffers (a stateful force-field /
light N-body sim), writes particle positions into an `RGBA32F` texture, and a
GLSL material reads that texture to draw each particle as a camera-facing
billboard sprite. Nothing round-trips through the CPU.

```
 ParticleCloudTOP.dll  (CPlusPlus TOP, CUDA execute mode)
   pos[i], vel[i] persist on the GPU; integrate a force field each frame
        │
        ▼  writes RGBA32F  texel = (x, y, z, life)
   position texture ──────────────┐  bound as sampler 'sPos'
                                   ▼
   GLSL MAT vertex shader: texelFetch(TDInstanceID()) places one
   instanced billboard quad per particle, sized in the shader
        │
        ▼
   Render TOP ──▶ Blur ──▶ additive Composite (bloom) ──▶ out
```

The force field is **O(N)** per frame (central gravity well + swirl +
turbulence + damping), not an O(N²) all-pairs gravity — that would need
tiling / Barnes-Hut and is a natural next step.

## Repository layout

| File | Purpose |
|------|---------|
| `ParticleCloudTOP.cu` | The custom operator: CUDA kernels (init + integrate) and the TouchDesigner `CPlusPlus TOP` plugin glue. |
| `CMakeLists.txt` | Build configuration. Targets `sm_89` (RTX 40-series / Ada) with fast-math. |
| `build.bat` | One-click Windows build (selects CUDA 12.8 + the TD headers). |
| `build_particle_cloud_cuda.py` | Builds the whole TD network (CUDA TOP, GLSL material, geometry, camera, bloom) when run inside TouchDesigner. |
| `.github/workflows/compile.yml` | CI that compiles the operator on every push (drives the badge above). |

## Prerequisites (local build, Windows)

- **Visual Studio 2022** with the "Desktop development with C++" workload.
- **CUDA Toolkit** matching your TouchDesigner build's **major** version.
  TouchDesigner 2025.x bundles CUDA 12.x; this project builds against **12.8**.
  A plugin built with any CUDA 12.x links the major-versioned
  `cudart64_12.dll`, so it runs against TD's 12.8 **or** 12.9 runtime.
  See the [CUDA version table](https://docs.derivative.ca/CUDA).
- The **TouchDesigner C++ API headers** (`TOP_CPlusPlusBase.h`,
  `CPlusPlus_Common.h`, …). These ship inside your TouchDesigner install with
  the bundled **CudaTOP** sample, e.g.
  `…/TouchDesigner/Samples/CPlusPlus/CudaTOP/`. They are *not* committed here
  (they belong to Derivative); point the build at your local copy.

## Build

```bat
:: Edit build.bat if your CUDA / TD paths differ, then just:
build.bat
```

…or drive CMake directly (Ninja shown; a VS generator also works):

```bat
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" ^
  -DTD_INCLUDE="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build --config Release
```

Output: `build/Release/ParticleCloudTOP.dll` (VS) or
`build/ParticleCloudTOP.dll` (Ninja).

> **Note on the host compiler:** recent MSVC toolsets are newer than older
> CUDA toolkits officially list. The `-allow-unsupported-compiler` flag (already
> in `CMakeLists.txt`) relaxes that gate; CUDA 12.8 + VS 2022 compiles cleanly.

## Run in TouchDesigner

1. Build the DLL (above).
2. Open `build_particle_cloud_cuda.py` and set `DLL_PATH` to your built DLL.
3. In TouchDesigner, create a **Text DAT**, paste the script in, right-click →
   **Run Script**.
4. Open `/particle_cloud_cuda` and view the `out` TOP.

## Parameters

On the `cuda_particles` TOP (the simulation):

| Parameter | Meaning |
|-----------|---------|
| **Particle Count** | Number of particles (buffers reallocate on change). |
| **Cloud Radius** | Size of the initial gaussian blob. |
| **Speed** | Time-step multiplier for the integration. |
| **Gravity Well** | Central spring pulling particles toward the origin. |
| **Swirl** | Tangential force — orbital motion around the Y axis. |
| **Damping** | Velocity retained per frame (lower = more drag). |
| **Turbulence** | Strength of the sinusoidal flow field. |
| **Reset** | Pulse — re-seed positions and velocities. |

Render-side constants (sprite size, camera distance, colour, bloom, count)
live at the top of `build_particle_cloud_cuda.py`.

## Roadmap

- True GL point sprites for 1M+ particles (billboards are the robust default).
- Per-particle colour by speed / age via extra texture channels.
- All-pairs or Barnes-Hut gravity for real N-body behaviour.

## License

The source in this repository is provided as-is for use with TouchDesigner.
The TouchDesigner C++ API headers are © Derivative Inc. and are **not**
included here — obtain them from your TouchDesigner installation.
