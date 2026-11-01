# Surface mode — a live isosurface from the particle cloud

Turns the CUDA particle cloud into a **watertight, smooth-shaded mesh** (a
glowing liquid / metaball surface) that moves with the simulation. It uses two
new operators alongside the existing `ParticleCloudTOP` ("Option B": the
particle work stays on the GPU, only a small grid comes back to the CPU).

```
ParticleCloudTOP   (CUDA)  positions, RGBA32F; alpha 0 marks padding texels
      │ wired input
ParticleDensityTOP (CUDA)  splat → blur → normalise, on the GPU
      │                    output: res³ density as a 2D slice atlas (R32F)
      │ TOP to CHOP        Download Type = next frame (no CPU/GPU stall)
ParticleSurfaceSOP (C++)   marching tetrahedra → welded mesh + normals
      │
Phong MAT → render1        drawn alongside (or instead of) the sprites
```

## Why it's built this way

- **The splat happens on the GPU, where the particles already live.** Each live
  particle deposits weight 1 trilinearly into the 8 surrounding grid points
  (`atomicAdd`). A separable box blur then turns that into a continuous field.
  Blur cost is `res³ · (2r+1) · 3` and does not depend on particle count.
- **Only the grid crosses to the CPU.** At 64³ that's 1 MB per frame whether you
  run 200k or 2M particles. Moving the raw particles back instead would cost
  3–17 MB per frame plus a per-particle CPU loop.
- **Padding is masked by alpha.** The position TOP writes `(0,0,0,0)` into
  texels past Count. The density TOP skips alpha < 0.5 and any non-finite
  position, so it doesn't depend on texel order or a second Count parameter.
- **The box is fixed and centred on the origin.** Particles that escape it (for
  example under negative gravity) are simply dropped; they can't stretch the
  grid, and the sampling lattice never shifts, so the surface doesn't shimmer.
  If the cloud does spill past the box, the surface is capped flat one cell
  inside it, so the mesh stays closed.
- **Density is normalised.** It's scaled so that 1.0 equals the mean density in
  the box. Isolevel 4 means "4× average", and that stays meaningful when you
  change the particle count.
- **The mesher is marching tetrahedra over the Kuhn triangulation.** Each grid
  cube splits into 6 tets the same way everywhere, which makes the result
  watertight by construction. Each tet uses Bourke's `PolygoniseTri` cases,
  with no 256-entry marching-cubes table to get wrong.
- **Vertices are welded through a rolling two-slab edge cache.** Every tet edge
  runs from a grid point in one of 7 positive directions, so `(point,
  direction)` is a unique vertex key. That costs about 6 MB even at 192³, with
  no hashing and no per-frame clear.
- **Empty space is skipped in 8³ bricks.** The cloud fills a small part of the
  box, so classifying bricks by min/max first removes most of the scan.
- **Normals come from the density gradient,** and every triangle is wound
  counter-clockwise around its outward normal.

## Controls (the component's **Surface** page)

| Parameter | Meaning |
|-----------|---------|
| **Show Surface / Show Sprites** | Toggle the mesh and the glowing sprites independently. |
| **Grid Resolution** | Grid points per axis. Cost grows cubically: 64 → ~2 ms, 96 → ~5 ms, 128 → ~12 ms of CPU per frame (measured, see below). |
| **Bounds Size** | Edge length of the origin-centred cube the grid covers. Particles outside it are ignored. |
| **Smoothing** | Blur radius in world units. Larger means rounder, more merged blobs. |
| **Isolevel** | Surface threshold as a multiple of the mean density. Lower gives a fatter blob; higher shrink-wraps the dense core. |
| **Surface Color** | Diffuse colour, plus a little emission so the surface reads with the bloom. |
| **Flip Normals** | Reverses normals and winding together, to render inside-out. |
| **Flip Atlas Rows** | Only needed if TOP to CHOP turns out to list texture rows top-first. Symptom: a scrambled or mirrored surface. |

Grid Resolution and Bounds Size are bound to both the density TOP and the SOP.
They must match, because both derive the atlas layout from them
(`DensityAtlas.h`). If they ever differ, the SOP shows a warning instead of
meshing garbage.

## Setup

1. Build: append `CMakeLists_optionB.txt` to `CMakeLists.txt`, then
   `cmake --build build --config Release`. This produces
   `ParticleDensityTOP.dll` and `ParticleSurfaceSOP.dll`.
2. In TouchDesigner, run `build_particle_cloud_cuda.py` (the base cloud) and then
   `build_particle_surface.py` (this add-on). Both are paste-into-a-Text-DAT,
   Run Script.
3. Set Isolevel by eye. Around 4 is a good start for the default cloud.

## Verification

**Live on Windows** (TouchDesigner 2025.33230, RTX 4070, CUDA 12.8 + MSVC):

- All three DLLs build against the installed 2025.x headers. They also compile
  against the public CustomOperatorSamples headers that CI uses.
- The TouchDesigner API usage was checked line by line against the bundled
  CudaTOP and SimpleShapesSOP samples. Checking turned up three things the
  final code now handles:
  - The input `cudaArray` has to be acquired **before**
    `beginCUDAOperations()`.
  - `executeVBO()` is pure virtual, so the SOP must implement it.
  - Winding must be set to `SOP_Winding::CCW`, otherwise the faces come out
    inverted.
- **The live surface exactly matches the CPU reference.** On the TOP's own
  seeding at 64³ / Isolevel 4, the GPU density field peaks at the same
  368.6 × mean, and the SOP produces the same **12,024 vertices / 24,044
  triangles** as the sandbox tests, to the vertex.
- The atlas row order was verified bit-for-bit against the texture (bottom row
  first), so *Flip Atlas Rows* stays off.
- Sustained **60 fps** with sprites and surface both live at 64³. Measured
  cook times per frame:

  | Stage | Where | Time |
  |-------|-------|------|
  | Particle sim | GPU | 0.7 ms |
  | Density splat + blur | GPU | 1.0 ms |
  | TOP to CHOP download | CPU | 5.5 ms |
  | Surface SOP | CPU | 10 ms |

  The mesher itself takes about 2 ms of that SOP time; the rest is TD
  ingesting the geometry. Those two CPU costs are the next optimisation
  targets.

**TOP to CHOP settings that matter** (`build_particle_surface.py` sets them):

- **Single Set: on.** Otherwise TOP to CHOP emits one channel *per image row*,
  and the SOP rejects that layout with a warning.
- **Output Color Space: passthrough.** The default sRGB would apply a transfer
  curve to the density *data* and skew every isolevel.
- **Channels:** `r` is named `density`, and `g/b/a` are left blank so they're
  omitted. These parameters are channel *names*, not on/off toggles.

**Sandbox** (Linux, no GPU; `check/test_surface.cpp`, run in CI by
`surface-tests.yml`): 86/86 checks, also under ASan + UBSan.

- **Analytic sphere:** vertices within 0.03–0.10 of a cell of the exact
  radius, normals radial to ≥ 0.9998, volume within 0.1%, closed, Euler 2.
- **Multi-blob and box-clipped bodies:** closed.
- **Hostile input:** garbage padding and NaN/inf positions are masked, and the
  splat conserves mass.
- **Atlas:** round trip checked at 16…192.
- **Edge cache:** identical output when re-used across cooks.

**Gravity note:** with the default Gravity Well (+0.8) the whole cloud falls
into the core within a couple of seconds of a Reset. The surface then
correctly shrinks to a small, blocky core a few cells wide. For a full-size
surface, lower Gravity (~0.1–0.2) so swirl and turbulence keep the cloud
spread out, or use Auto Reset.

## Limits and next steps

- The surface lags the sprites by one frame because of the next-frame download.
  That's the price of not stalling the GPU.
- Main-thread cost at 64³ is about 15 ms per frame (CHOP conversion plus TD's
  geometry ingest). Two next steps:
  - Read the density TOP directly in the SOP (`OP_TOPInput::downloadTexture`,
    delayed) and drop the CHOP.
  - Write the VBO directly (`directToGPU`).
- Grid atomics contend in the dense core at very high counts. If profiling ever
  shows it, a shared-memory pre-accumulation in the splat would fix it.
- Above about 128³ the CPU marching dominates. The next step would be marching
  on the GPU and writing the SOP's VBO directly (`directToGPU`).
