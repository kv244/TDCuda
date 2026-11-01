@echo off
REM ===========================================================================
REM Build the TouchDesigner custom operators for TD 2025.x (CUDA major 12):
REM   ParticleCloudTOP.dll   - CUDA particle simulation
REM   ParticleDensityTOP.dll - CUDA density splat (for the surface)
REM   ParticleSurfaceSOP.dll - isosurface mesh SOP
REM Uses CUDA 12.8 (matches TD's bundled 12.8 runtime). Double-click to build.
REM Result: build\Release\*.dll
REM
REM NOTE: Windows locks a loaded DLL. If TouchDesigner has these plugins loaded,
REM unload them first (or close TD), otherwise the link step can't overwrite them.
REM ===========================================================================

set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
REM Make the toolkit visible even if this shell started before 12.8 was installed.
set "CUDA_PATH=%CUDA_DIR%"
set "CUDA_PATH_V12_8=%CUDA_DIR%"
set "TD_SAMPLES=C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus"

cd /d C:\Users\julia\Downloads\TDClaude
rmdir /s /q build 2>nul

REM -T cuda=<full path> sets CudaToolkitDir directly (no reliance on env var).
REM CUDA flags relax the host-compiler gate (MSVC 14.44 is a touch newer than 12.8 lists).
cmake -B build -A x64 -T cuda="%CUDA_DIR%" ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" ^
  -DTD_INCLUDE="%TD_SAMPLES%/CudaTOP" ^
  -DTD_SOP_INCLUDE="%TD_SAMPLES%/SimpleShapesSOP"
if errorlevel 1 ( echo. & echo ===CONFIGURE_FAILED=== & exit /b 1 )

cmake --build build --config Release
if errorlevel 1 ( echo. & echo ===BUILD_FAILED=== & exit /b 1 )

echo. & echo ===BUILD_OK===
dir build\Release\*.dll
echo.
echo Next: run build_particle_cloud_cuda.py, then build_particle_surface.py,
echo       inside TouchDesigner (Text DAT ^> Run Script).
