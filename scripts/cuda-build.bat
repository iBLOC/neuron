@echo off
REM Run a command inside the MSVC x64 developer environment.
REM
REM The candle CUDA backend compiles .cu kernels with nvcc, and nvcc needs
REM the MSVC toolchain (cl.exe + INCLUDE/LIB) to be on the environment.
REM Git Bash has neither, so `cargo build --features cuda` fails with
REM "nvcc fatal : Cannot find compiler 'cl.exe' in PATH" unless this
REM wrapper is used.
REM
REM Usage: scripts\cuda-build.bat <command> [args...]
REM   scripts\cuda-build.bat cargo build --lib --features cuda
REM   scripts\cuda-build.bat cargo bench --bench model_bench --features cuda
REM
REM Override the Visual Studio location with NEURON_VS_PATH if the
REM auto-detection below picks the wrong install.

setlocal

set "VSPATH=%NEURON_VS_PATH%"
if not defined VSPATH set "VSPATH=C:\Program Files\Microsoft Visual Studio\18\Enterprise"

if not exist "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" (
  echo [cuda-build] vcvars64.bat not found under "%VSPATH%".
  echo [cuda-build] set NEURON_VS_PATH to the Visual Studio install root.
  exit /b 1
)

call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [cuda-build] vcvars64.bat failed.
  exit /b 1
)

%*
exit /b %ERRORLEVEL%
