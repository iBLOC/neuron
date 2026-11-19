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

REM ---------------------------------------------------------------------------
REM NOTE: keep this file ASCII-only. A .bat is parsed as GBK on a zh-CN
REM Windows, and non-ASCII bytes in a REM block shift the parser: the
REM lines get executed as commands and the MSVC env never reaches nvcc
REM (symptom: "nvcc fatal : Cannot find compiler 'cl.exe' in PATH").
REM
REM 2026-11-19: two upstream kernel bugs break under CUDA 13.x, so we
REM inject one extra flag for each. nvcc reads NVCC_PREPEND_FLAGS itself,
REM so it applies to every nvcc invocation.
REM   1) "-Xcompiler /Zc:preprocessor"
REM      candle-kernels' build.rs (rev 66a8cf1) is missing it. CUDA 13's
REM      CCCL headers hard-error (#error C1189) under MSVC's traditional
REM      preprocessor, so nvcc dies and the error text is EMPTY.
REM      candle's other rev (aebc405d) already has these two lines.
REM   2) "-include string"
REM      mistral.rs' own kernels (mistralrs-quant/kernels/marlin/**) call
REM      std::to_string without including <string>; CUDA 13.x no longer
REM      includes it transitively.
REM Both are upstream bugs; this is the local workaround.
if not defined NVCC_PREPEND_FLAGS (
  set "NVCC_PREPEND_FLAGS=-include string -Xcompiler /Zc:preprocessor"
)

%*
exit /b %ERRORLEVEL%
