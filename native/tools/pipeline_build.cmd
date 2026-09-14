@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cmake -S "%~dp0..\pipeline" -B "%~dp0..\pipeline\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=89 -DD5_PIPELINE_TEST=ON
if errorlevel 1 exit /b %errorlevel%
cmake --build "%~dp0..\pipeline\build"
