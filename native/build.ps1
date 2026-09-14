param([string]$BuildDir = "$PSScriptRoot/build")
$ErrorActionPreference = 'Stop'
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'MSVC x64 tools not found' }
$vcvars = "$vs/VC/Auxiliary/Build/vcvars64.bat"
$cmd = "`"$vcvars`" && cmake -S `"$PSScriptRoot`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -UD5_EMBED_DEPENDENCIES && cmake --build `"$BuildDir`" --parallel 6"
& cmd /c $cmd
if ($LASTEXITCODE) { throw "Native build failed ($LASTEXITCODE)" }
