param([string]$BuildDir="$PSScriptRoot/build",[string]$Destination="$PSScriptRoot/../dist",[switch]$IncludeModels)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path "$PSScriptRoot/..").Path
$dest=[IO.Path]::GetFullPath($Destination)
$bin="$dest/native-sdk/bin";$lib="$dest/native-sdk/lib";$inc="$dest/native-sdk/include"
New-Item -ItemType Directory -Force $bin,$lib,$inc,"$dest/licenses","$dest/native-sdk/examples" | Out-Null
Copy-Item "$BuildDir/dlss5.dll" $bin -Force
Copy-Item "$BuildDir/dlss5.lib" $lib -Force
Copy-Item "$PSScriptRoot/include/dlss5.h" $inc -Force
Copy-Item "$PSScriptRoot/third_party/ort/runtime/*.dll" $bin -Force
Copy-Item "$PSScriptRoot/tests/consumer.cpp" "$dest/native-sdk/examples/consumer.cpp" -Force
Copy-Item "$BuildDir/d5_consumer.exe" $bin -Force
Copy-Item "$BuildDir/dlss5_server.exe" "$dest/dlss5_server.exe" -Force
Copy-Item "$repo/docs/NATIVE_SDK.md" "$dest/README.md" -Force
Copy-Item "$PSScriptRoot/licenses/*" "$dest/licenses" -Recurse -Force
Copy-Item "$PSScriptRoot/third_party/httplib.LICENSE","$PSScriptRoot/third_party/json.LICENSE","$PSScriptRoot/reconstruction/FSR2_LICENSE.txt" "$dest/licenses" -Force
Copy-Item "$PSScriptRoot/guides/VDA_LICENSE","$PSScriptRoot/guides/VDA_NOTICE","$PSScriptRoot/guides/TORCHVISION_LICENSE","$repo/model/WEIGHT_RIGHTS.md" "$dest/licenses" -Force
$toolchain="$repo/.toolchains/cuda12.8"
Get-ChildItem $toolchain -Recurse -File | Where-Object { $_.Length -gt 0 -and $_.Extension -notin '.py','.pyc' } | ForEach-Object {
 $relative=$_.FullName.Substring($toolchain.Length).TrimStart('\','/')
 if($relative -like 'runtime*' -or $relative -like 'cccl*' -or $relative -like 'nvrtc*'){
  $target=Join-Path "$dest/native-sdk/.toolchains/cuda12.8" $relative
  New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
  Copy-Item $_.FullName $target -Force
 }
}
if($IncludeModels){
 New-Item -ItemType Directory -Force "$dest/model/native_guides" | Out-Null
 Copy-Item "$repo/model/weights_ht_blob.bin" "$dest/model" -Force
 Copy-Item "$repo/model/native_guides/raft_small_u8.onnx","$repo/model/native_guides/vda_small_518x924_init.onnx","$repo/model/native_guides/vda_small_518x924_step.onnx","$repo/model/native_guides/native_guides.json" "$dest/model/native_guides" -Force
}
@'
SDK: set NATIVE_NR_TOOLCHAIN to native-sdk/.toolchains/cuda12.8 before first NR prepare.
Link lib/dlss5.lib, include include/dlss5.h, place bin runtime DLLs next to your EXE.
Server: dlss5_server.exe --model model --port 7863 (no SDK DLL sidecars required).
See README.md for GPU target, supported models, variants and callback contracts.
'@ | Set-Content "$dest/START.txt" -Encoding UTF8
Get-FileHash "$dest/dlss5_server.exe","$bin/dlss5.dll","$lib/dlss5.lib" -Algorithm SHA256 | Format-List | Out-File "$dest/SHA256.txt"
Write-Host "Native delivery: $dest"
