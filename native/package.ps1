param([string]$BuildDir="$PSScriptRoot/build",[string]$Destination="$PSScriptRoot/../dist",[switch]$IncludeModels)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path "$PSScriptRoot/..").Path
$dest=[IO.Path]::GetFullPath($Destination)
if($dest -eq $repo -or $dest -eq [IO.Path]::GetPathRoot($dest)){throw 'Destination must be a separate distribution directory'}
$frontend="$PSScriptRoot/webui/dist"
if(!(Test-Path "$frontend/index.html")){throw 'Missing web assets: build native/webui with npm ci && npm run build first'}
$runtime="$dest/runtime"
$bin="$dest/native-sdk/bin";$lib="$dest/native-sdk/lib";$inc="$dest/native-sdk/include"
New-Item -ItemType Directory -Force $bin,$lib,$inc,"$dest/licenses","$dest/native-sdk/examples","$runtime/bin" | Out-Null
Copy-Item "$BuildDir/dlss5.dll" $bin -Force
Copy-Item "$BuildDir/dlss5.lib" $lib -Force
Copy-Item "$PSScriptRoot/include/dlss5.h" $inc -Force
# One shared runtime, not a second copy hidden inside either the EXE or SDK bin.
Get-ChildItem "$PSScriptRoot/third_party/ort/runtime/*.dll" | ForEach-Object {
 $target=Join-Path "$runtime/bin" $_.Name
 if(!(Test-Path $target) -or (Get-Item $target).Length -ne $_.Length -or (Get-Item $target).LastWriteTimeUtc -ne $_.LastWriteTimeUtc){Copy-Item $_.FullName $target -Force}
 $legacy=Join-Path $bin $_.Name
 if(Test-Path $legacy){Remove-Item $legacy -Force}
}
Copy-Item "$PSScriptRoot/tests/consumer.cpp" "$dest/native-sdk/examples/consumer.cpp" -Force
Copy-Item "$BuildDir/d5_consumer.exe" $bin -Force
Copy-Item "$BuildDir/dlss5_server.exe" "$dest/dlss5_server.exe" -Force
# Replace generated frontend output so superseded hashed bundles do not accumulate.
if(Test-Path "$dest/assets"){Remove-Item "$dest/assets" -Recurse -Force}
New-Item -ItemType Directory -Force "$dest/assets" | Out-Null
Copy-Item "$frontend/*" "$dest/assets" -Recurse -Force
Copy-Item "$repo/docs/NATIVE_SDK.md" "$dest/README.md" -Force
Copy-Item "$PSScriptRoot/licenses/*" "$dest/licenses" -Recurse -Force
Copy-Item "$PSScriptRoot/third_party/httplib.LICENSE","$PSScriptRoot/third_party/json.LICENSE","$PSScriptRoot/reconstruction/FSR2_LICENSE.txt" "$dest/licenses" -Force
Copy-Item "$PSScriptRoot/guides/VDA_LICENSE","$PSScriptRoot/guides/VDA_NOTICE","$PSScriptRoot/guides/TORCHVISION_LICENSE","$repo/model/WEIGHT_RIGHTS.md" "$dest/licenses" -Force
$toolchain="$repo/.toolchains/cuda12.8"
Get-ChildItem $toolchain -Recurse -File | Where-Object { $_.Length -gt 0 -and $_.Extension -notin '.py','.pyc' } | ForEach-Object {
 $relative=$_.FullName.Substring($toolchain.Length).TrimStart('\','/')
 if($relative -like 'runtime*' -or $relative -like 'cccl*' -or $relative -like 'nvrtc*'){
  $target=Join-Path "$runtime/cuda12.8" $relative
  New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
  if(!(Test-Path $target) -or (Get-Item $target).Length -ne $_.Length -or (Get-Item $target).LastWriteTimeUtc -ne $_.LastWriteTimeUtc){Copy-Item $_.FullName $target -Force}
 }
}
if(Test-Path "$dest/native-sdk/.toolchains"){Remove-Item "$dest/native-sdk/.toolchains" -Recurse -Force}
if($IncludeModels){
 New-Item -ItemType Directory -Force "$dest/model/native_guides" | Out-Null
 Copy-Item "$repo/model/weights_ht_blob.bin" "$dest/model" -Force
 Copy-Item "$repo/model/native_guides/raft_small_u8.onnx","$repo/model/native_guides/vda_small_518x924_init.onnx","$repo/model/native_guides/vda_small_518x924_step.onnx","$repo/model/native_guides/native_guides.json" "$dest/model/native_guides" -Force
}
@'
Server: dlss5_server.exe --port 7863
Keep runtime/, model/ and assets/ next to the EXE. No dependency extraction/cache.
Optional overrides: --runtime PATH --model PATH --assets PATH.
SDK: register runtime/bin with AddDllDirectory before estimator use;
set NATIVE_NR_TOOLCHAIN to runtime/cuda12.8 before NR preparation.
Link native-sdk/lib/dlss5.lib and include native-sdk/include/dlss5.h.
The packaged consumer detects this shared runtime layout automatically.
See README.md for GPU target, exported VDA grids and callback contracts.
'@ | Set-Content "$dest/START.txt" -Encoding UTF8
Get-FileHash "$dest/dlss5_server.exe","$bin/dlss5.dll","$lib/dlss5.lib" -Algorithm SHA256 | Format-List | Out-File "$dest/SHA256.txt"
$commit=(& git -C $repo rev-parse HEAD).Trim()
@{commit=$commit;layout='external-assets-v2';server='dlss5_server.exe';runtime='runtime';models='model';web_assets='assets';dependency_extraction=$false;python_runtime=$false} | ConvertTo-Json | Set-Content "$dest/BUILD.json" -Encoding UTF8
Write-Host "Native external-asset delivery: $dest"
