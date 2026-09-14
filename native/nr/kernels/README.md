# Generic native NR code assets

`sm89/` contains eight content-addressed `.nrbin` kernel packs. Each pack wraps a
CUBIN with source and payload SHA-256 checksums. These are **not spatial profiles**:
H/W, derived geometry and route/address buffers are runtime data. A single set
serves all supported grids. Model channels, layer roles and MMA tile shapes stay
compile-time constants.

The default lookup path is `<NATIVE_NR_TOOLCHAIN>/../nr/sm89`; deployed toolchain
`runtime/cuda12.8` therefore uses `runtime/nr/sm89`. `D5_NR_KERNELS` overrides it.
Missing matching source hashes use one generic NVRTC compilation per module per
Engine. Corrupt matching packs fail explicitly. Different source code never reuses
an old code pack. Installed packs were compiled with the qualified CUDA12.8 headers;
they do not depend on whatever headers a consuming machine might have installed.

## Rebuild (development only)

After changing NR CUDA code or its generated headers, build the native targets and
run the paired evaluator with `D5_NR_EXPORT_KERNELS` pointing to an empty staging
directory. Set `D5_NR_KERNELS` to that same initially empty directory. The first
prepare compiles all eight geometry-independent modules and writes their packs.
Subsequent never-seen grids must retain identical compile/module-load counters.

For example, from the repository in PowerShell:

```powershell
C:/Python311/python.exe native/tools/bench_nr_dynamic.py packets C:/tmp/nr-packets
$env:NATIVE_NR_TOOLCHAIN = "$PWD/dist/runtime/cuda12.8"
$env:D5_NR_KERNELS = 'C:/tmp/nr-kernels-new'
$env:D5_NR_EXPORT_KERNELS = $env:D5_NR_KERNELS
native/build/d5_nr_dynamic_bench.exe model C:/tmp/nr-packets C:/tmp/nr-results 50 100 320 384 384 320 384 512 448 576 512 640 320 384
```

Compare all output heads to the frozen reference, then replace `sm89/` with the
qualified eight packs. Use a fresh process without `D5_NR_EXPORT_KERNELS` to check
zero NVRTC calls and cold prepare time. `native/package.ps1` distributes the packs
outside the EXE. Runtime deployment does not execute Python or Node.js.
