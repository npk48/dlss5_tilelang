# DLSS5 Native Studio

React + TypeScript + Vite frontend for the native server. All data comes from the real server; there is no mock backend, progress simulation, cancellation endpoint, or runtime Node/npm requirement.

## Develop

Requires Node.js 20.19+ or 22.12+ (validated with Node 24).

```powershell
cd native/webui
npm ci
npm run dev
```

Open the Vite URL printed in the terminal. `/v1` and `/api` proxy to `http://127.0.0.1:7863`, with `changeOrigin: false` to preserve the server's Origin/Host check. To use another server:

```powershell
$env:VITE_API_PROXY_TARGET = 'http://127.0.0.1:17864'
npm run dev
```

Use the Connection button to enter an optional `--token` bearer secret. It stays only in React memory, never in a URL or persistent storage. Reloading clears it. Every API request, including result image retrieval, carries authorization when set. Images are displayed/downloaded through Blob URLs, not unprotected `<img>` API requests.

## Build and serve

```powershell
npm run typecheck
npm run build
# From repository root, use the actual path to your native server executable:
# dlss5_server.exe --assets native/webui/dist --model <model-directory>
```

`dist/index.html` and `dist/assets/*` are real production assets. Production base is `/`; the server mounts `native/webui/dist` at `/`. Packaging copies this folder's contents to the package's root `dist/assets` folder. Serve the generated folder, not the source index.html. No npm process is needed after building. This frontend does not modify backend or native build configuration.

## API contract

- `GET /v1/info`: server environment, assets/runtime presence, and available model filenames.
- `GET /v1/jobs`: `{jobs: [...]}`; newest-first server history, up to eight retained jobs.
- `POST /v1/jobs`: multipart `image` file and `config` JSON text; returns `{id, ...}`.
- `GET /v1/jobs/{id}`: actual submitted job status/details.
- `GET /v1/jobs/{id}/result.png`: authorized PNG retrieval.

Job objects contain `id`, `status`, `error`, `stream_id`, `metrics`, `result`, `config`, and `created_at` (Unix seconds). Status polling runs serially every 1.8 seconds. Stages are queued/preparing/processing/completed/failed; upload submission is shown separately and no percentage is fabricated. Config download uses the selected job's server-returned config, never the currently edited controls. Poll failures retain the workspace and show a retryable connection error.

## Controls and image ownership

- NR only is the initial preset. Full enables NR, VDA depth, RAFT flow, guide validation, and FSR2 reconstruction; Custom allows independent stage selection. Guide/reconstruction require depth and flow for still-image uploads.
- Structure 0–4, tone 0–2 and skin −1–4 are deliberate UI adjustment ranges, not claims about SDK hard limits. Style is 0–255, passes 1–30, mix/intensity/temporal strength 0–1. Every upload sends `reset: true`; these are independent stills, not a temporal stream editor.
- Output and NR dimensions are both zero for automatic sizing, or both 32–16384. VDA dimensions mirror native `depth_hw` (ties-to-even rounding, 14-pixel patch alignment and wide-aspect adjustment). The UI shows the exact required `vda_small_HxW_init.onnx` and `_step.onnx` filenames and checks reported model availability. A matching `raft_small_uN.onnx` is needed for the chosen update count. Models are not downloaded or substituted by the UI.
- PNG/JPEG/BMP up to 31 MiB leave room under the server's 32 MiB multipart limit. Input dimensions must be 32–16384 per axis. The server remains the final validator for GPU memory and runtime/model compatibility.
- Sources are associated with IDs only for this page's own submissions. Historical/server-only jobs explicitly show “Original not in this browser”; they never borrow a newer source. Object URLs are revoked on replacement, result switching, retention cleanup, or unmount. A selected expired job may retain its known source until the selection changes.
- Side-by-side comparison supports fit and pixel zoom. Wipe maps the result to the original's comparison canvas, including when output dimensions differ. The labeled range slider works with keyboard as well as pointer input.

## Real-server smoke checklist

1. Open with an empty queue; confirm actual connection info. With `--token`, confirm public static UI, unauthorized API error, and successful token connection.
2. Upload a PNG/JPEG/BMP, verify dimensions and thumbnail, process NR-only, observe actual stages, and download PNG/config.
3. Switch completed jobs and current source; ensure each job keeps its own original. Reload and verify old jobs show result without an unrelated original.
4. Try wipe/fit/100%/200%, a narrow viewport, invalid files, missing VDA models, and stopped/restarted server.
5. Check browser console/network for errors. This checklist requires the native server and GPU; frontend compilation alone does not validate inference.

## Real-browser native smoke test

Run the packaged native server on port 17864 with `--token studio-smoke-token`,
with default VDA518x924 and RAFT u8 assets available. Then:

```powershell
npx playwright install chromium
npm run test:e2e
# Or use installed Edge: $env:PLAYWRIGHT_CHANNEL="msedge"
```

Override `D5_TEST_URL`, `D5_TEST_TOKEN` and `D5_TEST_OUTPUT` as needed. This test
uses real GPU inference (no mocks), checks authentication, source/result pairing,
PNG/config downloads, wipe/zoom and mobile layout, and saves screenshots. Build
output (`dist/`) and test artifacts are generated and ignored by Git.
