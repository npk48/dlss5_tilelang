import React, { useEffect, useRef, useState } from 'react';
import { Aperture, Upload, SlidersHorizontal, Play, Download, Images, Columns2, SplitSquareHorizontal, Settings2, Wifi, X, AlertCircle, Check, Clock, LoaderCircle } from 'lucide-react';

type Config = typeof defaults;
const defaults = { width: 0, height: 0, nr_width: 0, nr_height: 0, structure: 2, tone: 1, mix: 1, passes: 1, style: 1, skin: -1, automatic_mask: false, nr: true, depth: false, flow: false, guide: false, reconstruction: false, flow_updates: 8, flow_longest_side: 512, depth_input_size: 518, temporal_strength: 1, intensity: 1, reset: true };
type Job = { id: string; status: string; error?: string; stream_id?: string; metrics?: Record<string, number>; result?: string; config?: Config; created_at?: string | number };
type Info = { assets_directory?: string; runtime_directory?: string; model_directory?: string; device?: string | number; gpu_target?: string; abi_version?: number; assets_present?: boolean; runtime_present?: boolean; available_models?: string[] };
type Source = { file: File; url: string; width: number; height: number };
const stages = ['queued', 'preparing', 'processing', 'completed'];
const message = (e: unknown) => e instanceof Error ? e.message : String(e);
function evenRound(x: number) { const n = Math.floor(x), f = x - n; return n + (f > .5 || (f === .5 && n % 2 !== 0) ? 1 : 0); }
function depthGrid(source: Source | null, size: number) {
  if (!source) return null;
  const { width: w, height: h } = source, ratio = Math.max(h, w) / Math.min(h, w);
  if (ratio > 1.78) size = Math.max(14, evenRound(Math.trunc(size * 1.777 / ratio) / 14) * 14);
  if (size < 14) return null;
  const scale = Math.max(size / h, size / w);
  const multiple = (x: number) => { const n = evenRound(x / 14) * 14; return n < size ? Math.ceil(x / 14) * 14 : n; };
  return `${multiple(h * scale)}x${multiple(w * scale)}`;
}
function Range({ label, value, min, max, step = .05, onChange }: { label: string; value: number; min: number; max: number; step?: number; onChange: (n: number) => void }) {
  return <label className="range"><span>{label}<output>{value.toFixed(step < 1 ? 2 : 0)}</output></span><input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(+e.target.value)} /><span className="ends"><small>{min}</small><small>{max}</small></span></label>;
}
export default function App() {
  const [config, setConfig] = useState<Config>({ ...defaults });
  const [preset, setPreset] = useState('nr');
  const [source, setSource] = useState<Source | null>(null);
  const sources = useRef(new Map<string, Source>());
  const urls = useRef(new Set<string>());
  const loadSequence = useRef(0);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [info, setInfo] = useState<Info | null>(null);
  const [token, setToken] = useState('');
  const [tokenDraft, setTokenDraft] = useState('');
  const [dialog, setDialog] = useState(false);
  const [connection, setConnection] = useState('Connecting');
  const [networkError, setNetworkError] = useState('');
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const [loadingSource, setLoadingSource] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState<{ id: string; url: string } | null>(null);
  const [resultError, setResultError] = useState('');
  const [compare, setCompare] = useState('side');
  const [wipe, setWipe] = useState(50);
  const [zoom, setZoom] = useState('fit');
  const fileInput = useRef<HTMLInputElement>(null);
  const job = jobs.find(j => j.id === selected);
  const original = selected ? sources.current.get(selected) : source;
  const resultUrl = result?.id === selected ? result.url : undefined;
  const grid = depthGrid(source, config.depth_input_size);
  const requiredModels = [...(config.depth ? ['vda_small_dynamic_init.onnx', 'vda_small_dynamic_step.onnx'] : []), ...(config.flow ? [`raft_small_u${config.flow_updates}.onnx`] : [])];
  const missing = info?.available_models ? requiredModels.filter(name => !info.available_models!.some(p => p.replaceAll('\\', '/').split('/').pop() === name)) : [];
  const invalidAxes = (w: number, h: number) => !((w === 0 && h === 0) || (Number.isInteger(w) && Number.isInteger(h) && w >= 32 && h >= 32 && w <= 16384 && h <= 16384));
  const validation = invalidAxes(config.nr_width, config.nr_height) || invalidAxes(config.width, config.height) ? 'Set both axes to 0 (automatic), or both to integers between 32 and 16384.' : (config.guide || config.reconstruction) && (!config.depth || !config.flow) ? 'Guide and reconstruction need both depth and flow estimators for image uploads.' : missing.length ? `Required model files not reported by server: ${missing.join(', ')}` : '';
  const update = <K extends keyof Config>(key: K, value: Config[K]) => setConfig(c => ({ ...c, [key]: value }));
  async function api(path: string, options: RequestInit = {}) {
    const headers = new Headers(options.headers); if (token) headers.set('Authorization', `Bearer ${token}`);
    const response = await fetch(path, { ...options, headers });
    if (!response.ok) { let detail = ''; try { detail = (await response.json()).error || ''; } catch { /* Non-JSON proxy errors use HTTP status. */ } throw new Error(response.status === 401 ? 'Authentication required. Open Connection and enter the server token.' : `${response.status} ${detail || response.statusText}`); }
    return response;
  }
  useEffect(() => {
    let stopped = false; let timer: ReturnType<typeof setTimeout>; const abort = new AbortController();
    async function poll() {
      try {
        const [i, list] = await Promise.all([api('/v1/info', { signal: abort.signal }).then(r => r.json()), api('/v1/jobs', { signal: abort.signal }).then(r => r.json())]);
        if (!Array.isArray(list.jobs)) throw new Error('Unexpected jobs response; server must support GET /v1/jobs.');
        if (!stopped) { setInfo(i); setJobs(list.jobs); setConnection('Connected'); setNetworkError(''); }
      } catch (e) { if (!stopped) { setConnection('Disconnected'); setNetworkError(message(e)); } }
      if (!stopped) timer = setTimeout(poll, 1800);
    }
    void poll(); return () => { stopped = true; abort.abort(); clearTimeout(timer); };
  }, [token, refresh]);
  useEffect(() => {
    const keep = new Set(jobs.map(j => j.id));
    for (const [id, value] of sources.current) if (!keep.has(id) && id !== selected) { sources.current.delete(id); if (value !== source && ![...sources.current.values()].includes(value)) { URL.revokeObjectURL(value.url); urls.current.delete(value.url); } }
  }, [jobs, selected, source]);
  useEffect(() => () => { loadSequence.current++; for (const url of urls.current) URL.revokeObjectURL(url); }, []);
  useEffect(() => {
    let active = true; let url: string | undefined; const abort = new AbortController(); setResult(null); setResultError('');
    if (selected && job?.status === 'completed') {
      // Never use an arbitrary server-provided URL: credentials stay on this origin.
      api(`/v1/jobs/${encodeURIComponent(selected)}/result.png`, { signal: abort.signal }).then(r => r.blob()).then(blob => {
        if (!active) return; url = URL.createObjectURL(blob); setResult({ id: selected, url });
      }).catch(e => { if (active) setResultError(message(e)); });
    }
    return () => { active = false; abort.abort(); if (url) URL.revokeObjectURL(url); };
  }, [selected, job?.status, token, refresh]);
  async function choose(file?: File) {
    if (!file || sending) return; const seq = ++loadSequence.current; setError(''); setLoadingSource(false);
    if (!/\.(png|jpe?g|bmp)$/i.test(file.name)) { setError('Choose a PNG, JPEG or BMP image.'); return; }
    if (file.size > 31 * 1024 * 1024) { setError('Image must be at most 31 MiB (server multipart limit: 32 MiB).'); return; }
    setLoadingSource(true); const url = URL.createObjectURL(file); urls.current.add(url);
    try {
      const img = new Image(); img.src = url; await img.decode();
      if (seq !== loadSequence.current) { URL.revokeObjectURL(url); urls.current.delete(url); return; }
      if (Math.min(img.naturalWidth, img.naturalHeight) < 32 || Math.max(img.naturalWidth, img.naturalHeight) > 16384) throw new Error('Image axes must be between 32 and 16384 pixels.');
      if (img.naturalWidth * img.naturalHeight > 16777216) throw new Error('The native image API accepts at most 16 megapixels.');
      if (source && ![...sources.current.values()].includes(source)) { URL.revokeObjectURL(source.url); urls.current.delete(source.url); }
      setSource({ file, url, width: img.naturalWidth, height: img.naturalHeight }); setSelected(null);
    } catch (e) { URL.revokeObjectURL(url); urls.current.delete(url); if (seq === loadSequence.current) setError(`Cannot open image: ${message(e)}`); }
    finally { if (seq === loadSequence.current) setLoadingSource(false); }
  }
  async function submit() {
    if (!source || validation || sending) return; setSending(true); setError('');
    const submitted = source, snapshot = { ...config }; const form = new FormData(); form.append('image', submitted.file); form.append('config', JSON.stringify(snapshot));
    try {
      const data = await (await api('/v1/jobs', { method: 'POST', body: form })).json();
      if (typeof data.id !== 'string') throw new Error('Server did not return a job ID.');
      sources.current.set(data.id, submitted); setSelected(data.id);
      // Retrieve actual status; do not infer a processing stage from an accepted upload.
      const detail: Job = await (await api(`/v1/jobs/${encodeURIComponent(data.id)}`)).json();
      setJobs(previous => [detail, ...previous.filter(j => j.id !== detail.id)]); setRefresh(n => n + 1);
    } catch (e) { setError(message(e)); setRefresh(n => n + 1); } finally { setSending(false); }
  }
  function downloadConfig() {
    if (!job?.config) return; const url = URL.createObjectURL(new Blob([JSON.stringify(job.config, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = `dlss5-${job.id}-config.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function numberField(key: keyof Config, label: string, min: number, max: number) {
    return <label>{label}<input type="number" min={min} max={max} step="1" value={Number(config[key])} onChange={e => { const value = Number(e.target.value); update(key, Math.max(min, Math.min(max, Math.round(value)))); }} /></label>;
  }
  return <div className="app">
    <header className="topbar"><div className="brand"><Aperture size={28} /><div>DLSS5 <span>STUDIO</span><small>Native image processing</small></div></div><div className="top-actions"><span className="chip">CUDA · {info?.gpu_target || 'Native'}</span><button className={`connection ${connection === 'Connected' ? 'online' : ''}`} onClick={() => { setTokenDraft(token); setDialog(true); }}><Wifi size={15} />{connection}<Settings2 size={14} /></button></div></header>
    <div className="layout"><aside className="sidebar"><form onSubmit={e => { e.preventDefault(); void submit(); }}>
      <section><div className="section-title"><span>01 / SOURCE</span><Upload size={15} /></div><button type="button" className={`dropzone ${dragging ? 'dragging' : ''}`} onClick={() => fileInput.current?.click()} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); void choose(e.dataTransfer.files[0]); }}>
        {source ? <><img src={source.url} alt="Source thumbnail"/><strong>{source.file.name}</strong><small>{source.width} × {source.height} · {(source.file.size / 1048576).toFixed(1)} MiB</small><span>Drop or click to replace</span></> : <><Upload size={25}/><strong>{loadingSource ? 'Opening image…' : 'Bring your image in'}</strong><span>Drop here or browse files</span><small>PNG, JPEG, BMP · up to 31 MiB</small></>}
      </button><input ref={fileInput} className="hidden" type="file" accept="image/png,image/jpeg,image/bmp,.bmp" onChange={e => { void choose(e.target.files?.[0]); e.target.value = ''; }}/></section>
      <section><div className="section-title"><span>02 / PIPELINE</span><SlidersHorizontal size={15}/></div><div className="presets">{[['nr', 'NR only'], ['full', 'Full pipeline'], ['custom', 'Custom']].map(([id, label]) => <button type="button" key={id} className={preset === id ? 'active' : ''} onClick={() => { setPreset(id); if (id !== 'custom') setConfig(c => ({ ...c, nr: true, depth: id === 'full', flow: id === 'full', guide: id === 'full', reconstruction: id === 'full' })); }}>{label}</button>)}</div><p className="hint">{preset === 'nr' ? 'Neural rendering without estimator models.' : preset === 'full' ? 'Depth → flow → guide → reconstruction → NR. Uses the same dynamic depth models for different image sizes.' : 'Choose individual stages in Advanced.'}</p></section>
      <section><div className="section-title"><span>03 / LOOK</span><Aperture size={15}/></div><Range label="Structure" value={config.structure} min={0} max={4} onChange={n => update('structure', n)}/><Range label="Tone" value={config.tone} min={0} max={2} onChange={n => update('tone', n)}/><Range label="Output mix" value={config.mix} min={0} max={1} onChange={n => update('mix', n)}/><p className="hint">Mix 0 keeps the processed base; 1 uses the full NR output.</p></section>
      <section><details><summary>Advanced <Settings2 size={15}/></summary><div className="advanced"><h4>Size & neural rendering</h4><div className="fields">{numberField('width', 'Output width', 0, 16384)}{numberField('height', 'Output height', 0, 16384)}{numberField('nr_width', 'NR work width', 0, 16384)}{numberField('nr_height', 'NR work height', 0, 16384)}{numberField('passes', 'Passes', 1, 30)}{numberField('style', 'Style byte', 0, 255)}</div><p className="hint">Both axes 0 = automatic. Explicit axes: 32–16384. Larger work sizes and more passes use more GPU memory.</p><Range label="Skin (−1 = default)" value={config.skin} min={-1} max={4} onChange={n => update('skin', n)}/><label className="toggle"><input type="checkbox" checked={config.automatic_mask} onChange={e => update('automatic_mask', e.target.checked)}/>Automatic mask</label><Range label="Temporal strength" value={config.temporal_strength} min={0} max={1} onChange={n => update('temporal_strength', n)}/><Range label="Intensity" value={config.intensity} min={0} max={1} onChange={n => update('intensity', n)}/><h4>Stages & estimators</h4>{(['nr', 'depth', 'flow', 'guide', 'reconstruction'] as const).map((key, i) => <label className="toggle" key={key}><input type="checkbox" checked={config[key]} onChange={e => { setPreset('custom'); update(key, e.target.checked); }}/>{['Neural rendering', 'VDA depth', 'RAFT optical flow', 'Guide validation', 'FSR2 reconstruction'][i]}</label>)}<div className="fields">{numberField('flow_updates', 'Flow updates', 1, 32)}{numberField('flow_longest_side', 'Flow longest side', 128, 4096)}{numberField('depth_input_size', 'VDA input size', 28, 2048)}</div><div className="notice"><strong>Dynamic VDA spatial dimensions</strong><p>One init/step model pair handles different image sizes and aspect ratios. Preprocessing aligns the internal grid to 14-pixel patches{grid ? `: ${grid} for this image` : ' (choose an image)'}. This is an internal processing size, not a required input resolution. No per-size model export is needed.</p>{requiredModels.map(m => <code key={m}>{m}</code>)}</div><p className="hint">Each upload is an independent still image (reset=true). Temporal controls do not preserve history between uploads.</p></div></details></section>
      <div className="submit-area">{validation && <p className="warning" role="alert">{validation}</p>}{error && <p className="warning" role="alert">{error}</p>}<button className="primary" type="submit" disabled={!source || sending || loadingSource || !!validation || connection !== 'Connected'}>{sending ? <LoaderCircle className="spin" size={17}/> : <Play size={17}/>} {sending ? 'Submitting image…' : 'Process image'}</button><small>Runs locally on your server GPU</small></div>
    </form></aside>
    <main><div className="workspace-heading"><div><p className="eyebrow">YOUR IMAGE, REIMAGINED</p><h1>Image workspace</h1></div><button onClick={() => { setSelected(null); setResultError(''); }}>Current source</button></div>
      {networkError && <div className="banner" role="alert"><AlertCircle size={18}/><div><strong>Server connection unavailable</strong><p>{networkError} Your local image and controls are preserved.</p></div><button onClick={() => setRefresh(n => n + 1)}>Retry</button></div>}
      <div className="viewer"><div className="viewer-toolbar"><div className="segmented"><button aria-label="Side-by-side comparison" className={compare === 'side' ? 'active' : ''} onClick={() => setCompare('side')}><Columns2 size={16}/>Side by side</button><button aria-label="Wipe comparison" className={compare === 'wipe' ? 'active' : ''} onClick={() => setCompare('wipe')} disabled={!original || !resultUrl}><SplitSquareHorizontal size={16}/>Wipe</button></div><label className="zoom">View<select aria-label="Image zoom" value={zoom} onChange={e => setZoom(e.target.value)}><option value="fit">Fit</option><option value="1">100%</option><option value="2">200%</option><option value=".5">50%</option></select></label></div>
      {!original && !selected ? <div className="empty hero"><div className="empty-icon"><Images size={40}/></div><h2>A clearer perspective starts here.</h2><p>Import an image, shape its look, and compare every detail.<br/>Your images are processed by your native server.</p><button onClick={() => fileInput.current?.click()}><Upload size={16}/> Choose an image</button><span className="muted">PNG / JPEG / BMP</span></div> : <>
        {compare === 'wipe' && original && resultUrl ? <div className="canvas-scroll"><div className={`wipe-canvas ${zoom === 'fit' ? 'fit' : ''}`} style={zoom === 'fit' ? { aspectRatio: `${original.width}/${original.height}` } : { width: original.width * Number(zoom), height: original.height * Number(zoom) }}><img src={original.url} alt="Original"/><img className="wipe-result" src={resultUrl} alt="Processed result" style={{ clipPath: `inset(0 0 0 ${wipe}%)` }}/><div className="wipe-line" style={{ left: `${wipe}%` }}/><span className="image-label left">Original</span><span className="image-label right">Result</span></div><label className="wipe-control">Comparison divider<input aria-label="Comparison divider" type="range" min="0" max="100" value={wipe} onChange={e => setWipe(+e.target.value)}/></label></div> : <div className="side-images"><div className="image-pane"><div className="pane-title">ORIGINAL <span>{original ? `${original.width} × ${original.height}` : 'Not in this browser'}</span></div><div className="image-scroll">{original ? <img src={original.url} alt="Original image" className={zoom === 'fit' ? 'fit-image' : ''} style={zoom !== 'fit' ? { width: original.width * Number(zoom), maxWidth: 'none' } : {}}/> : <div className="empty"><Images size={28}/><p>Original not in this browser</p><small>This job was submitted elsewhere or before this page opened.</small></div>}</div></div><div className="image-pane"><div className="pane-title">RESULT <span>{job?.metrics?.width ? `${job.metrics.width} × ${job.metrics.height}` : 'Native output'}</span></div><div className="image-scroll">{resultUrl ? <img src={resultUrl} alt="Processed result" className={zoom === 'fit' ? 'fit-image' : ''} style={zoom !== 'fit' ? { width: (job?.metrics?.width || original?.width || 512) * Number(zoom), maxWidth: 'none' } : {}}/> : <div className="empty">{job && ['preparing', 'processing'].includes(job.status) ? <LoaderCircle className="spin" size={28}/> : <Aperture size={28}/>}<h3>{resultError ? 'Result could not be loaded' : job?.status === 'failed' ? 'Processing failed' : job?.status === 'completed' ? 'Loading result…' : job ? job.status : selected ? 'Retrieving job…' : 'Ready when you are'}</h3><p>{resultError || job?.error || (job ? 'The server reports stages, not estimated percentages.' : 'Choose a look and press Process image.')}</p>{resultError && <button onClick={() => setRefresh(n => n + 1)}>Retry result</button>}</div>}</div></div></div>}
        <div className="viewer-footer"><div className="stages" aria-live="polite">{stages.map((s, i) => <span key={s} className={job?.status === s ? 'current' : job && stages.indexOf(job.status) > i ? 'done' : ''}>{job && stages.indexOf(job.status) > i ? <Check size={12}/> : <span className="stage-dot"/>}{s}</span>)}{job?.status === 'failed' && <span className="warning">Failed</span>}</div><span className="muted">{job?.metrics?.total_seconds != null ? `${job.metrics.total_seconds.toFixed(2)}s total` : 'No estimated progress'}</span></div>
      </>}
      </div>
      <div className="export-row"><div><strong>{job ? `Job ${job.id}` : 'Non-destructive comparison'}</strong><small>{job ? 'Downloads belong to the selected job, not current controls.' : 'The source stays untouched. Download the processed PNG when ready.'}</small></div><div className="actions"><button disabled={!job?.config} onClick={downloadConfig}><Download size={15}/>Config JSON</button>{resultUrl ? <a className="button primary compact" href={resultUrl} download={`dlss5-${selected}.png`}><Download size={15}/>Download PNG</a> : <button disabled><Download size={15}/>Download PNG</button>}</div></div>
      <section className="history"><div className="history-heading"><div><h2>Queue & history <span>{jobs.length}</span></h2><p>Latest server jobs · up to 8 retained · updates automatically</p></div><Clock size={18}/></div>{jobs.length ? <div className="job-list">{jobs.map(j => <button key={j.id} className={`job-row ${selected === j.id ? 'selected' : ''}`} onClick={() => { setSelected(j.id); setZoom('fit'); }}><span className={`job-icon ${j.status}`}>{j.status === 'completed' ? <Check size={18}/> : j.status === 'failed' ? <AlertCircle size={18}/> : j.status === 'queued' ? <Clock size={18}/> : <LoaderCircle size={18} className="spin"/>}</span><span className="job-name"><strong>{sources.current.get(j.id)?.file.name || `Job ${j.id}`}</strong><small>{j.id}{j.created_at ? ` · ${new Date(typeof j.created_at === 'number' ? j.created_at * 1000 : j.created_at).toLocaleString()}` : ''}</small></span><span className={`status ${j.status}`}>{j.status}</span><span className="job-time">{j.metrics?.total_seconds != null ? `${j.metrics.total_seconds.toFixed(2)}s` : '—'}</span></button>)}</div> : <div className="history-empty"><Clock size={21}/><p>{connection === 'Connected' ? 'No jobs yet. Your first image starts the story.' : 'Connect to the server to see its queue.'}</p></div>}{selected && !job && <p className="hint">Selected job is not in the retained server history. It may have expired.</p>}</section>
    </main></div>
    {dialog && <div className="modal-backdrop" onClick={() => setDialog(false)}><div className="modal" role="dialog" aria-modal="true" aria-labelledby="connection-title" onClick={e => e.stopPropagation()} onKeyDown={e => { if (e.key === 'Escape') setDialog(false); if (e.key === 'Tab') { const controls = [...e.currentTarget.querySelectorAll<HTMLElement>('button,input')]; const first = controls[0], last = controls.at(-1); if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); } } }}><div className="modal-title"><h2 id="connection-title">Server connection</h2><button aria-label="Close connection dialog" onClick={() => setDialog(false)}><X size={18}/></button></div><p>API requests use this origin. In development, Vite proxies them to the native server.</p><label>Bearer token <input autoFocus type="password" autoComplete="off" placeholder="Optional · only when server uses --token" value={tokenDraft} onChange={e => setTokenDraft(e.target.value)}/></label><p className="hint">Kept only in memory. Reloading clears it. Protected results are fetched with authorization, then displayed as local Blob URLs.</p><button className="primary" onClick={() => { setToken(tokenDraft.trim()); setDialog(false); setRefresh(n => n + 1); }}>Apply & reconnect</button><h4>Server environment</h4><dl>{(['device', 'gpu_target', 'abi_version', 'assets_directory', 'runtime_directory', 'model_directory', 'assets_present', 'runtime_present'] as const).map(key => <React.Fragment key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{info?.[key] === undefined ? 'Not reported' : String(info[key])}</dd></React.Fragment>)}</dl><details><summary>Available models ({info?.available_models?.length ?? 'unknown'})</summary>{info?.available_models?.map(m => <code key={m}>{m}</code>)}</details></div></div>}
  </div>;
}
