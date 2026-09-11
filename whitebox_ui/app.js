'use strict';
const $ = id => document.getElementById(id);
let active = null, lastPreview = '', busy = false, fileOffset = 0, refreshRequested = false;
const number = id => Number($(id).value);
const measured = value => value == null ? '未测' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const seconds = value => typeof value === 'number' ? `${value.toFixed(2)} s` : '未测';
const stages = {queued:'排队', loading_engine:'加载本地Engine', loading_model:'加载本地模型',
  preparing_pipeline:'准备管线 / shape（首次可能编译）', preparing_nr:'准备NR shape / 编译',
  compiling_nr:'编译NR', processing:'逐帧处理', completed:'完成', cancelled:'已取消', failed:'失败'};
function manifest() {
  const files = [...$('files').files].sort((a,b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  if (!files.length) throw Error('请选择输入文件');
  const kind = $('kind').value, enc = $('encoding').value;
  const passes = number('passes');
  if (!Number.isInteger(passes) || passes < 1 || passes > 30) throw Error('NR pass数量需为1–30');
  const s = {mode:'rgb_estimated', color_encoding:enc, output_size:[number('height'),number('width')],
    nr_passes:Array.from({length:passes}, () => ({structure:number('structure')})),
    output_settings:{mix:number('mix'), nr_resample:$('resample').value},
    fsr_settings:{camera_fov_y:number('fov')*Math.PI/180}};
  if ($('workH').value || $('workW').value) {
    if (!$('workH').value || !$('workW').value) throw Error('NR工作尺寸需同时填写宽高');
    s.nr_work_size = [number('workH'),number('workW')];
  }
  if (kind === 'video') {
    if (files.length !== 1) throw Error('视频任务请选择一个文件');
    s.video = {file:files[0].name};
    if ($('transfer').value) s.video.transfer = $('transfer').value;
    if ($('matrix').value) s.video.matrix = $('matrix').value;
    if ($('limit').value) s.video.max_frames = number('limit');
  } else s.frames = files.map((f,i) => ({color:f.name, reset:kind === 'images' || i === 0}));
  if (enc === 'PQ2020' || enc === 'linear709_nits') s.color_settings = {reference_white_nits:203, peak_nits:4000, output_encoding:enc};
  if ($('mp4').checked) s.video_output = {file:'preview.mp4', ...(kind === 'video' ? {} : {fps:24})};
  return s;
}
$('generate').onclick = () => {
  try { $('spec').value = JSON.stringify(manifest(), null, 2); $('error').textContent = ''; }
  catch(e) { $('error').textContent = e.message; }
};
$('transfer').onchange = () => {
  if ($('transfer').value === 'PQ') { $('encoding').value = 'PQ2020'; $('matrix').value = 'bt2020'; }
};
$('files').onchange = () => {
  const file = $('files').files[0]; if (!file) return;
  const url = URL.createObjectURL(file), media = document.createElement(file.type.startsWith('video/') ? 'video' : 'img');
  const done = () => {
    $('width').value = media.videoWidth || media.naturalWidth || 960;
    $('height').value = media.videoHeight || media.naturalHeight || 540;
    URL.revokeObjectURL(url);
  };
  media.onload = done; media.onloadedmetadata = done; media.onerror = () => URL.revokeObjectURL(url); media.src = url;
};
async function api(url, options) {
  const response = await fetch(url, options), value = await response.json();
  if (!response.ok) throw Error(value.error || response.statusText);
  return value;
}
function choose(id) { active = id; lastPreview = ''; fileOffset = 0; return poll(); }
$('start').onclick = async () => {
  try {
    $('start').disabled = true;
    const spec = $('advanced').checked ? JSON.parse($('spec').value) : manifest();
    $('spec').value = JSON.stringify(spec, null, 2);
    const data = new FormData();
    data.append('manifest', JSON.stringify(spec)); data.append('nr_backend', $('nrBackend').value);
    for (const file of $('files').files) data.append('files', file, file.name);
    const job = await api('/api/jobs', {method:'POST', body:data});
    $('error').textContent = ''; await choose(job.id);
  } catch(e) { $('error').textContent = e.message; }
  finally { $('start').disabled = false; }
};
for (const action of ['cancel','reset']) $(action).onclick = async () => {
  try { await api(`/api/jobs/${active}/${action}`, {method:'POST'}); await poll(); }
  catch(e) { $('error').textContent = e.message; }
};
function fileUrl(jid, name) { return `/api/jobs/${jid}/files/${encodeURIComponent(name)}`; }
function link(jid, label, name, target = 'downloads') {
  const anchor = document.createElement('a'); anchor.textContent = label;
  anchor.href = fileUrl(jid, name) + '?download=1'; $(target).append(anchor);
}
async function outputs(jid) {
  const offset = fileOffset, result = await api(`/api/jobs/${jid}/outputs?offset=${offset}`);
  if (active !== jid || fileOffset !== offset) return;
  $('outputFiles').replaceChildren();
  for (const name of result.files) link(jid, name, name, 'outputFiles');
  $('filePage').textContent = `文件 ${result.total ? offset+1 : 0}–${offset+result.files.length} / ${result.total}`;
  $('prevFiles').disabled = offset === 0; $('nextFiles').disabled = offset+100 >= result.total;
}
$('outputList').ontoggle = () => { if ($('outputList').open) poll(); };
$('prevFiles').onclick = () => { fileOffset = Math.max(0,fileOffset-100); poll(); };
$('nextFiles').onclick = () => { fileOffset += 100; poll(); };
function render(job) {
  const terminal = ['completed','failed','cancelled'].includes(job.status), native = job.native_nr || {};
  $('jobTitle').textContent = `任务 ${job.id.slice(0,8)} · ${job.status}`;
  $('stage').textContent = (stages[job.stage] || job.stage) + (job.message ? ` · ${job.message}` : '')
    + (job.cancel_requested && !terminal ? ' · 等待当前GPU安全边界取消，保留完成帧' : '')
    + (job.reset_requested ? ' · 下一帧将reset' : '');
  $('counts').textContent = `已完成 ${job.processed_frames} 帧` + (job.total_frames_estimate ? ` / 预计 ${job.total_frames_estimate}` : '')
    + (job.error ? ' · '+job.error : '');
  $('metrics').textContent = [
    `所选：${job.selected_nr} · 外围：${job.selected_backend}`,
    `实际NR：${measured(native.actual_backend)} · NR selected：${measured(native.selected_backend)}`,
    `实际neural尺寸（高×宽）：${measured(job.actual_neural_size)} · 新NR实际调用：${measured(job.new_nr_calls)}`,
    `NR shapes：${measured(native.shapes)} · fallback调用：${measured(native.fallback_calls)}`,
    `完整处理（不含排队，含本次加载/准备/写出）：${seconds(job.full_processing_seconds)}` + (!terminal ? ` · 已用 ${seconds(job.elapsed_seconds)}` : ''),
    `管线报告秒数：${seconds(job.seconds)} · shape准备：${seconds(native.prepare_seconds)} · 编译：${seconds(native.compile_seconds)}`,
    `最近帧输入→输出（含本帧写出，不含解码/排队）：${seconds(job.last_frame && job.last_frame.input_to_output_seconds)}`,
    `最近帧CPU分段（秒；调用含GPU等待）：${measured(job.last_frame && job.last_frame.host_stage_seconds)}`,
    `额外阶段时长（后端原字段）：${measured(job.stage_seconds)}`,
  ].join('\n');
  $('cancel').disabled = terminal; $('reset').disabled = terminal;
  const total = job.total_frames_estimate || 0; $('progress').max = total || 1;
  if (total) $('progress').value = job.processed_frames;
  else if (terminal) $('progress').value = 1;
  else $('progress').removeAttribute('value');
  $('details').textContent = JSON.stringify({last_frame:job.last_frame, backend_stats:job.backend_stats, last_reset_frame:job.last_reset_frame}, null, 2);
  $('downloads').replaceChildren();
  if (job.last_frame && job.last_frame.file) {
    const key = job.id + '/' + job.last_frame.file;
    if (key !== lastPreview) { $('preview').src = fileUrl(job.id, job.last_frame.file); $('preview').style.display = 'block'; lastPreview = key; }
    link(job.id,'PNG预览',job.last_frame.file);
    if (job.last_frame.data_file) link(job.id,'浮点真输出',job.last_frame.data_file);
    link(job.id,'完整帧记录','frames.jsonl'); link(job.id,'报告','report.json'); link(job.id,'有效manifest','run-manifest.json');
  } else { $('preview').style.display = 'none'; lastPreview = ''; }
  if (job.backend_stats) link(job.id,'后端记录','backend.json');
  if (job.status === 'completed' && job.video_output_file) link(job.id,'SDR MP4预览',job.video_output_file);
}
async function poll() {
  if (busy) { refreshRequested = true; return; }
  busy = true;
  try {
    const status = await api('/api/status');
    $('model').textContent = `Engine：${status.model_state} · ${status.device}`
      + (status.load_seconds != null ? ` · 加载 ${seconds(status.load_seconds)}` : '') + (status.model_error ? ' · '+status.model_error : '');
    $('jobs').replaceChildren();
    for (const job of [...status.jobs].reverse()) {
      const element = document.createElement('button'); element.className = 'job';
      element.textContent = `${job.id.slice(0,8)} · ${job.selected_nr} · ${job.status} · ${job.processed_frames}帧`;
      element.onclick = () => choose(job.id); $('jobs').append(element);
    }
    if (!active && status.jobs.length) active = status.jobs.at(-1).id;
    if (!active) return;
    const jid = active, job = await api('/api/jobs/'+jid);
    if (active !== jid) return;
    render(job);
    if ($('outputList').open) await outputs(jid);
  } catch(e) { $('error').textContent = e.message; }
  finally {
    busy = false;
    if (refreshRequested) { refreshRequested = false; queueMicrotask(poll); }
  }
}
poll(); setInterval(poll, 750);
