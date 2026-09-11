"""Local task UI for the root Engine. Importing never loads GPU modules or listens.
Only an explicit `python whitebox_app.py` starts the loopback server.
"""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'model'
TERMINAL = {'completed', 'failed', 'cancelled'}
SELECTIONS = {
    'tilelang-vit': ('tilelang', 'tilelang-vit', 'TileLang NR（VitJoint 797fd63）'),
    'torch': ('torch', 'tilelang-vit', '全Torch参考'),
}
INPUT_PLANES = ('color', 'depth', 'motion', 'reactive', 'composition', 'protect')


def load_engine():
    # bootstrap adds reference to sys.path. Load run by exact root path so neither
    # that precedence nor a caller's cwd can select the frozen reference app.
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location('_whitebox_root_run', ROOT / 'run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Engine(fast_load=True)
    finally:
        while str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
        sys.path.insert(0, str(ROOT))


def safe_basename(name):
    if (not isinstance(name, str) or not name or len(name) > 200
            or name in ('.', '..') or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in name)
            or name.endswith((' ', '.'))):
        raise ValueError('文件名必须是安全的basename，不能包含路径')
    stem = name.split('.')[0].upper()
    if stem in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('不能使用系统保留文件名')
    return name


class JobManager:
    def __init__(self, workspace, model_dir=None, engine_factory=None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(model_dir or MODEL).resolve()
        self.engine_factory = engine_factory or load_engine
        self.items = {}
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        self.worker = None
        self.engine = None
        self.model_status = 'unloaded'
        self.model_error = None
        self.load_started = None
        self.load_seconds = None

    def submit(self, spec, files=(), nr_backend='tilelang-vit'):
        if not isinstance(spec, dict):
            raise ValueError('Manifest must be an object')
        if nr_backend not in SELECTIONS:
            raise ValueError('Unknown NR selection')
        spec = copy.deepcopy(spec)
        with self.lock:
            if sum(j['status'] not in TERMINAL for j in self.items.values()) >= 16:
                raise ValueError('最多16个活跃或排队任务')
            jid = uuid.uuid4().hex
            folder = self.workspace / jid
            inputs = folder / 'inputs'
            inputs.mkdir(parents=True)
            try:
                seen = {'job.json'}
                for upload in files:
                    name = safe_basename(upload.filename)
                    if name.casefold() in seen:
                        raise ValueError('上传文件名必须唯一，job.json为保留名')
                    seen.add(name.casefold())
                    upload.save(inputs / name)

                def input_path(name):
                    safe_basename(name)
                    if not (inputs / name).is_file() or name.casefold() == 'job.json':
                        raise ValueError(f'输入文件未上传: {name}')

                frames = spec.get('frames', [])
                events = spec.get('frame_events', {})
                if not isinstance(frames, list) or not isinstance(events, dict):
                    raise ValueError('frames必须为数组，frame_events必须为对象')
                for frame in frames + list(events.values()):
                    if not isinstance(frame, dict):
                        raise ValueError('每帧配置必须为对象')
                    if '_color_tensor' in frame:
                        raise ValueError('不能上传内部tensor字段')
                    for field in INPUT_PLANES:
                        if frame.get(field) is not None:
                            input_path(frame[field])
                if spec.get('video'):
                    if not isinstance(spec['video'], dict):
                        raise ValueError('video必须为对象')
                    input_path(spec['video']['file'])
                if spec.get('video_output'):
                    safe_basename(spec['video_output'].get('file', 'preview.mp4'))
                if spec.get('mode') == 'rgb_estimated':
                    settings = spec.setdefault('estimator_settings', {})
                    if not isinstance(settings, dict):
                        raise ValueError('estimator_settings必须为对象')
                    if 'model_dir' in settings and Path(settings['model_dir']).resolve() != self.model_dir:
                        raise ValueError('估计器模型目录由本地启动配置指定')
                    settings['model_dir'] = str(self.model_dir)
                manifest = inputs / 'job.json'
                manifest.write_text(json.dumps(spec, indent=2), encoding='utf-8')
            except BaseException:
                shutil.rmtree(folder)
                raise
            backend, engine_nr, label = SELECTIONS[nr_backend]
            job = {
                'id': jid, 'status': 'queued', 'stage': 'queued', 'created': time.time(),
                'processed_frames': 0, 'last_frame': None, 'error': None,
                'cancel': threading.Event(), 'reset': threading.Event(),
                'manifest': manifest, 'output': folder / 'output',
                'nr_backend': nr_backend, 'selected_nr': label, 'selected_backend': backend,
                'engine_nr': engine_nr, 'actual_neural_size': None, 'new_nr_calls': None,
                'stage_seconds': None, 'full_processing_seconds': None,
            }
            self.items[jid] = job
            self.queue.put(jid)
            if self.worker is None:
                self.worker = threading.Thread(target=self._run, name='whitebox-engine-worker', daemon=True)
                self.worker.start()
            return jid

    def snapshot(self, jid):
        with self.lock:
            j = self.items[jid]
            result = {k: copy.deepcopy(v) for k, v in j.items()
                      if k not in ('cancel', 'reset', 'manifest', 'output', 'engine_nr', 'started_tick')}
            result.update(cancel_requested=j['cancel'].is_set(), reset_requested=j['reset'].is_set())
            if 'started_tick' in j and j['status'] not in TERMINAL:
                result['elapsed_seconds'] = time.perf_counter() - j['started_tick']
            return result

    def status(self):
        with self.lock:
            return {'device': 'cuda', 'model_state': self.model_status, 'model_error': self.model_error,
                    'load_seconds': time.perf_counter() - self.load_started if self.model_status == 'loading' else self.load_seconds,
                    'jobs': [self.snapshot(i) for i in list(self.items)[-20:]],
                    'gpu_workers': 1, 'runtime_downloads': False}

    def cancel(self, jid):
        with self.lock:
            j = self.items[jid]
            if j['status'] not in TERMINAL:
                j['cancel'].set()
                j['status'] = 'cancel_requested'

    def reset_next(self, jid):
        with self.lock:
            j = self.items[jid]
            if j['status'] in TERMINAL:
                raise ValueError('已结束任务不能reset，请提交新任务')
            j['reset'].set()

    def _metrics(self, job, data):
        # Only measured fields: do not infer call counts from passes or dimensions
        # from requested output_size (the pipeline owns neural_size policy).
        if data.get('last_frame') is not None:
            job['last_frame'] = copy.deepcopy(data['last_frame'])
        frame = job.get('last_frame') or {}
        if frame.get('neural_hw') is not None:
            job['actual_neural_size'] = frame['neural_hw']
        for source, target in (('neural_hw', 'actual_neural_size'), ('new_nr_calls', 'new_nr_calls'),
                               ('stage_timings', 'stage_seconds')):
            if data.get(source) is not None:
                job[target] = copy.deepcopy(data[source])
        if isinstance(data.get('nr'), dict):
            nr_state = copy.deepcopy(data['nr'])
            job['nr'] = nr_state
            if nr_state.get('calls') is not None:
                job['new_nr_calls'] = nr_state['calls']

    def _apply_report(self, job, report):
        for field in ('processed_frames', 'seconds'):
            if field in report:
                job[field] = report[field]
        if report.get('frames'):
            job['last_frame'] = report['frames'][-1]
        job['video_output_file'] = (report.get('video_output') or {}).get('file')
        self._metrics(job, report)

    def _run(self):
        while True:
            jid = self.queue.get()
            j = self.items[jid]
            try:
                if j['cancel'].is_set():
                    with self.lock:
                        j.update(status='cancelled', stage='cancelled')
                    continue
                with self.lock:
                    j.update(status='running', stage='loading_engine', started_tick=time.perf_counter())
                if self.engine is None:
                    with self.lock:
                        self.model_status = 'loading'
                        self.model_error = None
                        self.load_started = time.perf_counter()
                    self.engine = self.engine_factory()
                    with self.lock:
                        self.model_status = 'ready'
                        self.load_seconds = time.perf_counter() - self.load_started
                if j['cancel'].is_set():
                    with self.lock:
                        j.update(status='cancelled', stage='cancelled')
                    continue

                def progress(event):
                    with self.lock:
                        j.update({k: copy.deepcopy(v) for k, v in event.items()
                                  if k in ('stage', 'message', 'processed_frames', 'total_frames_estimate', 'last_frame', 'seconds')})
                        self._metrics(j, event)

                def before_frame(index):
                    with self.lock:
                        if j['reset'].is_set():
                            j['reset'].clear()
                            j['last_reset_frame'] = index
                            return {'reset': True, 'reset_reason': 'UI explicit reset'}
                    return None

                old_nr = self.engine.backend.nr_backend
                try:
                    self.engine.backend.nr_backend = j['engine_nr']
                    with self.lock:
                        j['stage'] = 'preparing_pipeline'
                    proc, report, stats = self.engine.run(
                        j['manifest'], j['output'], backend=j['selected_backend'], quiet=True,
                        cancel=j['cancel'].is_set, progress=progress, before_frame=before_frame)
                    with self.lock:
                        self._apply_report(j, report)
                        j['backend_stats'] = copy.deepcopy(stats)
                        self._metrics(j, stats)
                        status = report.get('status', 'completed')
                        if status not in TERMINAL:
                            status = 'cancelled' if j['cancel'].is_set() else 'completed'
                        j.update(status=status, stage=status)
                    del proc
                finally:
                    self.engine.backend.nr_backend = old_nr
            except BaseException as exc:
                detail = traceback.format_exc()
                with self.lock:
                    if self.engine is None:
                        self.model_status = 'error'
                        self.model_error = str(exc)
                        self.load_seconds = time.perf_counter() - self.load_started
                    try:
                        report_path = j['output'] / 'report.json'
                        if report_path.is_file():
                            self._apply_report(j, json.loads(report_path.read_text(encoding='utf-8')))
                    except (OSError, ValueError, TypeError, KeyError):
                        pass
                    status = 'cancelled' if j['cancel'].is_set() else 'failed'
                    j.update(status=status, stage=status, error=str(exc))
                try:
                    (j['manifest'].parent.parent / 'error.txt').write_text(detail, encoding='utf-8')
                except OSError:
                    pass  # A full disk must not kill the sole queue worker.
            finally:
                with self.lock:
                    if 'started_tick' in j:
                        j['full_processing_seconds'] = time.perf_counter() - j['started_tick']
                    j['finished'] = time.time()
                self.queue.task_done()


def create_app(*, workspace=None, model_dir=None, engine_factory=None):
    app = Flask(__name__, static_folder=str(ROOT / 'webui'), static_url_path='/ui')
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
    manager = JobManager(workspace or Path.home() / '.dlss5-tilelang-whitebox' / 'jobs', model_dir, engine_factory)
    app.extensions['whitebox_jobs'] = manager

    @app.before_request
    def loopback_guard():
        if request.host.split(':')[0] not in ('localhost', '127.0.0.1'):
            return jsonify(error='Loopback host required'), 403
        origin = request.headers.get('Origin')
        if origin and urlsplit(origin).netloc != request.host:
            return jsonify(error='Cross-origin control request rejected'), 403

    @app.get('/')
    def index():
        return send_file(ROOT / 'webui/index.html')

    @app.get('/api/status')
    def status():
        return jsonify(manager.status())

    @app.post('/api/jobs')
    def submit():
        try:
            if 'manifest' in request.form:
                spec = json.loads(request.form['manifest'])
                selection = request.form.get('nr_backend', 'tilelang-vit')
            else:
                payload = request.get_json(silent=True)
                spec = payload['manifest']
                selection = payload.get('nr_backend', 'tilelang-vit')
            jid = manager.submit(spec, request.files.getlist('files'), selection)
            return jsonify(id=jid), 202
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return jsonify(error=str(exc)), 400

    @app.get('/api/jobs/<jid>')
    def job(jid):
        try:
            return jsonify(manager.snapshot(jid))
        except KeyError:
            return jsonify(error='Unknown job'), 404

    @app.post('/api/jobs/<jid>/cancel')
    def cancel(jid):
        try:
            manager.cancel(jid)
            return jsonify(manager.snapshot(jid)), 202
        except KeyError:
            return jsonify(error='Unknown job'), 404

    @app.post('/api/jobs/<jid>/reset')
    def reset(jid):
        try:
            manager.reset_next(jid)
            return jsonify(manager.snapshot(jid)), 202
        except KeyError:
            return jsonify(error='Unknown job'), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @app.get('/api/jobs/<jid>/outputs')
    def outputs(jid):
        # Paginated access to every retained frame, not a latest-only output path.
        with manager.lock:
            j = manager.items.get(jid)
            if j is None:
                return jsonify(error='Unknown job'), 404
            root = j['output']
        try:
            offset = max(0, int(request.args.get('offset', 0)))
        except ValueError:
            return jsonify(error='Invalid offset'), 400
        paths = sorted(p.name for p in root.iterdir() if p.is_file()) if root.is_dir() else []
        return jsonify(files=paths[offset:offset + 100], offset=offset, total=len(paths))

    @app.get('/api/jobs/<jid>/files/<path:name>')
    def file(jid, name):
        with manager.lock:
            j = manager.items.get(jid)
            if j is None:
                return jsonify(error='Unknown job'), 404
            root = j['output'].resolve()
            path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return jsonify(error='Output file not found'), 404
        response = send_file(path, as_attachment=request.args.get('download') == '1', conditional=True)
        response.headers['Cache-Control'] = 'no-store'
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    app = create_app(workspace=args.workspace, model_dir=args.model_dir)
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == '__main__':
    main()
