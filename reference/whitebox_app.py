"""Standalone whitebox task UI. Importing does not import Torch or bind a port.
Only explicit `python whitebox_app.py` starts a loopback server.
"""
from pathlib import Path
import argparse,copy,json,queue,threading,time,uuid,traceback
from urllib.parse import urlsplit
from flask import Flask,request,jsonify,send_file
from werkzeug.utils import secure_filename

ROOT=Path(__file__).resolve().parent
TERMINAL={'completed','failed','cancelled'}

class JobManager:
    def __init__(self,workspace,device='cuda',weights=None,model_dir=None):
        self.workspace=Path(workspace).resolve();self.workspace.mkdir(parents=True,exist_ok=True)
        self.device=device;self.weights=Path(weights) if weights else ROOT/'weights_ht_blob.bin'
        self.model_dir=Path(model_dir) if model_dir else ROOT/'guide_models'
        self.items={};self.lock=threading.RLock();self.queue=queue.Queue();self.worker=None
        self.model=None;self.model_status='unloaded';self.model_error=None;self.load_started=None;self.load_seconds=None
    def submit(self,spec,files=()):
        if not isinstance(spec,dict):raise ValueError('Manifest must be an object')
        spec=copy.deepcopy(spec)
        with self.lock:
            if sum(j['status'] not in TERMINAL for j in self.items.values())>=16:raise ValueError('At most16 active/queued jobs')
            jid=uuid.uuid4().hex;folder=self.workspace/jid;inputs=folder/'inputs';inputs.mkdir(parents=True)
            seen=set()
            for upload in files:
                name=upload.filename
                if not name or len(name)>200 or name in ('.','..') or any(c in name for c in '<>:"/\\\\|?*') or name.endswith((' ','.')):
                    raise ValueError('Upload filename must be a safe basename')
                if Path(name).stem.upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:raise ValueError('Reserved upload filename')
                if name in seen:raise ValueError('Upload filenames must be distinct')
                seen.add(name);upload.save(inputs/name)
            if spec.get('mode')=='rgb_estimated':spec.setdefault('estimator_settings',{}).setdefault('model_dir',str(self.model_dir))
            manifest=inputs/'job.json';manifest.write_text(json.dumps(spec,indent=2),encoding='utf-8')
            job={'id':jid,'status':'queued','stage':'queued','created':time.time(),'processed_frames':0,'last_frame':None,
                 'cancel':threading.Event(),'reset':threading.Event(),'manifest':manifest,'output':folder/'output','error':None}
            self.items[jid]=job;self.queue.put(jid)
            if self.worker is None:
                self.worker=threading.Thread(target=self._run,name='whitebox-gpu-worker',daemon=True);self.worker.start()
            return jid
    def snapshot(self,jid):
        with self.lock:
            if jid not in self.items:raise KeyError(jid)
            j=self.items[jid];result={k:copy.deepcopy(v) for k,v in j.items() if k not in ('cancel','reset','manifest','output')}
            result.update(cancel_requested=j['cancel'].is_set(),reset_requested=j['reset'].is_set())
            return result
    def status(self):
        with self.lock:
            return {'device':self.device,'model_state':self.model_status,'model_error':self.model_error,
                    'load_seconds':time.monotonic()-self.load_started if self.model_status=='loading' else self.load_seconds,
                    'jobs':[self.snapshot(i) for i in list(self.items)[-20:]],'gpu_workers':1,'runtime_downloads':False}
    def cancel(self,jid):
        with self.lock:
            j=self.items[jid]
            if j['status'] not in TERMINAL:j['cancel'].set();j['status']='cancel_requested'
    def reset_next(self,jid):
        with self.lock:
            j=self.items[jid]
            if j['status'] in TERMINAL:raise ValueError('Completed jobs cannot be reset; submit a new job')
            j['reset'].set()
    def _run(self):
        while True:
            jid=self.queue.get();j=self.items[jid]
            try:
                if j['cancel'].is_set():
                    with self.lock:j.update(status='cancelled',stage='cancelled')
                    continue
                with self.lock:j.update(status='running',stage='loading_nr')
                if self.model is None:
                    with self.lock:self.model_status='loading';self.model_error=None;self.load_started=time.monotonic()
                    import dlss5_model as nr
                    self.model=nr.load_model(self.weights,device=self.device)
                    with self.lock:self.model_status='ready';self.load_seconds=time.monotonic()-self.load_started
                if j['cancel'].is_set():
                    with self.lock:j.update(status='cancelled',stage='cancelled')
                    continue
                from whitebox_pipeline.__main__ import run_manifest
                def progress(event):
                    with self.lock:
                        j.update({k:v for k,v in event.items() if k in ('stage','processed_frames','total_frames_estimate','last_frame','seconds')})
                def before_frame(index):
                    if j['reset'].is_set():j['reset'].clear();return {'reset':True,'reset_reason':'UI explicit reset'}
                    return None
                proc,report=run_manifest(j['manifest'],j['output'],device=self.device,weights=self.weights,model=self.model,
                                        cancel=j['cancel'].is_set,progress=progress,before_frame=before_frame,quiet=True)
                with self.lock:j.update(status='completed',stage='completed',processed_frames=report['processed_frames'],seconds=report['seconds'],video_output_file=(report.get('video_output') or {}).get('file'))
                del proc
            except BaseException as exc:
                with self.lock:
                    if self.model is None:self.model_status='error';self.model_error=str(exc)
                    j.update(status='cancelled' if j['cancel'].is_set() else 'failed',stage='cancelled' if j['cancel'].is_set() else 'failed',error=str(exc))
                (j['manifest'].parent.parent/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            finally:
                with self.lock:j['finished']=time.time()
                self.queue.task_done()

def create_app(*,workspace=None,device='cuda',weights=None,model_dir=None):
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=1024*1024*1024
    manager=JobManager(workspace or Path.home()/'.dlss5-whitebox'/'jobs',device,weights,model_dir)
    app.extensions['whitebox_jobs']=manager
    @app.before_request
    def loopback_guard():
        if request.host.split(':')[0] not in ('localhost','127.0.0.1'):return jsonify(error='Loopback host required'),403
        origin=request.headers.get('Origin')
        if origin and urlsplit(origin).netloc!=request.host:return jsonify(error='Cross-origin control request rejected'),403
    @app.get('/')
    def index():return send_file(ROOT/'whitebox_ui/index.html')
    @app.get('/api/status')
    def status():return jsonify(manager.status())
    @app.post('/api/jobs')
    def submit():
        try:
            spec=json.loads(request.form['manifest']) if 'manifest' in request.form else request.get_json()['manifest']
            jid=manager.submit(spec,request.files.getlist('files'));return jsonify(id=jid),202
        except (ValueError,KeyError,TypeError) as exc:return jsonify(error=str(exc)),400
    @app.get('/api/jobs/<jid>')
    def job(jid):
        try:return jsonify(manager.snapshot(jid))
        except KeyError:return jsonify(error='Unknown job'),404
    @app.post('/api/jobs/<jid>/cancel')
    def cancel(jid):
        try:manager.cancel(jid);return jsonify(manager.snapshot(jid)),202
        except KeyError:return jsonify(error='Unknown job'),404
    @app.post('/api/jobs/<jid>/reset')
    def reset(jid):
        try:manager.reset_next(jid);return jsonify(manager.snapshot(jid)),202
        except KeyError:return jsonify(error='Unknown job'),404
        except ValueError as exc:return jsonify(error=str(exc)),409
    @app.get('/api/jobs/<jid>/files/<path:name>')
    def file(jid,name):
        with manager.lock:
            j=manager.items.get(jid)
            if j is None:return jsonify(error='Unknown job'),404
            root=j['output'].resolve();path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():return jsonify(error='Output file not found'),404
        response=send_file(path,as_attachment=request.args.get('download')=='1');response.headers['Cache-Control']='no-store';return response
    return app

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=7861);p.add_argument('--device',default='cuda')
    p.add_argument('--workspace',type=Path);p.add_argument('--weights',type=Path);p.add_argument('--model-dir',type=Path)
    a=p.parse_args();app=create_app(workspace=a.workspace,device=a.device,weights=a.weights,model_dir=a.model_dir)
    app.run(host='127.0.0.1',port=a.port,debug=False,use_reloader=False,threaded=True)
if __name__=='__main__':main()
