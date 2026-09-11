"""Independent manifest application: explicit Torch or TileLang backend."""
import argparse,json,time
from pathlib import Path
import bootstrap

class Engine:
    """One frozen model/backend per process; jobs get independent pipeline states."""
    def __init__(self,*,fast_load=True):
        import dlss5_model as nr
        from backend import Backend
        tick=time.perf_counter()
        if fast_load:
            from fast_load import load_model
            self.model,self.load_report=load_model(nr,bootstrap.REFERENCE/'weights_ht_blob.bin',device='cuda')
        else:
            self.model=nr.load_model(bootstrap.REFERENCE/'weights_ht_blob.bin',device='cuda')
            self.load_report={'strategy':'original frozen decoder'}
        self.load_seconds=time.perf_counter()-tick;self.backend=Backend(nr,self.model)
    def close(self):
        self.backend.close()

    def run(self,manifest,output,*,backend='tilelang',quiet=False,cancel=None,progress=None,before_frame=None):
        from execution import run_manifest
        if backend not in ('torch','tilelang'):raise ValueError('Unknown backend')
        output=Path(output)
        before=self.backend.report()
        job_shapes=set()
        def observed_progress(event):
            frame=event.get('last_frame') or {}
            shape=frame.get('neural_hw')
            if shape is not None:job_shapes.add(tuple(shape))
            if progress is not None:progress(event)
        with self.backend.activate(backend=='tilelang'):
            proc,report=run_manifest(manifest,output,model=self.model,quiet=quiet,cancel=cancel,progress=observed_progress,before_frame=before_frame,
                                    compute_backend=before['backend'] if backend=='tilelang' else 'PyTorch')
        after=self.backend.report()
        native=dict(after['native_nr'])
        native['total_calls']=native['calls']
        for key in ('calls','samples','failures','prepare_seconds','compile_seconds'):
            native[key]=native.get(key,0)-before['native_nr'].get(key,0)
        if not native['calls']:native['last_frame']=None
        counters={k:v-before['counters'].get(k,0) for k,v in after['counters'].items()}
        actual='torch' if backend=='torch' else self.backend.nr_backend if native['calls'] else 'not-run'
        native.update(selected_backend='torch' if backend=='torch' else self.backend.nr_backend,
                      actual_backend=actual,shapes=[{'height':h,'width':w} for h,w in sorted(job_shapes)])
        stats={**after,'counters':counters,'native_nr':native,'selected_backend':backend,'nr_backend':actual,
               'model_load_seconds':self.load_seconds,'model_loader':self.load_report,'run_seconds':report['seconds']}
        if backend=='torch':
            stats['backend']='PyTorch';stats['precision']='Frozen Torch reference; TileLang NR is inactive'
        report.update(nr_backend=actual,native_nr=native)
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (output/'backend.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
        return proc,report,stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('torch','tilelang'),default='tilelang')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--original-loader',action='store_true',help='Use the original frozen byte-by-byte model decoder')
    p.add_argument('--no-fsr-depth-clip',action='store_true',help='Use original Torch FSR depth clip/reactivity')
    p.add_argument('--no-fsr-accumulate',action='store_true',help='Use original Torch FSR accumulation/upsampling/history sampling')
    p.add_argument('--no-fsr-frontend',action='store_true',help='Use original Torch FSR reconstruction and locks')
    p.add_argument('--no-nr-chain',action='store_true',help='Use original Torch NRChain input/history/output operations')
    p.add_argument('--no-flow-implementation',action='store_true',help='Use original RAFT-small adapter implementation, not disable flow')
    p.add_argument('--no-depth-attention',action='store_true',help='Restore original DINO probability publication; do not disable depth')
    p.add_argument('--no-temporal-inputs',action='store_true',help='Restore original Temporal input preparation, not disable temporal history')
    a=p.parse_args()
    engine=Engine(fast_load=not a.original_loader)
    engine.backend.fsr_depth_clip_enabled=not a.no_fsr_depth_clip;engine.backend.fsr_accumulate_enabled=not a.no_fsr_accumulate;engine.backend.fsr_frontend_enabled=not a.no_fsr_frontend;engine.backend.nr_chain_enabled=not a.no_nr_chain;engine.backend.flow_implementation_enabled=not a.no_flow_implementation;engine.backend.depth_attention_enabled=not a.no_depth_attention;engine.backend.temporal_inputs_enabled=not a.no_temporal_inputs
    try:
        _,_,stats=engine.run(a.manifest,a.output,backend=a.backend)
        print(json.dumps(stats,indent=2))
    finally:engine.close()
if __name__=='__main__':main()
