"""Development evaluator for frozen/current TileLang NR, with an IPC timing worker.
No native CUDA NR is imported or called. Packets are shared immutable F32 data.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path

def worker(args):
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ['TILELANG_CACHE_DIR']=str(args.cache.resolve())
    os.environ['DLSS5_FP8_TOOLCHAIN']=str(args.toolchain.resolve())
    sys.path.insert(0,str(root))
    from runtime import bootstrap
    import torch,numpy as np
    from runtime.model_loader import load_model
    import dlss5_model as M
    from tilelang_nr.runtime import VitJointTileLangNR
    if args.trace:
        from tilelang import tvm
        from tilelang.instrumentation import PassInstrumentationTool,register_pass_instrumentation_tool
        @tvm.instrument.pass_instrument
        class Trace:
            def run_before_pass(self,mod,info):print('PASS_BEGIN',time.time(),info.name,flush=True)
            def run_after_pass(self,mod,info):print('PASS_END',time.time(),info.name,flush=True)
        class TraceTool(PassInstrumentationTool):
            def create_pass_instrument(self):return Trace()
        register_pass_instrumentation_tool('nr-diagnostic',TraceTool)
    from runtime.fp8_compiler import install
    install()
    import importlib
    libgen=importlib.import_module('tilelang.jit.adapter.nvrtc.libgen')
    compiler=libgen.compile_cuda;compiles=[]
    def counted(*a,**kw):
        start=time.perf_counter()
        try:return compiler(*a,**kw)
        finally:compiles.append({'source_sha256':hashlib.sha256(a[0].encode() if isinstance(a[0],str) else bytes(a[0])).hexdigest(),'seconds':time.perf_counter()-start})
    libgen.compile_cuda=counted
    from tilelang.cache.kernel_cache import KernelCache
    cache_entries={};tag_entry=KernelCache._tag_kernel_cache_entry
    def tagged(kernel,key,path):
        tag_entry(kernel,key,path);cache_entries[key]=path
    KernelCache._tag_kernel_cache_entry=staticmethod(tagged)
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    start=time.perf_counter();model,loader=load_model(M,root/'model/weights_ht_blob.bin',device='cuda');model.eval()
    engine=VitJointTileLangNR(model,max_cached_shapes=1);packets=[];h=w=None
    def emit(value):print('D5_RESULT '+json.dumps(value,default=str),flush=True)
    emit({'ready':True,'load_seconds':time.perf_counter()-start,'root':str(root),'tilelang':__import__('tilelang').__version__})
    try:
        for line in sys.stdin:
            try:
                command=json.loads(line);op=command['op']
                if op=='close':break
                if op=='prepare':
                    h,w=command['h'],command['w'];packets=[]
                    for v in range(2):
                        x=np.fromfile(args.packets/f'{h}x{w}_{v}.packet.f32',dtype=np.float32).reshape(1,16,h,w)
                        packets.append(torch.from_numpy(x.copy()).cuda())
                    before=len(compiles);torch.cuda.synchronize();start=time.perf_counter()
                    head=engine.infer_minimal(packets[0]);torch.cuda.synchronize();first=time.perf_counter()-start
                    hashes=[]
                    for v,x in enumerate(packets):
                        y=engine.infer_minimal(x).cpu().numpy();assert np.isfinite(y).all()
                        data=y.tobytes();(out/f'{h}x{w}_{v}.head.f32').write_bytes(data);hashes.append(hashlib.sha256(data).hexdigest())
                    emit({'h':h,'w':w,'first_seconds':first,'new_nvrtc_compiles':len(compiles)-before,'total_nvrtc_compiles':len(compiles),'head_sha256':hashes,'report':engine.report()})
                elif op=='measure':
                    for i in range(command.get('warm',10)):engine.infer_minimal(packets[i%2])
                    torch.cuda.synchronize();host=[];gpu=[]
                    for i in range(command.get('n',30)):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        begin.record();start=time.perf_counter();engine.infer_minimal(packets[i%2]);end.record();end.synchronize()
                        host.append((time.perf_counter()-start)*1000);gpu.append(begin.elapsed_time(end))
                    emit({'h':h,'w':w,'host_ms':host,'gpu_span_ms':gpu,'total_nvrtc_compiles':len(compiles)})
                elif op=='report':emit({'report':engine.report(),'compiles':compiles,'cache_entries':cache_entries})
                else:raise ValueError(op)
            except BaseException as e:
                import traceback;traceback.print_exc();emit({'error':str(e)});raise
    finally:
        engine.close();(out/'compile-log.json').write_text(json.dumps(compiles,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--packets',type=Path,default=Path('C:/tmp/d5-nr-eval/packets'));p.add_argument('--toolchain',type=Path,default=Path('C:/work/dlss5_remake/.toolchains/cuda12.8'))
    p.add_argument('--trace',action='store_true');worker(p.parse_args())
