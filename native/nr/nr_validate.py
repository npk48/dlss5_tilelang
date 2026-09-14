"""Development-only differential validation; never imported by the native engine.
All generated files/caches are confined to native/nr/validation.
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('validation')
OUT.mkdir(exist_ok=True)
os.environ['TILELANG_CACHE_DIR'] = str(OUT / 'tilelang')
os.environ['NATIVE_NR_TOOLCHAIN'] = str(ROOT / '.toolchains/cuda12.8')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / 'reference')]
tempfile.tempdir = str(OUT)
from cuda_nr import layouts as L, source as S
EXE = Path(os.environ.get('NR_SMOKE_EXE', str(Path(__file__).with_name('build') / 'Release/nr_smoke.exe')))

def layouts(shapes):
    report = []
    for h,w in shapes:
        path = OUT / f'layouts_{h}x{w}'
        subprocess.run([str(EXE), '--dump-layouts', str(h), str(w), str(path)], check=True)
        shape = L.Shape(h,w)
        expected = {}
        ds,up = {},{}
        for heads,b in [(2,8),(4,14),(8,22)]:
            ds[heads] = L.pool_metadata(shape.outer(b), L.FAMILY, L.TWO)
            for name,v in zip(['input','output'], ds[heads]):
                expected[f'pool{heads}_{name}.i32'] = v
        for heads,b in [(2,62),(4,56),(8,48)]:
            ui,inv = L.up_metadata(shape.outer(b),L.FAMILY,L.TWO)
            im,_ = (L.maps2 if heads==2 else L.wide_maps)(shape.outer(b))
            up[heads] = ui,inv,im
            expected[f'up{heads}_input.i32'] = ui
            expected[f'up{heads}_inverse.i32'] = inv
        dh,dw = shape.deep_hw
        ph,pw = L.align(dh,8)//2,L.align(dw,8)//2
        pc = L.pointmap(ph,pw,1024)
        pool = L.raw16(ph,pw,pool=True)
        expected.update({'enc.i32':L.raw16(dh,dw), 'dec.i32':L.raw16(dh,dw,decoder=True),
            'pc.i32':pc, 'pi.i32':L.pointmap(ph,pw,1024,True), 'ff.i32':L.pointmap(ph,pw,4096),
            'pool.i32':pool, 'terminal.i32':L.raw16(ph,pw,pool=True,channels=1024),
            'half.i32':L.halfmap(pc),'repack99.i32':L.repack99(ph,pw)})
        for i in range(4):
            plan,members,_ = L.local_packets(dh,dw,i)
            expected[f'local{i}.i32'] = plan
            expected[f'members{i}.i32'] = members
        sources,routes = L.packets(pool,ph*pw,512)
        expected['sources.i32'],expected['routes.u16'] = sources,routes
        for name,value in expected.items():
            dtype='<u2' if name.endswith('.u16') else '<i4'
            actual=np.fromfile(path/name,dtype)
            value=np.asarray(value,dtype).ravel()
            if not np.array_equal(actual,value):
                bad=np.flatnonzero(actual!=value) if actual.size==value.size else []
                raise AssertionError((h,w,name,actual.size,value.size,[(int(i),int(actual[i]),int(value[i])) for i in bad[:8]]))
        route_arrays = 0
        for heads in [4,8]:
            # Geometry is now GPU data, not CUDA literals. Compare its bytes against
            # the original Python generator's arrays, independently of generic code.
            source = S.routes(heads,ds[heads],up[heads])
            for name in ['ds_ids','ds_rows','up_ids','up_local','up_global']:
                match = re.search(r'\b'+name+r'(?:\[\d+\])+\s*=\s*\{(.*?)\};', source, re.S)
                if match is None:
                    raise AssertionError(('missing reference route', heads, name))
                dtype = 'i1' if name in ['ds_rows','up_local'] else '<i4'
                value = np.asarray([int(x) for x in re.findall(r'-?\d+',match[1])], dtype=dtype)
                actual = np.fromfile(path/f'routes{heads}_{name}.bin', dtype=dtype)
                if not np.array_equal(actual,value):
                    raise AssertionError((h,w,'runtime routes differ',heads,name))
                route_arrays += 1
        for heads in [2,4,8]:
            value = np.asarray(ds[heads][1]).reshape(-1,heads*64)[:,0]
            value = np.where(value < 0, -1, value // 16).astype('<i4')
            actual = np.fromfile(path/f'pixels{heads}.bin','<i4')
            if not np.array_equal(actual,value):
                raise AssertionError((h,w,'runtime transition addresses differ',heads))
        report.append({'shape':[h,w],'integer_arrays_exact':len(expected),'runtime_route_arrays_exact':route_arrays,'transition_arrays_exact':3})
        print('LAYOUT PASS',h,w,flush=True)
    (OUT/'layouts.json').write_text(json.dumps(report,indent=2))

def reference(shapes):
    import torch
    import dlss5_model as M
    from runtime.model_loader import load_model
    import cuda_nr.device as D
    import cuda_nr.runtime as R
    D.ROOT=OUT
    R.ROOT=OUT
    torch.set_grad_enabled(False)
    model,load_report=load_model(M,ROOT/'model/weights_ht_blob.bin',device='cuda')
    runtime=R.NativeNR(model,max_cached_shapes=1)
    cases=[]
    try:
        for h,w in shapes:
            y=torch.arange(h,device='cuda',dtype=torch.float32)[:,None]/max(h-1,1)
            x=torch.arange(w,device='cuda',dtype=torch.float32)[None,:]/max(w-1,1)
            color=torch.stack((x.expand(h,w),y.expand(h,w),((x+y)/2).expand(h,w)),-1)[None]
            motion=torch.zeros((1,h,w,2),device='cuda');motion[...,0]=.25;motion[...,1]=-.5
            frame=M.FrameInputs(color=color,prev_output=color.roll(3,dims=2),mvec=motion,
                mvec_scale_xy=torch.ones(2,device='cuda'),output_dimensions_wh=torch.tensor([w,h]),
                style=1,local_structure_strength=2,local_tone_strength=1.2,use_auto_mask=False)
            packet=M.build_preblock_features(frame,frame=37).contiguous()
            for variant in range(2):
                p=packet if variant==0 else (packet*.9+torch.sin(torch.arange(packet.numel(),device='cuda',dtype=torch.float32).reshape(packet.shape))*.01).contiguous()
                name=f'{h}x{w}_{variant}'
                p.cpu().numpy().tofile(OUT/f'{name}.packet.f32')
                print('REFERENCE',name,flush=True)
                start=time.perf_counter();head=runtime.infer_minimal(p).cpu().numpy()
                if not np.isfinite(head).all():raise AssertionError('nonfinite reference')
                head.tofile(OUT/f'{name}.reference.f32')
                cases.append({'name':name,'h':h,'w':w,'reference_seconds':time.perf_counter()-start,
                    'packet_sha256':hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest(),
                    'reference_sha256':hashlib.sha256(head.tobytes()).hexdigest()})
                (OUT/'cases.json').write_text(json.dumps({'loader':load_report,'cases':cases},indent=2))
    finally:runtime.close()

def native():
    cases=json.loads((OUT/'cases.json').read_text())['cases']
    command=[str(EXE),str(ROOT/'model')]
    for c in cases:
        n=c['name'];command += [str(c['h']),str(c['w']),str(OUT/f'{n}.packet.f32'),str(OUT/f'{n}.native.f32')]
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (OUT/'native.log').write_text(result.stdout)
    print(result.stdout,flush=True)
    result.check_returncode()
    report=[]
    for c in cases:
        n=c['name'];a=np.fromfile(OUT/f'{n}.native.f32','<f4');b=np.fromfile(OUT/f'{n}.reference.f32','<f4')
        if a.size!=c['h']*c['w']*4 or not np.isfinite(a).all():raise AssertionError(n)
        error=np.abs(a.astype(np.float64)-b)
        row={**c,'native_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'max_abs':float(error.max()),
            'mean_abs':float(error.mean()),'different_values':int(np.count_nonzero(a!=b)),
            'bitwise_equal':a.tobytes()==b.tobytes()}
        report.append(row);print(json.dumps(row),flush=True)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))
    if not all(r['bitwise_equal'] for r in report):raise AssertionError('Native head differs from reference; see comparison.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['layouts','reference','native']);args=parser.parse_args()
    if args.mode=='layouts':layouts([(64,64),(128,192),(320,384),(384,320),(512,640)])
    elif args.mode=='reference':reference([(320,384),(128,192),(384,320)])
    else:native()
