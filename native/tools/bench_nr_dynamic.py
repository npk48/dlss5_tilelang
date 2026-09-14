"""Development-only packet generation and paired native NR comparison."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SHAPES=[(320,384),(384,320),(384,512),(448,576),(512,640)]
def packets(out):
    sys.path.insert(0,str(ROOT/'reference'))
    import torch
    import dlss5_model as M
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for h,w in SHAPES:
        y=torch.arange(h,dtype=torch.float32)[:,None]/max(h-1,1)
        x=torch.arange(w,dtype=torch.float32)[None,:]/max(w-1,1)
        color=torch.stack((x.expand(h,w),y.expand(h,w),((x+y)/2).expand(h,w)),-1)[None]
        motion=torch.zeros((1,h,w,2));motion[...,0]=.25;motion[...,1]=-.5
        frame=M.FrameInputs(color=color,prev_output=color.roll(3,dims=2),mvec=motion,
            mvec_scale_xy=torch.ones(2),output_dimensions_wh=torch.tensor([w,h]),style=1,
            local_structure_strength=2,local_tone_strength=1.2,use_auto_mask=False)
        packet=M.build_preblock_features(frame,frame=37).contiguous()
        for v in range(2):
            data=packet if v==0 else packet*.9+torch.sin(torch.arange(packet.numel(),dtype=torch.float32).reshape(packet.shape))*.01
            p=out/f'{h}x{w}_{v}.packet.f32';p.write_bytes(data.contiguous().numpy().tobytes())
            records.append({'h':h,'w':w,'variant':v,'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        print('PACKET',h,w,flush=True)
    (out/'manifest.json').write_text(json.dumps(records,indent=2))
def compare(baseline,candidate):
    rows=[]
    for h,w in SHAPES:
        for v in range(2):
            name=f'{h}x{w}_{v}.head.f32';a=np.fromfile(baseline/name,dtype=np.float32);b=np.fromfile(candidate/name,dtype=np.float32)
            assert a.size==b.size==h*w*4
            d=np.abs(a-b)
            rows.append({'h':h,'w':w,'variant':v,'bit_exact':bool(np.array_equal(a,b)),
                'max_abs':float(d.max()),'mean_abs':float(d.mean()),'pass':bool(np.allclose(a,b,rtol=1e-5,atol=1e-5))})
    old=[json.loads(s) for s in (baseline/'timings.jsonl').read_text().splitlines()]
    new=[json.loads(s) for s in (candidate/'timings.jsonl').read_text().splitlines()]
    perf=[]
    for b in new:
        a=next(a for a in old if (a['h'],a['w'])==(b['h'],b['w']))
        perf.append({'h':b['h'],'w':b['w'],'static':a,'dynamic':b,
            'gpu_span_change_percent':100*(b['gpu_span_p50_ms']/a['gpu_span_p50_ms']-1),
            'host_change_percent':100*(b['host_p50_ms']/a['host_p50_ms']-1)})
    report={'correctness':rows,'performance':perf,'pass':all(r['pass'] for r in rows)}
    (candidate/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert report['pass']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['packets','compare']);p.add_argument('directory',type=Path);p.add_argument('--baseline',type=Path);a=p.parse_args()
    if a.action=='packets':packets(a.directory)
    else:compare(a.baseline,a.directory)
