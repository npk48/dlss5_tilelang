"""Exercise the real Engine.run entry with image sizes/history and dynamic NR.
Use --generate without CUDA to create deterministic guided image inputs.
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
SHAPES=[(257,321),(333,481),(451,601),(321,257),(257,321)]
def generate(directory):
    directory.mkdir(parents=True,exist_ok=True);frames=[]
    for case,(h,w) in enumerate(SHAPES):
        y,x=np.mgrid[:h,:w].astype(np.float32)
        for v in range(2):
            color=np.stack([.5+.2*np.sin((x-v)*.071),.5+.2*np.cos(y*.083),.5+.2*np.sin((x-v)*.09+y*.07)],-1).astype(np.float32)
            depth=np.full((h,w,1),.5,dtype=np.float32);motion=np.zeros((h,w,2),np.float32);motion[...,0]=-.25 if v else 0
            key=f'{case}_{v}';np.save(directory/(key+'-color.npy'),color);np.save(directory/(key+'-depth.npy'),depth);np.save(directory/(key+'-motion.npy'),motion)
            frame={'color':key+'-color.npy','depth':key+'-depth.npy','motion':key+'-motion.npy','reset':v==0}
            if v==0:
                frame['output_size']=[h,w]
                frame['nr_passes']=([{'tone':.25,'temporal_strength':.65},{'intensity':.4}] if case==1 else [{'tone':.1,'temporal_strength':.8}])
            frames.append(frame)
    manifest={'mode':'guided','color_encoding':'sRGB','motion_convention':'current_to_previous_pixels','output_size':list(SHAPES[0]),'save_float':True,'guide_settings':{'validate':False},'frames':frames}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2));return directory/'manifest.json'
def run(a):
    os.environ['TILELANG_CACHE_DIR']=str(a.cache.resolve());os.environ['DLSS5_FP8_TOOLCHAIN']='C:/work/dlss5_remake/.toolchains/cuda12.8'
    sys.path.insert(0,str(a.root.resolve()))
    from run import Engine
    from tilelang_nr.common.runtime_jit import runtime_compilation_stats
    engine=Engine();events=[]
    try:
        def progress(event):
            if event.get('stage')!='processing':return
            stats=runtime_compilation_stats();record={'frame':event['processed_frames'],'compile':stats,'image':event.get('last_frame')};events.append(record)
            print(json.dumps({'frame':record['frame'],'jit_builds':stats['jit_builds'],'nvrtc':stats['private_nvrtc_calls']}),flush=True)
        proc,report,backend=engine.run(a.inputs/'manifest.json',a.output,backend='tilelang',quiet=True,progress=progress)
        assert report['status']=='completed' and report['processed_frames']==len(SHAPES)*2
        assert backend['nr_backend']=='tilelang-vit'
        assert backend['counters'].get('nr_chain_fallback_calls',0)==0
        first=events[1]['compile']['jit_builds'];nvrtc=events[1]['compile']['private_nvrtc_calls']
        assert all(e['compile']['jit_builds']==first and e['compile']['private_nvrtc_calls']==nvrtc for e in events[1:]),'NR compiled after initial cold/warm frames'
        for i,(h,w) in enumerate(SHAPES):
            for v in range(2):
                image=np.load(a.output/f'{i*2+v:06}.npy');assert image.shape[:2]==(h,w) and np.isfinite(image).all()
        (a.output/'runtime-spatial.json').write_text(json.dumps({'pass':True,'events':events,'backend':backend},indent=2))
    finally:engine.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');p.add_argument('--inputs',type=Path,default=Path('C:/tmp/d5-tl-pipeline/inputs'));p.add_argument('--root',type=Path,default=Path('C:/work/dlss5_remake'));p.add_argument('--cache',type=Path,default=Path('C:/tmp/d5-nr-eval/candidate-probe-cache'));p.add_argument('--output',type=Path,default=Path('C:/tmp/d5-tl-pipeline/output'));a=p.parse_args()
    if a.generate:print(generate(a.inputs))
    else:run(a)
