"""Explicit, hash-checked public model download; never called during inference."""
from pathlib import Path
import argparse,hashlib,json,time,urllib.request
ROOT=Path(__file__).resolve().parents[1]

def run(proxy):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
    def get(url):return opener.open(urllib.request.Request(url,headers={'User-Agent':'whitebox-guide-setup/1','Cache-Control':'no-cache'}),timeout=120)
    model='depth-anything/Metric-Video-Depth-Anything-Small'
    with get(f'https://huggingface.co/api/models/{model}?blobs=true') as r:meta=json.load(r)
    revision=meta['sha'];file='metric_video_depth_anything_vits.pth'
    asset=next(s for s in meta['siblings'] if s['rfilename']==file)
    expected=asset['lfs']['sha256']
    license_id=meta.get('cardData',{}).get('license')
    print(json.dumps({'model':model,'revision':revision,'license':license_id,'asset':asset}),flush=True)
    if license_id!='apache-2.0':raise RuntimeError('Metric Small license metadata needs explicit review before download')
    target=ROOT/'guide_models';target.mkdir(exist_ok=True)
    items=[{'name':'raft-small-C_T_V2','file':'raft_small_C_T_V2-01064c6d.pth','url':'https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth','prefix':'01064c6d'},
           {'name':'metric-video-depth-anything-small','file':file,'url':f'https://huggingface.co/{model}/resolve/{revision}/{file}','sha256':expected,'revision':revision,'model_license':license_id}]
    for item in items:
        p=target/item['file'];expected=item.get('sha256');prefix=item.get('prefix','')
        if not p.exists():
            tmp=p.with_suffix(p.suffix+'.partial');count=0;start=time.monotonic()
            with get(item['url']+('?download=true&cb='+str(time.time_ns()) if 'huggingface.co' in item['url'] else '')) as r,tmp.open('wb') as f:
                while True:
                    data=r.read(1024*1024)
                    if not data:break
                    f.write(data);count+=len(data)
            digest=hashlib.sha256(tmp.read_bytes()).hexdigest()
            if (expected and digest!=expected) or (prefix and not digest.startswith(prefix)):raise RuntimeError('Weight digest mismatch; partial file retained')
            tmp.replace(p);print(item['name'],count,'bytes',round(time.monotonic()-start,2),'seconds',flush=True)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if (expected and digest!=expected) or (prefix and not digest.startswith(prefix)):raise RuntimeError('Existing checkpoint digest mismatch')
        item.update(sha256=digest,bytes=p.stat().st_size)
    with get(f'https://huggingface.co/{model}/resolve/{revision}/README.md') as r:card=r.read().decode('utf-8')
    (target/'METRIC_SMALL_MODEL_CARD.md').write_text(card,encoding='utf-8')
    manifest={'models':items,'torch':'2.5.1+cu124','torchvision':'0.20.1+cu124',
              'depth_source_repo':'https://github.com/DepthAnything/Video-Depth-Anything','depth_source_commit':'4f5ae23172ba60fd7bc11ef671cca678842c7072',
              'raft_license_note':'Torchvision source BSD-3-Clause; checkpoint/data and redistribution terms require separate review; local research use here',
              'nr_weights_not_relicensed':True}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (target/'.gitignore').write_text('*.pth\n*.partial\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--proxy',default='http://127.0.0.1:7890');run(p.parse_args().proxy)
