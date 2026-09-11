"""Batch/video convenience commands; the same run_manifest remains authoritative."""
import argparse,json,tempfile
from pathlib import Path


def main(argv=None,*,model=None):
    p=argparse.ArgumentParser(description='Local Torch whitebox image/video pipeline. No auto downloads.')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest',type=Path);source.add_argument('--video',type=Path);source.add_argument('--input-dir',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--weights',type=Path)
    p.add_argument('--config',type=Path,help='Base pipeline settings JSON for convenience commands')
    p.add_argument('--size',type=int,nargs=2,metavar=('HEIGHT','WIDTH'));p.add_argument('--nr-work-size',type=int,nargs=2)
    p.add_argument('--sequence',action='store_true',help='Input directory is a temporal sequence; default is independent photos')
    p.add_argument('--encoding',choices=['sRGB','linear','linear709_nits','PQ2020'])
    p.add_argument('--transfer',choices=['sRGB','BT709','PQ']);p.add_argument('--matrix',choices=['bt709','bt601','bt2020'])
    p.add_argument('--fps',type=float);p.add_argument('--max-frames',type=int);p.add_argument('--mp4',action='store_true',help='Also write silent opaque SDR preview.mp4')
    a=p.parse_args(argv)
    from .__main__ import run_manifest
    if a.manifest:
        if a.config:p.error('--config is only for convenience input commands')
        run_manifest(a.manifest,a.output,device=a.device,weights=a.weights,model=model);return
    spec=json.loads(a.config.read_text(encoding='utf-8')) if a.config else {}
    if 'frames' in spec or 'video' in spec:p.error('Base config must not contain input sources; use --manifest')
    spec.setdefault('mode','rgb_estimated')
    if spec['mode']!='rgb_estimated':p.error('Convenience commands require RGB-estimated mode; guided inputs use a manifest')
    if a.nr_work_size:spec['nr_work_size']=a.nr_work_size
    if a.size:spec['output_size']=a.size
    if a.video:
        from .media import probe_video
        v={'file':str(a.video.resolve())}
        for key in ('transfer','matrix','fps','max_frames'):
            value=getattr(a,key)
            if value is not None:v[key]=value
        info=probe_video(a.video,v);spec['video']=v
        spec['color_encoding']=a.encoding or spec.get('color_encoding',info['encoding'])
        spec.setdefault('output_size',[info['height'],info['width']])
    else:
        from PIL import Image
        import numpy as np
        files=sorted(f for f in a.input_dir.iterdir() if f.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp','.npy'))
        if a.max_frames:files=files[:a.max_frames]
        if not files:p.error('No supported images')
        spec['color_encoding']=a.encoding or spec.get('color_encoding','sRGB');frames=[];previous=None
        for f in files:
            if f.suffix.lower()=='.npy':shape=np.load(f,mmap_mode='r',allow_pickle=False).shape;hw=list(shape[:2])
            else:
                with Image.open(f) as im:hw=[im.height,im.width]
            item={'color':str(f.resolve()),'reset':not a.sequence or previous!=hw,'delta_ms':1000/(a.fps or 24)}
            if not a.size and 'output_size' not in spec:item['output_size']=hw
            frames.append(item);previous=hw
        spec.setdefault('output_size',frames[0].get('output_size',previous));spec['frames']=frames
        spec['batch_policy']='temporal with dimension-change reset' if a.sequence else 'independent images; reset each frame'
    if a.mp4:
        spec['video_output']={'file':'preview.mp4'}
        if a.fps:spec['video_output']['fps']=a.fps
        elif not a.video:p.error('Image sequence MP4 requires --fps')
    with tempfile.TemporaryDirectory(prefix='whitebox-cli-') as temp:
        path=Path(temp)/'manifest.json';path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
        run_manifest(path,a.output,device=a.device,weights=a.weights,model=model)
