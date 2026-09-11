# Application execution fork of frozen whitebox pipeline, source 1fb9f4f.
# Reference remains byte-identical; backend metadata is explicit here.
"""Streaming guided or RGB-estimated SDR/HDR frame CLI; no runtime downloads."""
from runtime import bootstrap  # installs the frozen reference package path
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from whitebox_pipeline.pipeline import GuidedPipeline,GuidedSDRPipeline,NRSettings
from whitebox_pipeline.color import ColorSettings,decode_input,linear_to_srgb
from whitebox_pipeline.post import OutputSettings
from whitebox_pipeline.guides import GuideConfig
from whitebox_pipeline.nr_chain import FrameCancelled,check_cancel

def _plane(path,channels,device):
    path=Path(path)
    if path.suffix.lower()=='.npy':
        a=np.load(path,allow_pickle=False)
        if a.ndim==2:a=a[...,None]
        if a.ndim!=3 or a.shape[2]!=channels or not np.issubdtype(a.dtype,np.floating):raise ValueError(f'{path}: expected float HWC{channels}')
    else:
        if channels!=3:raise ValueError('Guide/mask inputs require float .npy')
        with Image.open(path) as im:
            if im.mode not in ('RGB','RGBA','L','LA','P'):raise ValueError('High-bit-depth input requires float .npy')
            rgba=im.convert('RGBA')
            if np.asarray(rgba.getchannel('A')).min()!=255:raise ValueError('Opaque plane reader does not discard alpha')
            a=np.asarray(rgba.convert('RGB'),dtype='f4')/255
    return torch.from_numpy(np.array(a,dtype='f4',copy=True)).permute(2,0,1)[None].to(device)

def _color_input(path,device,encoding,alpha_mode):
    path=Path(path)
    if path.suffix.lower()=='.npy':
        a=np.load(path,allow_pickle=False)
        if a.ndim!=3 or a.shape[2] not in (3,4) or not np.issubdtype(a.dtype,np.floating):raise ValueError('Color npy requires float HWC RGB/RGBA')
        value=torch.from_numpy(np.array(a,dtype='f4',copy=True)).permute(2,0,1)[None].to(device)
    else:
        if encoding not in ('sRGB','linear'):raise ValueError('HDR/PQ input requires float .npy, not 8-bit images')
        with Image.open(path) as im:
            if im.mode not in ('RGB','RGBA','L','LA','P'):raise ValueError('High-bit-depth image input requires float .npy')
            channels=4 if 'A' in im.getbands() or 'transparency' in im.info else 3
            a=np.asarray(im.convert('RGBA' if channels==4 else 'RGB'),dtype='f4')/255
        value=torch.from_numpy(a.copy()).permute(2,0,1)[None].to(device)
    c=value[:,:3];alpha=value[:,3:4] if value.shape[1]==4 else None
    if alpha is not None and (not bool(alpha.isfinite().all()) or not bool(((alpha>=0)&(alpha<=1)).all())):raise ValueError('Invalid alpha')
    if alpha_mode not in ('straight','premultiplied_linear'):raise ValueError('Unknown alpha mode')
    if alpha_mode=='premultiplied_linear':
        if encoding not in ('linear','linear709_nits'):raise ValueError('Premultiplication is supported only in explicit linear input domains')
        if alpha is None:raise ValueError('Premultiplied input requires alpha')
        denom=torch.where(alpha>0,alpha,torch.ones_like(alpha))
        c=torch.where(alpha>0,c/denom,torch.zeros_like(c))
    return c,alpha

def run_manifest(manifest,output,*,device='cuda',weights=None,model=None,cancel=None,progress=None,before_frame=None,quiet=False,compute_backend='PyTorch'):
    manifest=Path(manifest);output=Path(output);spec=json.loads(manifest.read_text(encoding='utf-8'))
    encoding=spec.get('color_encoding');mode=spec.get('mode','guided')
    if encoding not in ('linear','sRGB','linear709_nits','PQ2020'):raise ValueError('Explicit supported color_encoding required')
    hdr=encoding in ('linear709_nits','PQ2020')
    if mode not in ('guided','rgb_estimated'):raise ValueError('Unknown input mode')
    if mode=='guided' and spec.get('motion_convention')!='current_to_previous_pixels':raise ValueError('motion_convention must be current_to_previous_pixels')
    video_info=None;reader=None;writer=None;video_path=None;processed=0
    if spec.get('video'):
        if spec.get('frames'):raise ValueError('Use frames OR video')
        if mode!='rgb_estimated':raise ValueError('Video input requires explicit RGB-estimated mode')
        from whitebox_pipeline.media import probe_video,VideoFrames
        v=spec['video'];video_info=probe_video(manifest.parent/v['file'],v)
        if encoding!=video_info['encoding']:raise ValueError('video transfer and color_encoding disagree')
        spec.setdefault('output_size',[video_info['height'],video_info['width']])
        reader=VideoFrames(video_info,device,cancel)
        source_frames=reader
    else:
        if not spec.get('frames'):raise ValueError('Empty sequence')
        source_frames=spec['frames']
    def emit(event):
        if progress:progress(event)
        if not quiet:print(json.dumps(event),flush=True)
    if mode=='rgb_estimated':
        for f in list(spec.get('frames',[]))+list(spec.get('frame_events',{}).values()):
            if 'depth' in f or 'motion' in f:raise ValueError('RGB-estimated mode does not discard supplied guides')
            if tuple(f.get('jitter_xy',(0,0)))!=(0,0):raise ValueError('RGB-estimated mode requires zero jitter')
    cs=dict(spec.get('color_settings',{}))
    if 'hdr' in cs and bool(cs['hdr'])!=hdr:raise ValueError('Color settings conflict with input encoding')
    cs['hdr']=hdr;cs.setdefault('output_encoding','linear709_nits' if hdr else 'sRGB');colors=ColorSettings(**cs)
    passes=spec.get('nr_passes')
    if passes is not None and 'nr_settings' in spec:raise ValueError('Use nr_passes OR nr_settings')
    alpha_mode=spec.get('alpha_mode','straight');out_alpha_mode=spec.get('output_alpha_mode','straight')
    if out_alpha_mode not in ('straight','premultiplied_linear'):raise ValueError('Invalid output alpha mode')
    if out_alpha_mode=='premultiplied_linear' and colors.output_encoding not in ('linear','linear709_nits'):raise ValueError('Premultiplied output requires a linear encoding')
    output.mkdir(parents=True,exist_ok=True)
    if any(p.name!='.gitignore' for p in output.iterdir()):raise FileExistsError('Choose an empty output directory')
    (output/'run-manifest.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    emit({'stage':'loading_model','message':'Loading local models; no inference-time download'})
    try:
        common=dict(device=device,weights_path=weights,model=model,nr_settings=NRSettings(**spec.get('nr_settings',{})) if passes is None else None,
                    nr_passes=passes,nr_work_size=spec.get('nr_work_size'),color_settings=colors,output_settings=OutputSettings(**spec.get('output_settings',{})))
        if mode=='guided':
            ctor=GuidedPipeline if hdr else GuidedSDRPipeline
            proc=ctor(spec['output_size'],guide_config=GuideConfig(**spec.get('guide_settings',{'validate':False})),**common,**spec.get('fsr_settings',{}))
        else:
            from whitebox_pipeline.rgb_pipeline import RGBSequencePipeline
            options=dict(spec.get('estimator_settings',{}))
            if 'model_dir' in options:options['model_dir']=manifest.parent/options['model_dir']
            proc=RGBSequencePipeline(spec['output_size'],render_size=spec.get('render_size'),guide_config=GuideConfig(**spec.get('guide_settings',{})),**common,**options,**spec.get('fsr_settings',{}))
    except Exception as exc:
        (output/'report.json').write_text(json.dumps({'status':'failed','stage':'loading_model','mode':mode,'frames':[],'error':str(exc)},indent=2),encoding='utf-8');raise
    engine=proc if mode=='guided' else proc.guided
    provenance=getattr(proc,'provenance',{'guide_source':'provided'})
    root=manifest.parent;records=[];start=time.perf_counter()
    total=video_info['estimated_frames'] if video_info else len(spec['frames'])
    emit({'stage':'processing','processed_frames':0,'total_frames_estimate':total})
    log=(output/'frames.jsonl').open('w',encoding='utf-8')
    try:
        for index,f in enumerate(source_frames):
            check_cancel(cancel)
            frame_started=time.perf_counter()
            f={**f,**spec.get('frame_events',{}).get(str(index),{})}
            if before_frame:f={**f,**(before_frame(index) or {})}
            # Configuration events apply between frames. They invalidate their
            # affected histories immediately, even if the next frame fails.
            if 'output_size' in f:proc.resize(f['output_size'])
            if 'nr_passes' in f or 'nr_work_size' in f:proc.configure_nr(f.get('nr_passes',engine.chain.passes),f.get('nr_work_size'))
            if 'color_settings' in f:proc.set_color(replace(engine.bridge.settings,**f['color_settings']))
            if 'output_settings' in f:proc.set_output(replace(engine.output_settings,**f['output_settings']))
            if engine.bridge.settings.hdr!=hdr:raise ValueError('Changing HDR/SDR input mode requires a new manifest')
            if out_alpha_mode=='premultiplied_linear' and engine.bridge.settings.output_encoding not in ('linear','linear709_nits'):raise ValueError('Premultiplied output requires linear encoding')
            if '_color_tensor' in f:
                c=f['_color_tensor'];alpha=None
            else:c,alpha=_color_input(root/f['color'],device,encoding,alpha_mode)
            linear=decode_input(c,encoding)
            r=_plane(root/f['reactive'],1,device) if f.get('reactive') else None
            t=_plane(root/f['composition'],1,device) if f.get('composition') else None
            protect=_plane(root/f['protect'],1,device) if f.get('protect') else None
            tick=time.perf_counter();kwargs=dict(delta_ms=f.get('delta_ms',1000/60),reset=f.get('reset',False),reactive=r,composition=t,protect=protect,alpha=alpha,cancel=cancel)
            if mode=='guided':
                d=_plane(root/f['depth'],1,device);mv=_plane(root/f['motion'],2,device)
                result=proc.process_frame(linear,d,mv,jitter_xy=tuple(f.get('jitter_xy',(0,0))),**kwargs)
            else:
                source=linear if hdr else c if encoding=='sRGB' else linear_to_srgb(linear)
                result=proc.process_frame(source,**kwargs)
            process_returned=time.perf_counter()
            color=result['color'];a=result.get('alpha')
            raw=color if a is None or out_alpha_mode=='straight' else color*a
            data=raw if a is None else torch.cat((raw,a),1)
            data_file=None
            if hdr or result['info']['color_encoding']=='linear' or spec.get('save_float',False):
                data_file=f'{index:06}.npy';np.save(output/data_file,data[0].permute(1,2,0).cpu().numpy())
            preview=engine.bridge.encode(result['linear_color']) if hdr else linear_to_srgb(result['linear_color']).clamp(0,1)
            if not hdr and result['info']['color_encoding']=='sRGB':preview=color
            if a is not None:preview=torch.cat((preview,a),1)
            image=(preview[0].permute(1,2,0).cpu().numpy().clip(0,1)*255).round().astype('u1')
            filename=f'{index:06}.png';Image.fromarray(image).save(output/filename)
            if spec.get('video_output'):
                if writer is None:
                    from whitebox_pipeline.media import PreviewVideo
                    vo=spec['video_output'];name=vo.get('file','preview.mp4')
                    if Path(name).name!=name:raise ValueError('Video output must be a filename inside the output directory')
                    video_path=output/name
                    fps=vo.get('fps',video_info['fps'] if video_info else None)
                    if fps is None:raise ValueError('video_output.fps required for image sequences')
                    writer=PreviewVideo(video_path,image.shape[:2],fps,cancel)
                writer.write(image)
            if spec.get('save_guides',False):
                g=result['guides'];np.savez_compressed(output/f'{index:06}-guides.npz',**{k:g[k].cpu().numpy() for k in ('motion','distrust','tests','geometry_decision')})
            frame_finished=time.perf_counter()
            record={**result['info'],'compute_backend':compute_backend,'input_index':index,'file':filename,'data_file':data_file,
                    'png_is_sdr_preview':hdr,'output_alpha_mode':out_alpha_mode,'seconds':frame_finished-tick,
                    'input_to_output_seconds':frame_finished-frame_started,
                    'host_stage_seconds':{'input_and_configuration':tick-frame_started,
                                          'pipeline_call':process_returned-tick,
                                          'output_readback_and_export':frame_finished-process_returned},
                    'timing_scope':'Sequential host wall intervals, including any GPU waits; not GPU busy. Includes image decode/upload and output export. Video iterator decode before this frame and job queue/model loading are excluded.'};records.append(record)
            processed=index+1
            log.write(json.dumps(record)+'\n');log.flush()
            if len(records)>100:del records[:-100]
            emit({'stage':'processing','processed_frames':processed,'total_frames_estimate':total,'last_frame':record})
            report={'status':'running','frames':records,'processed_frames':processed,'frame_records':'frames.jsonl','seconds':time.perf_counter()-start,'source_manifest':str(manifest.resolve()),
                    'all_image_compute':compute_backend+'; fixed NR; no native reference fallback','mode':mode,'guide_provenance':provenance,'video_source':video_info,
                    'video_output':{'file':video_path.name,'meaning':'silent opaque SDR preview, not HDR master'} if video_path else None}
            tmp=output/'report.tmp';tmp.write_text(json.dumps(report,indent=2),encoding='utf-8');tmp.replace(output/'report.json')
        if not processed:raise ValueError('No frames decoded')
        if writer:writer.close();writer=None
    except (Exception,KeyboardInterrupt) as exc:
        status='cancelled' if isinstance(exc,(FrameCancelled,KeyboardInterrupt)) or (cancel and cancel()) else 'failed'
        if writer:
            try:writer.close(abort=True)
            finally:writer=None
        if video_path and video_path.exists():video_path.unlink()
        (output/'report.json').write_text(json.dumps({'status':status,'frames':records,'processed_frames':processed,'frame_records':'frames.jsonl','error':str(exc)},indent=2),encoding='utf-8')
        emit({'stage':status,'processed_frames':processed,'error':str(exc)});raise
    finally:
        log.close()
        if reader:reader.close()
    report['status']='completed';report['seconds']=time.perf_counter()-start
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    emit({'stage':'completed','processed_frames':processed,'seconds':report['seconds']})
    return proc,report
