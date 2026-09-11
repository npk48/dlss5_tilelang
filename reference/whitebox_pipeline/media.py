"""Local FFmpeg codecs, bounded frame streaming. No network media inputs.
Image/learned processing stays in Torch. Encoded video output is explicitly an
opaque, silent SDR preview; HDR truth remains float NPY with encoding metadata.
"""
from pathlib import Path
from fractions import Fraction
import collections,json,os,shutil,subprocess,threading,time
import numpy as np

class MediaError(RuntimeError):pass

def executables():
    ff=os.environ.get('WHITEBOX_FFMPEG') or shutil.which('ffmpeg')
    if not ff:
        try:
            import imageio_ffmpeg
            ff=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:pass
    if not ff:raise MediaError('Install FFmpeg and ffprobe on PATH, or set WHITEBOX_FFMPEG / WHITEBOX_FFPROBE')
    probe=os.environ.get('WHITEBOX_FFPROBE') or shutil.which('ffprobe')
    sibling=Path(ff).with_name('ffprobe.exe' if os.name=='nt' else 'ffprobe')
    if not probe and sibling.is_file():probe=str(sibling)
    if not probe:raise MediaError('ffprobe is required beside FFmpeg or on PATH')
    return str(ff),str(probe)

def probe_video(path,options):
    path=Path(path).resolve()
    if not path.is_file():raise MediaError('Video must be an existing local file')
    ff,probe=executables()
    p=subprocess.run([probe,'-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,color_transfer,color_space,color_range,color_primaries,pix_fmt','-of','json',str(path)],capture_output=True,text=True,timeout=30)
    if p.returncode:raise MediaError(p.stderr.strip())
    streams=json.loads(p.stdout).get('streams',[])
    if not streams:raise MediaError('No video stream')
    s=streams[0];fps_text=s.get('avg_frame_rate','0/0')
    if fps_text in ('0/0','0/1'):fps_text=s.get('r_frame_rate','0/0')
    try:fps=float(Fraction(options.get('fps',fps_text)))
    except (ValueError,ZeroDivisionError):raise MediaError('Missing frame rate; specify video.fps')
    if not 0<fps<=240:raise MediaError('Frame rate must be in (0,240]')
    transfer=options.get('transfer') or {'bt709':'BT709','iec61966-2-1':'sRGB','smpte2084':'PQ'}.get(s.get('color_transfer'))
    if transfer not in ('BT709','sRGB','PQ'):raise MediaError('Missing/unsupported transfer; explicitly set video.transfer to BT709, sRGB or PQ')
    matrix=options.get('matrix') or {'bt709':'bt709','bt2020nc':'bt2020','bt2020c':'bt2020','smpte170m':'bt601','bt470bg':'bt601'}.get(s.get('color_space'))
    if matrix not in ('bt709','bt601','bt2020'):raise MediaError('Missing/unsupported YUV matrix; explicitly set video.matrix')
    if transfer=='PQ' and (matrix!='bt2020' or s.get('color_primaries') not in (None,'bt2020')):raise MediaError('PQ video requires declared BT.2020 primaries/matrix')
    if transfer!='PQ' and s.get('color_primaries') not in (None,'bt709'):raise MediaError('Unsupported SDR video primaries')
    fmt=s.get('pix_fmt','')
    if any(t in fmt for t in ('yuva','rgba','bgra','argb','abgr','gbrap')):raise MediaError('Video alpha is not discarded: use an RGBA frame sequence instead')
    count=int(s['nb_frames']) if s.get('nb_frames','').isdigit() else None
    limit=options.get('max_frames')
    if limit is not None and (type(limit) is not int or limit<1):raise MediaError('max_frames must be a positive integer')
    if limit:count=min(count,limit) if count is not None else limit
    return {'path':str(path),'width':int(s['width']),'height':int(s['height']),'fps':fps,'estimated_frames':count,
            'transfer':transfer,'matrix':matrix,'range':options.get('range',s.get('color_range','tv')),
            'source_tags':s,'timing':'decoded at declared constant FPS using FFmpeg fps filter','explicit_assumptions':{k:options[k] for k in ('transfer','matrix','range','fps') if k in options},
            'encoding':'PQ2020' if transfer=='PQ' else 'sRGB','ffmpeg':ff,'max_frames':limit}

class Process:
    def __init__(self,args,cancel=None,read=False):
        self.errors=collections.deque(maxlen=80);self.stop=threading.Event();self.cancel=cancel
        self.p=subprocess.Popen(args,stdin=subprocess.DEVNULL if read else subprocess.PIPE,
            stdout=subprocess.PIPE if read else subprocess.DEVNULL,stderr=subprocess.PIPE)
        def drain():
            for line in iter(self.p.stderr.readline,b''):self.errors.append(line.decode('utf-8','replace').rstrip())
        self.drain=threading.Thread(target=drain,daemon=True);self.drain.start()
        def monitor():
            while not self.stop.wait(.1):
                if self.cancel and self.cancel():
                    if self.p.poll() is None:self.p.terminate()
                    return
        self.monitor=threading.Thread(target=monitor,daemon=True);self.monitor.start()
    def close(self,abort=False):
        self.stop.set()
        if abort and self.p.poll() is None:self.p.terminate()
        if self.p.stdin and not self.p.stdin.closed:
            try:self.p.stdin.close()
            except OSError:pass
        try:code=self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:self.p.kill();self.p.wait();raise MediaError('FFmpeg did not terminate')
        self.drain.join(timeout=2)
        if self.p.stdout:self.p.stdout.close()
        if self.p.stderr:self.p.stderr.close()
        if code and not abort and not (self.cancel and self.cancel()):raise MediaError('FFmpeg: '+'\n'.join(self.errors))

class VideoFrames:
    def __init__(self,info,device,cancel=None):self.info=info;self.device=device;self.cancel=cancel;self.process=None
    def __iter__(self):
        import torch
        from .color import linear_to_srgb
        from .nr_chain import check_cancel
        i=self.info;h,w=i['height'],i['width'];pq=i['transfer']=='PQ'
        fmt='gbrpf32le' if pq else 'rgb24';vrange='pc' if i['range'] in ('pc','full','jpeg') else 'tv'
        args=[i['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-i',i['path'],'-an','-sn','-dn',
              '-vf',f"fps={i['fps']},scale=in_color_matrix={i['matrix']}:in_range={vrange}:out_range=pc",'-fps_mode','passthrough']
        if i['max_frames']:args+=['-frames:v',str(i['max_frames'])]
        args+=['-pix_fmt',fmt,'-f','rawvideo','pipe:1']
        self.process=Process(args,self.cancel,read=True);n=h*w*3*(4 if pq else 1);index=0
        try:
            while True:
                check_cancel(self.cancel);parts=[];remaining=n
                while remaining:
                    b=self.process.p.stdout.read(remaining)
                    if not b:break
                    parts.append(b);remaining-=len(b)
                check_cancel(self.cancel)
                if not parts:break
                if remaining:raise MediaError('Truncated decoded frame')
                raw=b''.join(parts)
                if pq:
                    a=np.frombuffer(raw,dtype='<f4').reshape(3,h,w)[[2,0,1]].copy()
                    c=torch.from_numpy(a)[None].to(self.device).clamp(0,1)
                else:
                    a=np.frombuffer(raw,dtype='u1').reshape(h,w,3).copy()
                    c=torch.from_numpy(a).permute(2,0,1)[None].to(self.device,dtype=torch.float32)/255
                    if i['transfer']=='BT709':
                        linear=torch.where(c<.081,c/4.5,((c+.099)/1.099).pow(1/.45))
                        c=linear_to_srgb(linear).clamp(0,1)
                yield {'_color_tensor':c,'delta_ms':1000/i['fps'],'reset':index==0,'source_index':index}
                index+=1
            self.process.close();self.process=None
        finally:self.close()
    def close(self):
        if self.process:self.process.close(abort=True);self.process=None

class PreviewVideo:
    def __init__(self,path,size,fps,cancel=None):
        ff,_=executables();h,w=size
        if h%2 or w%2:raise MediaError('MP4 preview requires even output axes; float/PNG output has no such restriction')
        self.size=tuple(size);self.process=Process([ff,'-hide_banner','-loglevel','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-an','-vf','scale=out_color_matrix=bt709:out_range=tv,setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
            '-color_primaries','bt709','-color_trc','iec61966-2-1','-colorspace','bt709','-movflags','+faststart',str(path)],cancel)
    def write(self,image):
        if tuple(image.shape[:2])!=self.size:raise MediaError('Fixed-size MP4 cannot follow dynamic target resize; export frames instead')
        if image.shape[-1]==4:raise MediaError('MP4 preview does not silently discard alpha; export RGBA frames instead')
        self.process.p.stdin.write(np.ascontiguousarray(image).tobytes())
    def close(self,abort=False):
        if self.process:
            process=self.process;self.process=None;process.close(abort)
