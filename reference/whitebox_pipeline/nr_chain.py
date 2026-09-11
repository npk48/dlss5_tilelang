"""Frozen NR multi-pass host, one immutable history per pass; no graph edits."""
from dataclasses import dataclass,asdict
import math
import torch
import torch.nn.functional as F
from .fsr2 import half as half_rtz

@dataclass(frozen=True)
class NRSettings:
    style: int=1
    structure: float=2.
    tone: float=1.
    skin: float=-1.
    automatic_mask: bool=False
    temporal_strength: float=1.
    intensity: float=1.  # output mix, not learned-head gain
    def __post_init__(self):
        if not 0<=self.style<=255 or int(self.style)!=self.style:raise ValueError('Invalid style byte')
        if not all(math.isfinite(x) for x in (self.structure,self.tone,self.skin,self.temporal_strength,self.intensity)):
            raise ValueError('Nonfinite NR setting')
        if not 0<=self.temporal_strength<=1 or not 0<=self.intensity<=1:raise ValueError('NR temporal/output mixes must be [0,1]')

@dataclass(frozen=True)
class PassState:
    image: torch.Tensor
    frame_index: int

class FrameCancelled(RuntimeError):pass

def check_cancel(cancel):
    if cancel is not None and cancel():raise FrameCancelled('Frame cancelled before state commit')

class NRChain:
    def __init__(self,model,output_size,passes=None,work_size=None):
        import dlss5_model as nr
        self.nr=nr;self.model=model;self.output_size=tuple(output_size)
        self.passes=();self.states=();self.work_size=None
        self.configure((NRSettings(),) if passes is None else passes,work_size or output_size)
    def configure(self,passes,work_size=None):
        passes=tuple(p if isinstance(p,NRSettings) else NRSettings(**p) for p in passes)
        if not 1<=len(passes)<=30:raise ValueError('NR needs 1..30 passes; performance is not guaranteed')
        size=tuple(map(int,work_size or self.work_size or self.output_size))
        if len(size)!=2 or min(size)<32:raise ValueError('NR work axes must be >=32')
        prefix=0
        if size==self.work_size:
            for a,b in zip(self.passes,passes):
                if a!=b:break
                prefix+=1
        self.states=self.states[:prefix]+(None,)*(len(passes)-prefix)
        self.passes=passes;self.work_size=size
        h,w=size;self.neural_size=(max(320,math.ceil(h/64)*64),max(320,math.ceil((w+64)/64)*64))
        return prefix
    def reset(self):self.states=(None,)*len(self.passes)
    @torch.inference_mode()
    def propose(self,encoded,motion_uv,confidence,*,reset=False,cancel=None,resample='bilinear'):
        check_cancel(cancel);nr=self.nr;oh,ow=self.output_size;h,w=self.work_size;nh,nw=self.neural_size
        if resample not in ('bilinear','easu'):raise ValueError('Unknown NR resampling filter')
        if resample=='easu' and (h>oh or w>ow):raise ValueError('EASU requires NR work <= output; use bilinear to downsample')
        reference=encoded
        work=encoded if (h,w)==(oh,ow) else F.interpolate(encoded,size=(h,w),mode='bilinear',align_corners=False,antialias=True)
        base=half_rtz(work);work=base
        rh,rw=motion_uv.shape[:2];device=encoded.device
        y=torch.floor((torch.arange(h,device=device)+.5)*rh/h).long().clamp_max(rh-1)
        x=torch.floor((torch.arange(w,device=device)+.5)*rw/w).long().clamp_max(rw-1)
        pixels=motion_uv[y[:,None],x[None,:]]*motion_uv.new_tensor([w,h])
        confidence=confidence.permute(2,0,1)[None]
        if (h,w)!=(oh,ow):confidence=F.interpolate(confidence,size=(h,w),mode='bilinear',align_corners=False)
        conf=confidence[0].permute(1,2,0)
        yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        visible=((xx+.5+pixels[...,0]>=0)&(xx+.5+pixels[...,0]<=w)&(yy+.5+pixels[...,1]>=0)&(yy+.5+pixels[...,1]<=h))[...,None]
        states=[];records=[]
        for index,settings in enumerate(self.passes):
            check_cancel(cancel)
            old=None if reset else self.states[index]
            current=nr.pad_color_for_neural_buffer(half_rtz(work).permute(0,2,3,1).contiguous(),nh,nw)
            cold=old is None or settings.temporal_strength==0
            previous=current if cold else old.image
            mv=current.new_zeros(1,nh,nw,2);gate=current.new_zeros(1,nh,nw,1)
            if not cold:
                mv[0,:h,:w]=pixels;gate[0,:h,:w]=visible*conf*settings.temporal_strength
            frame=0 if old is None else old.frame_index+1
            rt=nr.Block70RuntimeInputs(color=current,prev_output=previous,mvec=mv,mvec_scale_xy=current.new_ones(2),
                output_dimensions_wh=current.new_tensor([nw,nh]),style=settings.style,local_structure_strength=settings.structure,
                local_tone_strength=settings.tone,skin_structure_strength=settings.skin,use_auto_mask=settings.automatic_mask)
            packet=nr.build_preblock_features(rt,frame=frame)
            packet[:,7:10]=torch.where((gate>0).permute(0,3,1,2),packet[:,7:10],packet[:,4:7])
            head=self.model.infer_minimal(packet)
            result=nr.reconstruct_block70_color(head[...,:3],head[...,3:4],rt,gate)
            if not bool(result.isfinite().all()):raise FloatingPointError('Nonfinite NR pass '+str(index))
            if settings.intensity!=1:result=current+settings.intensity*(result-current)
            states.append(PassState(half_rtz(result).detach(),frame))
            work=result[:,:h,:w].permute(0,3,1,2).contiguous()
            records.append({'pass':index,'frame_index':frame,'cold_nr':cold,'head_abs_mean':float(head[...,:3].abs().mean()),
                            'history_gate_mean':float(gate[:,:h,:w].mean()),'intensity':settings.intensity,'settings':asdict(settings)})
        if (h,w)==(oh,ow):image=work
        else:
            # Explicit encoded-domain residual transport, NOT full-size NR or
            # tiling equivalence. A zero pass effect preserves full-res context.
            if resample=='easu':
                from .spatial import easu
                delta=easu(work,(oh,ow))-easu(base,(oh,ow))
            else:delta=F.interpolate(work-base,size=(oh,ow),mode='bilinear',align_corners=False)
            image=(reference+delta).clamp(0,1)
        check_cancel(cancel)
        return image,tuple(states),records
