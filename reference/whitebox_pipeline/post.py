"""Declared output-only effects. No effect is fed into any temporal history."""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from .fsr2 import half as half_rtz,ycocg,rgb

@dataclass(frozen=True)
class OutputSettings:
    mix: float=1.
    nr_resample: str='bilinear'
    sharpness: float|None=None
    detail_only: bool=False
    max_luma_stops: float|None=None
    max_chroma_delta: float|None=None  # normalized by reference white in HDR
    def __post_init__(self):
        if self.nr_resample not in ('bilinear','easu'):raise ValueError('NR resampling must be bilinear or easu')
        if not math.isfinite(self.mix) or not 0<=self.mix<=1:raise ValueError('Output mix must be [0,1]')
        if self.sharpness is not None and not 0<=self.sharpness<=1:raise ValueError('Sharpness must be [0,1] or None')
        for x in (self.max_luma_stops,self.max_chroma_delta):
            if x is not None and (not math.isfinite(x) or x<0):raise ValueError('Guard bounds must be finite nonnegative')

def composite(context,reference,encoded,bridge,fsr,settings,protect=None,alpha=None):
    if settings.sharpness is not None:
        hwc=half_rtz(encoded)[0].permute(1,2,0)
        encoded=fsr._rcas(torch.cat((hwc,torch.zeros_like(hwc[...,:1])),-1),1.,1.,settings.sharpness).permute(2,0,1)[None].clamp(0,1)
    candidate=bridge.restore(context,reference,encoded)
    delta=candidate-context
    if settings.detail_only:
        low=F.avg_pool2d(F.pad(delta,(1,1,1,1),mode='replicate'),3,stride=1)
        delta=delta-low
    if settings.max_chroma_delta is not None:
        scale=bridge.settings.reference_white_nits if bridge.settings.hdr else 1.
        ycc=ycocg((delta/scale).permute(0,2,3,1))
        ycc=torch.cat((ycc[...,:1],ycc[...,1:].clamp(-settings.max_chroma_delta,settings.max_chroma_delta)),-1)
        delta=rgb(ycc).permute(0,3,1,2)*scale
    if settings.max_luma_stops is not None:
        # Explicit black floor: one 16-bit fraction of reference white. This is
        # part of this guard, not an inserted epsilon in the learned algorithm.
        white=bridge.settings.reference_white_nits if bridge.settings.hdr else 1.
        floor=white/65536
        coeff=context.new_tensor([.2126,.7152,.0722])[None,:,None,None]
        y0=(context*coeff).sum(1,keepdim=True).clamp_min(floor)
        y1=((context+delta)*coeff).sum(1,keepdim=True).clamp_min(floor)
        limit=settings.max_luma_stops
        target=y0*torch.exp2(torch.log2(y1/y0).clamp(-limit,limit))
        # Correct only luminance along neutral RGB; preserve zero delta exactly.
        correction=torch.where(delta.abs().amax(1,keepdim=True)>0,target-y1,torch.zeros_like(y1))
        delta=delta+correction
    strength=torch.full_like(context[:,:1],settings.mix)
    if protect is not None:strength*=1-protect
    if alpha is not None:strength*=alpha>0
    result=context+delta*strength
    # Explicit identity/protection bypass, no roundtrip loss or 0*nonfinite.
    return torch.where(strength==0,context,result)
