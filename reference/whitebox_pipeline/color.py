"""Explicit SDR / absolute-nit HDR color bridge. Not a DFC/RenoDX codec.

HDR compressor: u=positive BT.709 nits/white, t=u/(1+max(u)), then sRGB.
Decode uses its analytic inverse with a declared finite peak. Original linear
context is restored by applying decoded DIFFERENCES, retaining out-of-gamut or
negative components and exact zero-effect identity instead of a codec roundtrip.
"""
from dataclasses import dataclass
import math
import torch


def linear_to_srgb(a):
    a=a.clamp_min(0)
    return torch.where(a<=.0031308,12.92*a,1.055*a.pow(1/2.4)-.055)
def srgb_to_linear(a):
    a=a.clamp_min(0)
    return torch.where(a<=.04045,a/12.92,((a+.055)/1.055).pow(2.4))
def matrix_rgb(a,m):
    return torch.einsum('ij,bjhw->bihw',a.new_tensor(m),a)
# Rounded D65 linear BT.2020/BT.709 conversion matrices.
M2020_709=((1.660491,-.587641,-.072850),(-.124550,1.132900,-.008349),(-.018151,-.100579,1.118730))
M709_2020=((.627404,.329283,.043313),(.069097,.919540,.011362),(.016391,.088013,.895595))

def pq_to_nits(a):
    m1=2610/16384;m2=2523/32;c1=3424/4096;c2=2413/128;c3=2392/128
    p=a.clamp(0,1).pow(1/m2)
    return 10000*((p-c1).clamp_min(0)/(c2-c3*p)).pow(1/m1)
def nits_to_pq(a):
    m1=2610/16384;m2=2523/32;c1=3424/4096;c2=2413/128;c3=2392/128
    p=(a.clamp(0,10000)/10000).pow(m1)
    return ((c1+c2*p)/(1+c3*p)).pow(m2)

def decode_input(a,encoding):
    if not bool(torch.isfinite(a).all()):raise ValueError('Nonfinite input color')
    if encoding in ('sRGB','linear'):
        if not bool(((a>=0)&(a<=1)).all()):raise ValueError('SDR input must be [0,1]')
        return srgb_to_linear(a) if encoding=='sRGB' else a
    if encoding=='linear709_nits':return a
    if encoding=='PQ2020':
        if not bool(((a>=0)&(a<=1)).all()):raise ValueError('PQ code values must be [0,1]')
        return matrix_rgb(pq_to_nits(a),M2020_709)
    raise ValueError('Unknown color encoding '+str(encoding))

@dataclass(frozen=True)
class ColorSettings:
    hdr: bool=False
    reference_white_nits: float=203.
    peak_nits: float=1000.
    output_encoding: str='sRGB'
    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in (self.reference_white_nits,self.peak_nits)) or self.peak_nits>10000:
            raise ValueError('White/peak must be positive finite nits; peak <=10000')
        allowed=('linear709_nits','PQ2020') if self.hdr else ('linear','sRGB')
        if self.output_encoding not in allowed:raise ValueError('Output encoding conflicts with HDR/SDR mode')

class ColorBridge:
    def __init__(self,settings=None):self.settings=settings or ColorSettings()
    def encode(self,context):
        if not self.settings.hdr:return linear_to_srgb(context).clamp(0,1)
        u=context.clamp_min(0)/self.settings.reference_white_nits
        return linear_to_srgb(u/(1+u.amax(1,keepdim=True)))
    def decode(self,encoded):
        t=srgb_to_linear(encoded.clamp(0,1))
        if not self.settings.hdr:return t
        # Bound the inverse at the explicit display/working peak, not epsilon.
        peak=self.settings.peak_nits/self.settings.reference_white_nits
        high=t.amax(1,keepdim=True)
        inverse=(1-high).clamp_min(1/(1+peak)).reciprocal()
        peak_scale=peak/torch.where(high>0,high,torch.ones_like(high))
        scale=torch.minimum(inverse,peak_scale)
        return t*scale*self.settings.reference_white_nits
    def restore(self,context,reference,modified):
        delta=self.decode(modified)-self.decode(reference)
        return context+delta
    def output(self,linear):
        enc=self.settings.output_encoding
        if enc in ('linear','linear709_nits'):return linear
        if enc=='sRGB':return linear_to_srgb(linear).clamp(0,1)
        # Negative/out-of-gamut components are retained in linear output but
        # necessarily clipped for the bounded PQ/BT.2020 representation.
        return nits_to_pq(matrix_rgb(linear,M709_2020))
