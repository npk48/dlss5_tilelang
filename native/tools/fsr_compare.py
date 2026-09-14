"""Compare native CUDA FSR to original Python FSR on a nondefault CUDA stream."""
import ctypes as ct
import json
import pathlib
import sys
import torch
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'reference'))
from whitebox_pipeline.fsr2 import FSR2
lib=ct.CDLL(str(root/'native/reconstruction/build/Release/fsr_test_bridge.dll'))
ptr=ct.c_void_p
lib.fsr_create.argtypes=[ct.c_int]*4+[ct.c_float,ct.c_int,ct.c_float];lib.fsr_create.restype=ptr
lib.fsr_process.argtypes=[ptr]*9+[ct.c_float]*2;lib.fsr_process.restype=ct.c_int
lib.fsr_reset.argtypes=[ptr];lib.fsr_destroy.argtypes=[ptr]
lib.fsr_copy_motion.argtypes=[ptr,ptr,ct.c_int,ptr]
lib.fsr_copy_resource.argtypes=[ptr,ct.c_uint,ptr,ct.c_int,ptr]
lib.fsr_normalize.argtypes=[ptr]*3+[ct.c_int]*2+[ct.c_float]*2+[ptr]
def p(t):return None if t is None else t.data_ptr()
def hwc(t):return t[0].permute(1,2,0).contiguous()
results=[]
stream=torch.cuda.Stream()
with torch.cuda.stream(stream):
 for h,w,oh,ow,far,encoding,sharpness in [(48,64,72,96,100.,1,0.),(65,97,97,145,1000.,1,0.),(64,96,64,96,100.,1,0.),(64,96,96,144,1000.,0,.6),(360,640,720,1280,1000.,1,0.)]:
  engine=lib.fsr_create(w,h,ow,oh,far,encoding,sharpness);assert engine
  ref=FSR2((oh,ow),camera_far=far)
  yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
  try:
   for frame in range(18):
    x=(xx-frame*.7)/w;y=yy/h
    c=torch.stack((.2+.15*torch.sin(x*35),.3+.2*torch.cos(y*27),.25+.15*torch.sin(x*19+y*31)),0)[None]
    obj=(xx>12+frame)&(xx<30+frame)&(yy>12)&(yy<34)
    c=torch.where(obj[None,None],c*.4+.4,c)
    metric=torch.where(obj,torch.tensor(2.,device='cuda'),torch.tensor(8.,device='cuda'))[None,None]
    mv=torch.zeros((1,2,h,w),device='cuda');mv[:,0]=-.7 if 0<frame<8 else 0;mv[:,1]=.13 if 0<frame<8 else 0
    reactive=torch.zeros_like(metric);composition=torch.zeros_like(metric)
    if frame in (2,3):reactive[:,:,15:25,20:30]=.7;composition[:,:,30:40,35:45]=.4
    jx,jy=((0.,0.) if frame==0 else ((frame%3-1)*.23,(frame%2-.5)*.31))
    if frame==14:ref.reset();lib.fsr_reset(engine)
    a,b=ref._depth_factors(metric);d=(a+b/metric.clamp(.1,far)).clamp(0,1)
    encoded=torch.where(c<=.0031308,c*12.92,1.055*c.pow(1/2.4)-.055) if encoding==0 else c
    linear=torch.where(encoded<=.04045,encoded/12.92,((encoded+.055)/1.055).pow(2.4)) if encoding==0 else c
    expected=ref.process(linear,d,mv,reactive=reactive,composition=composition,jitter_xy=(jx,jy),sharpness=sharpness)
    if encoding==0:expected=torch.where(expected<=.0031308,expected*12.92,1.055*expected.pow(1/2.4)-.055)
    ec=(1-ref.diagnostics['depth_clip'])*(1-ref.diagnostics['reactivity'])
    C,D,M,R,T=map(hwc,(encoded,metric,mv,reactive,composition));output=torch.empty((oh,ow,3),device='cuda');conf=torch.empty((oh,ow,1),device='cuda')
    if frame==10:
     bad=C.clone();bad[0,0,0]=float('nan')
     assert lib.fsr_process(engine,p(bad),p(D),p(M),p(R),p(T),p(output),p(conf),stream.cuda_stream,jx,jy)==1
    rc=lib.fsr_process(engine,p(C),p(D),p(M),p(R),p(T),p(output),p(conf),stream.cuda_stream,jx,jy)
    assert rc==0,rc
    pixels=torch.empty_like(M);assert lib.fsr_copy_motion(engine,p(pixels),h*w,stream.cuda_stream)==0
    assert torch.equal(pixels,ref.state.dilated_motion*torch.tensor([w,h],device='cuda'))
    normalized=torch.empty_like(D);invalid=torch.empty_like(D)
    lib.fsr_normalize(p(D),p(normalized),p(invalid),w,h,.1,far,stream.cuda_stream)
    assert torch.allclose(normalized,hwc(ref._z(d)/ref._z(d.new_tensor(1.))),atol=1e-6)
    assert torch.equal(invalid,torch.zeros_like(invalid))
    diff=(output-hwc(expected)).abs();cd=(conf-ec).abs()
    result=dict(shape=[h,w,oh,ow],frame=frame,color_max=diff.max().item(),color_mean=diff.mean().item(),confidence_max=cd.max().item(),confidence_mean=cd.mean().item())
    states={}
    for slot,expected_state in enumerate([ref.state.color,ref.state.locks,ref.state.luma_history,ref.diagnostics['prepared'],ref.diagnostics['luma_mip4'],ref.diagnostics['new_locks']]):
     state=torch.empty_like(expected_state);assert lib.fsr_copy_resource(engine,slot,p(state),state.numel(),stream.cuda_stream)==0
     states[slot]=state
     error=(state-expected_state).abs();result['state'+str(slot)+'_max']=error.max().item()
    if result['color_max']>.003:
     pixel=diff.amax(-1).argmax().item();y0,x0=divmod(pixel,ow)
     print('MISMATCH',frame,y0,x0,'color',output[y0,x0].tolist(),hwc(expected)[y0,x0].tolist(),'locks',states[1][y0,x0].tolist(),ref.state.locks[y0,x0].tolist(),'luma',states[2][y0,x0].tolist(),ref.state.luma_history[y0,x0].tolist(),flush=True)
    results.append(result);print(result,flush=True)
    assert torch.isfinite(output).all() and torch.isfinite(conf).all()
  finally:lib.fsr_destroy(engine)
print(json.dumps({'stream':stream.cuda_stream,'results':results},indent=2))
assert max(x['color_max'] for x in results)<.001
assert max(x['confidence_max'] for x in results)<2e-6
assert max(x['state2_max'] for x in results)==0
