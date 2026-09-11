"""Complete NRChain candidate; unchanged Gaussian and texture math boundaries."""
from runtime import bootstrap
from dataclasses import asdict
import torch,torch.nn.functional as F,tilelang
import tilelang.language as T
from runtime.device import TARGET,CONFIG,EXECUTION_BACKEND
from pipeline.fsr.depth_clip import half_rtz,mx,mn
from pipeline.fsr.accumulate import recip
from whitebox_pipeline.nr_chain import NRChain,PassState,check_cancel
from whitebox_pipeline.fsr2 import half as torch_rtz
REFERENCE_PROPOSE=NRChain.propose

def clamp01(x):return T.if_then_else(x!=x,x,mn(mx(x,T.float32(0)),T.float32(1)))

@tilelang.jit(out_idx=[3,4,5],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def prepare_kernel(H,W,NH,NW,RH,RW,Cold,Strength):
 @T.prim_func
 def main(Work:T.Tensor((1,3,H,W),T.float32),Motion:T.Tensor((RH,RW,2),T.float32),Confidence:T.Tensor((H,W,1),T.float32),Current:T.Tensor((1,NH,NW,3),T.float32),MV:T.Tensor((1,NH,NW,2),T.float32),Gate:T.Tensor((1,NH,NW,1),T.float32)):
  with T.Kernel(T.ceildiv(NH*NW,128),threads=128) as bx:
   index=bx*128+T.get_thread_binding(0)
   if index<NH*NW:
    y=index//NW;x=index%NW;sy=T.max(0,T.min(T.if_then_else(y<H,y,2*H-2-y),H-1));sx=T.max(0,T.min(T.if_then_else(x<W,x,2*W-2-x),W-1))
    for k in T.unroll(3):Current[0,y,x,k]=half_rtz(Work[0,k,sy,sx])
    if not Cold:
     if T.And(y<H,x<W):
      iy=T.min(T.cast(T.floor(((T.cast(y,T.float32)+T.float32(.5))*T.float32(RH))*T.float32(recip(H))),T.int32),RH-1)
      ix=T.min(T.cast(T.floor(((T.cast(x,T.float32)+T.float32(.5))*T.float32(RW))*T.float32(recip(W))),T.int32),RW-1)
      dx=Motion[iy,ix,0]*T.float32(W);dy=Motion[iy,ix,1]*T.float32(H)
      px=T.cast(x,T.float32)+T.float32(.5)+dx;py=T.cast(y,T.float32)+T.float32(.5)+dy
      visible=T.And(T.And(px>=T.float32(0),px<=T.float32(W)),T.And(py>=T.float32(0),py<=T.float32(H)))
      MV[0,y,x,0]=dx;MV[0,y,x,1]=dy;Gate[0,y,x,0]=(T.cast(visible,T.float32)*Confidence[y,x,0])*T.float32(Strength)
     else:MV[0,y,x,0]=T.float32(0);MV[0,y,x,1]=T.float32(0);Gate[0,y,x,0]=T.float32(0)
    else:MV[0,y,x,0]=T.float32(0);MV[0,y,x,1]=T.float32(0);Gate[0,y,x,0]=T.float32(0)
 return main

@tilelang.jit(out_idx=[5],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def packet_kernel(H,W):
 @T.prim_func
 def main(Current:T.Tensor((1,H,W,3),T.float32),Previous:T.Tensor((1,H,W,3),T.float32),Gaussian:T.Tensor((1,H,W,3),T.float32),Gate:T.Tensor((1,H,W,1),T.float32),Settings:T.Tensor((5,),T.float32),Out:T.Tensor((1,16,H,W),T.float32)):
  with T.Kernel(T.ceildiv(H*W,128),threads=128) as bx:
   i=bx*128+T.get_thread_binding(0)
   if i<H*W:
    y=i//W;x=i%W
    Out[0,0,y,x]=Gaussian[0,y,x,1];Out[0,1,y,x]=Gaussian[0,y,x,2];Out[0,2,y,x]=Gaussian[0,y,x,0];Out[0,3,y,x]=T.float32(1)
    for k in T.unroll(3):
     current=T.cast(T.cast(T.cast(Current[0,y,x,k],T.float16)-T.float16(.5),T.float16)*T.float16(.125),T.float16)
     prev=T.cast(T.cast(T.cast(Previous[0,y,x,k],T.float16)-T.float16(.5),T.float16)*T.float16(.125),T.float16)
     Out[0,k+4,y,x]=T.cast(current,T.float32);Out[0,k+7,y,x]=T.if_then_else(Gate[0,y,x,0]>T.float32(0),T.cast(prev,T.float32),T.cast(current,T.float32))
    for k in T.unroll(5):Out[0,k+10,y,x]=Settings[k]
    Out[0,15,y,x]=T.float32(0)
 return main

@tilelang.jit(out_idx=[5],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def output_kernel(H,W):
 @T.prim_func
 def main(Current:T.Tensor((1,H,W,3),T.float32),Previous:T.Tensor((1,H,W,3),T.float32),Residual:T.Tensor((1,H,W,3),T.float32),Sigmoid:T.Tensor((1,H,W,1),T.float32),Gate:T.Tensor((1,H,W,1),T.float32),Out:T.Tensor((1,H,W,3),T.float32)):
  with T.Kernel(T.ceildiv(H*W,128),threads=128) as bx:
   i=bx*128+T.get_thread_binding(0)
   if i<H*W:
    y=i//W;x=i%W;alpha=clamp01(Sigmoid[0,y,x,0]*clamp01(Gate[0,y,x,0]))
    for k in T.unroll(3):
     centered=Current[0,y,x,k]*T.float32(.125)-T.float32(.0625)
     corrected=centered+Residual[0,y,x,k]*T.float32(.03125)
     corrected=clamp01(corrected*T.float32(8)+T.float32(.5))
     Out[0,y,x,k]=corrected+alpha*(Previous[0,y,x,k]-corrected)
 return main

class Chain:
 def __init__(self,chain):self.chain=chain;self.reference=REFERENCE_PROPOSE.__get__(chain,type(chain));self.accelerated_calls=0
 @torch.inference_mode()
 def __call__(self,encoded,motion_uv,confidence,*,reset=False,cancel=None,resample='bilinear'):
  c=self.chain;nr=c.nr;oh,ow=c.output_size;h,w=c.work_size;nh,nw=c.neural_size
  # Preserve the reference's wider standalone input behavior; accelerated image
  # pipelines supply a single RGB frame and F32 motion/confidence resources.
  if (encoded.ndim!=4 or tuple(encoded.shape[:2])!=(1,3) or tuple(encoded.shape[-2:])!=(oh,ow) or encoded.device.type!='cuda'
      or motion_uv.ndim!=3 or motion_uv.shape[-1]!=2 or min(motion_uv.shape[:2])<1 or motion_uv.dtype!=torch.float32 or motion_uv.device!=encoded.device
      or tuple(confidence.shape)!=(oh,ow,1) or confidence.dtype!=torch.float32 or confidence.device!=encoded.device):
   return self.reference(encoded,motion_uv,confidence,reset=reset,cancel=cancel,resample=resample)
  check_cancel(cancel)
  if torch.cuda.current_stream(encoded.device)!=torch.cuda.default_stream(encoded.device):raise RuntimeError('Default CUDA stream required')
  if resample not in ('bilinear','easu'):raise ValueError('Unknown NR resampling filter')
  if resample=='easu' and (h>oh or w>ow):raise ValueError('EASU requires NR work <= output; use bilinear to downsample')
  reference=encoded;work=encoded if (h,w)==(oh,ow) else F.interpolate(encoded,size=(h,w),mode='bilinear',align_corners=False,antialias=True)
  base=torch_rtz(work);work=base;rh,rw=motion_uv.shape[:2]
  conf=confidence.permute(2,0,1)[None]
  if (h,w)!=(oh,ow):conf=F.interpolate(conf,size=(h,w),mode='bilinear',align_corners=False)
  conf=conf[0].permute(1,2,0).contiguous();states=[];records=[]
  for index,settings in enumerate(c.passes):
   check_cancel(cancel);old=None if reset else c.states[index];cold=old is None or settings.temporal_strength==0
   current,mv,gate=prepare_kernel(h,w,nh,nw,rh,rw,cold,settings.temporal_strength)(work.contiguous(),motion_uv.contiguous(),conf)
   previous=current if cold else old.image;frame=0 if old is None else old.frame_index+1
   # Dimensions are constructed from owned shape metadata on CPU, not copied
   # back from a newly-created GPU tensor. Runtime validation remains intact.
   rt=nr.Block70RuntimeInputs(color=current,prev_output=previous,mvec=mv,mvec_scale_xy=current.new_ones(2),output_dimensions_wh=torch.tensor([nw,nh]),style=settings.style,local_structure_strength=settings.structure,local_tone_strength=settings.tone,skin_structure_strength=settings.skin,use_auto_mask=settings.automatic_mask)
   rt.validate(1,nh,nw)
   uv=nr._output_uv(1,nh,nw,current.device);color_uv=nr._rect_uv(uv,rt.color,rt.color_rect_xywh)
   sampled_current=nr.bilinear_texture(rt.color,color_uv)[...,:3]
   mvec_uv=nr._rect_uv(uv,rt.mvec,rt.mvec_rect_xywh);motion=nr.bilinear_texture(rt.mvec,mvec_uv)[...,:2]
   history_uv=uv+motion*rt.normalized_mvec_scale(nw,nh).to(current.device).reshape(1,1,1,2)
   sampled_previous=nr.catmull_rom_texture(previous,history_uv,output_size=(nh,nw),rect=rt.prev_rect_xywh)[...,:3]
   structure=settings.structure
   if settings.automatic_mask:
    skin=settings.skin if settings.skin>=0 else structure;auto=structure;both=skin>=0 and auto>=0
    ss,sk,au=(1.,-1.,-1.) if both else (structure,skin,auto)
   else:ss,sk,au=structure,-1.,-1.
   scalars=current.new_tensor([settings.style/128,settings.tone,ss,sk,au])
   gaussian=nr.gaussian_dither(nh,nw,frame=frame,device=current.device)
   sampled_current=sampled_current.contiguous();sampled_previous=sampled_previous.contiguous()
   packet=packet_kernel(nh,nw)(sampled_current,sampled_previous,gaussian.contiguous(),gate,scalars)
   head=c.model.infer_minimal(packet)
   result=output_kernel(nh,nw)(sampled_current,sampled_previous,head[...,:3].float().contiguous(),torch.sigmoid(head[...,3:4].float()).contiguous(),gate)
   if not bool(result.isfinite().all()):raise FloatingPointError('Nonfinite NR pass '+str(index))
   if settings.intensity!=1:result=current+settings.intensity*(result-current)
   states.append(PassState(torch_rtz(result).detach(),frame));work=result[:,:h,:w].permute(0,3,1,2).contiguous()
   records.append({'pass':index,'frame_index':frame,'cold_nr':cold,'head_abs_mean':float(head[...,:3].abs().mean()),'history_gate_mean':float(gate[:,:h,:w].mean()),'intensity':settings.intensity,'settings':asdict(settings)})
  if (h,w)==(oh,ow):image=work
  else:
   if resample=='easu':
    from whitebox_pipeline.spatial import easu
    delta=easu(work,(oh,ow))-easu(base,(oh,ow))
   else:delta=F.interpolate(work-base,size=(oh,ow),mode='bilinear',align_corners=False)
   image=(reference+delta).clamp(0,1)
  check_cancel(cancel)
  self.accelerated_calls+=1
  return image,tuple(states),records
