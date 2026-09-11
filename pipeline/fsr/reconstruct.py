"""Complete FSR reconstruction and lock publication candidate, no state ownership."""
from runtime import bootstrap
import torch,tilelang
import tilelang.language as T
from runtime.device import TARGET,CONFIG,EXECUTION_BACKEND
from whitebox_pipeline import fsr2 as ref
from pipeline.fsr.depth_clip import load,inside,mx,mn,div,power,half_rtz
from pipeline.fsr.accumulate import recip
REFERENCE_RECONSTRUCT=ref.FSR2._reconstruct
REFERENCE_LOCKS=ref.FSR2._locks

@tilelang.jit(out_idx=[6,7,8],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def reconstruct_kernel(H,W,OH,OW,MH,MW,Inverted,DisplayMV,HDR):
 X=(1,0,0,-1,-1,1,-1,1);Y=(0,1,-1,0,1,1,-1,-1)
 @T.prim_func
 def main(D:T.Tensor((H,W,1),T.float32),MV:T.Tensor((MH,MW,2),T.float32),Color:T.Tensor((H,W,1),T.float32),Magnitude:T.Tensor((MH,MW,1),T.float32),P:T.Tensor((4,),T.float32),Rec:T.Tensor((H,W,1),T.float32),Nearest:T.Tensor((H,W,1),T.float32),Dilated:T.Tensor((H,W,2),T.float32),Luma:T.Tensor((H,W,1),T.float32)):
  with T.Kernel(T.ceildiv(H*W,128),threads=128) as bx:
   tx=T.get_thread_binding(0);index=bx*128+tx
   if index<H*W:
    y=index//W;x=index%W
    nearest=T.alloc_var('float32',init=D[y,x,0]);cx=T.alloc_var('int32',init=x);cy=T.alloc_var('int32',init=y)
    for i in T.unroll(8):
     ox=T.if_then_else(T.Or(i==0,T.Or(i==5,i==7)),1,T.if_then_else(T.Or(i==3,T.Or(i==4,i==6)),-1,0))
     oy=T.if_then_else(T.Or(i==1,T.Or(i==4,i==5)),1,T.if_then_else(T.Or(i==2,T.Or(i==6,i==7)),-1,0))
     sx=x+ox;sy=y+oy;d=load(D,sy,sx,0,H,W)
     take=T.And(inside(sy,sx,H,W),d>nearest if Inverted else d<nearest)
     cx=T.if_then_else(take,sx,cx);cy=T.if_then_else(take,sy,cy);nearest=T.if_then_else(take,d,nearest)
    if DisplayMV:
     qx=T.cast(T.floor(div(T.cast(cx,T.float32)+T.float32(.5)-P[2],T.float32(W))*T.float32(OW)),T.int32)
     qy=T.cast(T.floor(div(T.cast(cy,T.float32)+T.float32(.5)-P[3],T.float32(H))*T.float32(OH)),T.int32)
    else:qx=cx;qy=cy
    dx=load(MV,qy,qx,0,MH,MW);dy=load(MV,qy,qx,1,MH,MW)
    moving=load(Magnitude,qy,qx,0,MH,MW)>T.float32(.1)
    ux=div(T.cast(x,T.float32)+T.float32(.5),T.float32(W))+dx*T.cast(moving,T.float32)
    uy=div(T.cast(y,T.float32)+T.float32(.5),T.float32(H))+dy*T.cast(moving,T.float32)
    px=ux*T.float32(W)-T.float32(.5);py=uy*T.float32(H)-T.float32(.5)
    ix=T.cast(T.floor(px),T.int32);iy=T.cast(T.floor(py),T.int32);fx=px-T.cast(ix,T.float32);fy=py-T.cast(iy,T.float32)
    for i in T.unroll(4):
     sx=ix+i%2;sy=iy+i//2
     wt=T.if_then_else(i%2==0,T.float32(1)-fx,fx)*T.if_then_else(i//2==0,T.float32(1)-fy,fy)
     if T.And(inside(sy,sx,H,W),wt>T.float32(.01)):
      if Inverted:T.atomic_max(Rec[sy,sx,0],nearest)
      else:T.atomic_min(Rec[sy,sx,0],nearest)
    lum=Color[y,x,0]
    perceived=T.if_then_else(lum<=T.float32(216/24389),lum*T.float32(24389/27),power(lum,T.float32(1/3))*T.float32(116)-T.float32(16))*T.float32(.01)
    Nearest[y,x,0]=nearest;Dilated[y,x,0]=half_rtz(dx);Dilated[y,x,1]=half_rtz(dy);Luma[y,x,0]=half_rtz(power(perceived,T.float32(1/6)))
 return main

@tilelang.jit(out_idx=[],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def locks_kernel(H,W,OH,OW):
 @T.prim_func
 def main(Luma:T.Tensor((H,W,1),T.float32),J:T.Tensor((2,),T.float32),Out:T.Tensor((OH,OW,1),T.float32)):
  with T.Kernel(T.ceildiv(H*W,128),threads=128) as bx:
   tx=T.get_thread_binding(0);index=bx*128+tx
   if index<H*W:
    y=index//W;x=index%W;value=Luma[y,x,0]
    bits=T.alloc_var('int32',init=16);low=T.alloc_var('float32',init=T.float32(3.4028234663852886e38));high=T.alloc_var('float32',init=0)
    for i in T.unroll(9):
     if i!=4:
      s=load(Luma,y+i//3-1,x+i%3-1,0,H,W,True);ratio=div(mx(s,value),mn(s,value));same=T.And(ratio>T.float32(0),ratio<T.float32(1.05))
      bits=bits|(T.cast(same,T.int32)<<i);low=T.if_then_else(same,low,mn(low,s));high=T.if_then_else(same,high,mx(high,s))
    ridge=T.And(T.Or(value>high,value<low),T.And((bits&27)!=27,T.And((bits&54)!=54,T.And((bits&216)!=216,(bits&432)!=432))))
    qx=T.cast(T.floor(div(T.cast(x,T.float32)+T.float32(.5)-J[0],T.float32(W))*T.float32(OW)),T.int32)
    qy=T.cast(T.floor(div(T.cast(y,T.float32)+T.float32(.5)-J[1],T.float32(H))*T.float32(OH)),T.int32)
    if T.And(ridge,inside(qy,qx,OH,OW)):T.atomic_max(Out[qy,qx,0],T.float32(1))
 return main

class ReconstructLocks:
 def __init__(self,fsr):
  self.fsr=fsr;self.reference_reconstruct=REFERENCE_RECONSTRUCT.__get__(fsr,type(fsr));self.reference_locks=REFERENCE_LOCKS.__get__(fsr,type(fsr))
 def reconstruct(self,depth,mv,color,exposure,pre,jitter,first_execution):
  f=self.fsr;h,w=depth.shape[:2];oh,ow=f.output_size;mh,mw=mv.shape[:2]
  if torch.cuda.current_stream(depth.device)!=torch.cuda.default_stream(depth.device):raise RuntimeError('Default CUDA stream required')
  mag=ref.norm(mv*mv.new_tensor([ow,oh])).contiguous();params=torch.cat((exposure.reshape(1),depth.new_tensor([recip(pre)]),jitter.reshape(2)))
  reconstructed=depth.new_full((h,w,1),0. if first_execution or f.inverted else 1.)
  prepared=color.clamp_min(0)/pre*exposure
  if f.hdr:prepared=ref.tonemap(prepared)
  luma=ref.luma(prepared) # Preserve Torch's layout-dependent F32 reduction boundary.
  inputs=(depth,mv,luma,mag,params,reconstructed)
  if any(v.dtype!=torch.float32 for v in inputs):raise ValueError('F32 operands required')
  nearest,dilated,luma=reconstruct_kernel(h,w,oh,ow,mh,mw,f.inverted,f.display_mv,f.hdr)(*(v.contiguous() for v in inputs))
  return nearest,dilated,reconstructed,luma
 def locks(self,lock_luma,jitter):
  if torch.cuda.current_stream(lock_luma.device)!=torch.cuda.default_stream(lock_luma.device):raise RuntimeError('Default CUDA stream required')
  h,w=lock_luma.shape[:2];oh,ow=self.fsr.output_size;out=lock_luma.new_zeros(oh,ow,1)
  locks_kernel(h,w,oh,ow)(lock_luma.contiguous(),jitter.contiguous(),out)
  return out
