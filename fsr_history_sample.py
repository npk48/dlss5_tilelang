"""Exact history interpolation stencil, retaining original Torch weight functions."""
import bootstrap
import torch,tilelang
import tilelang.language as T
from device_policy import TARGET,CONFIG,EXECUTION_BACKEND
from fsr_depth_clip import load,mx,mn,div
from whitebox_pipeline import fsr2 as ref

@tilelang.jit(out_idx=[3],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def kernel(H,W,OH,OW):
 @T.prim_func
 def main(A:T.Tensor((H,W,4),T.float32),Base:T.Tensor((OH,OW,2),T.int64),Weight:T.Tensor((OH,OW,8),T.float32),Out:T.Tensor((OH,OW,4),T.float32)):
  with T.Kernel(T.ceildiv(OH*OW,128),threads=128) as bx:
   row=T.alloc_local((4,),T.float32);out=T.alloc_local((4,),T.float32);lo=T.alloc_local((4,),T.float32);hi=T.alloc_local((4,),T.float32)
   tx=T.get_thread_binding(0);index=bx*128+tx
   if index<OH*OW:
    y=index//OW;x=index%OW;ix=T.cast(Base[y,x,0],T.int32);iy=T.cast(Base[y,x,1],T.int32)
    wx=((Weight[y,x,0]+Weight[y,x,1])+Weight[y,x,2])+Weight[y,x,3]
    wy=((Weight[y,x,4]+Weight[y,x,5])+Weight[y,x,6])+Weight[y,x,7]
    for c in T.unroll(4):out[c]=T.float32(0)
    for j in T.unroll(4):
     oy=j-1;sy=T.if_then_else(oy<0,T.max(iy+oy,0),T.if_then_else(oy>0,T.min(iy+oy,H-1),iy+oy))
     for c in T.unroll(4):row[c]=T.float32(0)
     for k in T.unroll(4):
      ox=k-1;sx=T.if_then_else(ox<0,T.max(ix+ox,0),T.if_then_else(ox>0,T.min(ix+ox,W-1),ix+ox))
      for c in T.unroll(4):
       v=load(A,sy,sx,c,H,W);row[c]=row[c]+Weight[y,x,k]*v
       if j==1 and k==1:lo[c]=v;hi[c]=v
       elif (j==1 or j==2) and (k==1 or k==2):lo[c]=mn(lo[c],v);hi[c]=mx(hi[c],v)
     for c in T.unroll(4):out[c]=out[c]+Weight[y,x,j+4]*div(row[c],wx)
    for c in T.unroll(4):Out[y,x,c]=mn(mx(div(out[c],wy),lo[c]),hi[c])
 return main

def sample(a,uv,lut=None):
 h,w=a.shape[:2];oh,ow=uv.shape[:2];p=uv*uv.new_tensor([w,h])-.5;p=ref.mx(torch.zeros_like(p),ref.mn(p,p.new_tensor([w,h])));base=p.floor().long();frac=p-base
 def weight(x):
  if lut is None:return ref.lanczos(x)
  q=(x.abs()/2*128-.5).clamp(0,127);i=q.floor().long();f=q-i
  return ref.mix(lut[i],lut[(i+1).clamp_max(127)],f)
 weights=torch.cat([weight(frac[...,axis:axis+1]-i) for axis in [0,1] for i in [-1,0,1,2]],-1)
 return kernel(h,w,oh,ow)(a.contiguous(),base.contiguous(),weights.contiguous())
