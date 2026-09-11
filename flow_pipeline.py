"""Complete RAFT-small adapter; all recurrent updates retained.
Forward structure adapted from torchvision0.20.1 optical_flow/raft.py;
Copyright (c) Soumith Chintala 2016. BSD-3-Clause: FLOW_TORCHVISION_LICENSE.txt.
"""
import bootstrap
import math
import torch,torch.nn.functional as F,tilelang
import tilelang.language as T
from device_policy import TARGET,CONFIG,EXECUTION_BACKEND
from fsr_accumulate import recip
from whitebox_pipeline.estimators import RaftSmall,valid_rgb
from torchvision.models.optical_flow._utils import make_coords_grid,upsample_flow
REFERENCE_FORWARD=RaftSmall.forward

@tilelang.jit(out_idx=[1,2,3,4],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def grid_kernel(B,H,W,R):
 S=2*R+1;N=B*H*W
 RX0=recip(W-1);RX1=recip(W//2-1);RX2=recip(W//4-1);RX3=recip(W//8-1)
 RY0=recip(H-1);RY1=recip(H//2-1);RY2=recip(H//4-1);RY3=recip(H//8-1)
 @T.prim_func
 def main(C:T.Tensor((B,2,H,W),T.float32),G0:T.Tensor((N,S,S,2),T.float32),G1:T.Tensor((N,S,S,2),T.float32),G2:T.Tensor((N,S,S,2),T.float32),G3:T.Tensor((N,S,S,2),T.float32)):
  with T.Kernel(T.ceildiv(N*S*S,128),threads=128) as bx:
   i=bx*128+T.get_thread_binding(0)
   if i<N*S*S:
    n=i//(S*S);k=i%(S*S);b=n//(H*W);p=n%(H*W);y=p//W;x=p%W
    cx=T.alloc_var('float32',init=C[b,0,y,x]);cy=T.alloc_var('float32',init=C[b,1,y,x])
    for level in T.unroll(4):
     # Torch meshgrid(di,dj) stores di as the first coordinate, not dj.
     sx=cx+T.cast(k//S-R,T.float32);sy=cy+T.cast(k%S-R,T.float32)
     rx=T.if_then_else(level==0,T.float32(RX0),T.if_then_else(level==1,T.float32(RX1),T.if_then_else(level==2,T.float32(RX2),T.float32(RX3))))
     ry=T.if_then_else(level==0,T.float32(RY0),T.if_then_else(level==1,T.float32(RY1),T.if_then_else(level==2,T.float32(RY2),T.float32(RY3))))
     gx=(T.float32(2)*sx)*rx-T.float32(1);gy=(T.float32(2)*sy)*ry-T.float32(1)
     if level==0:G0[n,k//S,k%S,0]=gx;G0[n,k//S,k%S,1]=gy
     elif level==1:G1[n,k//S,k%S,0]=gx;G1[n,k//S,k%S,1]=gy
     elif level==2:G2[n,k//S,k%S,0]=gx;G2[n,k//S,k%S,1]=gy
     else:G3[n,k//S,k%S,0]=gx;G3[n,k//S,k%S,1]=gy
     cx=cx*T.float32(.5);cy=cy*T.float32(.5)
 return main

@tilelang.jit(out_idx=[4],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def pack_kernel(B,H,W,R):
 S=2*R+1;N=B*H*W;K=S*S
 @T.prim_func
 def main(A:T.Tensor((N,1,S,S),T.float32),B1:T.Tensor((N,1,S,S),T.float32),C:T.Tensor((N,1,S,S),T.float32),D:T.Tensor((N,1,S,S),T.float32),Out:T.Tensor((B,4*K,H,W),T.float32)):
  with T.Kernel(T.ceildiv(B*4*K*H*W,128),threads=128) as bx:
   i=bx*128+T.get_thread_binding(0)
   if i<B*4*K*H*W:
    b=i//(4*K*H*W);ch=i//(H*W)%(4*K);p=i%(H*W);n=b*H*W+p;k=ch%K;level=ch//K
    v=T.alloc_var('float32',init=0)
    if level==0:v=A[n,0,k//S,k%S]
    elif level==1:v=B1[n,0,k//S,k%S]
    elif level==2:v=C[n,0,k//S,k%S]
    else:v=D[n,0,k//S,k%S]
    Out[b,ch,p//W,p%W]=v
 return main

class Flow:
 def __init__(self,estimator):self.estimator=estimator;self.reference=REFERENCE_FORWARD.__get__(estimator,type(estimator));self.accelerated_calls=0
 def index(self,coords):
  corr=self.estimator.network.corr_block;b,_,h,w=coords.shape;r=corr.radius
  grids=grid_kernel(b,h,w,r)(coords.contiguous())
  sampled=[F.grid_sample(volume,grid,mode='bilinear',padding_mode='zeros',align_corners=True) for volume,grid in zip(corr.corr_pyramid,grids)]
  result=pack_kernel(b,h,w,r)(*sampled)
  if result.shape!=(b,corr.out_channels,h,w):raise ValueError('Output shape of index pyramid is incorrect')
  return result
 @torch.inference_mode()
 def __call__(self,current,previous):
  e=self.estimator;net=e.network
  valid_rgb(current);valid_rgb(previous)
  if current.shape!=previous.shape:raise ValueError('Motion estimator requires same frame shape')
  if current.device.type!='cuda' or torch.is_autocast_enabled() or e.updates<1 or net.mask_predictor is not None or net.corr_block.num_levels!=4:
   return self.reference(current,previous)
  if torch.cuda.current_stream(current.device)!=torch.cuda.default_stream(current.device):raise RuntimeError('Default CUDA stream required')
  h,w=current.shape[-2:];scale=min(1.,e.limit/max(h,w));rh,rw=max(1,round(h*scale)),max(1,round(w*scale));nh,nw=max(128,math.ceil(rh/8)*8),max(128,math.ceil(rw/8)*8)
  def prep(x):
   x=F.interpolate(x.float(),size=(rh,rw),mode='bilinear',align_corners=False,antialias=False)
   return F.pad(x,(0,nw-rw,0,nh-rh),mode='replicate')*2-1
  image1=prep(current);image2=prep(previous);batch=image1.shape[0]
  fmaps=net.feature_encoder(torch.cat([image1,image2],dim=0));fmap1,fmap2=torch.chunk(fmaps,2,dim=0)
  if fmap1.shape[-2:]!=(nh//8,nw//8):raise ValueError('The feature encoder should downsample H and W by 8')
  net.corr_block.build_pyramid(fmap1,fmap2)
  context_out=net.context_encoder(image1)
  if context_out.shape[-2:]!=(nh//8,nw//8):raise ValueError('The context encoder should downsample H and W by 8')
  hidden_size=net.update_block.hidden_state_size;context_size=context_out.shape[1]-hidden_size
  if context_size<=0:raise ValueError('Context encoder must output more channels than hidden state')
  hidden,context=torch.split(context_out,[hidden_size,context_size],dim=1);hidden=torch.tanh(hidden);context=F.relu(context)
  coords0=make_coords_grid(batch,nh//8,nw//8).to(fmap1.device);coords1=make_coords_grid(batch,nh//8,nw//8).to(fmap1.device)
  for _ in range(e.updates):
   coords1=coords1.detach();corr_features=self.index(coords1);flow=coords1-coords0
   hidden,delta=net.update_block(hidden,context,corr_features,flow);coords1=coords1+delta
  # Earlier upsampled predictions are not inputs to any update and the adapter
  # only consumes [-1]. No recurrent update or externally returned frame is lost.
  flow=upsample_flow(flow=coords1-coords0,up_mask=None)[:,:,:rh,:rw]
  flow=F.interpolate(flow,size=(h,w),mode='bilinear',align_corners=False)*flow.new_tensor([w/rw,h/rh])[None,:,None,None]
  if not bool(flow.isfinite().all()):raise FloatingPointError('Nonfinite RAFT output')
  self.accelerated_calls+=1
  return flow,{'network_hw':[nh,nw],'resized_image_hw':[rh,rw],'updates':e.updates,'direction':'current_to_previous','units':'source pixels','padding':'bottom/right replicate'}
