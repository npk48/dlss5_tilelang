"""Complete FSR depth-clip/reactivity implementation.
Original Torch norm and temporal bilinear sampling remain explicit boundaries.
No state commit or diagnostics are changed by this callable.
"""
import bootstrap
import math,struct
import torch,tilelang
import tilelang.language as T
from device_policy import TARGET,CONFIG,EXECUTION_BACKEND
from whitebox_pipeline import fsr2 as ref
REFERENCE_DEPTH_CLIP=ref.FSR2._depth_clip

def mx(a,b):return T.call_extern('float32','fmaxf',a,b)
def mn(a,b):return T.call_extern('float32','fminf',a,b)
def sat(a):return mn(mx(a,T.float32(0)),T.float32(1))
def div(a,b):return T.call_extern('float32','__fdiv_rn',a,b)
def power(a,b):return T.call_extern('float32','powf',a,b)
def scalar_power(a,b):
 if b==2:return a*a
 if b==3:return (a*a)*a
 return power(a,T.float32(b))
def inside(y,x,h,w):return T.And(T.And(y>=0,y<h),T.And(x>=0,x<w))
def load(A,y,x,k,h,w,clamp=False):
 value=A[T.max(0,T.min(y,h-1)),T.max(0,T.min(x,w-1)),k]
 return value if clamp else T.if_then_else(inside(y,x,h,w),value,T.float32(0))
def z(d,P):return div(P[1],d-P[0])
def half_rtz(v):
 q=T.cast(v,T.float16);bits=T.reinterpret(q,T.uint16)
 bits=T.cast(bits-T.cast(T.abs(T.cast(q,T.float32))>T.abs(v),T.uint16),T.uint16)
 return T.cast(T.reinterpret(bits,T.float16),T.float32)
def unorm(v):return T.call_extern('float32','nearbyintf',sat(v)*T.float32(255))*T.float32(1/255)

@tilelang.jit(out_idx=[11,12],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def kernel(H,W,OH,OW,MH,MW,DisplayMV,FastColorSum):
 ScalarPower=struct.unpack('f',struct.pack('f',1+2*min(1,math.hypot(W,H)/math.hypot(1920,1080))))[0]
 @T.prim_func
 def main(C:T.Tensor((H,W,3),T.float32),DD:T.Tensor((H,W,1),T.float32),Rec:T.Tensor((H,W,1),T.float32),DM:T.Tensor((H,W,2),T.float32),MV:T.Tensor((MH,MW,2),T.float32),React:T.Tensor((H,W,1),T.float32),Comp:T.Tensor((H,W,1),T.float32),VM:T.Tensor((MH,MW,2),T.float32),DI:T.Tensor((H,W,3),T.float32),CN:T.Tensor((H,W,2),T.float32),P:T.Tensor((10,),T.float32),Prepared:T.Tensor((H,W,4),T.float32),Masks:T.Tensor((H,W,2),T.float32)):
  with T.Kernel(T.ceildiv(H*W,128),threads=128) as bx:
   for i in T.Parallel(128):
    idx=bx*128+i
    if idx<H*W:
     y=idx//W;x=idx%W
     ux=div(T.cast(x,T.float32)+T.float32(.5),T.float32(W))+T.if_then_else(DI[y,x,1]>T.float32(.01),DM[y,x,0],T.float32(0))
     uy=div(T.cast(y,T.float32)+T.float32(.5),T.float32(H))+T.if_then_else(DI[y,x,1]>T.float32(.01),DM[y,x,1],T.float32(0))
     px=ux*T.float32(W)-T.float32(.5);py=uy*T.float32(H)-T.float32(.5)
     ix=T.cast(T.floor(px),T.int32);iy=T.cast(T.floor(py),T.int32);fx=px-T.cast(ix,T.float32);fy=py-T.cast(iy,T.float32)
     zz=z(DD[y,x,0],P);total=T.alloc_var('float32',init=0);ws=T.alloc_var('float32',init=0)
     for q in T.unroll(4):
      qx=ix+q%2;qy=iy+q//2
      wt=T.if_then_else(q%2==0,T.float32(1)-fx,fx)*T.if_then_else(q//2==0,T.float32(1)-fy,fy)
      prev=z(load(Rec,qy,qx,0,H,W),P);difference=zz-prev;take=T.And(inside(qy,qx,H,W),T.And(wt>T.float32(.01),difference>T.float32(0)))
      sep=P[2]*mx(zz,prev)
      total=total+T.if_then_else(take,scalar_power(sat(div(sep,difference)),ScalarPower)*wt,T.float32(0))
      ws=ws+T.if_then_else(take,wt,T.float32(0))
     clip=T.if_then_else(ws>T.float32(0),sat(T.float32(1)-div(total,ws)),T.float32(0))
     d0=z(load(Rec,y-1,x,0,H,W),P);d1=z(Rec[y,x,0],P);d2=z(load(Rec,y+1,x,0,H,W),P)
     clip=clip*T.cast(T.Not(T.And(d0-d1>d1*T.float32(.01),d1-d2>d2*T.float32(.01))),T.float32)
     if DisplayMV:
      qx=T.cast(T.floor(div(T.cast(x,T.float32)+T.float32(.5)-P[8],T.float32(W))*T.float32(OW)),T.int32)
      qy=T.cast(T.floor(div(T.cast(y,T.float32)+T.float32(.5)-P[9],T.float32(H))*T.float32(OH)),T.int32)
     else:qx=x;qy=y
     nx=load(MV,qy,qx,0,MH,MW);ny=load(MV,qy,qx,1,MH,MW);velocity=load(VM,qy,qx,1,MH,MW)
     maximum=T.alloc_var('float32',init=load(VM,qy,qx,0,MH,MW));convergence=T.alloc_var('float32',init=1)
     for j in T.unroll(9):
      ox=j%3-1;oy=j//3-1
      sx=T.if_then_else(ox<0,T.max(qx+ox,0),T.if_then_else(ox>0,T.min(qx+ox,W-1),qx+ox))
      sy=T.if_then_else(oy<0,T.max(qy+oy,0),T.if_then_else(oy>0,T.min(qy+oy,H-1),qy+oy))
      vx=load(MV,sy,sx,0,MH,MW);vy=load(MV,sy,sx,1,MH,MW)
      maximum=mx(load(VM,sy,sx,0,MH,MW),maximum)
      convergence=mn(convergence,div(div(vx,maximum)*nx,maximum)+div(div(vy,maximum)*ny,maximum))
     convergence=T.if_then_else(velocity>T.float32(.01),convergence,T.float32(1))
     divergence=sat(T.float32(1)-convergence)*sat(maximum*T.float32(100))
     dist=DI[y,x,1]
     temporal=T.if_then_else(dist>T.float32(1),(T.float32(1)-sat(div(DI[y,x,2],DI[y,x,0])))*sat(scalar_power(dist*T.float32(.05),3)),T.float32(0))
     mind=T.alloc_var('float32',init=P[5]);maxd=T.alloc_var('float32',init=0);farfound=T.alloc_var('bool',init=False)
     for j in T.unroll(9):
      sx=x+j%3-1;sy=y+j//3-1
      dd=z(load(DD,sy,sx,0,H,W),P)*P[4]*T.cast(inside(sy,sx,H,W),T.float32)
      farfound=T.Or(farfound,dd==P[5]);mind=mn(mind,dd);maxd=mx(maxd,dd)
     depthdiv=(T.float32(1)-div(mind,maxd))*T.cast(T.Not(farfound),T.float32)
     divergence=mx(divergence,sat(temporal-depthdiv))
     rr=T.alloc_var('float32',init=0);tc=T.alloc_var('float32',init=divergence)
     for j in T.unroll(9):
      sx=x+j%3-1;sy=y+j//3-1
      r=load(C,sy,sx,0,H,W,True);g=load(C,sy,sx,1,H,W,True);b=load(C,sy,sx,2,H,W,True)
      # Fast reduced dimension: two lanes (0+2)+1. Slow dimension:
      # three values remain in one thread, accumulating (0+1)+2.
      if FastColorSum:dot=(C[y,x,0]*r+C[y,x,2]*b)+C[y,x,1]*g
      else:dot=(C[y,x,0]*r+C[y,x,1]*g)+C[y,x,2]*b
      similarity=div(dot,mx(CN[y,x,0],load(CN,sy,sx,1,H,W,True)))
      pw=T.float32(1)+(T.float32(6)-similarity*T.float32(6))
      rr=mx(rr,power(load(React,sy,sx,0,H,W,True),pw));tc=mx(tc,power(load(Comp,sy,sx,0,H,W,True),pw))
     Masks[y,x,0]=unorm(rr);Masks[y,x,1]=unorm(tc)
     r=mn(mx(mx(C[y,x,0],T.float32(0))*P[6]*P[7],T.float32(0)),T.float32(65504))
     g=mn(mx(mx(C[y,x,1],T.float32(0))*P[6]*P[7],T.float32(0)),T.float32(65504))
     b=mn(mx(mx(C[y,x,2],T.float32(0))*P[6]*P[7],T.float32(0)),T.float32(65504))
     Prepared[y,x,0]=half_rtz((T.float32(.25)*r+T.float32(.5)*g)+T.float32(.25)*b)
     Prepared[y,x,1]=half_rtz(T.float32(.5)*r-T.float32(.5)*b)
     Prepared[y,x,2]=half_rtz((-T.float32(.25)*r+T.float32(.5)*g)-T.float32(.25)*b)
     Prepared[y,x,3]=half_rtz(clip)
 return main

class DepthClip:
 def __init__(self,fsr):self.fsr=fsr;self.reference=REFERENCE_DEPTH_CLIP.__get__(fsr,type(fsr));self.cache={}
 def __call__(self,color,depth,dm,dd,reconstructed,mv,old,reactive,composition,pre,exposure,jitter):
  if torch.cuda.current_stream(color.device)!=torch.cuda.default_stream(color.device):raise RuntimeError('FSR TileLang depth clip is qualified only on the default CUDA stream')
  f=self.fsr;h,w=depth.shape[:2];oh,ow=f.output_size;mh,mw=mv.shape[:2]
  key=(h,w,oh,ow,f.near,f.far,f.fov,f.inverted,f.infinite,f.meters,color.device)
  if key not in self.cache:
   units=depth.new_tensor([w,h]);display=depth.new_tensor([ow,oh]);p=ref.grid(h,w,depth.device)
   ty=math.tan(f.fov*.5);tx=ty*w/h;center=depth.new_tensor([2*(w//2)/w-1,1-2*(h//2)/h])
   kfov=math.sqrt(tx*tx+ty*ty+1)/math.sqrt(float((center*depth.new_tensor([tx,ty])).square().sum())+1)
   ab=f._depth_factors(depth);maxdist=f._z(depth.new_tensor(0. if f.inverted else 1.))*f.meters
   const=torch.cat((ab,depth.new_tensor([1.37e-5*kfov*math.hypot(w,h),1+2*min(1,math.hypot(w,h)/math.hypot(1920,1080)),f.meters]),maxdist.reshape(1),depth.new_tensor([1/struct.unpack('f',struct.pack('f',float(pre)))[0]])))
   self.cache[key]=(units,display,p,const,float(pre))
  units,display,p,const,last_pre=self.cache[key]
  if last_pre!=float(pre):
   const=torch.cat((const[:6],depth.new_tensor([1/struct.unpack('f',struct.pack('f',float(pre)))[0]])))
   self.cache[key]=(units,display,p,const,float(pre))
  previous=ref.sample(old.dilated_motion,(p+.5)/units+dm) if old else torch.zeros_like(dm)
  vm=torch.cat((ref.norm(mv),ref.norm(mv*units)),-1).contiguous()
  di=torch.cat((ref.norm(dm),ref.norm(dm*display),ref.norm(previous)),-1).contiguous()
  squared=color.square();fast_color_sum=squared.stride(-1)<min(squared.stride(0),squared.stride(1))
  center_norm=squared.sum(-1,keepdim=True)
  neighbor_norm=center_norm if fast_color_sum else color.contiguous().square().sum(-1,keepdim=True)
  cn=torch.cat((center_norm,neighbor_norm),-1).contiguous();params=torch.cat((const,exposure.reshape(1),jitter.reshape(2)))
  args=(color,dd,reconstructed,dm,mv,reactive,composition,vm,di,cn,params)
  if any(x.dtype!=torch.float32 or not x.is_cuda for x in args):raise ValueError('FSR depth clip requires CUDA F32 operands')
  return kernel(h,w,oh,ow,mh,mw,f.display_mv,fast_color_sum)(*(x.contiguous() for x in args))
