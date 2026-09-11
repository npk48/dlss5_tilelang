"""Complete FSR accumulation candidate; original history/bilinear sampling boundary.
Not installed by default until whole-stage and actual pipeline qualification.
"""
import bootstrap
import math,struct
import torch,tilelang
import tilelang.language as T
from device_policy import TARGET,CONFIG,EXECUTION_BACKEND
from fsr_history_sample import sample as history_sample
from whitebox_pipeline import fsr2 as ref
from fsr_depth_clip import mx,mn,sat,div,power,scalar_power,half_rtz,unorm,load,inside
REFERENCE_ACCUMULATE=ref.FSR2._accumulate

def f32(x):return struct.unpack('f',struct.pack('f',float(x)))[0]
def recip(x):return f32(1/f32(x))
def mix(a,b,t):return a+(b-a)*t
def ratio(a,b):
 m=mx(a,b)
 return T.if_then_else(m!=T.float32(0),div(mn(a,b),m),T.float32(0))
def sign(x):return T.if_then_else(x>T.float32(0),T.float32(1),T.if_then_else(x<T.float32(0),T.float32(-1),T.float32(0)))

@tilelang.jit(out_idx=[10,11,12,13,14,15,16,17,18],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def kernel(H,W,OH,OW,HDR,HasOld):
 DSX=f32(W/OW);DSY=f32(H/OH);BiasMax=f32(min(1.99,OW/W));Influence=f32(min(20,(1/(W/OW*H/OH))**3))
 @T.prim_func
 def main(Prepared:T.Tensor((H,W,4),T.float32),Motion:T.Tensor((OH,OW,2),T.float32),Velocity:T.Tensor((OH,OW,1),T.float32),DC:T.Tensor((OH,OW,1),T.float32),Masks:T.Tensor((OH,OW,2),T.float32),Raw:T.Tensor((OH,OW,4),T.float32),OldLock:T.Tensor((OH,OW,2),T.float32),Shade:T.Tensor((OH,OW,2),T.float32),LH:T.Tensor((OH,OW,4),T.float32),Params:T.Tensor((8,),T.float32),Resolved:T.Tensor((OH,OW,3),T.float32),History:T.Tensor((OH,OW,4),T.float32),Locks:T.Tensor((OH,OW,2),T.float32),Luma:T.Tensor((OH,OW,4),T.float32),Up:T.Tensor((OH,OW,3),T.float32),Weight:T.Tensor((OH,OW,1),T.float32),Mean:T.Tensor((OH,OW,3),T.float32),Reactivity:T.Tensor((OH,OW,1),T.float32),HistOut:T.Tensor((OH,OW,3),T.float32)):
  with T.Kernel(T.ceildiv(OH*OW,128),threads=128) as bx:
   hist=T.alloc_local((3,),T.float32);up=T.alloc_local((3,),T.float32);mean=T.alloc_local((3,),T.float32);moment=T.alloc_local((3,),T.float32);lo=T.alloc_local((3,),T.float32);hi=T.alloc_local((3,),T.float32);std=T.alloc_local((3,),T.float32);accum=T.alloc_local((3,),T.float32);rgbh=T.alloc_local((3,),T.float32);rgbu=T.alloc_local((3,),T.float32);blended=T.alloc_local((3,),T.float32)
   lh=T.alloc_local((4,),T.float32)
   i=T.get_thread_binding(0)
   index=bx*128+i
   if index<OH*OW:
    y=index//OW;x=index%OW
    ux=div(T.cast(x,T.float32)+T.float32(.5),T.float32(OW));uy=div(T.cast(y,T.float32)+T.float32(.5),T.float32(OH))
    rx=ux+Motion[y,x,0];ry=uy+Motion[y,x,1]
    existing=T.And(T.And(rx>=T.float32(0),rx<=T.float32(1)),T.And(ry>=T.float32(0),ry<=T.float32(1)))
    isnew=T.Not(existing) if HasOld else T.bool(True)
    reactive=Masks[y,x,0];accmask=Masks[y,x,1];dc=DC[y,x,0];velocity=Velocity[y,x,0]
    r=T.if_then_else(existing,mn(mx(Raw[y,x,0]*Params[2]*Params[0],T.float32(0)),T.float32(65504)),T.float32(0)) if HasOld else T.float32(0)
    g=T.if_then_else(existing,mn(mx(Raw[y,x,1]*Params[2]*Params[0],T.float32(0)),T.float32(65504)),T.float32(0)) if HasOld else T.float32(0)
    b=T.if_then_else(existing,mn(mx(Raw[y,x,2]*Params[2]*Params[0],T.float32(0)),T.float32(65504)),T.float32(0)) if HasOld else T.float32(0)
    hist[0]=(T.float32(.25)*r+T.float32(.5)*g)+T.float32(.25)*b;hist[1]=T.float32(.5)*r-T.float32(.5)*b;hist[2]=(-T.float32(.25)*r+T.float32(.5)*g)-T.float32(.25)*b
    temporal=T.if_then_else(existing,sat(T.abs(Raw[y,x,3])),T.float32(0)) if HasOld else T.float32(0)
    moving=T.And(existing,Raw[y,x,3]<T.float32(0)) if HasOld else T.bool(False)
    life=T.if_then_else(existing,OldLock[y,x,0],T.float32(0)) if HasOld else T.float32(0)
    lum0=T.if_then_else(existing,OldLock[y,x,1],T.float32(0)) if HasOld else T.float32(0)
    newlock=T.And(Shade[y,x,1]>T.float32(127/255),T.And(existing,T.Not(isnew)))
    shade=Shade[y,x,0];lum=T.if_then_else(lum0==T.float32(0),shade,lum0);diff=T.float32(1)-ratio(lum,shade)
    newlife=T.if_then_else(life!=T.float32(0),T.float32(2),T.float32(1))
    updated=T.if_then_else(newlock,shade,T.if_then_else(life<=T.float32(1),mix(lum,shade,T.float32(.5)),lum))
    life=T.if_then_else(newlock,newlife,T.if_then_else(T.And(life>T.float32(1),diff>T.float32(.1)),T.float32(0),life))
    thisreact=mx(mx(reactive,temporal),sat((diff-T.float32(.1))*T.float32(10)))
    life=((life*(T.float32(1)-thisreact))*sat(T.float32(1)-accmask))*T.cast(dc<T.float32(.1),T.float32)
    lockcon=sat(sat(sat(life-T.float32(1))*T.float32(4))*sat(ratio(updated,shade)))
    sx=(T.cast(x,T.float32)+T.float32(.5))*T.float32(DSX);sy=(T.cast(y,T.float32)+T.float32(.5))*T.float32(DSY)
    ix=T.cast(T.floor(sx),T.int32);iy=T.cast(T.floor(sy),T.int32)
    jx=T.cast(ix,T.float32)+T.float32(.5)-Params[3];jy=T.cast(iy,T.float32)+T.float32(.5)-Params[4]
    flipx=jx>sx;flipy=jy>sy;offsetx=jx-sx;offsety=jy-sy
    kr=mx(thisreact,T.cast(isnew,T.float32));bmax=T.float32(BiasMax)*(T.float32(1)-kr);bmin=mx(T.float32(1),(T.float32(1)+bmax)*T.float32(.3))
    bias=mix(bmax,bmin,mx(T.float32(.25)*dc,kr));curve=mix(T.float32(-2),T.float32(-3),sat(velocity*T.float32(recip(50))))
    weight=T.alloc_var('float32',init=0);bw=T.alloc_var('float32',init=0)
    for k in T.unroll(3):up[k]=T.float32(0);mean[k]=T.float32(0);moment[k]=T.float32(0)
    for j in T.unroll(9):
     ox=T.if_then_else(flipx,1-j%3,j%3-1);oy=T.if_then_else(flipy,1-j//3,j//3-1)
     qx=ix+ox;qy=iy+oy;dx=offsetx+T.cast(ox,T.float32);dy=offsety+T.cast(oy,T.float32)
     distance=dx*dx+dy*dy
     d=mn((distance*bias)*bias,T.float32(4));aa=T.float32(.4)*d-T.float32(1);bb=T.float32(.25)*d-T.float32(1)
     wt=T.cast(inside(qy,qx,H,W),T.float32)*((T.float32(1.5625)*aa*aa-T.float32(.5625))*(bb*bb))
     boxweight=T.call_extern('float32','expf',curve*distance)
     weight=weight+wt;bw=bw+boxweight
     for k in T.unroll(3):
      value=load(Prepared,qy,qx,k,H,W)
      up[k]=up[k]+value*wt;mean[k]=mean[k]+value*boxweight;moment[k]=moment[k]+(value*value)*boxweight
      if j==0:lo[k]=value;hi[k]=value
      else:lo[k]=mn(lo[k],value);hi[k]=mx(hi[k],value)
    bw=T.if_then_else(T.abs(bw)>T.float32(.001),bw,T.float32(1))
    good=weight>T.float32(.001)
    for k in T.unroll(3):
     mean[k]=div(mean[k],bw);std[k]=T.call_extern('float32','sqrtf',T.abs(div(moment[k],bw)-mean[k]*mean[k]))
     up[k]=T.if_then_else(good,mn(mx(div(up[k],weight),lo[k]),hi[k]),up[k])
    weight=T.if_then_else(good,weight*T.float32(recip(12)),T.float32(0))
    current=mean[0]
    if HDR:current=div(current,T.float32(1)+mx(current,T.float32(0)))
    current=T.call_extern('float32','nearbyintf',current*T.float32(255))*T.float32(recip(255))
    useluma=T.And(mx(mx(dc,accmask),diff)<T.float32(.1),T.Not(isnew))
    for k in T.unroll(4):lh[k]=T.if_then_else(useluma,LH[y,x,k],T.float32(0))
    d0=current-lh[0];minimum=T.alloc_var('float32',init=T.abs(d0))
    for k in T.unroll(3):
     dd=current-lh[k+1];minimum=T.if_then_else(sign(d0)==sign(dd),mn(minimum,T.abs(dd)),minimum)
    instability=T.cast(minimum!=T.abs(d0),T.float32)*power(sat(std[0]*T.float32(recip(.1))),T.float32(6))
    instability=(T.cast(instability>T.float32(1/255),T.float32)*T.cast(T.abs(d0)>=T.float32(1/255),T.float32))*(T.float32(1)-mx(accmask,power(thisreact,T.float32(1/6))))
    instability=instability*T.cast(lh[2]!=T.float32(0),T.float32)
    base=(T.cast(existing,T.float32)*(T.float32(1)-thisreact))*(T.float32(1)-dc)
    base=mn(base,mix(base,weight*T.float32(10),mx(T.cast(moving,T.float32),sat(velocity*T.float32(10)))))
    base=mn(base,mix(base,weight,sat(velocity*T.float32(recip(20)))))
    boxscale=mix(T.float32(Influence),T.float32(1),mx(dc,mx(accmask,sat(velocity*T.float32(recip(20))))))
    outside=T.alloc_var('bool',init=False)
    for k in T.unroll(3):
     lo[k]=mx(lo[k],mean[k]-std[k]*boxscale);hi[k]=mn(hi[k],mean[k]+std[k]*boxscale)
     outside=T.Or(outside,T.Or(hist[k]<lo[k],hist[k]>hi[k]))
    contribution=sat(mx(instability,lockcon)*(T.float32(1)-T.call_extern('float32','sqrtf',reactive)))
    for k in T.unroll(3):
     hist[k]=T.if_then_else(outside,mix(mn(mx(hist[k],lo[k]),hi[k]),hist[k],contribution),hist[k])
     accum[k]=T.if_then_else(outside,mix(mn(base,T.float32(.1)),base,contribution),base)
     HistOut[y,x,k]=hist[k];Up[y,x,k]=up[k];Mean[y,x,k]=mean[k]
    if HDR:
     rgbh[0]=(hist[0]+hist[1])-hist[2];rgbh[1]=hist[0]+hist[2];rgbh[2]=(hist[0]-hist[1])-hist[2]
     rgbu[0]=(up[0]+up[1])-up[2];rgbu[1]=up[0]+up[2];rgbu[2]=(up[0]-up[1])-up[2]
     denh=mx(mx(mx(rgbh[0],rgbh[1]),rgbh[2]),T.float32(0))+T.float32(1);denu=mx(mx(mx(rgbu[0],rgbu[1]),rgbu[2]),T.float32(0))+T.float32(1)
     for k in T.unroll(3):rgbh[k]=div(rgbh[k],denh);rgbu[k]=div(rgbu[k],denu)
     hist[0]=(T.float32(.25)*rgbh[0]+T.float32(.5)*rgbh[1])+T.float32(.25)*rgbh[2];hist[1]=T.float32(.5)*rgbh[0]-T.float32(.5)*rgbh[2];hist[2]=(-T.float32(.25)*rgbh[0]+T.float32(.5)*rgbh[1])-T.float32(.25)*rgbh[2]
     rgbh[0]=(T.float32(.25)*rgbu[0]+T.float32(.5)*rgbu[1])+T.float32(.25)*rgbu[2];rgbh[1]=T.float32(.5)*rgbu[0]-T.float32(.5)*rgbu[2];rgbh[2]=(-T.float32(.25)*rgbu[0]+T.float32(.5)*rgbu[1])-T.float32(.25)*rgbu[2]
     for k in T.unroll(3):blended[k]=mix(hist[k],rgbh[k],div(weight,mx(accum[k]+weight,T.float32(.001))))
    else:
     for k in T.unroll(3):blended[k]=mix(hist[k],up[k],div(weight,mx(accum[k]+weight,T.float32(.001))))
    rgbh[0]=(blended[0]+blended[1])-blended[2];rgbh[1]=blended[0]+blended[2];rgbh[2]=(blended[0]-blended[1])-blended[2]
    if HDR:
     inverse=mx(T.float32(1)-mx(mx(rgbh[0],rgbh[1]),rgbh[2]),T.float32(1/65504))
     for k in T.unroll(3):rgbh[k]=div(rgbh[k],inverse)
    rgbu[0]=(up[0]+up[1])-up[2];rgbu[1]=up[0]+up[2];rgbu[2]=(up[0]-up[1])-up[2]
    for k in T.unroll(3):
     value=div(T.if_then_else(isnew,rgbu[k],rgbh[k]),Params[0])*Params[1]
     Resolved[y,x,k]=value;History[y,x,k]=half_rtz(value)
    backx=ux-Motion[y,x,0];backy=uy-Motion[y,x,1]
    validback=T.And(T.And(backx>=T.float32(0),backx<=T.float32(1)),T.And(backy>=T.float32(0),backy<=T.float32(1)))
    life=T.if_then_else(validback,mx(life-weight*Params[5],T.float32(0)),T.float32(0))
    temporal=mn(thisreact,T.float32(.99));temporal=mx(temporal,mix(temporal,T.float32(.4),sat(velocity)));temporal=mx(temporal*temporal,mx(dc*T.float32(.1),reactive));temporal=T.if_then_else(isnew,T.float32(1),temporal)
    temporal=T.if_then_else(sat(velocity*T.float32(10))>=T.float32(1),-mx(temporal,T.float32(.001)),temporal)
    History[y,x,3]=half_rtz(temporal);Locks[y,x,0]=half_rtz(life);Locks[y,x,1]=half_rtz(updated)
    Luma[y,x,0]=unorm(current)
    for k in T.unroll(3):Luma[y,x,k+1]=unorm(lh[k])
    Weight[y,x,0]=weight;Reactivity[y,x,0]=thisreact
 return main

class Accumulate:
 def __init__(self,fsr):self.fsr=fsr;self.reference=REFERENCE_ACCUMULATE.__get__(fsr,type(fsr));self.cache={}
 def __call__(self,prepared,masks,dm,mip,newlocks,old,exposure,pre,jitter,phase,raw_motion):
  f=self.fsr
  if torch.cuda.current_stream(dm.device)!=torch.cuda.default_stream(dm.device):raise RuntimeError('Default CUDA stream required')
  h,w=dm.shape[:2];oh,ow=f.output_size;key=(h,w,oh,ow,dm.device)
  if key not in self.cache:
   p=ref.grid(oh,ow,dm.device).float();units=p.new_tensor([w,h]);display=p.new_tensor([ow,oh]);uv=(p+.5)/display
   self.cache[key]=(units,display,uv,(uv*units).long())
  units,display,uv,pos=self.cache[key];motion=raw_motion if f.display_mv else ref.load(dm,pos)
  velocity=ref.norm(motion*display);reprojected=uv+motion;lr_uv=uv+jitter/units
  dc=ref.sat(ref.sample(prepared,lr_uv)[...,3:4]);sampled_masks=ref.sample(masks,lr_uv)
  shade=(ref.sample(mip,uv).exp()*exposure).pow(1/6)
  if old:
   lut=None
   if f.use_lut:
    x=torch.arange(128,device=dm.device,dtype=torch.float32)*(2/127);y=ref.lanczos(x);lut=torch.sign(y)*(y.abs()*32767+.5).floor()/32767
   raw=history_sample(old.color,reprojected,lut);lock=ref.sample(old.locks,reprojected);lh=ref.sample(old.luma_history,reprojected)
  else:
   raw=dm.new_zeros(oh,ow,4);lock=dm.new_zeros(oh,ow,2);lh=dm.new_zeros(oh,ow,4)
  # Params exposure is a GPU scalar; scalar reciprocals use original F32 opmath.
  params=torch.cat((exposure.reshape(1),dm.new_tensor([pre,recip(old.pre_exposure) if old else 1.]),jitter.reshape(2),dm.new_tensor([recip(phase*(.74/12)),0.,0.])))
  sampled_shade=torch.cat((shade,newlocks),-1)
  inputs=(prepared,motion,velocity,dc,sampled_masks,raw,lock,sampled_shade,lh,params)
  if any(v.dtype!=torch.float32 for v in inputs):raise ValueError('F32 operands required')
  result=kernel(h,w,oh,ow,f.hdr,old is not None)(*(v.contiguous() for v in inputs))
  resolved,history,locks,luma,up,weight,mean,reactivity,hist=result
  f.diagnostics.update(prepared=prepared,masks=masks,new_locks=newlocks,upsampled=up,upsampled_weight=weight,box_mean=mean,depth_clip=dc,reactivity=reactivity,history_reprojected=hist)
  return resolved,history,locks,luma
