"""FSR2 v2.2.1 temporal reconstruction in ordinary PyTorch.

Algorithm source: GPUOpen FidelityFX-FSR2, 1680d1edd5c034f88ebbbb793d8b88f8842cf804
Copyright (c) 2022-2023 Advanced Micro Devices, Inc. MIT; see FSR2_LICENSE.txt.
No shader, executable, custom CUDA or vendor runtime is used by this module.

The public interface is BCHW; internal tensors use HWC, one sequence per context.
All ordinary stages run: log-luma/SPD, nearest depth/MV + depth reconstruction,
depth clip/mask dilation, thin-feature locks, history reprojection/rectification,
accumulation/luma history and optional RCAS. F32 shader equations are used with
explicit F16/UNORM resource boundaries. Hardware sampler/ALU bit identity is not
claimed. External masks are supported; experimental opaque-only TCR autogen is
not exposed. Unsupported input contracts are rejected rather than substituted.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F

EPS=1e-3
MIN16=6.10e-5
MAX16=65504.
mx=torch.fmax
mn=torch.fmin

def sat(x):
    # HLSL saturate(NaN) is zero. No invented denominator epsilon.
    return mn(mx(x,torch.zeros_like(x)),torch.ones_like(x))
def half(x):
    # NVIDIA D3D12 typed R16 UAV stores in the reference use toward-zero.
    # Signed IEEE binary16 magnitude decreases by decrementing its bit pattern.
    # Preserve this resource boundary; do not change the F32 shader equations.
    q=x.half();bits=q.contiguous().view(torch.int16)
    return (bits-(q.float().abs()>x.abs()).to(torch.int16)).view(torch.float16).float()
def unorm(x):return sat(x).mul(255).round().div(255)
def mix(a,b,t):return a+(b-a)*t
def norm(x):return torch.linalg.vector_norm(x,dim=-1,keepdim=True)
def grid(h,w,device):
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    return torch.stack((x,y),-1)
def valid(p,h,w):return ((p[...,0]>=0)&(p[...,0]<w)&(p[...,1]>=0)&(p[...,1]<h))[...,None]
def uvinside(uv):return ((uv>=0)&(uv<=1)).all(-1,keepdim=True)
def load(a,p,clamp=False):
    h,w=a.shape[:2];p=p.long();v=a[p[...,1].clamp(0,h-1),p[...,0].clamp(0,w-1)]
    return v if clamp else torch.where(valid(p,h,w),v,torch.zeros_like(v))
def sample(a,uv):
    return F.grid_sample(a.permute(2,0,1)[None],(uv*2-1)[None],mode='bilinear',padding_mode='border',align_corners=False)[0].permute(1,2,0)
def bilinear_positions(uv,h,w):
    pos=uv*uv.new_tensor([w,h])-.5;base=pos.floor().long();f=pos-base
    for x,y in ((0,0),(1,0),(0,1),(1,1)):
        weight=(f[...,0:1] if x else 1-f[...,0:1])*(f[...,1:2] if y else 1-f[...,1:2])
        yield base+base.new_tensor([x,y]),weight

def ycocg(x):
    r,g,b=x.unbind(-1);return torch.stack((.25*r+.5*g+.25*b,.5*r-.5*b,-.25*r+.5*g-.25*b),-1)
def rgb(x):
    y,co,cg=x.unbind(-1);return torch.stack((y+co-cg,y+cg,y-co-cg),-1)
def luma(x):return (x*x.new_tensor([.2126,.7152,.0722])).sum(-1,keepdim=True)
def tonemap(x):return x/(x.amax(-1,keepdim=True).clamp_min(0)+1)
def inverse_tonemap(x):return x/(1-x.amax(-1,keepdim=True)).clamp_min(1/MAX16)
def minmax_ratio(a,b):
    m=mx(a,b);return torch.where(m!=0,mn(a,b)/m,torch.zeros_like(m))
def lanczos(x):
    x=x.abs().clamp_max(2);p=math.pi*x
    return torch.where(x<EPS,torch.ones_like(x),(p.sin()/p)*((p*.5).sin()/(p*.5)))
def approx_lanczos_sq(x):
    x=x.clamp_max(4);a=.4*x-1;b=.25*x-1
    return (1.5625*a*a-.5625)*(b*b)
def rcp_medium(x):
    b=(0x7ef19fff-x.contiguous().view(torch.int32)).view(torch.float32)
    return b*(-b*x+2)

def history_sample(a,uv,lut=None):
    h,w=a.shape[:2];p=uv*uv.new_tensor([w,h])-.5
    p=mx(torch.zeros_like(p),mn(p,p.new_tensor([w,h])))
    base=p.floor().long();frac=p-base
    def weight(x):
        if lut is None:return lanczos(x)
        q=(x.abs()/2*128-.5).clamp(0,127);i=q.floor().long();f=q-i
        return mix(lut[i],lut[(i+1).clamp_max(127)],f)
    wx=[weight(frac[...,0:1]-i) for i in (-1,0,1,2)]
    wy=[weight(frac[...,1:2]-i) for i in (-1,0,1,2)]
    rows=[];central=[]
    for j,y in enumerate((-1,0,1,2)):
        row=torch.zeros_like(load(a,base))
        for i,x in enumerate((-1,0,1,2)):
            # ClampCoord clamps only by offset direction; base in valid viewport
            # for the existing samples that consume this value.
            pp=base+base.new_tensor([x,y])
            pp=torch.stack((pp[...,0].clamp_min(0) if x<0 else pp[...,0].clamp_max(w-1) if x>0 else pp[...,0],
                            pp[...,1].clamp_min(0) if y<0 else pp[...,1].clamp_max(h-1) if y>0 else pp[...,1]),-1)
            s=load(a,pp);row=row+wx[i]*s
            if x in (0,1) and y in (0,1):central.append(s)
        rows.append(row/sum(wx))
    out=sum(w*r for w,r in zip(wy,rows))/sum(wy)
    bounds=torch.stack(central);return mn(mx(out,bounds.amin(0)),bounds.amax(0))

@dataclass
class FSR2State:
    color: torch.Tensor
    locks: torch.Tensor
    luma_history: torch.Tensor
    dilated_motion: torch.Tensor
    exposure_log: torch.Tensor
    pre_exposure: float
    jitter_xy: tuple
    phase_count: int
    frame_index: int

class FSR2:
    """Stateful full-frame temporal reconstructor; call reset() between sequences.

    color linear RGB, device-depth normal/reversed per constructor, motion
    current->previous PIXELS (not normalized UV). Jitter is in render pixels.
    Exposure is a positive scalar; None follows auto_exposure or source default1.
    Render dimensions may change only with explicit reset in this initial API.
    """
    def __init__(self,output_size,*,camera_near=.1,camera_far=100.,camera_fov_y=math.pi/3,
                 inverted_depth=False,infinite_depth=False,hdr=False,auto_exposure=False,
                 display_motion_vectors=False,jittered_motion_vectors=False,
                 view_space_to_meters=1.,lanczos_lut=False):
        self.output_size=tuple(map(int,output_size))
        self.near=float(camera_near);self.far=float(camera_far);self.fov=float(camera_fov_y)
        if min(self.output_size)<=0 or min(self.near,self.far)<=0 or self.near==self.far or math.isnan(self.near) or math.isnan(self.far) or (not infinite_depth and not all(math.isfinite(v) for v in (self.near,self.far))) or not 0<self.fov<math.pi:
            raise ValueError('invalid dimensions or camera')
        if not math.isfinite(view_space_to_meters) or view_space_to_meters<=0:raise ValueError('invalid meter scale')
        self.inverted=bool(inverted_depth);self.infinite=bool(infinite_depth);self.hdr=bool(hdr)
        self.auto_exposure=bool(auto_exposure);self.display_mv=bool(display_motion_vectors)
        self.jittered_mv=bool(jittered_motion_vectors);self.meters=float(view_space_to_meters)
        self.use_lut=bool(lanczos_lut);self.state=None;self.diagnostics={}
    def reset(self):self.state=None;self.diagnostics={}
    def _depth_factors(self,like):
        n,f=sorted((self.near,self.far))
        if self.inverted:n,f=f,n
        q=f/(n-f);c=q;e=q*n
        if self.infinite:
            c=(2**-23) if self.inverted else -1.-2**-23
            e=f if self.inverted else -n-2**-23
        return like.new_tensor([-c,e])
    def _z(self,d):
        ab=self._depth_factors(d);return ab[1]/(d-ab[0])
    def _luminance(self,color,jitter,pre,dt,old):
        h,w=color.shape[:2];p=grid(h,w,color.device).float();uv=(p+.5+jitter)/p.new_tensor([w,h])
        a=luma(sample(color,uv)/pre).clamp_min(EPS).log()
        # SPD processes 64x64 workgroups; off-screen source samples contribute 0.
        ph=math.ceil(h/64)*64;pw=math.ceil(w/64)*64
        a=F.pad(a.permute(2,0,1),(0,pw-w,0,ph-h)).permute(1,2,0)
        count=min(int(math.floor(math.log2(max(h,w)))),12)
        shading=None;last=None
        for level in range(count):
            ah,aw=a.shape[:2]
            a=F.pad(a.permute(2,0,1),(0,aw%2,0,ah%2)).permute(1,2,0)
            a=(a[0::2,0::2]+a[0::2,1::2]+a[1::2,0::2]+a[1::2,1::2])*.25
            if level==4:shading=half(a[:max(1,h//32),:max(1,w//32)])
            last=a[0,0,0]
            if level==5 and count>6:
                # Global continuation reads the published R16 mip5 resource.
                a=half(a[:max(1,h//64),:max(1,w//64)])
        if shading is None:raise ValueError('FSR2 shading-change mip requires render axes >=32')
        lavg=last if old is None else mix(old.exposure_log,last,1-math.exp(-dt))
        exposure=1/((78/(.65*100))*torch.exp2(torch.log2(lavg.exp()*100/12.5)))
        return shading,lavg,exposure
    def _reconstruct(self,depth,mv,color,exposure,pre,jitter,first_execution):
        h,w=depth.shape[:2];oh,ow=self.output_size;p=grid(h,w,depth.device);units=depth.new_tensor([w,h]);display=depth.new_tensor([ow,oh])
        nearest=depth.clone();coords=p.clone()
        for x,y in ((1,0),(0,1),(0,-1),(-1,0),(-1,1),(1,1),(-1,-1),(1,-1)):
            q=p+p.new_tensor([x,y]);d=load(depth,q)
            take=valid(q,h,w)&((d>nearest) if self.inverted else (d<nearest))
            nearest=torch.where(take,d,nearest);coords=torch.where(take, q, coords)
        if self.display_mv:coords=((coords.float()+.5-jitter)/units*display).floor().long()
        dilated=load(mv,coords)
        motion=dilated*(norm(dilated*display)>.1)
        # The pinned DX12 implementation does not clear this resource before
        # its first dispatch: committed resources start zeroed. Lock clears it
        # to far-Z for subsequent frames. A first-frame output ignores history,
        # but its depth-clip diagnostic still reflects this actual lifecycle.
        initial=0. if first_execution or self.inverted else 1.
        reconstructed=torch.full((h*w,1),initial,device=depth.device)
        for q,weight in bilinear_positions((p+.5)/units+motion,h,w):
            take=(valid(q,h,w)&(weight>.01))[...,0]
            index=(q[...,1]*w+q[...,0])[take]
            reconstructed.scatter_reduce_(0,index[:,None],nearest[take],reduce='amax' if self.inverted else 'amin',include_self=True)
        prepared=(color.clamp_min(0)/pre*exposure)
        if self.hdr:prepared=tonemap(prepared)
        lum=luma(prepared);perceived=torch.where(lum<=216/24389,lum*(24389/27),lum.pow(1/3)*116-16)*.01
        return nearest,half(dilated),reconstructed.reshape(h,w,1),half(perceived.pow(1/6))
    def _depth_clip(self,color,depth,dm,dd,reconstructed,mv,old,reactive,composition,pre,exposure,jitter):
        h,w=depth.shape[:2];oh,ow=self.output_size;p=grid(h,w,depth.device);units=depth.new_tensor([w,h]);display=depth.new_tensor([ow,oh])
        motion=dm*(norm(dm*display)>.01);uv=(p+.5)/units+motion;z=self._z(dd)
        total=torch.zeros_like(z);ws=total.clone()
        # Center and corner at the same plane depth; Z cancels in Kfov.
        tan_y=math.tan(self.fov*.5);tan_x=tan_y*w/h
        center_ndc=depth.new_tensor([2*(w//2)/w-1,1-2*(h//2)/h])
        kfov=math.sqrt(tan_x*tan_x+tan_y*tan_y+1)/math.sqrt(float((center_ndc*depth.new_tensor([tan_x,tan_y])).square().sum())+1)
        power=1+2*min(1,math.hypot(w,h)/math.hypot(1920,1080))
        for q,weight in bilinear_positions(uv,h,w):
            prevz=self._z(load(reconstructed,q));diff=z-prevz
            take=valid(q,h,w)&(weight>.01)&(diff>0)
            sep=1.37e-5*kfov*math.hypot(w,h)*mx(z,prevz)
            total+=torch.where(take,sat(sep/diff).pow(power)*weight,torch.zeros_like(z));ws+=torch.where(take,weight,torch.zeros_like(z))
        clip=torch.where(ws>0,sat(1-total/ws),torch.zeros_like(z))
        d0=self._z(load(reconstructed,p+p.new_tensor([0,-1])));d1=self._z(reconstructed);d2=self._z(load(reconstructed,p+p.new_tensor([0,1])))
        clip*=~(((d0-d1)>d1*.01)&((d1-d2)>d2*.01))
        q=p if not self.display_mv else ((p+.5-jitter)/units*display).floor().long()
        nucleus=load(mv,q);velocity=norm(nucleus*units);maxv=norm(nucleus);convergence=torch.ones_like(z)
        mh,mw=mv.shape[:2]
        for y in (-1,0,1):
            for x in (-1,0,1):
                pos=q+q.new_tensor([x,y])
                # DepthClip explicitly passes RenderSize even for display MV.
                px=pos[...,0].clamp_min(0) if x<0 else pos[...,0].clamp_max(w-1) if x>0 else pos[...,0]
                py=pos[...,1].clamp_min(0) if y<0 else pos[...,1].clamp_max(h-1) if y>0 else pos[...,1]
                other=load(mv,torch.stack((px,py),-1));v=norm(other);maxv=mx(v,maxv);v=mx(v,maxv)
                convergence=mn(convergence,(other/v*nucleus/v).sum(-1,keepdim=True))
        convergence=torch.where(velocity>1e-2,convergence,torch.ones_like(z))
        divergence=sat(1-convergence)*sat(maxv/.01)
        previous=sample(old.dilated_motion,(p+.5)/units+dm) if old else torch.zeros_like(dm)
        dist=norm(dm*display)
        temporal=torch.where(dist>1,(1-sat(norm(previous)/norm(dm)))*sat((dist/20).pow(3)),torch.zeros_like(z))
        maxdist=self._z(depth.new_tensor(0. if self.inverted else 1.))*self.meters
        mind=torch.ones_like(z)*maxdist;maxd=torch.zeros_like(z);farfound=torch.zeros_like(z,dtype=torch.bool)
        for y in (-1,0,1):
            for x in (-1,0,1):
                q=p+p.new_tensor([x,y]);d=self._z(load(dd,q))*self.meters*valid(q,h,w)
                farfound|=(d==maxdist);mind=mn(mind,d);maxd=mx(maxd,d)
        depthdiv=(1-mind/maxd)*(~farfound)
        divergence=mx(divergence,sat(temporal-depthdiv))
        r=torch.zeros_like(z);tc=divergence
        for y in (-1,0,1):
            for x in (-1,0,1):
                q=p+p.new_tensor([x,y]);c=load(color,q,True)
                denom=mx(color.square().sum(-1,keepdim=True),c.square().sum(-1,keepdim=True))
                similarity=(color*c).sum(-1,keepdim=True)/denom
                power=1+(6-similarity*6)
                r=mx(r,load(reactive,q,True).pow(power));tc=mx(tc,load(composition,q,True).pow(power))
        masks=unorm(torch.cat((r,tc),-1))
        prepared=half(torch.cat((ycocg((color.clamp_min(0)/pre*exposure).clamp(0,MAX16)),clip),-1))
        return prepared,masks
    def _locks(self,lock_luma,jitter):
        h,w=lock_luma.shape[:2];oh,ow=self.output_size;p=grid(h,w,lock_luma.device)
        bits=torch.full((h,w,1),1<<4,dtype=torch.int32,device=p.device)
        low=torch.full_like(lock_luma,torch.finfo(torch.float32).max);high=torch.zeros_like(low)
        for y in (-1,0,1):
            for x in (-1,0,1):
                if x==0 and y==0:continue
                s=load(lock_luma,p+p.new_tensor([x,y]),True);ratio=mx(s,lock_luma)/mn(s,lock_luma)
                same=(ratio>0)&(ratio<1.05);bits|=same.to(torch.int32)<<(3*(y+1)+x+1)
                low=torch.where(same,low,mn(low,s));high=torch.where(same,high,mx(high,s))
        ridge=(lock_luma>high)|(lock_luma<low)
        for indices in ((0,1,3,4),(1,2,4,5),(3,4,6,7),(4,5,7,8)):
            mask=sum(1<<i for i in indices);ridge&=((bits&mask)!=mask)
        q=((p+.5-jitter)/p.new_tensor([w,h])*p.new_tensor([ow,oh])).floor().long()
        take=(ridge&valid(q,oh,ow))[...,0];out=torch.zeros(oh*ow,1,device=p.device)
        out.scatter_reduce_(0,(q[...,1]*ow+q[...,0])[take][:,None],torch.ones_like(lock_luma[take]),reduce='amax',include_self=True)
        return out.reshape(oh,ow,1)
    def _upsample(self,prepared,reactive,isnew,depthclip,velocity,jitter):
        h,w=prepared.shape[:2];oh,ow=self.output_size;p=grid(oh,ow,prepared.device).float();ds=p.new_tensor([w/ow,h/oh])
        source=(p+.5)*ds;base=source.floor().long();unjittered=base+.5-jitter
        flip=unjittered>source;tl=torch.where(flip,-2,-1);offset0=unjittered-source
        kr=mx(reactive,isnew.float());biasmax=min(1.99,ow/w)*(1-kr)
        biasmin=mx(torch.ones_like(kr),(1+biasmax)*.3)
        bias=mix(biasmax,biasmin,mx(.25*depthclip,kr))
        curve=mix(-2.,-3.,sat(velocity/50))
        color=torch.zeros(oh,ow,3,device=p.device);weight=torch.zeros_like(kr)
        mean=torch.zeros_like(color);moment=torch.zeros_like(color);bw=torch.zeros_like(weight)
        lo=None;hi=None
        for row in range(3):
            for col in range(3):
                cr=torch.stack((torch.where(flip[...,0],3-col,col),torch.where(flip[...,1],3-row,row)),-1)
                off=tl+cr;coord=base+off;s=load(prepared,coord)[...,:3]
                delta=offset0+off;distance=delta.square().sum(-1,keepdim=True)
                wt=valid(coord,h,w)*approx_lanczos_sq(distance*bias*bias)
                color+=s*wt;weight+=wt
                boxweight=(curve*distance).exp();mean+=s*boxweight;moment+=s*s*boxweight;bw+=boxweight
                lo=s if lo is None else mn(lo,s);hi=s if hi is None else mx(hi,s)
        bw=torch.where(bw.abs()>EPS,bw,torch.ones_like(bw));mean/=bw;std=(moment/bw-mean.square()).abs().sqrt()
        good=weight>EPS;color=torch.where(good,mn(mx(color/weight,lo),hi),color)
        weight=torch.where(good,weight/12,torch.zeros_like(weight))
        return color,weight,(mean,std,lo,hi)
    def _accumulate(self,prepared,masks,dm,mip,newlocks,old,exposure,pre,jitter,phase,raw_motion):
        h,w=dm.shape[:2];oh,ow=self.output_size;p=grid(oh,ow,dm.device).float();units=p.new_tensor([w,h]);display=p.new_tensor([ow,oh])
        uv=(p+.5)/display;lr_uv=uv+jitter/units
        motion=raw_motion if self.display_mv else load(dm,(uv*units).long())
        velocity=norm(motion*display);reprojected=uv+motion
        existing=uvinside(reprojected);isnew=~existing if old else torch.ones_like(existing)
        dc=sat(sample(prepared,lr_uv)[...,3:4]);m=sample(masks,lr_uv);reactive=m[...,:1];accmask=m[...,1:2]
        hist=torch.zeros(oh,ow,3,device=p.device);temporal=torch.zeros_like(reactive);wasmoving=torch.zeros_like(existing);lock=torch.zeros(oh,ow,2,device=p.device)
        if old:
            lut=None
            if self.use_lut:
                x=torch.arange(128,device=p.device,dtype=torch.float32)*(2/127);y=lanczos(x)
                lut=torch.sign(y)*(y.abs()*32767+.5).floor()/32767
            raw=history_sample(old.color,reprojected,lut)
            hist=torch.where(existing,ycocg((raw[...,:3]/old.pre_exposure*exposure).clamp(0,MAX16)),hist)
            temporal=torch.where(existing,sat(raw[...,3:4].abs()),temporal);wasmoving=existing&(raw[...,3:4]<0)
            lock=torch.where(existing,sample(old.locks,reprojected),lock)
        thisreact=mx(reactive,temporal);newlock=(newlocks>127/255)&existing&(~isnew)
        shade=(sample(mip,uv).exp()*exposure).pow(1/6)
        life=lock[...,:1];lum=torch.where(lock[...,1:2]==0,shade,lock[...,1:2]);diff=1-minmax_ratio(lum,shade)
        newlife=torch.where(life!=0,torch.full_like(life,2.),torch.ones_like(life))
        updated_lum=torch.where(newlock,shade,torch.where(life<=1,mix(lum,shade,.5),lum))
        life=torch.where(newlock,newlife,torch.where((life>1)&(diff>.1),torch.zeros_like(life),life))
        thisreact=mx(thisreact,sat((diff-.1)*10));life=life*(1-thisreact)*sat(1-accmask)*(dc<.1)
        lockcon=sat(sat(sat(life-1)*4)*sat(minmax_ratio(updated_lum,shade)))
        up,weight,(mean,std,lo,hi)=self._upsample(prepared,thisreact,isnew,dc,velocity,jitter)
        # Four-frame UNORM luma history and oscillation protection.
        current=mean[...,:1]
        if self.hdr:current=current/(1+current.clamp_min(0))
        current=(current*255).round()/255
        use_luma=(mx(mx(dc,accmask),diff)<.1)&(~isnew)
        lh=sample(old.luma_history,reprojected) if old else torch.zeros(oh,ow,4,device=p.device)
        lh=torch.where(use_luma,lh,torch.zeros_like(lh));d0=current-lh[...,:1];minimum=d0.abs()
        for k in range(1,4):
            d=current-lh[...,k:k+1];minimum=torch.where(d0.sign()==d.sign(),mn(minimum,d.abs()),minimum)
        instability=(minimum!=d0.abs()).float()*sat(std[...,:1]/.1).pow(6)
        instability=(instability>1/255).float()*(d0.abs()>=1/255)*(1-mx(accmask,thisreact.pow(1/6)))
        luma_next=torch.cat((current,lh[...,:3]),-1);instability*=luma_next[...,3:4]!=0
        accumulation=existing.float()*(1-thisreact)*(1-dc)
        accumulation=mn(accumulation,mix(accumulation,weight*10,mx(wasmoving.float(),sat(velocity*10))))
        accumulation=mn(accumulation,mix(accumulation,weight,sat(velocity/20))).expand_as(hist).clone()
        influence=min(20,(1/(w/ow*h/oh))**3);boxscale=mix(influence,1.,mx(dc,mx(accmask,sat(velocity/20))))
        boxlo=mx(lo,mean-std*boxscale);boxhi=mn(hi,mean+std*boxscale)
        outside=((hist<boxlo)|(hist>boxhi)).any(-1,keepdim=True)
        contribution=sat(mx(instability,lockcon)*(1-reactive.sqrt()))
        hist=torch.where(outside,mix(mn(mx(hist,boxlo),boxhi),hist,contribution),hist)
        accumulation=torch.where(outside,mix(mn(accumulation,accumulation.new_full((),.1)),accumulation,contribution),accumulation)
        total=(accumulation+weight).clamp_min(EPS)
        if self.hdr:
            blended=mix(ycocg(tonemap(rgb(hist))),ycocg(tonemap(rgb(up))),weight/total)
            resolved=inverse_tonemap(rgb(blended))
        else:resolved=rgb(mix(hist,up,weight/total))
        resolved=torch.where(isnew,rgb(up),resolved)/exposure*pre
        life=torch.where(uvinside(uv-motion),(life-weight/(phase*(.74/12))).clamp_min(0),torch.zeros_like(life))
        temporal=mn(thisreact,thisreact.new_full((),.99))
        temporal=mx(temporal,mix(temporal,.4,sat(velocity)))
        temporal=mx(temporal.square(),mx(dc*.1,reactive));temporal=torch.where(isnew,torch.ones_like(temporal),temporal)
        temporal=torch.where(sat(velocity*10)>=1,-temporal.clamp_min(EPS),temporal)
        self.diagnostics.update(prepared=prepared,masks=masks,new_locks=newlocks,upsampled=up,upsampled_weight=weight,
                                box_mean=mean,depth_clip=dc,reactivity=thisreact,history_reprojected=hist)
        return resolved,half(torch.cat((resolved,temporal),-1)),half(torch.cat((life,updated_lum),-1)),unorm(luma_next)
    def _rcas(self,history,exposure,pre,sharpness):
        a=(history[...,:3]/pre*exposure).clamp(0,MAX16);h,w=a.shape[:2];p=grid(h,w,a.device)
        b,d,f,hh=[load(a,p+p.new_tensor(o)) for o in ((0,-1),(-1,0),(1,0),(0,1))]
        ll=lambda x:x[...,2:3]*.5+(x[...,0:1]*.5+x[...,1:2])
        bl,dl,el,fl,hl=map(ll,(b,d,a,f,hh));vals=torch.stack((bl,dl,el,fl,hl))
        noise=1-.5*sat((.25*bl+.25*dl+.25*fl+.25*hl-el).abs()*rcp_medium(vals.amax(0)-vals.amin(0)))
        ring=torch.stack((b,d,f,hh));low=ring.amin(0);high=ring.amax(0)
        lobes=mx(-low/(4*high),(1-high)/(4*low-4))
        lobe=mx(mn(lobes.amax(-1,keepdim=True),lobes.new_zeros(())),lobes.new_full((),-.1875))*(2**(-2+2*sharpness))*noise
        return (lobe*b+lobe*d+lobe*hh+lobe*f+a)*rcp_medium(4*lobe+1)/exposure*pre
    @torch.inference_mode()
    def process(self,color,depth,motion,*,jitter_xy=(0.,0.),delta_ms=1000/60,reset=False,
                reactive=None,composition=None,exposure=None,pre_exposure=1.,sharpness=0.):
        if color.ndim!=4 or color.shape[0]!=1 or color.shape[1]!=3:raise ValueError('color requires 1x3xHxW linear RGB')
        h,w=color.shape[-2:];oh,ow=self.output_size
        if min(h,w)<32 or h>oh or w>ow:raise ValueError('render must be >=32 on each axis and <=output')
        if depth.shape!=(1,1,h,w):raise ValueError('device depth requires 1x1xrenderHxW')
        mh,mw=self.output_size if self.display_mv else (h,w)
        if motion.shape!=(1,2,mh,mw):raise ValueError('motion dimensions do not match declared grid')
        if not math.isfinite(pre_exposure) or pre_exposure<=0 or not 0<=sharpness<=1 or not math.isfinite(delta_ms):raise ValueError('invalid exposure/time/sharpness')
        if len(jitter_xy)!=2 or not all(math.isfinite(v) for v in jitter_xy):raise ValueError('invalid jitter')
        for a in (color,depth,motion,reactive,composition):
            if a is not None and (a.device!=color.device or not bool(torch.isfinite(a).all())):raise ValueError('nonfinite or cross-device input')
        if not bool(((depth>=0)&(depth<=1)).all()):raise ValueError('device depth must be [0,1]')
        def mask(a):
            if a is None:return color.new_zeros(h,w,1)
            if a.shape!=(1,1,h,w) or not bool(((a>=0)&(a<=1)).all()):raise ValueError('mask requires [0,1] 1x1xrenderHxW')
            return a[0].permute(1,2,0).float()
        r,t=mask(reactive),mask(composition)
        old=None if reset else self.state
        if old is not None and (old.dilated_motion.shape[:2]!=(h,w) or old.color.device!=color.device):raise ValueError('size/device change requires reset')
        c=color[0].permute(1,2,0).float();d=depth[0].permute(1,2,0).float();mv=motion[0].permute(1,2,0).float()/c.new_tensor([mw,mh]);j=c.new_tensor(jitter_xy)
        if self.jittered_mv:
            prevj=c.new_tensor(self.state.jitter_xy if self.state else (0.,0.));mv=mv-(prevj-j)/c.new_tensor([mw,mh])
        mip,lavg,auto=self._luminance(c,j,pre_exposure,max(0,min(1,delta_ms/1000)),old)
        e=auto if self.auto_exposure else c.new_tensor(1. if exposure is None or exposure==0 else exposure)
        if e.numel()!=1 or not bool(torch.isfinite(e)&(e>0)):raise ValueError('invalid effective exposure')
        dd,dm,rd,lock_luma=self._reconstruct(d,mv,c,e,pre_exposure,j,self.state is None)
        prepared,masks=self._depth_clip(c,d,dm,dd,rd,mv,old,r,t,pre_exposure,e,j)
        newlocks=self._locks(lock_luma,j);targetphase=int(8*(ow/w)**2)
        phase=targetphase if old is None else old.phase_count+(1 if targetphase>old.phase_count else -1 if targetphase<old.phase_count else 0)
        self.diagnostics={'dilated_depth':dd,'dilated_motion':dm,'reconstructed_depth':rd,'lock_luma':lock_luma,'luma_mip4':mip,'auto_exposure':auto}
        resolved,history,locks,lh=self._accumulate(prepared,masks,dm,mip,newlocks,old,e,pre_exposure,j,phase,mv)
        output=self._rcas(history,e,pre_exposure,sharpness) if sharpness>0 else resolved
        if not bool(torch.isfinite(output).all()):raise FloatingPointError('FSR2 nonfinite output; previous state not committed')
        self.state=FSR2State(history,locks,lh,dm,lavg,float(pre_exposure),tuple(jitter_xy),phase,0 if old is None else old.frame_index+1)
        return output.permute(2,0,1)[None].contiguous()
