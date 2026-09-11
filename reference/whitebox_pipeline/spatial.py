"""FSR1 EASU FP32 12-tap algorithm in Torch, AMD MIT (FSR2_LICENSE.txt).
Source: pinned FSR2 SDK shaders/ffx_fsr1.h:144-385 and core GPU approximations.
Whole-image RGB, clamped gather borders. No tiling/global-NR equivalence claim.
"""
import torch
from .fsr2 import grid,load,sat

def rcp_low(a):return (0x7ef07ebb-a.contiguous().view(torch.int32)).view(torch.float32)
def rsqrt_low(a):return (0x5f347d74-(a.contiguous().view(torch.int32)>>1)).view(torch.float32)

@torch.inference_mode()
def easu(image,size):
    if image.ndim!=4 or image.shape[:2]!=(1,3):raise ValueError('EASU requires BCHW RGB, batch1')
    h,w=image.shape[-2:];oh,ow=map(int,size)
    if oh<h or ow<w:raise ValueError('EASU is an upsampler, not a downsampling filter')
    a=image[0].permute(1,2,0).float();p=grid(oh,ow,a.device).float()
    ratio=a.new_tensor([w/ow,h/oh]);pos=p*ratio+(.5*ratio-.5);base=pos.floor().long();frac=pos-base
    offsets={'b':(0,-1),'c':(1,-1),'i':(-1,1),'j':(0,1),'f':(0,0),'e':(-1,0),
             'k':(1,1),'l':(2,1),'h':(2,0),'g':(1,0),'o':(1,2),'n':(0,2)}
    taps={key:load(a,base+base.new_tensor(off),True) for key,off in offsets.items()}
    luma={key:c[...,2:3]*.5+(c[...,0:1]*.5+c[...,1:2]) for key,c in taps.items()}
    direction=a.new_zeros(oh,ow,2);length=a.new_zeros(oh,ow,1)
    fx,fy=frac[...,0:1],frac[...,1:2]
    for keys,weight in zip(('befgj','cfghk','fijkn','gjklo'),((1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy)):
        aa,b,c,d,e=(luma[k] for k in keys)
        dx=d-b;dy=e-aa
        lx=sat(dx.abs()*rcp_low(torch.maximum((d-c).abs(),(c-b).abs())))
        ly=sat(dy.abs()*rcp_low(torch.maximum((e-c).abs(),(c-aa).abs())))
        direction+=torch.cat((dx,dy),-1)*weight
        length+=lx.square()*weight;length+=ly.square()*weight
    magnitude=direction.square().sum(-1,keepdim=True);small=magnitude<1/32768
    inv=torch.where(small,torch.ones_like(magnitude),rsqrt_low(magnitude))
    direction=torch.cat((torch.where(small,torch.ones_like(magnitude),direction[...,:1]),direction[...,1:2]),-1)*inv
    length=(length*.5).square()
    stretch=direction.square().sum(-1,keepdim=True)*rcp_low(direction.abs().amax(-1,keepdim=True))
    axis=torch.cat((1+(stretch-1)*length,1-.5*length),-1)
    lobe=.5+(.25-.04-.5)*length;clip=rcp_low(lobe)
    total=a.new_zeros(oh,ow,3);weights=a.new_zeros(oh,ow,1)
    for key,off in offsets.items():
        v=frac.new_tensor(off)-frac
        rotated=torch.stack((v[...,0]*direction[...,0]+v[...,1]*direction[...,1],
                             v[...,0]*(-direction[...,1])+v[...,1]*direction[...,0]),-1)*axis
        d=torch.minimum(rotated.square().sum(-1,keepdim=True),clip)
        b=(.4*d-1).square();window=(lobe*d-1).square();weight=(25/16*b-(25/16-1))*window
        total+=taps[key]*weight;weights+=weight
    central=torch.stack([taps[k] for k in 'fgjk'])
    out=torch.minimum(central.amax(0),torch.maximum(central.amin(0),total/weights))
    return out.permute(2,0,1)[None].contiguous()
