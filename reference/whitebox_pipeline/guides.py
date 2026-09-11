"""Readable PyTorch port of DLSS5_Feed.fx guide decisions and history.

Source: Jean-Laurent ROUZIES, DLSS5-Feeder 03da7d9 (MIT).
Normalized linear depth is REQUIRED here; it is neither device-Z nor meters.
Motion input/output is current->previous in color-grid pixels. History retains
unfiltered provider UV, matching the FX rather than the public pixel interface.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class GuideConfig:
    validate: bool=True
    static: bool=True
    luma: bool=False
    depth: bool=True
    consistency: bool=True
    static_bias: float=.15
    min_contrast: float=.012
    luma_tolerance: float=.25
    depth_tolerance: float=.10
    mv_consistency: float=1.4
    mask_strength: float=1.
    sign_x: float=1.
    sign_y: float=1.
    mv_scale: float=1.
    geometry: bool=False
    geometry_diagnostics: bool=False
    geom_parallax: float=.02
    geom_outlier_px: float=4.
    geom_agree_px: float=1.5
    geom_dynamic_margin: float=.25
    geom_mask_rejected: float=.35

def _uv(h,w,device):
    y,x=torch.meshgrid((torch.arange(h,device=device,dtype=torch.float32)+.5)/h,
                       (torch.arange(w,device=device,dtype=torch.float32)+.5)/w,indexing='ij')
    return torch.stack((x,y),-1)[None]

def _sample(a,uv,linear=False):
    return F.grid_sample(a.float(),uv*2-1,mode='bilinear' if linear else 'nearest',padding_mode='border',align_corners=False)

def _patch(a,uv,linear=False):
    h,w=a.shape[-2:]
    return torch.stack([_sample(a,uv+uv.new_tensor([x/w,y/h]),linear) for y in (-1,0,1) for x in (-1,0,1)],dim=2)

def _patch_error(luma,prev,uv,puv):
    c,p=_patch(luma,uv),_patch(prev,puv,True)
    c,p=c-c.mean(2,keepdim=True),p-p.mean(2,keepdim=True)
    return (c-p).abs().mean(2),c.abs().mean(2)

def resize_flow(flow,size,*,linear=True):
    """Resize a PIXEL flow and its vector units together, including 1/8 providers."""
    h,w=flow.shape[-2:];oh,ow=size
    a=F.interpolate(flow.float(),size=size,mode='bilinear',align_corners=False) if linear else F.interpolate(flow.float(),size=size,mode='nearest')
    return a*a.new_tensor([ow/w,oh/h])[None,:,None,None]

def _basis(uv,depth,s):
    x,y=uv[...,0]-.5,uv[...,1]-.5
    z=s/(depth[:,0]+s)
    return torch.stack((torch.ones_like(x),x,y,x*x,x*y,y*y,z,x*z,y*z),-1)

class GuideProcessor:
    def __init__(self,config=None):
        self.config=config or GuideConfig();self.reset()
    def reset(self):
        self.previous=None
    def _fit(self,depth,flow_uv,shape):
        cfg=self.config;b,_,h,w=flow_uv.shape
        uv=_uv(23,40,depth.device).expand(b,-1,-1,-1)
        d=_sample(depth,uv);f=_sample(flow_uv,uv).permute(0,2,3,1).reshape(b,-1,2)
        a=_basis(uv,d,cfg.geom_parallax).reshape(b,-1,9)
        units=f.new_tensor([w,h]);valid=(d[:,0].flatten(1)>.001)&((f*units).abs()<512).all(-1)
        p=f.new_zeros(b,9,2);used=f.new_zeros(b);share=used.clone();rms=used.clone()
        # Native solve uses an ordered 920-sample normal equation + pivoting.
        # Torch batching preserves the equations; reduction/solve ISA is not emulated.
        for iteration in range(2):
            residual=torch.linalg.vector_norm((a@p-f)*units,dim=-1)
            take=valid if iteration==0 else valid&(residual<=cfg.geom_outlier_px)
            n=take.sum(1);weighted=a*take[...,None]
            normal=a.transpose(1,2)@weighted
            rhs=weighted.transpose(1,2)@f
            normal=normal+torch.eye(9,device=a.device)[None]*(1e-5*n)[:,None,None]
            aug=torch.cat((normal,rhs),-1)
            good=n>=40
            ids=torch.arange(b,device=a.device)
            for col in range(9):
                pivot=aug[:,col:,col].abs().argmax(1)+col
                best=aug[ids,pivot,col].abs();good=good&(best>=1e-12)
                row=aug[:,col].clone();aug[:,col]=aug[ids,pivot];aug[ids,pivot]=row
                divisor=aug[:,col,col]
                # Singular batches are not accepted; divisor replacement is only
                # to avoid calculating NaN on lanes whose old model is retained.
                divisor=torch.where(good,divisor,torch.ones_like(divisor))
                aug[:,col]=aug[:,col]/divisor[:,None]
                factor=aug[:,:,col].clone();factor[:,col]=0
                aug=aug-factor[:,:,None]*aug[:,col:col+1]
            p=torch.where(good[:,None,None],aug[:,:,9:],p)
            used=torch.where(good,n.float(),used);share=torch.where(good,n/920.,share)
            if iteration:
                r=torch.sqrt((residual.square()*take).sum(1)/n.clamp_min(1))
                rms=torch.where(good,r,rms)
        return p,{'used':used,'inlier_share':share,'rms_px':rms,'usable':(used>=40)&(share>=.25)}

    @torch.inference_mode()
    def process(self,color,provider_flow,linear_depth,*,reset=False):
        cfg=self.config
        if color.ndim!=4 or color.shape[1]<3:raise ValueError('color must be BCHW RGB')
        b,_,h,w=color.shape
        if provider_flow.shape!=(b,2,h,w) or linear_depth.shape!=(b,1,h,w):raise ValueError('guides must use color grid; depth must be normalized linear depth')
        if not all(bool(torch.isfinite(a).all()) for a in (color,provider_flow,linear_depth)):raise ValueError('nonfinite input')
        if not bool(((linear_depth>=0)&(linear_depth<=1)).all()):raise ValueError('linear_depth must use declared normalized [0,1] convention, not meters')
        luma=(color[:,:3].float()*color.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        uv=_uv(h,w,color.device).expand(b,-1,-1,-1);units=color.new_tensor([w,h])[None,:,None,None]
        flow_uv=provider_flow.float()/units
        if reset:self.reset()
        if self.previous is not None and (self.previous[0].shape!=luma.shape or self.previous[0].device!=color.device):
            raise ValueError('guide dimensions/device changed; explicit reset required')
        first=self.previous is None
        # The FX lacks an explicit first-history rule. Our declared adapter policy
        # is cold-start passthrough flow + full distrust, then genuine history.
        previous=(luma,linear_depth.float(),flow_uv) if first else self.previous
        pl,pd,pm=previous
        puv=uv+flow_uv.permute(0,2,3,1)
        inside=((puv>=0)&(puv<=1)).all(-1)[:,None]
        bad=torch.zeros(b,4,h,w,device=color.device)
        if cfg.static:
            es,contrast=_patch_error(luma,pl,uv,uv);ef,_=_patch_error(luma,pl,uv,puv)
            winner=(es+.25*contrast<=ef*(1+cfg.static_bias))&(contrast>=cfg.min_contrast)&(torch.linalg.vector_norm(provider_flow,dim=1,keepdim=True)>.5)
            bad[:,3:4]=winner.float()
        if cfg.luma:
            patch=_patch(luma,uv);lo=patch.amin(2);hi=patch.amax(2)
            prev=_sample(pl,puv,True);margin=cfg.luma_tolerance*hi.clamp_min(.05)+2/255
            bad[:,0:1]=(torch.maximum(lo-prev,prev-hi)/margin).clamp(0,1)
        if cfg.depth:
            dp=_sample(pd,puv);tol=cfg.depth_tolerance*linear_depth.clamp_min(.001)
            bad[:,1:2]=((dp-linear_depth).abs()-tol).div(tol+1e-5).clamp(0,1)*(linear_depth<.999)
        if cfg.consistency and cfg.mv_consistency>0:
            prev=_sample(pm,puv);diff=torch.linalg.vector_norm((flow_uv-prev)*units,dim=1,keepdim=True)
            allow=cfg.mv_consistency+.5*torch.linalg.vector_norm(provider_flow,dim=1,keepdim=True)
            bad[:,2:3]=((diff-allow)/allow).clamp(0,1)
        bad=torch.where(inside,bad,bad.new_tensor([1,0,0,0])[None,:,None,None])
        normal_mv=flow_uv*(1-bad[:,1:].amax(1,keepdim=True)) if cfg.validate else flow_uv
        distrust=bad[:,:3].amax(1,keepdim=True) if cfg.validate else torch.zeros_like(luma)
        fit=None;decision=torch.zeros_like(luma)
        if cfg.geometry or cfg.geometry_diagnostics:
            p,fit=self._fit(linear_depth,flow_uv,(h,w))
            pred=torch.einsum('bhwk,bkc->bchw',_basis(uv,linear_depth,cfg.geom_parallax),p)
            difference=torch.linalg.vector_norm((flow_uv-pred)*units,dim=1,keepdim=True)
            agree=cfg.geom_agree_px+.1*torch.linalg.vector_norm(pred*units,dim=1,keepdim=True)
            ep,cp=_patch_error(luma,pl,uv,uv+pred.permute(0,2,3,1));ef,_=_patch_error(luma,pl,uv,puv)
            dynamic=(cp>=cfg.min_contrast)&(ef<=ep*(1-cfg.geom_dynamic_margin)-1/255)
            disagree=difference>agree
            useflow=disagree&dynamic
            geom_mv=torch.where(useflow,flow_uv,pred)
            geom_bad=torch.where(disagree&~dynamic,cfg.geom_mask_rejected*((difference-agree)/(4*agree)).clamp(0,1),torch.zeros_like(luma))
            if cfg.depth:
                guv=uv+geom_mv.permute(0,2,3,1);gi=((guv>=0)&(guv<=1)).all(-1)[:,None]
                tol=cfg.depth_tolerance*linear_depth.clamp_min(.001)
                gb=((_sample(pd,guv)-linear_depth).abs()-tol).div(tol+1e-5).clamp(0,1)*gi*(linear_depth<.999)
                geom_bad=torch.maximum(geom_bad,gb)
            usable=fit['usable'][:,None,None,None]
            if cfg.geometry:
                normal_mv=torch.where(usable,geom_mv,normal_mv);distrust=torch.where(usable,geom_bad,distrust)
            decision=torch.where(usable,torch.where(disagree,torch.where(dynamic,1.,2.),0.),-1.)
        if first:
            normal_mv=flow_uv;distrust=torch.ones_like(luma)
        scale=color.new_tensor([cfg.sign_x,cfg.sign_y])[None,:,None,None]*cfg.mv_scale
        output=(normal_mv*units*scale).half().float()
        # R8_UNORM publication, round-to-nearest-even on finite [0,1].
        mask=(distrust*cfg.mask_strength).clamp(0,1).mul(255).round().div(255)
        self.previous=tuple(a.detach().half().float().clone() for a in (luma,linear_depth,flow_uv))
        return {'motion':output,'distrust':mask,'tests':bad,'geometry':fit,'geometry_decision':decision,'cold_start':first,
                'history_policy':'input luma/normalized linear depth/raw provider UV; FP16; cold start full distrust'}
