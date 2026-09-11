"""Hash-locked, local-only RGB guide inference in Torch.

RAFT-small C_T_V2 (torchvision0.20.1) and Metric Video Depth Anything Small.
The VDA neural architecture is vendored with notices; streaming cache selection
matches the experimental upstream single-frame algorithm, not offline VDA.
Metric output is an ESTIMATE in meters, never ground-truth device depth.
"""
from dataclasses import dataclass
import hashlib,json,math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]

def checked_models(directory=None):
    root=Path(directory) if directory else ROOT/'model'
    manifest=json.loads((root/'guide_models.json').read_text(encoding='utf-8'))
    result={}
    for info in manifest['models']:
        file=root/info['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('Guide model hash mismatch: '+str(file))
        result[info['name']]=(file,info)
    return result,manifest

def valid_rgb(rgb):
    if rgb.ndim!=4 or rgb.shape[:2]!=(1,3) or not bool(torch.isfinite(rgb).all()) or not bool(((rgb>=0)&(rgb<=1)).all()):
        raise ValueError('Estimator input must be finite 1x3xHxW sRGB [0,1]')

def depth_network_size(h,w,input_size=518):
    ratio=max(h,w)/min(h,w)
    if ratio>1.78:input_size=round(int(input_size*1.777/ratio)/14)*14
    if input_size<14:raise ValueError('Aspect ratio too extreme for VDA preprocessing')
    scale=max(input_size/h,input_size/w)
    def multiple(x):
        value=round(x/14)*14
        return math.ceil(x/14)*14 if value<input_size else value
    return multiple(h*scale),multiple(w*scale)

class RaftSmall(nn.Module):
    def __init__(self,checkpoint,*,device='cuda',longest_side=512,updates=12):
        super().__init__()
        from torchvision.models.optical_flow import raft_small
        if longest_side<128 or updates<1:raise ValueError('Invalid RAFT work size/update count')
        self.network=raft_small(weights=None,progress=False)
        self.network.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.network.to(device).eval();self.limit=int(longest_side);self.updates=int(updates)
    @torch.inference_mode()
    def forward(self,current,previous):
        valid_rgb(current);valid_rgb(previous)
        if current.shape!=previous.shape:raise ValueError('Motion estimator requires same frame shape')
        h,w=current.shape[-2:];scale=min(1.,self.limit/max(h,w))
        rh,rw=max(1,round(h*scale)),max(1,round(w*scale))
        nh,nw=max(128,math.ceil(rh/8)*8),max(128,math.ceil(rw/8)*8)
        def prep(x):
            x=F.interpolate(x.float(),size=(rh,rw),mode='bilinear',align_corners=False,antialias=False)
            return F.pad(x,(0,nw-rw,0,nh-rh),mode='replicate')*2-1
        # RAFT returns first->second pixels: current MUST be the first image.
        flow=self.network(prep(current),prep(previous),num_flow_updates=self.updates)[-1][:,:,:rh,:rw]
        flow=F.interpolate(flow,size=(h,w),mode='bilinear',align_corners=False)*flow.new_tensor([w/rw,h/rh])[None,:,None,None]
        if not bool(flow.isfinite().all()):raise FloatingPointError('Nonfinite RAFT output')
        return flow,{'network_hw':[nh,nw],'resized_image_hw':[rh,rw],'updates':self.updates,
                     'direction':'current_to_previous','units':'source pixels','padding':'bottom/right replicate'}

class _MetricNetwork(nn.Module):
    """Same pretrained/head names and forward equations as upstream stream class."""
    def __init__(self):
        super().__init__()
        from ._vda.dinov2 import DINOv2
        from ._vda.dpt_temporal import DPTHeadTemporal
        self.pretrained=DINOv2(model_name='vits')
        self.head=DPTHeadTemporal(self.pretrained.embed_dim,64,False,out_channels=[48,96,192,384],use_clstoken=False,num_frames=32,pe='ape')
    def forward(self,x,cache):
        b,c,h,w=x.shape
        features=self.pretrained.get_intermediate_layers(x,[2,5,8,11],return_class_token=True)
        depth,new_cache=self.head(features,h//14,w//14,1,cached_hidden_state_list=cache)
        depth=F.relu(F.interpolate(depth,size=(h,w),mode='bilinear',align_corners=True))
        return depth,tuple(v.detach() for v in new_cache)

@dataclass(frozen=True)
class MetricDepthState:
    caches: tuple
    frame_index: int
    source_hw: tuple
    network_hw: tuple

class MetricVideoDepth(nn.Module):
    def __init__(self,checkpoint,*,device='cuda',input_size=518,fp32=False):
        super().__init__()
        if input_size<28:raise ValueError('Invalid depth input size')
        self.network=_MetricNetwork()
        self.network.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.network.to(device).eval();self.input_size=int(input_size);self.fp32=bool(fp32);self.state=None
    def reset(self):self.state=None
    @torch.inference_mode()
    def propose(self,rgb,*,reset=False):
        valid_rgb(rgb);h,w=rgb.shape[-2:];old=None if reset else self.state
        if old and old.source_hw!=(h,w):raise ValueError('Depth cache shape changed; reset required')
        shape=depth_network_size(h,w,self.input_size)
        if old and old.network_hw!=shape:raise ValueError('Depth work size changed; reset required')
        # Both cv2 INTER_CUBIC and Torch bicubic use the -0.75 cubic kernel.
        # This is a declared Torch preprocessing port, not pixel-bit equivalence
        # to a particular cv2 build. No OpenCV/native image processor is called.
        x=F.interpolate(rgb.float(),size=shape,mode='bicubic',align_corners=False,antialias=False)
        x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        cache=None
        if old:
            selected=old.caches[:2]+old.caches[-29:]
            assert len(selected)==31
            cache=[torch.cat([frame[i] for frame in selected],dim=1) for i in range(len(selected[0]))]
        with torch.autocast(device_type=x.device.type,dtype=torch.float16,enabled=x.is_cuda and not self.fp32):
            depth,current_cache=self.network(x,cache)
        depth=F.interpolate(depth.float(),size=(h,w),mode='bilinear',align_corners=True)
        if not bool(depth.isfinite().all()):raise FloatingPointError('Nonfinite metric depth')
        index=0 if old is None else old.frame_index+1
        # Original streaming bootstrap replicates first-frame hidden states;
        # these are NOT claimed to be 32 observed frames. No future frames used.
        frames=(current_cache,)*32 if old is None else old.caches+(current_cache,)
        if index+32>42:frames=frames[:1]+frames[2:]
        state=MetricDepthState(frames,index,(h,w),shape)
        return depth,state,{'network_hw':list(shape),'observed_frames_since_reset':index+1,'cache_entries':len(frames),
                            'bootstrap':'upstream first-frame hidden-state replication','causal':True,
                            'variant':'experimental streaming metric Small, not offline VDA','depth_units':'estimated meters'}

@dataclass(frozen=True)
class RGBGuideState:
    previous_rgb: torch.Tensor
    depth_state: MetricDepthState
    source_hw: tuple

class RGBGuideEstimator:
    """Propose/commit separates inference from a successful whole image pipeline."""
    def __init__(self,*,model_dir=None,device='cuda',flow_longest_side=512,flow_updates=12,depth_input_size=518,depth_fp32=False):
        models,self.manifest=checked_models(model_dir)
        self.flow=RaftSmall(models['raft-small-C_T_V2'][0],device=device,longest_side=flow_longest_side,updates=flow_updates)
        self.depth=MetricVideoDepth(models['metric-video-depth-anything-small'][0],device=device,input_size=depth_input_size,fp32=depth_fp32)
        self.state=None
        self.provenance={'estimated_guides':True,'flow_model':models['raft-small-C_T_V2'][1],
                         'depth_model':models['metric-video-depth-anything-small'][1],
                         'depth_source_commit':self.manifest['depth_source_commit'],
                         'streaming_accuracy_warning':'Upstream explicitly reports a drop from offline evaluation; no metric accuracy guarantee'}
    def reset(self):self.state=None;self.depth.reset()
    @torch.inference_mode()
    def propose(self,rgb,*,reset=False):
        valid_rgb(rgb);old=None if reset else self.state;size=tuple(rgb.shape[-2:])
        if old and old.source_hw!=size:raise ValueError('RGB sequence size changed; reset required')
        # Depth uses the same last committed state, never a failed frame's cache.
        self.depth.state=old.depth_state if old else None
        z,ds,di=self.depth.propose(rgb,reset=reset)
        if old:
            flow,fi=self.flow(rgb,old.previous_rgb)
        else:
            flow=rgb.new_zeros(1,2,*size);fi={'available':False,'reason':'first frame has no previous observation'}
        proposal=RGBGuideState(rgb.detach().clone(),ds,size)
        return {'metric_depth':z,'motion':flow,'state':proposal,'info':{'depth':di,'flow':fi,'cold_start':old is None}}
    def commit(self,state):self.state=state;self.depth.state=state.depth_state
