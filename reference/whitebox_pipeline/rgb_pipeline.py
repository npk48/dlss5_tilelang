"""Ordinary SDR/HDR RGB -> estimated guides -> full Torch guided pipeline."""
import math
import torch
import torch.nn.functional as F
from .estimators import RGBGuideEstimator,valid_rgb
from .pipeline import GuidedPipeline,srgb_to_linear
from .nr_chain import check_cancel
from .guides import GuideConfig

class RGBSequencePipeline:
    """Estimated causal mode with an explicit assumed camera.

    Default input is SOURCE-grid sRGB BCHW. With HDR ColorSettings, input is
    linear BT.709 nits and an explicit compressed SDR estimator view is used.
    No fabricated jitter is supplied. Metric VDA output
    is treated as estimated view-Z meters; physical scale/FOV accuracy is not
    guaranteed. Near/far clipping is fixed by camera settings, not per-frame
    normalization. Source/render/output/flow/depth-working sizes stay distinct.
    """
    def __init__(self,output_size,*,device='cuda',model=None,weights_path=None,model_dir=None,
                 render_size=None,nr_settings=None,nr_passes=None,nr_work_size=None,color_settings=None,output_settings=None,guide_config=None,flow_longest_side=512,
                 flow_updates=12,depth_input_size=518,depth_fp32=False,
                 camera_near=.1,camera_far=1000.,camera_fov_y=math.pi/3,inverted_depth=False):
        if not all(math.isfinite(v) for v in (camera_near,camera_far,camera_fov_y)) or not 0<camera_near<camera_far:
            raise ValueError('Estimated mode needs a finite, explicitly declared camera range')
        # Check local guide checkpoints before the slower NR load, never fetch.
        self.estimator=RGBGuideEstimator(model_dir=model_dir,device=device,flow_longest_side=flow_longest_side,
            flow_updates=flow_updates,depth_input_size=depth_input_size,depth_fp32=depth_fp32)
        self.guided=GuidedPipeline(output_size,device=device,model=model,weights_path=weights_path,
            nr_settings=nr_settings,nr_passes=nr_passes,nr_work_size=nr_work_size,color_settings=color_settings,output_settings=output_settings,
            guide_config=guide_config or GuideConfig(),camera_near=camera_near,
            camera_far=camera_far,camera_fov_y=camera_fov_y,inverted_depth=inverted_depth)
        self.render_size=tuple(render_size) if render_size else None
        self.camera={'near_m':camera_near,'far_m':camera_far,'vertical_fov_radians':camera_fov_y,
                     'provenance':'user-supplied or explicit default assumption, not inferred camera calibration',
                     'depth_interpretation':'metric model output treated as estimated view-Z, not measured geometry'}
        self.provenance={**self.estimator.provenance,'camera':self.camera,
                         'jitter_policy':'zero: existing RGB frames were not rendered with a supplied jitter'}
    def reset(self):self.guided.reset();self.estimator.reset()
    def configure_nr(self,passes,work_size=None):return self.guided.configure_nr(passes,work_size)
    def set_output(self,settings):self.guided.set_output(settings)
    def set_color(self,settings):
        if self.guided.set_color(settings):self.estimator.reset()
    def resize(self,size):self.guided.resize(size)
    @torch.inference_mode()
    def process_frame(self,srgb,*,reset=False,delta_ms=1000/60,reactive=None,composition=None,protect=None,alpha=None,cancel=None):
        check_cancel(cancel)
        hdr=self.guided.bridge.settings.hdr
        if hdr:
            if srgb.ndim!=4 or srgb.shape[:2]!=(1,3) or not bool(srgb.isfinite().all()) or not bool((srgb.abs()<=20000).all()):raise ValueError('HDR RGB input is finite BT.709 nits in +/-20000')
            linear_input=srgb;estimator_rgb=self.guided.bridge.encode(srgb)
        else:
            valid_rgb(srgb);linear_input=srgb_to_linear(srgb);estimator_rgb=srgb
        h,w=srgb.shape[-2:];oh,ow=self.guided.fsr.output_size
        if alpha is not None:
            if alpha.shape!=(1,1,h,w) or not bool(alpha.isfinite().all()) or not bool(((alpha>=0)&(alpha<=1)).all()):raise ValueError('Source alpha must be [0,1] BCHW1')
            estimator_rgb=estimator_rgb*alpha
        rh,rw=self.render_size or (min(h,oh),min(w,ow))
        if min(rh,rw)<32 or rh>oh or rw>ow:raise ValueError('Render must be >=32 and fit output')
        saved=(self.estimator.state,self.estimator.depth.state)
        try:
            proposed=self.estimator.propose(estimator_rgb,reset=reset);z=proposed['metric_depth'];flow=proposed['motion']
            check_cancel(cancel)
            near,far=self.camera['near_m'],self.camera['far_m']
            positive=z>0
            if not bool(positive.any()):raise ValueError('Metric model returned no positive depth; refusing a constant-depth substitute')
            clipped=(z<near)|(z>far)
            if not bool((~clipped).any()):raise ValueError('No depth estimate lies within the camera range; refusing a constant boundary-depth substitute')
            bounded=z.clamp(near,far)
            # Invert the same configured projection used by FSR; no min-max fit.
            ab=self.guided.fsr._depth_factors(z)
            device_z=ab[0]+ab[1]/bounded
            device_z=device_z.clamp(0,1)
            color=F.interpolate(linear_input,size=(rh,rw),mode='bilinear',align_corners=False,antialias=True)
            render_alpha=F.interpolate(alpha,size=(rh,rw),mode='bilinear',align_corners=False) if alpha is not None else None
            depth=F.interpolate(device_z,size=(rh,rw),mode='nearest-exact')
            motion=F.interpolate(flow,size=(rh,rw),mode='nearest-exact')*flow.new_tensor([rw/w,rh/h])[None,:,None,None]
            # This mask has an explicit FSR-reactive meaning: exclude unreliable
            # depth pixels from history, not a relabeled Feeder distrust mask.
            invalid=F.interpolate(clipped.float(),size=(rh,rw),mode='nearest-exact')
            def input_mask(mask):
                if mask is None:return None
                if mask.shape!=(1,1,h,w) or not bool(torch.isfinite(mask).all()) or not bool(((mask>=0)&(mask<=1)).all()):
                    raise ValueError('RGB-mode masks must be source-grid [0,1] 1x1xHxW')
                return F.interpolate(mask,size=(rh,rw),mode='nearest-exact')
            user_reactive=input_mask(reactive);tc=input_mask(composition)
            r=invalid if user_reactive is None else torch.maximum(invalid,user_reactive)
            details={**proposed['info'],'source_hw':[h,w],'render_hw':[rh,rw],
                'depth_estimate_min_m':float(z.min()),'depth_estimate_median_m':float(z.median()),'depth_estimate_max_m':float(z.max()),
                'depth_near_far_clipped_fraction':float(clipped.float().mean()),'depth_nonpositive_fraction':float((~positive).float().mean()),
                'depth_mapping':'fixed camera projection of metric estimate; no per-frame normalization',
                'jitter_xy':[0.,0.],'camera':self.camera,
                'estimator_color_view':'sRGB of positive max-RGB-compressed HDR/reference-white' if hdr else 'source sRGB',
                'estimator_alpha_view':'composited over black' if alpha is not None else 'opaque'}
            result=self.guided.process_frame(color,depth,motion,jitter_xy=(0.,0.),delta_ms=delta_ms,
                reset=reset,reactive=r,composition=tc,protect=protect,alpha=render_alpha,cancel=cancel)
            self.estimator.commit(proposed['state'])
            result['info'].update(guide_source='estimated',estimated_guides=True,estimation=details)
            return result
        except (Exception,KeyboardInterrupt):
            self.estimator.state,self.estimator.depth.state=saved
            raise
