"""Guided SDR/HDR sequence pipeline, fixed NR model and explicit host policies."""
from pathlib import Path
import torch
import torch.nn.functional as F
from .fsr2 import FSR2,half as half_rtz
from .guides import GuideProcessor,GuideConfig
from .color import ColorSettings,ColorBridge,linear_to_srgb,srgb_to_linear
from .nr_chain import NRSettings,NRChain,check_cancel
from .post import OutputSettings,composite

class GuidedPipeline:
    """BCHW linear709 input: SDR [0,1], or absolute HDR nits per ColorSettings.

    Render depth and current->previous pixel motion are supplied. Protection is
    a DISPLAY-grid NR/output-effect mask (1=protect), not a Feature1/guide mask.
    Alpha is straight, render-grid; premultiplication is handled explicitly at
    the file/codec boundary. One instance must not be called concurrently.
    """
    def __init__(self,output_size,*,model=None,weights_path=None,device='cuda',nr_settings=None,
                 nr_passes=None,nr_work_size=None,guide_config=None,color_settings=None,output_settings=None,**fsr_settings):
        import dlss5_model as nr
        self.nr=nr;self.device=torch.device(device)
        self.bridge=ColorBridge(color_settings)
        self.output_settings=output_settings or OutputSettings()
        if fsr_settings.get('display_motion_vectors'):raise ValueError('Guided entry takes render-resolution motion')
        if 'hdr' in fsr_settings and bool(fsr_settings['hdr'])!=self.bridge.settings.hdr:raise ValueError('FSR HDR flag conflicts with color settings')
        if not self.bridge.settings.hdr and fsr_settings.get('auto_exposure'):raise ValueError('SDR entry does not silently change exposure')
        self.fsr_settings={**fsr_settings,'hdr':self.bridge.settings.hdr}
        self.fsr=FSR2(output_size,**self.fsr_settings)
        self.guides=GuideProcessor(guide_config or GuideConfig(validate=False))
        self.model=model if model is not None else nr.load_model(Path(weights_path) if weights_path else Path(__file__).resolve().parents[1]/'weights_ht_blob.bin',device=str(self.device))
        if nr_settings is not None and nr_passes is not None:raise ValueError('Use nr_settings OR nr_passes')
        self.nr_work_requested=nr_work_size
        self.chain=NRChain(self.model,output_size,nr_passes if nr_passes is not None else (nr_settings or NRSettings(),),nr_work_size)
        self.previous_alpha=None;self.frame_index=0
    @property
    def settings(self):return self.chain.passes[0]
    @settings.setter
    def settings(self,value):self.configure_nr((value,)+self.chain.passes[1:])
    @property
    def previous_nr(self):return self.chain.states[0].image if self.chain.states[0] is not None else None
    @property
    def neural_size(self):return self.chain.neural_size
    def configure_nr(self,passes,work_size=None):
        if work_size is not None:self.nr_work_requested=tuple(work_size)
        return self.chain.configure(passes,work_size)
    def set_output(self,settings):self.output_settings=settings
    def set_color(self,settings):
        old=self.bridge.settings
        if settings==old:return False
        reset=(settings.hdr,settings.reference_white_nits)!=(old.hdr,old.reference_white_nits)
        self.bridge=ColorBridge(settings)
        # Output transfer and inverse-delta peak never enter model history.
        if reset:
            self.fsr_settings['hdr']=settings.hdr
            self.fsr=FSR2(self.fsr.output_size,**self.fsr_settings);self.reset()
        return reset
    def resize(self,output_size):
        if tuple(output_size)==self.fsr.output_size:return
        self.fsr=FSR2(output_size,**self.fsr_settings);self.chain.output_size=tuple(output_size)
        self.chain.configure(self.chain.passes,self.nr_work_requested or output_size);self.reset()
    def reset(self):
        self.fsr.reset();self.guides.reset();self.chain.reset();self.previous_alpha=None;self.frame_index=0
    @torch.inference_mode()
    def process_frame(self,color,depth,motion,*,jitter_xy=(0.,0.),delta_ms=1000/60,
                      reset=False,reactive=None,composition=None,protect=None,alpha=None,cancel=None):
        check_cancel(cancel)
        if color.device.type!=self.device.type or (self.device.index is not None and color.device.index!=self.device.index):raise ValueError('Explicit device transfer required')
        if not bool(color.isfinite().all()):raise ValueError('Nonfinite color')
        hdr=self.bridge.settings.hdr
        if not hdr and not bool(((color>=0)&(color<=1)).all()):raise ValueError('SDR requires linear [0,1]')
        if hdr and not bool((color.abs()<=20000).all()):raise ValueError('HDR working range is +/-20000 BT.709 nits (includes converted PQ/2020 gamut)')
        oh,ow=self.fsr.output_size;h,w=color.shape[-2:]
        def mask(value,shape,name):
            if value is not None and (value.shape!=(1,1,*shape) or value.device!=color.device or not bool(value.isfinite().all()) or not bool(((value>=0)&(value<=1)).all())):
                raise ValueError(name+' requires [0,1] BCHW1 at the declared size/device')
        mask(alpha,(h,w),'alpha');mask(protect,(oh,ow),'protection')
        saved=(self.fsr.state,self.guides.previous,self.chain.states,self.previous_alpha,self.frame_index)
        if reset:self.reset()
        try:
            white=self.bridge.settings.reference_white_nits if hdr else 1.
            far=self.fsr._z(depth.float().new_tensor(0. if self.fsr.inverted else 1.))
            normalized=self.fsr._z(depth.float())/far
            guide_view=self.bridge.encode(color)
            if alpha is not None:guide_view=guide_view*alpha
            guide=self.guides.process(guide_view,motion,normalized,reset=reset)
            r=reactive
            if alpha is not None:
                ar=1-alpha
                if self.previous_alpha is not None:ar=torch.maximum(ar,(alpha-self.previous_alpha).abs())
                r=ar if r is None else torch.maximum(r,ar)
            temporal=self.fsr.process(color.clamp_min(0)/white,depth,guide['motion'],jitter_xy=jitter_xy,
                delta_ms=delta_ms,reset=reset,reactive=r,composition=composition,sharpness=0.)*white
            context=temporal
            if hdr:
                negative=F.interpolate(color.clamp_max(0),size=(oh,ow),mode='bilinear',align_corners=False)
                context=context+negative
            check_cancel(cancel)
            reference=half_rtz(self.bridge.encode(context))
            confidence=(1-self.fsr.diagnostics['depth_clip'])*(1-self.fsr.diagnostics['reactivity'])
            encoded,states,records=self.chain.propose(reference,self.fsr.state.dilated_motion,confidence,reset=reset,cancel=cancel,resample=self.output_settings.nr_resample)
            out_alpha=F.interpolate(alpha,size=(oh,ow),mode='bilinear',align_corners=False) if alpha is not None else None
            legacy=(not hdr and self.output_settings==OutputSettings() and protect is None and alpha is None
                    and all(s.intensity==1 for s in self.chain.passes))
            if legacy:
                linear=srgb_to_linear(encoded);image=encoded if self.bridge.settings.output_encoding=='sRGB' else linear
            else:
                linear=composite(context,reference,encoded,self.bridge,self.fsr,self.output_settings,protect,out_alpha)
                image=self.bridge.output(linear)
            if not bool(image.isfinite().all()):raise FloatingPointError('Nonfinite final color')
            check_cancel(cancel)
            info={'frame_index':self.frame_index,'cold_nr':records[0]['cold_nr'],'neural_hw':list(self.neural_size),
                'nr_work_hw':list(self.chain.work_size),'nr_passes':records,'nr_head_abs_mean':records[-1]['head_abs_mean'],
                'nr_history_gate_mean':records[-1]['history_gate_mean'],'fsr_backend':'pytorch','nr_backend':'existing frozen dlss5_model',
                'guide_source':'provided','nr_history_policy':'per-pass explicit visibility/reactivity gate',
                'color_encoding':self.bridge.settings.output_encoding,'output_hw':[oh,ow],
                'hdr':hdr,'linear_units':'BT.709 nits' if hdr else 'relative linear BT.709',
                'reference_white_nits':white if hdr else None,'inverse_delta_peak_nits':self.bridge.settings.peak_nits if hdr else None,'codec':'own positive max-RGB compression + retained linear context' if hdr else 'sRGB',
                'alpha':'straight' if alpha is not None else 'opaque','protection_scope':'NR/output effects, not FSR',
                'work_resize_policy':self.output_settings.nr_resample+' encoded-difference transport, not full-size NR equivalence',
                'final_sharpening':self.output_settings.sharpness,
                'guide_geometry':{k:v.detach().flatten().tolist() for k,v in guide['geometry'].items()} if guide['geometry'] is not None else None,
                'guide_test_means':guide['tests'].mean(dim=(0,2,3)).tolist()}
            # Publish only after every pass, output guard, encoding and cancel check.
            self.chain.states=states;self.previous_alpha=alpha.detach().clone() if alpha is not None else None
            self.frame_index+=1
            return {'color':image,'linear_color':linear,'temporal_linear':context,'alpha':out_alpha,'guides':guide,'info':info}
        except (Exception,KeyboardInterrupt):
            self.fsr.state,self.guides.previous,self.chain.states,self.previous_alpha,self.frame_index=saved
            self.fsr.diagnostics={}
            raise

class GuidedSDRPipeline(GuidedPipeline):
    def __init__(self,*args,**kwargs):
        if kwargs.get('color_settings') is not None and kwargs['color_settings'].hdr:raise ValueError('Use GuidedPipeline for HDR')
        super().__init__(*args,**kwargs)
