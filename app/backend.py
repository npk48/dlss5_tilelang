"""Scoped TileLang dispatch for the frozen DLSS5 pipeline.

The frozen reference snapshot is byte-identical and never modified on disk.
Dispatch replaces the frozen NR graph with the TileLang VitJoint implementation
and accelerates the surrounding FSR / optical-flow / metric-depth / temporal /
NR-chain stages. It is explicit and thread-local; the old project is never
imported.
"""
from runtime import bootstrap  # installs the frozen reference package path
from contextlib import contextmanager
from contextvars import ContextVar
from collections import Counter
import torch

_active=ContextVar('dlss5_tilelang_backend',default=None)


class Backend:
    """One frozen model per process; the NR graph is replaced by TileLang."""

    def __init__(self,module,model,*,nr_backend='tilelang-vit'):
        if getattr(module,'_tilelang_dispatch_installed',False):raise RuntimeError('Reuse the existing Engine/Backend; this prototype owns one frozen model per process')
        if nr_backend!='tilelang-vit':raise ValueError('Unknown NR backend')
        self.nr_backend=nr_backend;self.nr=None
        module._tilelang_dispatch_installed=True
        self.fsr_depth_clip_enabled=True;self.fsr_accumulate_enabled=True;self.fsr_frontend_enabled=True;self.nr_chain_enabled=True;self.flow_implementation_enabled=True;self.depth_attention_enabled=True;self.temporal_inputs_enabled=True
        self.module=module;self.model=model;self.stats=Counter()
        self.device=next(model.buffers()).device
        self.original_infer=model.infer_minimal
        def checked_infer(*args,**kwargs):
            current=_active.get()
            if current is None:return self.original_infer(*args,**kwargs)
            if current is not self:raise RuntimeError('NR inference entered a different model backend')
            if torch.cuda.current_stream(self.device).cuda_stream!=torch.cuda.default_stream(self.device).cuda_stream:
                raise RuntimeError('This TileLang NR backend is qualified only on the default CUDA stream; nondefault-stream/Graph execution is not qualified')
            if current.nr is None:
                from tilelang_nr.runtime import VitJointTileLangNR
                current.nr=VitJointTileLangNR(current.model)
            result=current.nr.infer_minimal(*args,**kwargs)
            current.stats['nr_calls']+=1
            current.stats['tilelang_vit_nr_calls']+=1
            return result
        model.infer_minimal=checked_infer
        from pipeline.fsr import depth_clip as fsr_depth_clip
        from whitebox_pipeline.fsr2 import FSR2
        original_depth_clip=fsr_depth_clip.REFERENCE_DEPTH_CLIP
        def fsr_clip(block,*inputs,**kwargs):
            current=_active.get()
            if current is None or not current.fsr_depth_clip_enabled:return original_depth_clip(block,*inputs,**kwargs)
            impl=getattr(block,'_tilelang_depth_clip',None)
            if impl is None:
                impl=fsr_depth_clip.DepthClip(block);block._tilelang_depth_clip=impl
            result=impl(*inputs,**kwargs);current.stats['fsr_depth_clip_calls']+=1
            return result
        FSR2._depth_clip=fsr_clip
        from pipeline.fsr import accumulate as fsr_accumulate
        original_accumulate=fsr_accumulate.REFERENCE_ACCUMULATE
        def fsr_accum(block,*inputs,**kwargs):
            current=_active.get()
            if current is None or not current.fsr_accumulate_enabled:return original_accumulate(block,*inputs,**kwargs)
            impl=getattr(block,'_tilelang_accumulate',None)
            if impl is None:
                impl=fsr_accumulate.Accumulate(block);block._tilelang_accumulate=impl
            result=impl(*inputs,**kwargs);current.stats['fsr_accumulate_calls']+=1
            return result
        FSR2._accumulate=fsr_accum
        from pipeline.fsr import reconstruct as fsr_reconstruct
        for method,reference in [('reconstruct',fsr_reconstruct.REFERENCE_RECONSTRUCT),('locks',fsr_reconstruct.REFERENCE_LOCKS)]:
            def fsr_front(block,*inputs,_method=method,_reference=reference,**kwargs):
                current=_active.get()
                if current is None or not current.fsr_frontend_enabled:return _reference(block,*inputs,**kwargs)
                impl=getattr(block,'_tilelang_frontend',None)
                if impl is None:
                    impl=fsr_reconstruct.ReconstructLocks(block);block._tilelang_frontend=impl
                result=getattr(impl,_method)(*inputs,**kwargs);current.stats['fsr_'+_method+'_calls']+=1
                return result
            setattr(FSR2,'_'+method,fsr_front)
        from pipeline import nr_chain as nr_chain_pipeline
        from whitebox_pipeline.nr_chain import NRChain
        original_chain=nr_chain_pipeline.REFERENCE_PROPOSE
        def nr_propose(chain,*inputs,**kwargs):
            current=_active.get()
            if current is None or not current.nr_chain_enabled:return original_chain(chain,*inputs,**kwargs)
            impl=getattr(chain,'_tilelang_nr_chain',None)
            if impl is None:
                impl=nr_chain_pipeline.Chain(chain);chain._tilelang_nr_chain=impl
            before=impl.accelerated_calls;result=impl(*inputs,**kwargs)
            current.stats['nr_chain_calls' if impl.accelerated_calls>before else 'nr_chain_fallback_calls']+=1
            return result
        NRChain.propose=nr_propose
        from pipeline.guides import flow as flow_pipeline
        from whitebox_pipeline.estimators import RaftSmall
        original_flow=flow_pipeline.REFERENCE_FORWARD
        def raft_forward(estimator,*inputs,**kwargs):
            current=_active.get()
            if current is None or not current.flow_implementation_enabled:return original_flow(estimator,*inputs,**kwargs)
            impl=getattr(estimator,'_tilelang_flow',None)
            if impl is None:
                impl=flow_pipeline.Flow(estimator);estimator._tilelang_flow=impl
            before=impl.accelerated_calls;result=impl(*inputs,**kwargs)
            current.stats['flow_implementation_calls' if impl.accelerated_calls>before else 'flow_implementation_fallback_calls']+=1
            return result
        RaftSmall.forward=raft_forward
        from pipeline.guides import depth as depth_attention
        from whitebox_pipeline.estimators import MetricVideoDepth
        original_depth=depth_attention.REFERENCE_PROPOSE
        from pipeline.guides import temporal as temporal_inputs
        def depth_propose(estimator,*inputs,**kwargs):
            current=_active.get()
            if current is None:return original_depth(estimator,*inputs,**kwargs)
            def selected_depth(*args,**kw):
                if not current.depth_attention_enabled:return original_depth(estimator,*args,**kw)
                impl=getattr(estimator,'_tilelang_depth_attention',None)
                if impl is None:
                    impl=depth_attention.Depth(estimator);estimator._tilelang_depth_attention=impl
                result=impl(*args,**kw);current.stats['depth_attention_calls']+=1
                return result
            if not current.temporal_inputs_enabled:return selected_depth(*inputs,**kwargs)
            impl=getattr(estimator,'_tilelang_temporal_inputs',None)
            if impl is None:
                impl=temporal_inputs.Temporal(estimator,reference=selected_depth);estimator._tilelang_temporal_inputs=impl
            before=impl.prepared_calls();result=impl(*inputs,**kwargs)
            current.stats['temporal_inputs_calls' if impl.prepared_calls()>before else 'temporal_inputs_fallback_calls']+=1
            return result
        MetricVideoDepth.propose=depth_propose

    @contextmanager
    def activate(self,enabled=True):
        token=_active.set(self if enabled else None)
        try:yield self
        finally:_active.reset(token)

    def close(self):
        if self.nr is not None:
            self.nr.close();self.nr=None

    def report(self):
        report={'backend':'TileLang SM89 NVRTC + frozen Torch graph','counters':dict(self.stats),
                'fsr_depth_clip_enabled':self.fsr_depth_clip_enabled,
                'fsr_accumulate_enabled':self.fsr_accumulate_enabled,
                'fsr_frontend_enabled':self.fsr_frontend_enabled,
                'nr_chain_enabled':self.nr_chain_enabled,
                'flow_implementation_enabled':self.flow_implementation_enabled,
                'depth_attention_enabled':self.depth_attention_enabled,
                'temporal_inputs_enabled':self.temporal_inputs_enabled,
                'temporal_inputs_scope':'F32 cache/current concatenation and APE addition to exact Half inputs; K/V projections share one packed GEMM, while Q, attention and raw 32-frame cache semantics are unchanged.',
                'depth_attention_scope':'DINO self-attention uses fused PyTorch SDPA with the original QKV/scale/projection; numerically close rather than bit-exact. Temporal DPT/cache semantics are unchanged.',
                'flow_implementation_scope':'Full RAFT-small adapter with fused correlation grids/layout and final-only consumed upsample; application default is 8 recurrent updates, manifest may request 12.',
                'nr_chain_scope':'Complete single-frame RGB chain; fused preparation/packet/output and reused original texture sampling. Original Gaussian, model, validation, pass history and resampling.',
                'fsr_frontend_scope':'Complete reconstruct and locks, original norm/luma boundaries, ordered selection, first-use initialization and atomic depth/lock publication.',
                'fsr_accumulate_scope':'Complete accumulate/upsample/history stencil; original weight and bilinear boundaries, F32/RTZ/UNORM, state and diagnostics preserved.',
                'fsr_depth_clip_scope':'Complete depth clip/reactivity, original norm/bilinear prelude, F32 operation order, RTZ/UNORM stores; no FSR state/quality change.',
                'qualified_stream':'default CUDA stream only; nondefault stream/Graph diagnostics did not qualify'}
        report['nr_backend']=self.nr_backend
        report['nr']=self.nr.report() if self.nr is not None else {'calls':0,'samples':0,'failures':0,'fallback_calls':0,'cached_shapes':0,'prepare_seconds':0.,'compile_seconds':0.,'last_frame':None,'trace_supported':False}
        nr_state=report['nr']
        nr_state['selected_backend']='tilelang-vit'
        nr_state['actual_backend']='tilelang-vit' if nr_state['calls'] else 'not-run'
        if nr_state.get('last_frame') is not None:
            nr_state['last_frame']={**nr_state['last_frame'],'nr_backend':'tilelang-vit'}
        report.update(backend='TileLang host pipeline + VitJoint TileLang NR (797fd63)',precision='TileLang computation on the recovered packet/layout plan; E4M3/F16 K32 initial-C arithmetic with original raw/aux/counter publication',runtime_lossless_half_guard=False,native_packet_half_cast=True,coverage='Complete 71-layer TileLang plan with ViT/Eight/Shallow/wide organizations; see actual dispatch counters',nr_baseline='nr plan metadata',nr_implementation='tilelang_nr.runtime.VitJointTileLangNR',nr_qualification_base='797fd63')
        return report
