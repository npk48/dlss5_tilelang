"""Temporal input preparation candidate; original projections/attention/cache.
Adapted forward structure: AnimateDiff/ByteDance temporal attention, Apache-2.0
(see reference/whitebox_pipeline/_vda/LICENSE). Reference sources are unchanged.
"""
import bootstrap
import torch,tilelang
import tilelang.language as T
from einops import rearrange
from device_policy import TARGET,CONFIG,EXECUTION_BACKEND
from whitebox_pipeline._vda.motion_module.motion_module import TemporalAttention
REFERENCE_FORWARD=TemporalAttention.forward

@tilelang.jit(out_idx=[3,4],target=TARGET,execution_backend=EXECUTION_BACKEND,pass_configs=CONFIG,compile_flags=['--fmad=false'])
def prepare_kernel(N,F,C,Cold):
 Old=1 if Cold else F-1
 @T.prim_func
 def main(Current:T.Tensor((N,1,C),T.float32),Cache:T.Tensor((N,Old,C),T.float32),PE:T.Tensor((1,F,C),T.float32),Q:T.Tensor((N,1,C),T.float16),KV:T.Tensor((N,F,C),T.float16)):
  with T.Kernel(T.ceildiv(N*F*C,128),threads=128) as bx:
   i=bx*128+T.get_thread_binding(0)
   if i<N*F*C:
    n=i//(F*C);f=i//C%F;c=i%C
    v=T.alloc_var('float32',init=0)
    if Cold:v=Current[n,0,c]
    else:
     if f==F-1:v=Current[n,0,c]
     else:v=Cache[n,f,c]
    # Original F32 APE addition followed by the linear consumer's Half cast.
    value=T.cast(v+PE[0,f,c],T.float16)
    KV[n,f,c]=value
    if f==F-1:Q[n,0,c]=value
 return main

def forward(module,hidden_states,encoder_hidden_states=None,attention_mask=None,video_length=None,cached_hidden_states=None):
 if (module.training or encoder_hidden_states is not None or attention_mask is not None or video_length!=1
     or not torch.is_autocast_enabled() or torch.get_autocast_dtype('cuda')!=torch.float16
     or hidden_states.device.type!='cuda' or hidden_states.dtype!=torch.float32
     or module.pos_encoder is None or module.pos_encoder.training or module.group_norm is not None
     or module.freqs_cis is not None or module.added_kv_proj_dim is not None or module._slice_size is not None
     or (cached_hidden_states is not None and cached_hidden_states.dtype!=torch.float32)):
  return REFERENCE_FORWARD(module,hidden_states,encoder_hidden_states,attention_mask,video_length,cached_hidden_states)
 if torch.cuda.current_stream(hidden_states.device)!=torch.cuda.default_stream(hidden_states.device):raise RuntimeError('Default CUDA stream required')
 d=hidden_states.shape[1];raw=rearrange(hidden_states,'(b f) d c -> (b d) f c',f=1)
 n,_,channels=raw.shape;count=1 if cached_hidden_states is None else cached_hidden_states.shape[1]+1
 pe=module.pos_encoder.pe[:,:count].to(raw.dtype)
 if (pe.shape!=(1,count,channels) or pe.device!=raw.device
     or (cached_hidden_states is not None and (cached_hidden_states.shape!=(n,count-1,channels) or cached_hidden_states.device!=raw.device))):
  return REFERENCE_FORWARD(module,hidden_states,encoder_hidden_states,attention_mask,video_length,cached_hidden_states)
 cache=raw if cached_hidden_states is None else cached_hidden_states
 qin,kvin=prepare_kernel(n,count,channels,cached_hidden_states is None)(raw.contiguous(),cache.contiguous(),pe.contiguous())
 module._temporal_prepared_calls=getattr(module,'_temporal_prepared_calls',0)+1
 query=module.to_q(qin);key=module.to_k(kvin);value=module.to_v(kvin)
 query=module.reshape_heads_to_batch_dim(query);key=module.reshape_heads_to_batch_dim(key);value=module.reshape_heads_to_batch_dim(value)
 result=module._attention(query,key,value,attention_mask)
 result=module.to_out[1](module.to_out[0](result))
 result=rearrange(result,'(b d) f c -> (b f) d c',d=d)
 return result,raw

class Temporal:
 def __init__(self,estimator,reference=None):
  self.estimator=estimator;self.reference=estimator.propose if reference is None else reference;self.enabled=False;self.accelerated_calls=0;self.modules=[]
  for motion in estimator.network.head.motion_modules:
   for block in motion.temporal_transformer.transformer_blocks:
    for module in block.attention_blocks:
     self.modules.append(module);original=module.forward
     def dispatch(*args,_module=module,_original=original,**kwargs):
      if self.enabled:return forward(_module,*args,**kwargs)
      return _original(*args,**kwargs)
     module.forward=dispatch
 def prepared_calls(self):return sum(getattr(m,'_temporal_prepared_calls',0) for m in self.modules)
 def __call__(self,*args,**kwargs):
  old=self.enabled;self.enabled=True
  try:
   result=self.reference(*args,**kwargs);self.accelerated_calls+=1;return result
  finally:self.enabled=old
