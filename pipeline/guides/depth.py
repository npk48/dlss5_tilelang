"""DINO attention candidate: retain F32 softmax accumulation, direct consumer dtype.
Forward structure follows vendored DINOv2 attention; original source is unchanged.
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Apache-2.0, see DEPTH_DINO_LICENSE.txt. Modified to publish the consumer dtype.
"""
from runtime import bootstrap
import torch
from whitebox_pipeline._vda.dinov2_layers.attention import MemEffAttention
REFERENCE_ATTENTION=MemEffAttention.forward

def forward(module,x,attn_bias=None):
 if module.training or module.attn_drop.training or attn_bias is not None:
  return REFERENCE_ATTENTION(module,x,attn_bias=attn_bias)
 b,n,c=x.shape
 qkv=module.qkv(x).reshape(b,n,3,module.num_heads,c//module.num_heads).permute(2,0,3,1,4)
 q,k,v=qkv[0]*module.scale,qkv[1],qkv[2]
 scores=q@k.transpose(-2,-1)
 # Half input softmax still accumulates in F32. Under autocast the original
 # outputs F32 then its sole matmul consumer casts to Half. Publish that Half
 # boundary directly, keeping the original QK and PV matrix operations.
 # Keep the caller's autocast mode unchanged; explicit output dtype bypasses
 # only the otherwise-unused F32 probability publication.
 probs=scores.softmax(dim=-1,dtype=scores.dtype)
 probs=module.attn_drop(probs)
 out=(probs@v).transpose(1,2).reshape(b,n,c)
 return module.proj_drop(module.proj(out))

from whitebox_pipeline.estimators import MetricVideoDepth
REFERENCE_PROPOSE=MetricVideoDepth.propose

class Depth:
 def __init__(self,estimator):
  self.estimator=estimator;self.reference=REFERENCE_PROPOSE.__get__(estimator,type(estimator));self.enabled=False;self.accelerated_calls=0
  for block in estimator.network.pretrained.blocks:
   module=block.attn;original=module.forward
   def dispatch(x,*args,_module=module,_original=original,**kwargs):
    if self.enabled:return forward(_module,x,*args,**kwargs)
    return _original(x,*args,**kwargs)
   module.forward=dispatch
 def __call__(self,*args,**kwargs):
  previous=self.enabled;self.enabled=True
  try:
   result=self.reference(*args,**kwargs);self.accelerated_calls+=1
   return result
  finally:self.enabled=previous
