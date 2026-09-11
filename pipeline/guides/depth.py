"""Fused DINO self-attention for the metric-depth estimator.

The QKV projection, scale and output projection are unchanged. PyTorch SDPA avoids
publishing the full attention matrix; its fused reduction is numerically close,
not bit-exact, to the frozen reference. The reference remains available through
the backend's depth-attention switch.
"""
from runtime import bootstrap
import torch
import torch.nn.functional as F
from whitebox_pipeline._vda.dinov2_layers.attention import MemEffAttention
REFERENCE_ATTENTION=MemEffAttention.forward

def forward(module,x,attn_bias=None):
 if module.training or module.attn_drop.training or attn_bias is not None:
  return REFERENCE_ATTENTION(module,x,attn_bias=attn_bias)
 b,n,c=x.shape
 qkv=module.qkv(x).reshape(b,n,3,module.num_heads,c//module.num_heads).permute(2,0,3,1,4)
 q,k,v=qkv[0],qkv[1],qkv[2]
 out=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False,scale=module.scale)
 return module.proj_drop(module.proj(out.transpose(1,2).reshape(b,n,c)))

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
