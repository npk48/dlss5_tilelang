"""Verified execution policy, separate from recovered model semantics.

Only Windows/NVIDIA SM89 is qualified today. Layout address formulas, P32,
K32 Half publication, and family sum trees remain model semantics on every
future target. A HIP/Metal policy must qualify its native FP8 representation,
MMA/math lowering and runtime; changing a target string is not validation.
"""
TARGET={'kind':'cuda','arch':'sm_89'}
EXECUTION_BACKEND='nvrtc'
CONFIG={'tl.enable_fast_math':False}
NATIVE_E4M3='float8_e4m3fn' # FNUZ/block-scaled encodings are not interchangeable.
