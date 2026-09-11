"""Pure-Python NR plan: allocation, geometry and typed launch arguments.

The plan reproduces the learned NR's structure exactly — arena layout, saved
skips and auxiliary rails, buffer ownership, and per-step launch geometry and
arguments — without compiling or binding any CUDA kernel. Every step body is a
TileLang kernel chosen by step name.
"""
from .runtime import PlanRuntime, OuterRuntime, DeepRuntime

__all__ = ['PlanRuntime', 'OuterRuntime', 'DeepRuntime']
