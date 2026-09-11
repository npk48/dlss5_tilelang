"""Whitebox guided SDR sequence pipeline, with pure-Torch FSR2 and frozen NR.

Guided and RGB-estimated SDR/HDR entries support independent multi-pass NR
history and explicit output policies. Native code is never a fallback backend.
"""
from .guides import GuideConfig, GuideProcessor, resize_flow
from .fsr2 import FSR2, FSR2State
from .pipeline import GuidedPipeline, GuidedSDRPipeline, NRSettings
from .color import ColorSettings, ColorBridge
from .post import OutputSettings
from .nr_chain import FrameCancelled
from .rgb_pipeline import RGBSequencePipeline

__all__ = ['GuideConfig','GuideProcessor','resize_flow','FSR2','FSR2State','GuidedSDRPipeline','GuidedPipeline','NRSettings','RGBSequencePipeline','ColorSettings','ColorBridge','OutputSettings','FrameCancelled']
