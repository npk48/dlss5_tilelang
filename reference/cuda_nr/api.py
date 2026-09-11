"""Construct the CUDA NR reference from the repository's shared frozen model."""
from pathlib import Path
from runtime import bootstrap
from runtime.model_loader import load_model
import dlss5_model as model_module
from .runtime import NativeNR


def load_runtime(weights_path=None, *, model=None, device='cuda', max_cached_shapes=2,
                 return_details=False):
    """Construct the CUDA reference using ``./model`` by default.

    Passing an already loaded frozen model avoids a second 147 MB weight decode.
    The caller owns ``runtime.close()``. ``return_details=True`` additionally
    returns the model and loader report for diagnostics.
    """
    weights_path = Path(weights_path or bootstrap.MODEL / 'weights_ht_blob.bin').resolve()
    report = {'strategy': 'reuse caller frozen model'}
    if model is None:
        model, report = load_model(model_module, weights_path, device=device)
        model.eval()
    elif Path(model.weights_path).resolve() != weights_path:
        raise ValueError('CUDA reference and frozen model must use the same weights file')
    runtime = NativeNR(model, weights_path=weights_path, max_cached_shapes=max_cached_shapes)
    return (runtime, model, report) if return_details else runtime
