"""Shape metadata and logical units for the TileLang NR plan.

Only the pool/up ownership tables are computed here. The plan itself is built in
``plan/runtime.py`` and its computation is supplied entirely by TileLang kernels.
"""
from . import layouts


def prepare(shape):
    """Pool and up ownership tables for the three wide families."""
    ds, up = {}, {}
    for heads, block in ((2, 8), (4, 14), (8, 22)):
        ds[heads] = layouts.pool_metadata(shape.outer(block), layouts.FAMILY, layouts.TWO)
    for heads, block in ((2, 62), (4, 56), (8, 48)):
        p = shape.outer(block)
        ui, inverse = layouts.up_metadata(p, layouts.FAMILY, layouts.TWO)
        im, _ = (layouts.maps2 if heads == 2 else layouts.wide_maps)(p)
        up[heads] = ui, inverse, im
    return ds, up


# Logical plan units for the deep region. The TileLang factory for each step is
# chosen by step name, not by unit, so these are descriptive labels only.
DEEP_UNITS = {
    'sixteen': 'deep-16H',
    'local': 'deep-local64',
    'matrix': 'vit-matrix',
    'qkv': 'vit-qkv',
    'attention': 'vit-attention',
    'transition': 'physical-bridge-transition',
    'repack': 'physical-bridge-repack',
    'split': 'physical-bridge-split',
}
