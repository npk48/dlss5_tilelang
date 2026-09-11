"""Explicit algorithm organization; only the selected factory is constructed.

The public ViT organization selects the 797fd63 shallow/2H-4H/8H/ViT factories
before construction, rather than installing earlier implementations and
overwriting them.
"""
from importlib import import_module

STRICT_MODULES = {
    "utility": "tilelang_nr.utility",
    "deep": "tilelang_nr.heads16_bridge",
}
ORGANIZATIONS = {
    "vit": ("vit_bridge", "heads8", "shallow_endpoints", "heads24"),
}


def available_families(enabled=None):
    for name, module in STRICT_MODULES.items():
        if enabled is None or name in enabled:
            yield import_module(module)


def resolve(name, enabled=None, *, organization="vit"):
    if organization in ORGANIZATIONS:
        candidates = [
            import_module("tilelang_nr." + module) for module in ORGANIZATIONS[organization]
        ]
        found = [family for family in candidates if family.supports(name)]
        if len(found) > 1:
            raise RuntimeError(f"Ambiguous TileLang organization entry: {name}")
        if found:
            return found[0]
        for family_name in ("utility", "deep"):
            family = import_module(STRICT_MODULES[family_name])
            if family.supports(name):
                return family
    found = [family for family in available_families(enabled) if family.supports(name)]
    if len(found) > 1:
        raise RuntimeError(f"Ambiguous TileLang NR entry: {name}")
    return found[0] if found else None
