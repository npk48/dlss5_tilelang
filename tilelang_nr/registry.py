"""Factory registry for the single selected VitJoint organization."""
from importlib import import_module

PRIMARY_MODULES = (
    "tilelang_nr.kernels.vit_bridge",
    "tilelang_nr.kernels.heads8",
    "tilelang_nr.kernels.shallow_endpoints",
    "tilelang_nr.kernels.heads24",
)
FALLBACK_MODULES = (
    "tilelang_nr.kernels.utility",
    "tilelang_nr.kernels.heads16_bridge",
)


def _matches(name, modules):
    found = []
    for module_name in modules:
        family = import_module(module_name)
        if family.supports(name):
            found.append(family)
    if len(found) > 1:
        raise RuntimeError(f"Ambiguous TileLang NR entry: {name}")
    return found


def resolve(name, enabled=None, *, organization="vit"):
    """Resolve exactly one TileLang factory for a logical plan step."""
    if organization != "vit":
        raise ValueError("Only the VitJoint TileLang organization is available")
    if enabled is not None:
        raise ValueError("Partial family selection is not supported")
    found = _matches(name, PRIMARY_MODULES)
    if not found:
        found = _matches(name, FALLBACK_MODULES)
    return found[0] if found else None
