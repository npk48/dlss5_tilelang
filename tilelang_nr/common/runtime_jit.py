"""Architecture-cached TileLang builders with an explicit runtime spatial ABI.

Only architecture arguments participate in the Python cache. Symbolic variables
are created on a cache miss, never for a new image/layout. The existing builders
still describe the same math; their spatial arguments become scalar PrimFunc
parameters, including buffer extents and flattened descriptor fields.
"""
from functools import wraps
import inspect
import threading

import tilelang
import tilelang.language as T
from tilelang import tvm
from runtime.fp8_compiler import private_compile, compilation_stats

_LOCK = threading.RLock()
_FACTORIES = {}


def _flatten(value):
    if isinstance(value, (tuple, list)):
        return tuple(leaf for part in value for leaf in _flatten(part))
    return (int(value),)


def _symbols(value, name, variables):
    if isinstance(value, (tuple, list)):
        return tuple(_symbols(part, f"{name}_{i}", variables) for i, part in enumerate(value))
    symbol = T.symbolic(name, dtype="int32")
    variables.append(symbol)
    return symbol


def _keep_spatial_bindings(main, variables):
    """Keep dynamic address DAGs compact for TIR; CUDA folds max(x,x) to x.

    Only integer bindings dependent on runtime spatial arguments are opaque.
    Architecture constants, memory guards and all floating/MMA math are intact.
    """
    dependent = set(variables)
    # TIR represents bit operations and lazy selects as Calls too. Rejecting
    # every Call left the large tiled-address / sentinel DAGs fully inlineable.
    # Never hide a load (including an address_of operand) or an external call:
    # those carry memory/side-effect semantics that must remain visible to TIR.
    arithmetic_calls = frozenset({
        "tirx.bitwise_and", "tirx.bitwise_or", "tirx.bitwise_xor",
        "tirx.bitwise_not", "tirx.shift_left", "tirx.shift_right",
        "tirx.if_then_else",
    })

    def preserve(node):
        if not isinstance(node, tvm.tirx.Bind) or str(node.var.dtype) != "int32":
            return None
        uses = []
        unsafe = []

        def inspect_value(value):
            if isinstance(value, tvm.tirx.Var) and value in dependent:
                uses.append(value)
            elif isinstance(value, tvm.tirx.BufferLoad):
                unsafe.append(value)
            elif isinstance(value, tvm.tirx.Call) and getattr(value.op, "name", "") not in arithmetic_calls:
                unsafe.append(value)

        tvm.tirx.stmt_functor.post_order_visit(node.value, inspect_value)
        if not uses:
            return None
        dependent.add(node.var)
        if unsafe:
            return None
        value = tvm.tirx.call_extern("int32", "max", node.value, node.value)
        return tvm.tirx.Bind(node.var, value)
    body = tvm.tirx.stmt_functor.ir_transform(main.body, None, preserve)
    return tvm.tirx.PrimFunc(main.params, body, main.ret_type, main.buffer_map, main.attrs, main.span)


class BoundKernel:
    """A geometry binding, not a compiled specialization."""

    def __init__(self, kernel, scalars):
        self.compiled_kernel = kernel
        self.runtime_scalars = scalars
        adapter = kernel.adapter
        # These factories own their output buffers and explicitly bind every
        # symbolic scalar. Resolve that invariant once, not for every Step.
        direct = (not adapter.result_idx and
                  all(ref[0] == 2 for ref in adapter.dynamic_symbolic_map.values()))
        self._direct = adapter._forward_from_prebuild_lib if direct else None
        self._input_count = len(adapter.params) - len(scalars)
        self._adapter = adapter
        self._prepared = None
        self._prepare_eligible = None

    def __call__(self, *args, **kwargs):
        if self._direct is None or any(key != "stream" for key in kwargs):
            return self.compiled_kernel(*args, *self.runtime_scalars, **kwargs)
        if len(args) != self._input_count:
            raise ValueError(f"Expected {self._input_count} bound inputs, got {len(args)}")
        stream = kwargs.get("stream")
        if stream is None:
            import torch
            stream = torch.cuda.current_stream().cuda_stream
        if self._prepare_eligible is None:
            import torch
            self._prepare_eligible = all(isinstance(arg, torch.Tensor) for arg in args)
        if self._prepare_eligible:
            if self._prepared is None:
                from .bound_launch import BoundLaunch
                self._prepared = BoundLaunch(self._adapter, args, self.runtime_scalars, stream)
            self._prepared(args, stream)
        else:
            self._direct(*args, *self.runtime_scalars, stream=stream)
        return []

    def __getattr__(self, name):
        return getattr(self.compiled_kernel, name)


def spatial_jit(*, dynamic, **options):
    """Declare spatial factory arguments; all other arguments are architecture."""
    dynamic = frozenset(dynamic.split())
    # Runtime geometry predicates must stay inside their loops. Unswitching
    # those predicates duplicates the architecture-unrolled matrix body.
    options = dict(options)
    options["pass_configs"] = dict(options.get("pass_configs") or {})
    options["pass_configs"].setdefault("tl.disable_loop_unswitching", True)
    # Normal let simplification remains enabled: address_of requires load
    # aliases resolved. Runtime integer DAGs get selective boundaries below.

    def decorate(factory):
        signature = inspect.signature(factory)
        unknown = dynamic.difference(signature.parameters)
        if unknown:
            raise ValueError(f"Unknown spatial arguments for {factory.__name__}: {unknown}")
        name = factory.__module__ + "." + factory.__name__
        cache = {}
        records = {}
        _FACTORIES[name] = records

        @wraps(factory)
        def build(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            architecture = tuple((key, value) for key, value in bound.arguments.items() if key not in dynamic)
            scalars = tuple(leaf for key, value in bound.arguments.items() if key in dynamic for leaf in _flatten(value))
            with _LOCK:
                kernel = cache.get(architecture)
                if kernel is None:
                    variables = []
                    symbolic = {key: _symbols(value, key, variables) if key in dynamic else value
                                for key, value in bound.arguments.items()}
                    owner = name + repr(architecture)
                    with private_compile(owner=owner):
                        main = factory(**symbolic)
                        main = tvm.tirx.PrimFunc(
                            list(main.params) + variables, main.body, main.ret_type,
                            main.buffer_map, main.attrs, main.span
                        )
                        main = _keep_spatial_bindings(main, variables).with_attr("nr_spatial_lowering", "opaque_integer_arithmetic_v3")
                        kernel = tilelang.compile(main, **options)
                    cache[architecture] = kernel
                    records[repr(architecture)] = {
                        "jit_builds": 1,
                        "private_nvrtc_calls": compilation_stats()["by_owner"].get(owner, 0),
                        "runtime_scalar_count": len(variables),
                        "cache_key": getattr(kernel, "_tilelang_cache_key", None),
                        "cache_path": getattr(kernel, "_tilelang_cache_path", None),
                        "bindings": 0,
                    }
                record = records[repr(architecture)]
                if len(scalars) != record["runtime_scalar_count"]:
                    raise ValueError(f"Spatial ABI cardinality changed for {name}; use a runtime buffer, not a variable argument list")
                record["bindings"] += 1
            return BoundKernel(kernel, scalars)

        return build

    return decorate


def runtime_compilation_stats():
    with _LOCK:
        families = {name: {key: dict(record) for key, record in records.items()}
                    for name, records in _FACTORIES.items()}
    return {
        "scope": "process; architecture cache persists across plan eviction",
        "jit_builds": sum(record["jit_builds"] for records in families.values() for record in records.values()),
        "private_nvrtc_calls": compilation_stats()["private_nvrtc_calls"],
        "factories": families,
    }
