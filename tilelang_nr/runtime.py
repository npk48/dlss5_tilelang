"""TileLang NR: the plan's steps, computed by TileLang.

The plan supplies allocation, launch geometry and typed arguments only. Every
logical Step is replaced by its TileLang implementation; there is no original
math Step and no fallback path.
"""
from collections import Counter
from pathlib import Path
import time

from .plan.runtime import PlanRuntime
from .plan.source import DEEP_UNITS
from .spec import BufferViews, StepSpec
from .registry import resolve


class PortedStep:

    def __init__(self, spec, function, counters):
        self.spec = spec
        self.function = function
        self.counters = counters
        self.geometry = spec.geometry
        self.values = spec.values

    def __call__(self):
        self.function()
        self.counters["tilelang_steps"] += 1
        launches = getattr(self.function, "kernel_launches", None)
        if launches is None:
            self.counters["tilelang_calls_with_unknown_launch_count"] += 1
        else:
            self.counters["tilelang_declared_kernel_launches"] += int(launches)
        self.counters["tilelang:" + self.spec.name] += 1


def outer_source(name):
    """Logical origin of an outer-region step, recorded on its step spec."""
    if name.startswith("outer1_") or name in ("block0_native_packet", "post70_native_head"):
        return "shallow-endpoints"
    if name in ("streamed2", "packet2_ds", "streamed_project2"):
        return "heads2"
    if name.startswith(("four_", "packet4_")):
        return "heads4"
    if name.startswith(("full8_", "packet8_")):
        return "heads8"
    if name in ("clear_counters", "pool8_padding"):
        return "utility"
    raise NotImplementedError("Unknown outer entry: " + name)


class TileLangNR(PlanRuntime):

    def __init__(self, model, weights_path=None, max_cached_shapes=2):
        super().__init__(model, weights_path, max_cached_shapes)
        self.step_counts = Counter()
        self.coverage = {}
        self.port_compile_seconds = 0.0

    def _resolve_family(self, name):
        return resolve(name)

    def _build_function(self, family, spec):
        return family.build_step(spec)

    def _finish_coverage(self, key, records):
        """Validate a selected plan before it is published to the shape cache."""

    def _prepare(self, key, features):
        outer, deep = super()._prepare(key, features)
        memory = BufferViews(outer.storage.buffers + deep.storage.buffers)
        memory.bind_input(features)
        outer_defines = {"NR_H": outer.shape.height, "NR_W": outer.shape.width}
        records = []

        def outer_spec(step):
            return StepSpec(
                step.name, outer_source(step.name), step.geometry, step.values, outer_defines,
                memory
            )

        def add(owner, attr, index, step, spec):
            records.append((owner, attr, index, step, spec, self._resolve_family(spec.name)))

        add(outer, "clear", None, outer.clear, outer_spec(outer.clear))
        add(outer, "pre", None, outer.pre, outer_spec(outer.pre))
        for attr in ("encoder_steps", "decoder_steps"):
            for index, step in enumerate(getattr(outer, attr)):
                add(outer, attr, index, step, outer_spec(step))
        add(outer, "post", None, outer.post, outer_spec(outer.post))
        for index, (step, launch) in enumerate(zip(deep.steps, deep.launches)):
            spec = StepSpec(
                step.name,
                DEEP_UNITS[launch["unit"]],
                step.geometry,
                step.values,
                deep.defines,
                memory
            )
            add(deep, "steps", index, step, spec)
        missing = sorted({spec.name for _, _, _, _, spec, family in records if family is None})
        try:
            if missing:
                raise NotImplementedError(
                    "TileLang NR has missing entries: " + ", ".join(missing)
                )
            replacements = []
            started = time.perf_counter()
            for owner, attr, index, original, spec, family in records:
                # A compilation or numerical failure propagates; there is no CUDA fallback.
                function = self._build_function(family, spec)
                if not callable(function):
                    raise TypeError("TileLang factory returned no callable: " + spec.name)
                replacements.append(
                    (owner, attr, index, PortedStep(spec, function, self.step_counts))
                )
            self.port_compile_seconds += time.perf_counter() - started
            for owner, attr, index, replacement in replacements:
                if index is None:
                    setattr(owner, attr, replacement)
                else:
                    getattr(owner, attr)[index] = replacement
            # Input pointer ownership is refreshed once per infer, not copied.
            outer._tilelang_memory = memory
            self.coverage[key] = {
                "shape": list(features.shape),
                "steps": len(records),
                "tilelang_steps": len(records)
            }
            self._finish_coverage(key, records)
            return outer, deep
        except BaseException:
            self.entries.pop(key, None)
            deep.close()
            outer.close()
            raise

    def infer_minimal(self, prepared_features, **kwargs):
        key = (*prepared_features.shape, prepared_features.device.index, prepared_features.dtype)
        entry = self.entries.get(key)
        if entry is not None:
            entry[0]._tilelang_memory.bind_input(prepared_features)
        output = super().infer_minimal(prepared_features, **kwargs)
        self.last_frame["nr_backend"] = "tilelang-vit"
        return output

    def report(self):
        result = super().report()
        result.update(
            implementation="TileLang computation on the native packet/layout plan",
            actual_backend="tilelang-vit",
            step_counts=dict(self.step_counts),
            step_count_scope=
            'tilelang_steps counts replaced original logical Steps, not physical kernel launches; declared launches include factory tails; original deep counter memset is separate',
            coverage=[self.coverage[k] for k in self.entries if k in self.coverage],
            tilelang_compile_seconds=self.port_compile_seconds,
            reference_preparation=
            "pure-Python plan metadata; every logical Step executes a TileLang kernel"
        )
        return result


class OrganizedTileLangNR(TileLangNR):
    """Choose one factory per logical Step, with the original cache lifecycle."""
    organization = "vit"

    def _resolve_family(self, name):
        return resolve(name, organization=self.organization)

    def _build_function(self, family, spec):
        function = family.build_step(spec)
        if family.__name__ != "tilelang_nr.kernels.vit_bridge":
            return function

        def dispatch():
            function()
            self.step_counts["vit_joint:" + spec.name] += 1

        dispatch.kernels = function.kernels
        dispatch.kernel_launches = function.kernel_launches
        dispatch.workspace_bytes = 0
        dispatch.organization = "CUDA-equivalent-shared-pages-and-physical-raw-boundaries"
        return dispatch

    def _finish_coverage(self, key, records):
        counts = Counter(family.__name__ for *_, family in records if family is not None)
        coverage = self.coverage[key]
        coverage["selected_factories"] = dict(counts)
        coverage["factory_builds"] = sum(counts.values())
        coverage["superseded_factory_builds"] = 0
        coverage["joint_packet_steps"] = counts["tilelang_nr.kernels.heads24"]
        # The latest Eight organization owns UP8, so no separate UP8 factory is built.
        coverage["physical_up8_steps"] = 0
        assert coverage["joint_packet_steps"] == 21, "Incomplete 2H/4H packet plan"
        if self.organization in ("shallow", "eight", "vit"):
            coverage["shallow_joint_steps"] = counts["tilelang_nr.kernels.shallow_endpoints"]
            assert coverage["shallow_joint_steps"] == 10, "Incomplete shallow plan"
        if self.organization in ("eight", "vit"):
            coverage["eight_joint_steps"] = counts["tilelang_nr.kernels.heads8"]
            assert coverage["eight_joint_steps"] == 16, "Incomplete Eight plan"
        if self.organization == "vit":
            names = Counter(
                spec.name for *_,
                spec,
                family in records if family.__name__ == "tilelang_nr.kernels.vit_bridge"
            )
            expected = {
                name: (2 if name == "matrix_phase" else 8)
                for name in (
                    "matrix_expand_completion",
                    "matrix_contract",
                    "matrix_projection",
                    "joint_qkv_completion",
                    "joint_attention_completion",
                    "matrix_phase"
                )
            }
            assert dict(names) == expected, ("Incomplete ViT/bridge plan", dict(names))
            assert len(records) == 193 and coverage["factory_builds"] == 193
            coverage["vit_joint_steps"] = counts["tilelang_nr.kernels.vit_bridge"]
            coverage["vit_joint_families"] = dict(names)

    def report(self):
        result = super().report()
        result["wide_experiment"] = "joint-packet-bounded-vector-io"
        if self.organization in ("shallow", "eight", "vit"):
            result["shallow_experiment"] = "complete-1H-endpoints-joint-M32-consume"
        if self.organization in ("eight", "vit"):
            result["eight_experiment"] = "complete-eight-joint-five-roles"
        if self.organization == "vit":
            result["vit_experiment"] = "complete-eight-block-ViT-and-57-100-physical-bridge"
        result["factory_selection"
               ] = "direct once per logical Step; no superseded kernel construction"
        result["physical_up8_steps_scope"
               ] = "selected historical UP8 factories only; zero when Eight owns UP8"
        return result



class VitJointTileLangNR(OrganizedTileLangNR):
    """The single public 797fd63 VitJoint algorithm combination."""
    organization = "vit"
