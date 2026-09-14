"""Regression for runtime address DAG boundaries (no GPU or model required)."""
import unittest

from runtime import bootstrap  # noqa: F401 -- DLL discovery before TileLang
from tilelang import tvm
from tilelang_nr.common.runtime_jit import _keep_spatial_bindings


class SpatialBindingsTest(unittest.TestCase):
    def test_select_and_bitwise_addresses_preserve_bindings_not_memory(self):
        tir = tvm.tirx
        width = tir.Var("width", "int32")
        tile = tir.Var("tile", "int32")
        address = tir.Var("address", "int32")
        derived = tir.Var("derived", "int32")
        loaded = tir.Var("loaded", "int32")
        external = tir.Var("external", "int32")
        constant = tir.Var("constant", "int32")
        buffer = tir.decl_buffer((width,), "int32", name="data")
        body = tir.SeqStmt([
            tir.Bind(tile, tir.shift_right(width, 2)),
            tir.Bind(address, tir.if_then_else(tile > 0, tile * 16, -1)),
            tir.Bind(derived, address + 4),
            tir.Bind(loaded, tir.BufferLoad(buffer, [address])),
            tir.Bind(external, tir.call_extern("int32", "read_state", width)),
            tir.Bind(constant, tir.IntImm("int32", 8)),
            tir.Evaluate(derived),
        ])
        main = tir.PrimFunc([width, buffer.data], body)
        result = _keep_spatial_bindings(main, [width])
        bindings = []
        tir.stmt_functor.post_order_visit(
            result.body, lambda node: bindings.append(node) if isinstance(node, tir.Bind) else None
        )
        for binding in bindings[:3]:
            self.assertIsInstance(binding.value, tir.Call)
            self.assertEqual(binding.value.op.name, "tirx.call_extern")
            self.assertEqual(str(binding.value.args[0]), '"max"')
            self.assertTrue(tvm.ir.structural_equal(binding.value.args[1], binding.value.args[2]))
        # Load aliases must remain available to Simplify/address_of; calls must
        # not be duplicated or hidden from memory/side-effect analysis.
        for before, after in zip(body.seq[3:6], bindings[3:6]):
            self.assertTrue(tvm.ir.structural_equal(before, after))


if __name__ == "__main__":
    unittest.main()
