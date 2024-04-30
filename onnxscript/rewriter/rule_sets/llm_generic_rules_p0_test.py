from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import onnx
import onnx.reference

import onnxscript.rewriter.rule_sets
import onnxscript.rewriter.rule_sets.llm_generic_rules_p0 as p0
from onnxscript import ir

FLOAT = onnx.TensorProto.FLOAT


class GenericPatternTest(unittest.TestCase):

    def _check_model(
        self,
        model: onnx.ModelProto,
        optimized_model: onnx.ModelProto,
        feeds: dict[str, Any],
        atol: float = 0.0,
        rtol: float = 1e-7,
    ):
        ref = onnx.reference.ReferenceEvaluator(model)
        opt = onnx.reference.ReferenceEvaluator(optimized_model)
        expected = ref.run(None, feeds)
        got = opt.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    def test_rule_set(self):

        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node("Mul", ["X", "one"], ["Y"]),
                ],
                "name",
                [onnx.helper.make_tensor_value_info("X", FLOAT, [None])],
                [onnx.helper.make_tensor_value_info("Y", FLOAT, [None])],
                [onnx.numpy_helper.from_array(np.array([1], dtype=np.float32), name="one")],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )
        ir_model = ir.serde.deserialize_model(model)
        rule_set = onnxscript.rewriter.rule_sets.llm_generic_rule_set_p0()
        rule_set.apply_to_model(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)

        self.assertEqual(["Identity"], [n.op_type for n in rewritten_model.graph.node])
        self._check_model(model, rewritten_model, dict(X=np.array([4, 5.6], dtype=np.float32)))

    def test_rule_multiply_by_one(self):

        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node("Mul", ["X", "one"], ["Y"]),
                ],
                "name",
                [onnx.helper.make_tensor_value_info("X", FLOAT, [None])],
                [onnx.helper.make_tensor_value_info("Y", FLOAT, [None])],
                [onnx.numpy_helper.from_array(np.array([1], dtype=np.float32), name="one")],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )
        ir_model = ir.serde.deserialize_model(model)
        rule = p0.rule_multiply_by_one()
        rule.apply_to_model(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)

        self.assertEqual(["Identity"], [n.op_type for n in rewritten_model.graph.node])
        self._check_model(model, rewritten_model, dict(X=np.array([4, 5.6], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
