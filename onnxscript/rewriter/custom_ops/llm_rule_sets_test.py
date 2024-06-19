# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import onnx
import onnx.helper as oh
import onnx.reference
import onnx.reference.op_run as op_run
import parameterized

import onnxscript.rewriter.custom_ops.llm_rule_sets as llm_rule_sets
from onnxscript import ir

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16


class Transpose2DCastFP16(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return (X.T.astype(np.float16),)


class Transpose2DCastFP32(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return (X.T.astype(np.float32),)


class LlmRuleSetsTest(unittest.TestCase):
    def _get_random_inputs(self, model: onnx.ModelProto) -> dict[str, Any]:
        feeds: dict[str, Any] = {}
        for i in model.graph.input:
            ish = tuple(i.type.tensor_type.shape.dim)
            # Creates an input tensor with a dimension defined by the onnx model
            # or equals to i + 2 with i being the dimension index.
            # The tensor is kept small to make the test fast.
            shape = tuple(
                (d.dim_value if d.dim_value > 0 else i + 2) for i, d in enumerate(ish)
            )
            if i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                feeds[i.name] = np.random.randn(*shape).astype(np.float32)
            elif i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                feeds[i.name] = np.random.randn(*shape).astype(np.float16)
            else:
                raise AssertionError(f"Not implemented for input {i}")
        return feeds

    def _check_model(
        self,
        model: onnx.ModelProto,
        optimized_model: onnx.ModelProto,
        feeds: dict[str, Any] | None = None,
        atol: float = 0.0,
        rtol: float = 1e-7,
    ):
        if not feeds:
            feeds = self._get_random_inputs(model)
        ref = onnx.reference.ReferenceEvaluator(model)
        opt = onnx.reference.ReferenceEvaluator(
            optimized_model, new_ops=[Transpose2DCastFP16, Transpose2DCastFP32]
        )
        expected = ref.run(None, feeds)
        got = opt.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    @classmethod
    def _transpose_cast(cls, in_type, cast_before):
        out_type = TFLOAT16 if in_type == TFLOAT else TFLOAT

        if cast_before:
            nodes = [
                oh.make_node("Cast", ["X"], ["xc"], to=out_type),
                oh.make_node("Transpose", ["xc"], ["Y"], perm=[1, 0]),
            ]
        else:
            nodes = [
                oh.make_node("Transpose", ["X"], ["xt"], perm=[1, 0]),
                oh.make_node("Cast", ["xt"], ["Y"], to=out_type),
            ]

        model = oh.make_model(
            oh.make_graph(
                nodes,
                "dummy",
                [oh.make_tensor_value_info("X", in_type, ["a", "b"])],
                [oh.make_tensor_value_info("Y", out_type, ["b", "a"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model

    @parameterized.parameterized.expand(
        [
            (TFLOAT16, True),
            (TFLOAT16, False),
            (TFLOAT, True),
            (TFLOAT, False),
        ]
    )
    def test_llm_transpose_cast(self, in_type, cast_before):
        model_proto = self._transpose_cast(in_type, cast_before)
        ir_model = ir.serde.deserialize_model(model_proto)
        rule_set = llm_rule_sets.llm_rule_set()
        rule_set.apply_to_model(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)

        suffix = "16" if in_type == TFLOAT else "32"
        self.assertEqual(
            [f"Transpose2DCastFP{suffix}"], [n.op_type for n in rewritten_model.graph.node]
        )
        self._check_model(model_proto, rewritten_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
