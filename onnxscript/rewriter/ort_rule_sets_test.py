# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import onnx
import onnx.reference
import onnx.reference.op_run

import onnxscript.rewriter.ort_rule_sets as ort_rule_sets
from onnxscript import ir

FLOAT = onnx.TensorProto.FLOAT


class FusedMatMul(onnx.reference.op_run.OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        A,
        B,
        alpha: float = 1,
        transA: int = 0,
        transB: int = 0,
        transBatchA: int = 0,
        transBatchB: int = 0,
    ):
        assert transBatchA == 0, f"Not implemented for transBatchA==1 and {A.shape}x{B.shape}"
        assert transBatchB == 0, f"Not implemented for transBatchB==1 and {A.shape}x{B.shape}"
        if transA:
            perm = list(range(len(A.shape)))
            dim = len(perm)
            perm[dim - 2], perm[dim - 1] = perm[dim - 1], perm[dim - 2]
            A = np.transpose(A, perm)
        if transB:
            perm = list(range(len(B.shape)))
            dim = len(perm)
            perm[dim - 2], perm[dim - 1] = perm[dim - 1], perm[dim - 2]
            B = np.transpose(B, perm)
        a = np.array(alpha, dtype=A.dtype)
        return (np.matmul(A, B) * a,)


class OrtRuleSetsTest(unittest.TestCase):
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
        ref = onnx.reference.ReferenceEvaluator(model, new_ops=[FusedMatMul])
        opt = onnx.reference.ReferenceEvaluator(optimized_model, new_ops=[FusedMatMul])
        expected = ref.run(None, feeds)
        got = opt.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    @classmethod
    def _fused_matmul_div_models(cls):
        models = [
            onnx.helper.make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node(
                            "FusedMatMul",
                            ["X", "Y"],
                            ["xyc"],
                            transA=1,
                            transB=0,
                            alpha=0.4,
                            transBatchA=0,
                            transBatchB=0,
                            domain="com.microsoft",
                        ),
                        onnx.helper.make_node("Div", ["xyc", "D"], ["Z"]),
                    ],
                    "name",
                    [
                        onnx.helper.make_tensor_value_info("X", FLOAT, [6, "a"]),
                        onnx.helper.make_tensor_value_info("Y", FLOAT, [6, "b"]),
                    ],
                    [onnx.helper.make_tensor_value_info("Z", FLOAT, [None, None])],
                    [
                        onnx.numpy_helper.from_array(
                            np.array([0.8], dtype=np.float32), name="D"
                        ),
                    ],
                ),
                opset_imports=[
                    onnx.helper.make_opsetid("", 18),
                    onnx.helper.make_opsetid("com.microsoft", 1),
                ],
            ),
            onnx.helper.make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node("MatMul", ["X", "Y"], ["xy"]),
                        onnx.helper.make_node("Div", ["xy", "C"], ["Z"]),
                    ],
                    "name",
                    [
                        onnx.helper.make_tensor_value_info("X", FLOAT, ["a", 6]),
                        onnx.helper.make_tensor_value_info("Y", FLOAT, [6, "b"]),
                    ],
                    [onnx.helper.make_tensor_value_info("Z", FLOAT, [None, None])],
                    [
                        onnx.numpy_helper.from_array(
                            np.array([0.6], dtype=np.float32), name="C"
                        )
                    ],
                ),
                opset_imports=[onnx.helper.make_opsetid("", 18)],
            ),
            onnx.helper.make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node("MatMul", ["X", "Y"], ["xy"]),
                        onnx.helper.make_node("Div", ["xy", "C"], ["xyc"]),
                        onnx.helper.make_node("Div", ["xyc", "D"], ["Z"]),
                    ],
                    "name",
                    [
                        onnx.helper.make_tensor_value_info("X", FLOAT, ["a", 6]),
                        onnx.helper.make_tensor_value_info("Y", FLOAT, [6, "b"]),
                    ],
                    [onnx.helper.make_tensor_value_info("Z", FLOAT, [None, None])],
                    [
                        onnx.numpy_helper.from_array(
                            np.array([0.6], dtype=np.float32), name="C"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([0.8], dtype=np.float32), name="D"
                        ),
                    ],
                ),
                opset_imports=[
                    onnx.helper.make_opsetid("", 18),
                ],
            ),
        ]
        return models

    def test_ort_rule_set_fused_matmul_div(self):
        for model_proto in self._fused_matmul_div_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = ort_rule_sets.ort_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["FusedMatMul"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
