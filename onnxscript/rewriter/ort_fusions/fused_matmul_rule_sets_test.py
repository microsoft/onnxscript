# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Any, Tuple

import numpy as np
import onnx
import onnx.reference
import onnx.reference.op_run
import onnx_ir.passes.common as common_passes
import parameterized

import onnxscript.rewriter.ort_fusions.fused_matmul_rule_sets as fused_matmul_rule_sets
from onnxscript import FLOAT, ir, script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.values import Opset

ms_op = Opset("com.microsoft", 1)


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
        if transBatchA != 0 or transBatchB != 0:
            assert len(A.shape) >= 3 and len(B.shape) >= 3, (
                f"Batch dimensions must be at least 3 for A: {A.shape} and B: {B.shape}"
            )
            assert len(A.shape) == len(B.shape), (
                f"Batch dimensions must match for A: {A.shape} and B: {B.shape}"
            )
        if transBatchA:
            perm = list(range(len(A.shape)))
            dim = len(perm)
            perm = [*perm[1 : dim - 1], perm[0], perm[dim - 1]]
            A = np.transpose(A, perm)
        if transBatchB:
            perm = list(range(len(B.shape)))
            dim = len(perm)
            perm = [*perm[1 : dim - 1], perm[0], perm[dim - 1]]
            B = np.transpose(B, perm)
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


@script()
def _fused_matmul_div(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    C = 0.6
    ab = ms_op.FusedMatMul(A, B, alpha=0.4, transA=1)
    out = op.Div(ab, C)
    return out


@script()
def _matmul_div(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    C = 0.8
    ab = op.MatMul(A, B)
    out = op.Div(ab, C)
    return out


@script()
def _matmul_div_div(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    C = 0.6
    ab = op.MatMul(A, B)
    abd = op.Div(ab, C)
    out = op.Div(abd, C)
    return out


@script()
def _fused_matmul_transpose(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    ab = ms_op.FusedMatMul(A, B, alpha=0.5)
    out = op.Transpose(ab, perm=[1, 0])
    return out


@script()
def _matmul_transpose(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    ab = op.MatMul(A, B)
    out = op.Transpose(ab, perm=[1, 0])
    return out


@script()
def _transpose_matmul_1(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    At = op.Transpose(A, perm=[1, 0])
    out = op.MatMul(At, B)
    return out


@script()
def _transpose_fused_matmul_1(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    At = op.Transpose(A, perm=[1, 0])
    out = ms_op.FusedMatMul(At, B)
    return out


@script()
def _transpose_matmul_2(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    Bt = op.Transpose(B, perm=[1, 0])
    out = op.MatMul(A, Bt)
    return out


@script()
def _transpose_fused_matmul_2(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    Bt = op.Transpose(B, perm=[1, 0])
    out = ms_op.FusedMatMul(A, Bt)
    return out


@script()
def _should_not_match(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> Tuple[FLOAT[4, 4], FLOAT[4, 4]]:
    At = op.Transpose(A, perm=[1, 0])
    ab = op.MatMul(At, B)
    C = op.Transpose(At, perm=[1, 0])
    return ab, C


# Add unit tests to check that fusion rewrite can work even if MatMul is not the first node.
@script()
def _fused_matmul_with_identity_before_matmul(A: FLOAT[4, 4]) -> FLOAT[4, 4]:
    B = op.Identity(A)
    ab = op.MatMul(A, B)
    out = op.Transpose(ab, perm=[1, 0])
    return out


@script()
def _fused_matmul_with_identity_before_transpose(A: FLOAT[4, 4]) -> FLOAT[4, 4]:
    B = op.Identity(A)
    ab = op.Transpose(A, perm=[1, 0])
    out = op.MatMul(ab, B)
    return out


@script()
def _transpose_fused_matmul_flip_transBatchA_0_and_transA(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Xt = op.Transpose(X, perm=[1, 2, 3, 0])
    out = ms_op.FusedMatMul(Xt, Y, alpha=0.5, transA=0, transB=0, transBatchA=0, transBatchB=0)
    return out


@script()
def _transpose_fused_matmul_flip_transBatchA_1_and_transA(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Xt = op.Transpose(X, perm=[3, 0, 1, 2])
    out = ms_op.FusedMatMul(Xt, Y, transBatchA=1)
    return out


@script()
def _transpose_fused_matmul_flip_transBatchA_0(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Xt = op.Transpose(X, perm=[1, 2, 0, 3])
    out = ms_op.FusedMatMul(Xt, Y)
    return out


@script()
def _transpose_fused_matmul_flip_transBatchA_1(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Xt = op.Transpose(X, perm=[2, 0, 1, 3])
    out = ms_op.FusedMatMul(Xt, Y, transBatchA=1)
    return out


@script()
def _transpose_fused_matmul_flip_transA(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Xt = op.Transpose(X, perm=[3, 1, 2, 0])
    out = ms_op.FusedMatMul(Xt, Y, transBatchA=1)
    return out


@script()
def _transpose_fused_matmul_flip_transBatchB_0_and_transB(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Yt = op.Transpose(Y, perm=[1, 2, 3, 0])
    out = ms_op.FusedMatMul(X, Yt)
    return out


@script()
def _transpose_fused_matmul_flip_transBatchB_1_and_transB(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Yt = op.Transpose(Y, perm=[3, 0, 1, 2])
    out = ms_op.FusedMatMul(X, Yt, transBatchB=1)
    return out


@script()
def _transpose_fused_matmul_flip_transBatchB_0(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Yt = op.Transpose(Y, perm=[1, 2, 0, 3])
    out = ms_op.FusedMatMul(X, Yt)
    return out


@script()
def _transpose_fused_matmul_flip_transBatchB_1(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Yt = op.Transpose(Y, perm=[2, 0, 1, 3])
    out = ms_op.FusedMatMul(X, Yt, transBatchB=1)
    return out


@script()
def _transpose_fused_matmul_flip_transB(
    X: FLOAT[4, 4, 4, 4], Y: FLOAT[4, 4, 4, 4]
) -> FLOAT[4, 4, 4, 4]:
    Yt = op.Transpose(Y, perm=[3, 1, 2, 0])
    out = ms_op.FusedMatMul(X, Yt, transBatchB=1)
    return out


class TestFusedMatmulRules(unittest.TestCase):
    def _apply_fusion_rules(self, ir_model: ir.Model):
        rule_set = fused_matmul_rule_sets.fused_matmul_rule_sets()
        rule_set.apply_to_model(ir_model)

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
                if shape:
                    feeds[i.name] = np.random.randn(*shape).astype(np.float32)
                else:
                    feeds[i.name] = np.random.randn(1).astype(np.float32)
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

    @parameterized.parameterized.expand(
        [
            (
                "fused_matmul_div",
                _fused_matmul_div,
                [FLOAT[6, "a"], FLOAT[6, "b"]],
                [FLOAT[None, None]],
            ),
            (
                "matmul_div",
                _matmul_div,
                [FLOAT["a", 6], FLOAT[6, "b"]],
                [FLOAT[None, None]],
            ),
            (
                "matmul_div_div",
                _matmul_div_div,
                [FLOAT["a", 6], FLOAT[6, "b"]],
                [FLOAT[None, None]],
            ),
        ]
    )
    def test_fused_matmul_div_models(self, name, script_func, input_types, output_types):
        model_proto = script_func.to_model_proto(
            input_types=input_types,
            output_types=output_types,
        )
        ir_model = ir.serde.deserialize_model(model_proto)
        rule_set = fused_matmul_rule_sets.fused_matmul_rule_sets()
        rule_set.apply_to_model(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)
        self.assertEqual(["Constant", "FusedMatMul"], [n.op_type for n in ir_model.graph])
        self._check_model(model_proto, rewritten_model, atol=1e-6)

    @parameterized.parameterized.expand(
        [
            (
                "fused_matmul_transpose",
                _fused_matmul_transpose,
            ),
            (
                "matmul_transpose",
                _matmul_transpose,
            ),
            (
                "transpose_matmul_1",
                _transpose_matmul_1,
            ),
            (
                "transpose_fused_matmul_1",
                _transpose_fused_matmul_1,
            ),
            ("transpose_matmul_2", _transpose_matmul_2),
            (
                "transpose_fused_matmul_2",
                _transpose_fused_matmul_2,
            ),
        ]
    )
    def test_fused_matmul_with_transpose(self, _, script_func):
        model_proto = script_func.to_model_proto(
            input_types=[FLOAT[4, 4], FLOAT[4, 4]], output_types=[FLOAT[4, 4]]
        )
        ir_model = ir.serde.deserialize_model(model_proto)
        self._apply_fusion_rules(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)
        self.assertEqual(["FusedMatMul"], [n.op_type for n in ir_model.graph])
        self._check_model(model_proto, rewritten_model, atol=1e-6)

    @parameterized.parameterized.expand([("should_not_match", _should_not_match)])
    def test_should_not_match(self, _, script_func):
        model_proto = script_func.to_model_proto(
            input_types=[FLOAT[4, 4], FLOAT[4, 4]], output_types=[FLOAT[4, 4], FLOAT[4, 4]]
        )
        ir_model = ir.serde.deserialize_model(model_proto)
        self._apply_fusion_rules(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)
        self.assertEqual(
            ["Transpose", "MatMul", "Transpose"],
            [n.op_type for n in ir_model.graph],
        )
        self._check_model(model_proto, rewritten_model, atol=1e-6)

    @parameterized.parameterized.expand(
        [
            (
                "fused_matmul_with_identity_before_matmul",
                _fused_matmul_with_identity_before_matmul,
            ),
            (
                "fused_matmul_with_identity_before_transpose",
                _fused_matmul_with_identity_before_transpose,
            ),
        ]
    )
    def test_fused_matmul_with_other_node_in_middle(self, _, script_func):
        model_proto = script_func.to_model_proto(
            input_types=[FLOAT[4, 4]], output_types=[FLOAT[4, 4]]
        )
        ir_model = ir.serde.deserialize_model(model_proto)
        common_passes.ShapeInferencePass()(ir_model)
        self._apply_fusion_rules(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)
        self.assertEqual(["Identity", "FusedMatMul"], [n.op_type for n in ir_model.graph])
        self._check_model(model_proto, rewritten_model, atol=1e-6)

    @parameterized.parameterized.expand(
        [
            (
                "transpose_fused_matmul_flip_transBatchA_0_and_transA",
                _transpose_fused_matmul_flip_transBatchA_0_and_transA,
            ),
            (
                "transpose_fused_matmul_flip_transBatchA_1_and_transA",
                _transpose_fused_matmul_flip_transBatchA_1_and_transA,
            ),
            (
                "transpose_fused_matmul_flip_transBatchA_0",
                _transpose_fused_matmul_flip_transBatchA_0,
            ),
            (
                "transpose_fused_matmul_flip_transBatchA_1",
                _transpose_fused_matmul_flip_transBatchA_1,
            ),
            ("transpose_fused_matmul_flip_transA", _transpose_fused_matmul_flip_transA),
            (
                "transpose_fused_matmul_flip_transBatchB_0_and_transB",
                _transpose_fused_matmul_flip_transBatchB_0_and_transB,
            ),
            (
                "transpose_fused_matmul_flip_transBatchB_1_and_transB",
                _transpose_fused_matmul_flip_transBatchB_1_and_transB,
            ),
            (
                "transpose_fused_matmul_flip_transBatchB_0",
                _transpose_fused_matmul_flip_transBatchB_0,
            ),
            (
                "transpose_fused_matmul_flip_transBatchB_1",
                _transpose_fused_matmul_flip_transBatchB_1,
            ),
            ("transpose_fused_matmul_flip_transB", _transpose_fused_matmul_flip_transB),
        ]
    )
    def test_transpose_fused_matmul_with_batch(self, _, script_func):
        model_proto = script_func.to_model_proto(
            input_types=[FLOAT[4, 4, 4, 4], FLOAT[4, 4, 4, 4]],
            output_types=[FLOAT[4, 4, 4, 4]],
        )
        ir_model = ir.serde.deserialize_model(model_proto)
        self._apply_fusion_rules(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)
        self.assertEqual(["FusedMatMul"], [n.op_type for n in ir_model.graph])
        self._check_model(model_proto, rewritten_model, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
