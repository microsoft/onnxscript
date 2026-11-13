# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Any, Tuple

import numpy as np
import onnx
import onnx.reference
import onnx.reference.op_run
import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import parameterized

import onnxscript.rewriter.ort_fusions.fused_matmul_rule_sets as fused_matmul_rule_sets
import onnxscript.rewriter.pattern as orp
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.rewriter.ort_fusions import _test_utils, optimize_for_ort
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
    C = op.Constant(value_float=0.6)
    ab = ms_op.FusedMatMul(A, B, alpha=0.4, transA=1)
    out = op.Div(ab, C)
    return out


@script()
def _matmul_div(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    C = op.Constant(value_float=0.8)
    ab = op.MatMul(A, B)
    out = op.Div(ab, C)
    return out


@script()
def _matmul_div_div(A: FLOAT[4, 4], B: FLOAT[4, 4]) -> FLOAT[4, 4]:
    C = op.Constant(value_float=0.6)
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


# Issue #3156: a batch Transpose folded into FusedMatMul sets transBatchA/B.
# ONNX Runtime only accepts those flags when both inputs have the same rank >= 3.
@script()
def _transpose_batch_matmul_div_rank3_a_rank2_b(
    X: FLOAT[4, 2, 8], W: FLOAT[8, 16]
) -> FLOAT[2, 4, 16]:
    two = op.Constant(value_float=2.0)
    xt = op.Transpose(X, perm=[1, 0, 2])
    product = op.MatMul(xt, W)
    return op.Div(product, two)


@script()
def _transpose_batch_matmul_div_rank2_a_rank3_b(
    X: FLOAT[4, 8], Y: FLOAT[8, 2, 16]
) -> FLOAT[2, 4, 16]:
    two = op.Constant(value_float=2.0)
    yt = op.Transpose(Y, perm=[1, 0, 2])
    product = op.MatMul(X, yt)
    return op.Div(product, two)


@script()
def _transpose_batch_matmul_div_rank3(
    X: FLOAT[4, 2, 8], Y: FLOAT[2, 8, 16]
) -> FLOAT[2, 4, 16]:
    two = op.Constant(value_float=2.0)
    xt = op.Transpose(X, perm=[1, 0, 2])
    product = op.MatMul(xt, Y)
    return op.Div(product, two)


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
        self.assertEqual(len(got), len(expected))
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
        self.assertEqual([n.op_type for n in ir_model.graph], ["Constant", "FusedMatMul"])
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
        self.assertEqual([n.op_type for n in ir_model.graph], ["FusedMatMul"])
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
            [n.op_type for n in ir_model.graph],
            ["Transpose", "MatMul", "Transpose"],
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
        self.assertEqual([n.op_type for n in ir_model.graph], ["Identity", "FusedMatMul"])
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
        self.assertEqual([n.op_type for n in ir_model.graph], ["FusedMatMul"])
        self._check_model(model_proto, rewritten_model, atol=1e-6)
        feeds = self._get_random_inputs(model_proto)
        expected = _test_utils.ort_run(
            "original_batch", ir.serde.deserialize_model(model_proto), feeds
        )[0]
        got = _test_utils.ort_run("rewritten_batch", ir_model, feeds)[0]
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-7)

    @parameterized.parameterized.expand(
        [
            (
                "rank3_x_rank2_a",
                _transpose_batch_matmul_div_rank3_a_rank2_b,
                [FLOAT[4, 2, 8], FLOAT[8, 16]],
                [FLOAT[2, 4, 16]],
                ["Constant", "Transpose", "FusedMatMul"],
                0,
                0,
            ),
            (
                "rank2_x_rank3_b",
                _transpose_batch_matmul_div_rank2_a_rank3_b,
                [FLOAT[4, 8], FLOAT[8, 2, 16]],
                [FLOAT[2, 4, 16]],
                ["Constant", "Transpose", "FusedMatMul"],
                0,
                0,
            ),
            (
                "equal_rank3",
                _transpose_batch_matmul_div_rank3,
                [FLOAT[4, 2, 8], FLOAT[2, 8, 16]],
                [FLOAT[2, 4, 16]],
                ["Constant", "FusedMatMul"],
                1,
                0,
            ),
            (
                "equal_rank3_b",
                _transpose_batch_matmul_div_rank2_a_rank3_b,
                [FLOAT[2, 4, 8], FLOAT[8, 2, 16]],
                [FLOAT[2, 4, 16]],
                ["Constant", "FusedMatMul"],
                0,
                1,
            ),
        ]
    )
    def test_trans_batch_requires_equal_rank_at_least_3(
        self,
        _,
        script_func,
        input_types,
        output_types,
        expected_ops,
        trans_batch_a,
        trans_batch_b,
    ):
        model_proto = script_func.to_model_proto(
            input_types=input_types,
            output_types=output_types,
        )
        ir_model = ir.serde.deserialize_model(model_proto)
        self._apply_fusion_rules(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)
        self.assertEqual([n.op_type for n in ir_model.graph], expected_ops)
        fused = next(n for n in ir_model.graph if n.op_type == "FusedMatMul")
        attrs = {key: attr.value for key, attr in fused.attributes.items()}
        self.assertEqual(attrs.get("transBatchA", 0), trans_batch_a)
        self.assertEqual(attrs.get("transBatchB", 0), trans_batch_b)
        self.assertAlmostEqual(float(attrs["alpha"]), 0.5)
        feeds = self._get_random_inputs(model_proto)
        self._check_model(model_proto, rewritten_model, feeds=feeds, atol=1e-6)
        expected = _test_utils.ort_run(
            "original", ir.serde.deserialize_model(model_proto), feeds
        )[0]
        got = _test_utils.ort_run("rewritten", ir_model, feeds)[0]
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-7)

        # Match the projection in #3156: the non-transposed operand is an initializer.
        weight_index = 0 if script_func is _transpose_batch_matmul_div_rank2_a_rank3_b else 1
        weight = model_proto.graph.input[weight_index]
        model_proto.graph.initializer.append(
            onnx.numpy_helper.from_array(feeds[weight.name], name=weight.name)
        )
        del feeds[weight.name]
        model_proto.graph.input.remove(weight)
        optimized, _ = optimize_for_ort(ir.serde.deserialize_model(model_proto))
        self.assertEqual(
            [n.op_type for n in optimized.graph if n.op_type != "Constant"],
            [name for name in expected_ops if name != "Constant"],
        )
        fused = next(n for n in optimized.graph if n.op_type == "FusedMatMul")
        self.assertEqual(fused.attributes.get_int("transBatchA", 0), trans_batch_a)
        self.assertEqual(fused.attributes.get_int("transBatchB", 0), trans_batch_b)
        self.assertAlmostEqual(fused.attributes["alpha"].as_float(), 0.5)
        got = _test_utils.ort_run("optimized", optimized, feeds)[0]
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-7)

    @parameterized.parameterized.expand(
        [
            (
                "batch_a",
                fused_matmul_rule_sets.TransposeFusedMatMulWithFlippedBatch1,
                1,
                True,
                False,
            ),
            (
                "batch_b",
                fused_matmul_rule_sets.TransposeFusedMatMulWithFlippedBatch2,
                2,
                True,
                False,
            ),
            (
                "batch_transpose_a",
                fused_matmul_rule_sets.TransposeFusedMatMulWithFlippedBatchAndTranspose1,
                1,
                True,
                True,
            ),
            (
                "batch_transpose_b",
                fused_matmul_rule_sets.TransposeFusedMatMulWithFlippedBatchAndTranspose2,
                2,
                True,
                True,
            ),
            (
                "transpose_a",
                fused_matmul_rule_sets.TransposeFusedMatMulWithBatchAndTranspose1,
                1,
                False,
                True,
            ),
            (
                "transpose_b",
                fused_matmul_rule_sets.TransposeFusedMatMulWithBatchAndTranspose2,
                2,
                False,
                True,
            ),
        ]
    )
    def test_batch_transpose_rank_guard(self, _, rule_class, pos, flip_batch, flip_transpose):
        for x_rank, y_rank in [
            (None, 3),
            (3, None),
            (None, None),
            (2, 2),
            (3, 2),
            (2, 3),
            (3, 4),
            (4, 3),
            (3, 3),
            (4, 4),
        ]:
            for batch_a, batch_b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                # The transpose-only rules require the operand's batch flag to be set.
                batch = batch_a if pos == 1 else batch_b
                if not flip_batch and not batch:
                    continue
                with self.subTest(ranks=(x_rank, y_rank), flags=(batch_a, batch_b)):
                    rank = (x_rank if pos == 1 else y_rank) or 3
                    dims = list(range(rank))
                    if flip_batch and flip_transpose:
                        perm = [dims[-1], *dims[:-1]] if batch else [*dims[1:], dims[0]]
                    elif flip_batch:
                        perm = (
                            [dims[-2], *dims[:-2], dims[-1]]
                            if batch
                            else [*dims[1:-1], dims[0], dims[-1]]
                        )
                    else:
                        perm = [dims[-1], *dims[1:-1], dims[0]]
                    inputs = [
                        onnx.helper.make_tensor_value_info(
                            name, onnx.TensorProto.FLOAT, None if r is None else [None] * r
                        )
                        for name, r in [("x", x_rank), ("y", y_rank)]
                    ]
                    proto = onnx.helper.make_model(
                        onnx.helper.make_graph(
                            [
                                onnx.helper.make_node(
                                    "Transpose", ["x" if pos == 1 else "y"], ["t"], perm=perm
                                ),
                                onnx.helper.make_node(
                                    "FusedMatMul",
                                    ["t", "y"] if pos == 1 else ["x", "t"],
                                    ["out"],
                                    domain="com.microsoft",
                                    alpha=0.5,
                                    transBatchA=batch_a,
                                    transBatchB=batch_b,
                                ),
                            ],
                            "rank_guard",
                            inputs,
                            [
                                onnx.helper.make_tensor_value_info(
                                    "out", onnx.TensorProto.FLOAT, None
                                )
                            ],
                        ),
                        opset_imports=[
                            onnx.helper.make_opsetid("", 18),
                            onnx.helper.make_opsetid("com.microsoft", 1),
                        ],
                    )
                    model = ir.serde.deserialize_model(proto)
                    result_a = 1 - batch_a if flip_batch and pos == 1 else batch_a
                    result_b = 1 - batch_b if flip_batch and pos == 2 else batch_b
                    compatible = x_rank is not None and x_rank == y_rank and x_rank >= 3
                    should_fuse = not (result_a or result_b) or compatible
                    count = orp.RewriteRuleSet([rule_class.rule()]).apply_to_model(model)
                    self.assertEqual(count, int(should_fuse))
                    if should_fuse:
                        self.assertEqual([n.op_type for n in model.graph], ["FusedMatMul"])
                        fused = next(iter(model.graph))
                        self.assertEqual(fused.attributes.get_int("transBatchA", 0), result_a)
                        self.assertEqual(fused.attributes.get_int("transBatchB", 0), result_b)
                        self.assertAlmostEqual(fused.attributes["alpha"].as_float(), 0.5)
                    else:
                        self.assertEqual(
                            [n.op_type for n in model.graph], ["Transpose", "FusedMatMul"]
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
