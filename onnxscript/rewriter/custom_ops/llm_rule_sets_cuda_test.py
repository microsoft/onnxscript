# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import itertools
import unittest
from typing import Any

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnx.reference
import onnx.reference.op_run as op_run
import onnx.reference.ops.op_scatternd as op_scat
import parameterized

import onnxscript.rewriter.custom_ops.llm_rule_sets_cuda as llm_rule_sets_cuda
from onnxscript import ir

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16


class AddAdd(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, x, y, z):
        return (x + y + z,)


class MulMul(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, x, y, z):
        return (x * y * z,)


class AddMul(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, x, y, z, transposeMiddle=None):
        res = (x + y) * z
        if transposeMiddle:
            res = np.transpose(res, axes=[0, 2, 1, 3])
        return (res,)


class MulAdd(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, x, y, z, transposeMiddle=None):
        res = (x * y) + z
        if transposeMiddle:
            res = np.transpose(res, axes=[0, 2, 1, 3])
        return (res,)


class SubMul(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, x, y, z, negative=None):
        if negative:
            return ((y - x) * z,)
        return ((x - y) * z,)


class MulSub(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, x, y, z, negative=None):
        if negative:
            return (z - (x * y),)
        return ((x * y) - z,)


class AddSharedInput(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, x, y, z):
        return (x + y, x + z)


class MulSharedInput(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, x, y, z):
        return (x * y, x * z)


class MaskedScatterNDOfShape(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, shape, indices, updates, reduction=None, maskedValue=None):
        data = np.zeros(shape, dtype=updates.dtype)
        new_updates = np.where(indices == maskedValue, 0, updates)
        y = op_scat._scatter_nd_impl(data, indices, new_updates, reduction=reduction)
        return (y,)


def sigmoid(x):
    if x > 0:
        return 1 / (1 + np.exp(-x))
    return np.exp(x) / (1 + np.exp(x))


class MulSigmoid(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def __init__(self, onnx_node, run_params):
        op_run.OpRun.__init__(self, onnx_node, run_params)
        self.vf = np.vectorize(sigmoid)

    def _run(self, X):
        if len(X.shape) == 0:
            return ((X * sigmoid(X)).astype(X.dtype),)
        if X.size == 0:
            return (X,)
        return ((X * self.vf(X)).astype(X.dtype),)


class NegXplus1(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return ((1 - X).astype(X.dtype),)


class ReplaceZero(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X, by=None, equal=None):
        x2 = X.copy().flatten()
        if equal:
            x2[x2 == 0] = by
        else:
            x2[x2 != 0] = by
        return (x2.reshape(X.shape),)


class Rotary(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X, splits=None, side=None):
        assert splits is None or (
            splits.shape == (2,) and splits[0] == splits[1]
        ), f"Unexpected split value {splits}"
        last_dim = X.shape[-1] // 2
        cp = X.copy()
        if side == "left":
            cp[..., :last_dim] = X[..., last_dim:]
            cp[..., last_dim:] = -X[..., :last_dim]
        else:
            cp[..., :last_dim] = -X[..., last_dim:]
            cp[..., last_dim:] = X[..., :last_dim]
        return (cp,)


class ScatterNDOfShape(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, shape, indices, updates, reduction=None, strategy=None):
        data = np.zeros(shape, dtype=updates.dtype)
        y = op_scat._scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)


class Transpose2DCastFP16(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return (X.T.astype(np.float16),)


class Transpose2DCastFP32(op_run.OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return (X.T.astype(np.float32),)


def ExtendedReferenceEvaluator(
    model_proto: onnx.ModelProto,
) -> onnx.reference.ReferenceEvaluator:
    return onnx.reference.ReferenceEvaluator(
        model_proto,
        new_ops=[
            AddAdd,
            MulMul,
            AddMul,
            MulAdd,
            AddMul,
            SubMul,
            MulSub,
            AddSharedInput,
            MulSharedInput,
            MulSigmoid,
            NegXplus1,
            ReplaceZero,
            Rotary,
            ScatterNDOfShape,
            MaskedScatterNDOfShape,
            Transpose2DCastFP16,
            Transpose2DCastFP32,
        ],
    )


class LlmRuleSetsTest(unittest.TestCase):
    def _range(self, *shape, bias: float | None = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

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
            elif i.type.tensor_type.elem_type == onnx.TensorProto.INT64:
                if tuple(shape) == (2,):
                    feeds[i.name] = np.array([7, 5], dtype=np.int64)
                else:
                    feeds[i.name] = np.zeros(tuple(shape)).astype(np.int64)
                    feeds[i.name][::2] = 1
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
            optimized_model,
            new_ops=[
                ScatterNDOfShape,
                MaskedScatterNDOfShape,
                Transpose2DCastFP16,
                Transpose2DCastFP32,
            ],
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
        rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
        rule_set.apply_to_model(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)

        suffix = "16" if in_type == TFLOAT else "32"
        self.assertEqual(
            [f"Transpose2DCastFP{suffix}"], [n.op_type for n in rewritten_model.graph.node]
        )
        self._check_model(model_proto, rewritten_model)

    @classmethod
    def _masked_scatternd_of_shape(cls, reduction, itype):
        dtype = np.float32 if itype == onnx.TensorProto.FLOAT else np.float16

        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["data"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node("Equal", ["indices", "mone"], ["masked_indices"]),
                    oh.make_node(
                        "Where",
                        ["masked_indices", "zero", "updates"],
                        ["masked_updates"],
                    ),
                    oh.make_node(
                        "ScatterND",
                        inputs=["data", "indices", "masked_updates"],
                        outputs=["y"],
                        reduction=reduction,
                    ),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, [2]),
                    oh.make_tensor_value_info("indices", onnx.TensorProto.INT64, [5, 3, 1]),
                    oh.make_tensor_value_info("updates", itype, [5, 3, 5]),
                ],
                [oh.make_tensor_value_info("y", itype, [None, None])],
                [
                    onh.from_array(np.array([-1], dtype=np.int64), name="mone"),
                    onh.from_array(np.array([0], dtype=dtype), name="zero"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

    @parameterized.parameterized.expand([("add", TFLOAT), ("add", TFLOAT16)])
    def test_llm_masked_scatter(self, reduction, itype):
        model_proto = self._masked_scatternd_of_shape(reduction, itype)
        ir_model = ir.serde.deserialize_model(model_proto)
        rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
        rule_set.apply_to_model(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)

        self.assertEqual(
            ["MaskedScatterNDOfShape"], [n.op_type for n in rewritten_model.graph.node]
        )
        self._check_model(model_proto, rewritten_model, atol=1e-2)

    def _get_aamm_model(
        self, op_type: str, left: bool, other_type: str | None = None, negative: bool = False
    ) -> onnx.ModelProto:
        if other_type is None:
            other_type = op_type
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(op_type, ["Y", "X"] if negative else ["X", "Y"], ["xy"]),
                    oh.make_node(
                        other_type,
                        (
                            (["Z", "xy"] if left else ["xy", "Z"])
                            if negative
                            else (["xy", "Z"] if left else ["Z", "xy"])
                        ),
                        ["F"],
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("Z", TFLOAT, ["d"]),
                ],
                [oh.make_tensor_value_info("F", TFLOAT, ["d"])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        return model

    @parameterized.parameterized.expand(itertools.product(["Add", "Mul"], [True, False]))
    def test_add_add_mul_mul_pattern(self, op_type, left):
        model = self._get_aamm_model(op_type=op_type, left=left)
        self.assertEqual(len(model.graph.node), 2)

        ir_model = ir.serde.deserialize_model(model)
        rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
        rule_set.apply_to_model(ir_model)
        opt_onx = ir.serde.serialize_model(ir_model)

        self.assertEqual([op_type * 2], [_.op_type for _ in opt_onx.graph.node])
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("ai.onnx.contrib", opsets)
        self.assertEqual(opsets["ai.onnx.contrib"], 1)

        feeds = {
            "X": np.array([10, 11], dtype=np.float32),
            "Y": np.array([10, 12], dtype=np.float32),
            "Z": np.array([10, 13], dtype=np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        np.testing.assert_allclose(expected[0], got[0])

    def test_mul_sigmoid(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["X"], ["xs"]),
                    oh.make_node("Mul", ["X", "xs"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        ir_model = ir.serde.deserialize_model(model)
        rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
        rule_set.apply_to_model(ir_model)
        opt_onx = ir.serde.serialize_model(ir_model)

        self.assertEqual(
            ["MulSigmoid"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": np.arange(18).reshape((3, 6)).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(opt_onx.graph.initializer))
        onnx.checker.check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("ai.onnx.contrib", opsets)
        self.assertEqual(opsets["ai.onnx.contrib"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        np.testing.assert_allclose(expected[0], got[0], atol=1e-5)

    def _simple_rotary(self, side):
        models = [
            oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Split", ["X"], ["s1", "s2"], axis=-1, num_outputs=2),
                        (
                            oh.make_node("Neg", ["s1"], ["ns1"])
                            if side == "left"
                            else oh.make_node("Neg", ["s2"], ["ns2"])
                        ),
                        (
                            oh.make_node("Concat", ["s2", "ns1"], ["Y"], axis=-1)
                            if side == "left"
                            else oh.make_node("Concat", ["ns2", "s1"], ["Y"], axis=-1)
                        ),
                    ],
                    "dummy",
                    [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                    [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
                ),
                opset_imports=[
                    oh.make_opsetid("", 18),
                ],
                ir_version=9,
            ),
            oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Split", ["X", "splits"], ["s1", "s2"], axis=-1),
                        (
                            oh.make_node("Neg", ["s1"], ["ns1"])
                            if side == "left"
                            else oh.make_node("Neg", ["s2"], ["ns2"])
                        ),
                        (
                            oh.make_node("Concat", ["s2", "ns1"], ["Y"], axis=-1)
                            if side == "left"
                            else oh.make_node("Concat", ["ns2", "s1"], ["Y"], axis=-1)
                        ),
                    ],
                    "dummy",
                    [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                    [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
                    [onh.from_array(np.array([4, 4], dtype=np.int64), name="splits")],
                ),
                opset_imports=[
                    oh.make_opsetid("", 18),
                ],
                ir_version=9,
            ),
        ]
        for i, model in enumerate(models):
            with self.subTest(i=i):
                onnx.checker.check_model(model)
                ir_model = ir.serde.deserialize_model(model)
                rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
                rule_set.apply_to_model(ir_model)
                opt_onx = ir.serde.serialize_model(ir_model)
                self.assertEqual(
                    ["Rotary"],
                    [n.op_type for n in opt_onx.graph.node],
                )

                feeds = {
                    "X": np.arange(24).reshape((3, 8)).astype(np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                onnx.checker.check_model(opt_onx)
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("ai.onnx.contrib", opsets)
                self.assertEqual(opsets["ai.onnx.contrib"], 1)

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                np.testing.assert_allclose(expected[0], got[0], atol=1e-5)

    def test_simple_rotary(self):
        self._simple_rotary("right")
        self._simple_rotary("left")

    def test_add_mul_pattern(self):
        for op_type, left in itertools.product(["Add", "Mul"], [True, False]):
            other_type = "Add" if op_type == "Mul" else "Mul"
            with self.subTest(op_type=op_type, left=left):
                model = self._get_aamm_model(op_type=op_type, left=left, other_type=other_type)
                self.assertEqual(len(model.graph.node), 2)

                ir_model = ir.serde.deserialize_model(model)
                rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
                rule_set.apply_to_model(ir_model)
                opt_onx = ir.serde.serialize_model(ir_model)

                self.assertEqual(
                    [f"{op_type}{other_type}"], [_.op_type for _ in opt_onx.graph.node]
                )
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("ai.onnx.contrib", opsets)
                self.assertEqual(opsets["ai.onnx.contrib"], 1)

                feeds = {
                    "X": np.array([10, 11], dtype=np.float32),
                    "Y": np.array([10, 12], dtype=np.float32),
                    "Z": np.array([10, 13], dtype=np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)

                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                np.testing.assert_allclose(expected[0], got[0])

    def test_replace_zero(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cast", ["X"], ["xb"], to=onnx.TensorProto.BOOL),
                    oh.make_node("Where", ["xb", "cst", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
                [onh.from_array(np.array([5.67], dtype=np.float32), name="cst")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        onnx.checker.check_model(model)

        ir_model = ir.serde.deserialize_model(model)
        rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
        rule_set.apply_to_model(ir_model)
        opt_onx = ir.serde.serialize_model(ir_model)

        self.assertEqual(
            ["ReplaceZero"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": (np.arange(18).reshape((3, 6)) - 3).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        # self.assertEqual(0, len(opt_onx.graph.initializer))
        onnx.checker.check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("ai.onnx.contrib", opsets)
        self.assertEqual(opsets["ai.onnx.contrib"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        np.testing.assert_allclose(expected[0], got[0], atol=1e-5)

    def test_negx_plus1(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sub", ["one", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None])],
                [onh.from_array(np.array([1], dtype=np.float32), name="one")],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        onnx.checker.check_model(model)

        ir_model = ir.serde.deserialize_model(model)
        rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
        rule_set.apply_to_model(ir_model)
        opt_onx = ir.serde.serialize_model(ir_model)

        self.assertEqual(
            ["NegXplus1"],
            [n.op_type for n in opt_onx.graph.node],
        )

        feeds = {
            "X": (np.arange(18).reshape((3, 6)) - 3).astype(np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        # self.assertEqual(0, len(opt_onx.graph.initializer))
        onnx.checker.check_model(opt_onx)
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("ai.onnx.contrib", opsets)
        self.assertEqual(opsets["ai.onnx.contrib"], 1)

        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        np.testing.assert_allclose(expected[0], got[0], atol=1e-5)

    @parameterized.parameterized.expand(
        itertools.product(["Sub", "Mul"], [True, False], [False, True])
    )
    def test_sub_mul_pattern(self, op_type, left, negative):
        other_type = "Sub" if op_type == "Mul" else "Mul"
        model = self._get_aamm_model(
            op_type=op_type,
            left=left,
            other_type=other_type,
            negative=negative,
        )
        self.assertEqual(len(model.graph.node), 2)

        ir_model = ir.serde.deserialize_model(model)
        rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
        rule_set.apply_to_model(ir_model)
        opt_onx = ir.serde.serialize_model(ir_model)

        self.assertEqual([f"{op_type}{other_type}"], [_.op_type for _ in opt_onx.graph.node])
        opsets = {v.domain: v.version for v in opt_onx.opset_import}
        self.assertIn("ai.onnx.contrib", opsets)
        self.assertEqual(opsets["ai.onnx.contrib"], 1)

        feeds = {
            "X": np.array([10, 11], dtype=np.float32),
            "Y": np.array([10, 12], dtype=np.float32),
            "Z": np.array([10, 13], dtype=np.float32),
        }
        ref1 = ExtendedReferenceEvaluator(model)
        expected = ref1.run(None, feeds)
        ref2 = ExtendedReferenceEvaluator(opt_onx)
        got = ref2.run(None, feeds)
        np.testing.assert_allclose(expected[0], got[0])

    def _get_shared_input_model(self, op_type: str, left: bool) -> onnx.ModelProto:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(op_type, ["X", "Y"], ["F1"]),
                    oh.make_node(op_type, ["X", "Z"] if left else ["Y", "Z"], ["F2"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("Z", TFLOAT, ["d"]),
                ],
                [
                    oh.make_tensor_value_info("F1", TFLOAT, ["d"]),
                    oh.make_tensor_value_info("F2", TFLOAT, ["d"]),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        return model

    def test_add_mul_shared_input_pattern(self):
        for op_type, left in itertools.product(["Add", "Mul"], [True, False]):
            with self.subTest(op_type=op_type, left=left):
                model = self._get_shared_input_model(
                    op_type=op_type,
                    left=left,
                )
                self.assertEqual(len(model.graph.node), 2)

                ir_model = ir.serde.deserialize_model(model)
                rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
                rule_set.apply_to_model(ir_model)
                opt_onx = ir.serde.serialize_model(ir_model)

                self.assertEqual(
                    [f"{op_type}SharedInput"], [_.op_type for _ in opt_onx.graph.node]
                )
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("ai.onnx.contrib", opsets)
                self.assertEqual(opsets["ai.onnx.contrib"], 1)

                feeds = {
                    "X": np.array([10, 11], dtype=np.float32),
                    "Y": np.array([10, 12], dtype=np.float32),
                    "Z": np.array([10, 13], dtype=np.float32),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)
                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                np.testing.assert_allclose(expected[0], got[0])

    def _get_add_mul_transpose_model(
        self, op_type1: str, op_type2: str, left: bool
    ) -> onnx.ModelProto:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        f"{op_type1}{op_type2}",
                        ["X", "Y", "Z"],
                        ["F1"],
                        domain="ai.onnx.contrib",
                    ),
                    oh.make_node("Transpose", ["F1"], ["final"], perm=[0, 2, 1, 3]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c", "d"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c", "d"])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("ai.onnx.contrib", 1),
            ],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        return model

    def test_add_mul_transpose_pattern(self):
        for op_type, left in itertools.product(["Add", "Mul"], [True, False]):
            with self.subTest(op_type=op_type, left=left):
                op_type1 = op_type
                op_type2 = "Mul" if op_type == "Add" else "Add"
                model = self._get_add_mul_transpose_model(
                    op_type1=op_type1,
                    op_type2=op_type2,
                    left=left,
                )
                self.assertEqual(len(model.graph.node), 2)

                ir_model = ir.serde.deserialize_model(model)
                rule_set = llm_rule_sets_cuda.llm_rule_set_cuda()
                rule_set.apply_to_model(ir_model)
                opt_onx = ir.serde.serialize_model(ir_model)

                self.assertEqual(
                    [f"{op_type1}{op_type2}"], [_.op_type for _ in opt_onx.graph.node]
                )
                opsets = {v.domain: v.version for v in opt_onx.opset_import}
                self.assertIn("ai.onnx.contrib", opsets)
                self.assertEqual(opsets["ai.onnx.contrib"], 1)

                feeds = {
                    "X": self._range(2, 3, 4, 5),
                    "Y": self._range(2, 3, 4, 5),
                    "Z": self._range(2, 3, 4, 5),
                }
                ref1 = ExtendedReferenceEvaluator(model)
                expected = ref1.run(None, feeds)
                ref2 = ExtendedReferenceEvaluator(opt_onx)
                got = ref2.run(None, feeds)
                np.testing.assert_allclose(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
