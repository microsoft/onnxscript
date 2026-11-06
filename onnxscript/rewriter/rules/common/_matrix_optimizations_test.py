# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import shape_inference

from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _broadcast_to_matmul, _matrix_optimizations


def _value(name: str, shape, dtype=onnx.TensorProto.FLOAT):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers=(), *, opset=20) -> ir.Model:
    proto = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes, "matrix_optimization_test", inputs, outputs, initializers
        ),
        opset_imports=[onnx.helper.make_opsetid("", opset)],
        ir_version=10,
    )
    return shape_inference.infer_shapes(ir.from_proto(proto))


def _linear_attention_decomposition(
    *, q_heads: int = 2, kv_heads: int = 2, scale: float = 0.5
) -> ir.Model:
    batch, d_k, d_v = 1, 2, 3
    initializers = [
        onnx.numpy_helper.from_array(
            np.array([batch, 1, q_heads, d_k], np.int64), "query_shape"
        ),
        onnx.numpy_helper.from_array(
            np.array([batch, 1, kv_heads, d_k], np.int64), "key_shape"
        ),
        onnx.numpy_helper.from_array(
            np.array([batch, 1, kv_heads, d_v], np.int64), "value_shape"
        ),
        onnx.numpy_helper.from_array(
            np.array([batch, 1, q_heads * d_v], np.int64), "output_shape"
        ),
        onnx.numpy_helper.from_array(np.array([2], np.int64), "axis_2"),
        onnx.numpy_helper.from_array(np.array([3], np.int64), "axis_3"),
        onnx.numpy_helper.from_array(np.array(scale, np.float32), "scale"),
    ]
    nodes = []
    for name in ("query", "key", "value"):
        nodes.extend(
            [
                onnx.helper.make_node("Reshape", [name, f"{name}_shape"], [f"{name}_4d"]),
                onnx.helper.make_node(
                    "Transpose",
                    [f"{name}_4d"],
                    [f"{name}_transposed"],
                    perm=[0, 2, 1, 3],
                ),
                onnx.helper.make_node(
                    "Squeeze", [f"{name}_transposed", "axis_2"], [f"{name}_unpacked"]
                ),
            ]
        )
    nodes.extend(
        [
            onnx.helper.make_node("Unsqueeze", ["key_unpacked", "axis_3"], ["key_column"]),
            onnx.helper.make_node("Unsqueeze", ["value_unpacked", "axis_2"], ["value_row"]),
            onnx.helper.make_node("Mul", ["key_column", "value_row"], ["outer"]),
            onnx.helper.make_node("Add", ["past_state", "outer"], ["present_state"]),
            onnx.helper.make_node("Unsqueeze", ["query_unpacked", "axis_2"], ["query_row"]),
            onnx.helper.make_node("MatMul", ["query_row", "present_state"], ["read_4d"]),
            onnx.helper.make_node("Squeeze", ["read_4d", "axis_2"], ["read"]),
            onnx.helper.make_node("Mul", ["read", "scale"], ["scaled"]),
            onnx.helper.make_node("Unsqueeze", ["scaled", "axis_2"], ["packed_4d"]),
            onnx.helper.make_node(
                "Transpose",
                ["packed_4d"],
                ["output_transposed"],
                perm=[0, 2, 1, 3],
            ),
            onnx.helper.make_node(
                "Reshape", ["output_transposed", "output_shape"], ["output"]
            ),
        ]
    )
    return _model(
        nodes,
        [
            _value("query", [batch, 1, q_heads * d_k]),
            _value("key", [batch, 1, kv_heads * d_k]),
            _value("value", [batch, 1, kv_heads * d_v]),
            _value("past_state", [batch, kv_heads, d_k, d_v]),
        ],
        [
            _value("output", [batch, 1, q_heads * d_v]),
            _value("present_state", [batch, kv_heads, d_k, d_v]),
        ],
        initializers,
        opset=27,
    )


class MatrixOptimizationsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261105)

    def test_matmul_input_transposes_fold_to_gemm(self):
        model = _model(
            [
                onnx.helper.make_node("Transpose", ["A"], ["AT"], perm=[1, 0]),
                onnx.helper.make_node("Transpose", ["B"], ["BT"], perm=[1, 0]),
                onnx.helper.make_node("MatMul", ["AT", "BT"], ["Y"]),
            ],
            [_value("A", [3, 2]), _value("B", [4, 3])],
            [_value("Y", [2, 4])],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        count = _matrix_optimizations.rules.apply_to_model(rewritten)

        self.assertEqual(count, 2)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Gemm"])
        gemm = next(iter(rewritten.graph))
        self.assertEqual(gemm.attributes.get_int("transA"), 1)
        self.assertEqual(gemm.attributes.get_int("transB"), 1)
        inputs = (
            self.rng.normal(size=(3, 2)).astype(np.float32),
            self.rng.normal(size=(4, 3)).astype(np.float32),
        )
        testing.assert_numerically_equal(model, rewritten, inputs)

    def test_matmul_input_transpose_rejects_integer_dtype(self):
        model = _model(
            [
                onnx.helper.make_node("Transpose", ["A"], ["AT"], perm=[1, 0]),
                onnx.helper.make_node("MatMul", ["AT", "B"], ["Y"]),
            ],
            [
                _value("A", [3, 2], onnx.TensorProto.INT64),
                _value("B", [3, 4], onnx.TensorProto.INT64),
            ],
            [_value("Y", [2, 4], onnx.TensorProto.INT64)],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_matmul_input_transpose_bfloat16_requires_opset_13(self):
        for opset, expected in ((12, 0), (13, 1)):
            with self.subTest(opset=opset):
                model = _model(
                    [
                        onnx.helper.make_node("Transpose", ["A"], ["AT"], perm=[1, 0]),
                        onnx.helper.make_node("MatMul", ["AT", "B"], ["Y"]),
                    ],
                    [
                        _value("A", [3, 2], onnx.TensorProto.BFLOAT16),
                        _value("B", [3, 4], onnx.TensorProto.BFLOAT16),
                    ],
                    [_value("Y", [2, 4], onnx.TensorProto.BFLOAT16)],
                    opset=opset,
                )
                self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), expected)

    def test_gemm_input_transpose_preserves_bias_and_attributes(self):
        model = _model(
            [
                onnx.helper.make_node("Transpose", ["A"], ["AT"], perm=[1, 0]),
                onnx.helper.make_node(
                    "Gemm",
                    ["AT", "B", "C"],
                    ["Y"],
                    alpha=0.5,
                    beta=2.0,
                    transB=1,
                ),
            ],
            [_value("A", [3, 2]), _value("B", [4, 3]), _value("C", [4])],
            [_value("Y", [2, 4])],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        count = _matrix_optimizations.rules.apply_to_model(rewritten)

        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Gemm"])
        gemm = next(iter(rewritten.graph))
        self.assertEqual(gemm.attributes.get_int("transA"), 1)
        self.assertEqual(gemm.attributes.get_int("transB"), 1)
        self.assertEqual(gemm.attributes.get_float("alpha"), 0.5)
        self.assertEqual(gemm.attributes.get_float("beta"), 2.0)
        inputs = (
            self.rng.normal(size=(3, 2)).astype(np.float32),
            self.rng.normal(size=(4, 3)).astype(np.float32),
            self.rng.normal(size=(4,)).astype(np.float32),
        )
        testing.assert_numerically_equal(model, rewritten, inputs)

    def test_gemm_output_transpose(self):
        model = _model(
            [
                onnx.helper.make_node("Gemm", ["A", "B"], ["AB"], alpha=0.25, transA=1),
                onnx.helper.make_node("Transpose", ["AB"], ["Y"], perm=[1, 0]),
            ],
            [_value("A", [3, 2]), _value("B", [3, 4])],
            [_value("Y", [4, 2])],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        count = _matrix_optimizations.rules.apply_to_model(rewritten)

        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Gemm"])
        gemm = next(iter(rewritten.graph))
        self.assertEqual(gemm.inputs[0].name, "B")
        self.assertEqual(gemm.inputs[1].name, "A")
        self.assertEqual(gemm.attributes.get_int("transA"), 1)
        self.assertEqual(gemm.attributes.get_int("transB"), 0)
        inputs = (
            self.rng.normal(size=(3, 2)).astype(np.float32),
            self.rng.normal(size=(3, 4)).astype(np.float32),
        )
        testing.assert_numerically_equal(model, rewritten, inputs)

    def test_gemm_output_transpose_rejects_bias(self):
        model = _model(
            [
                onnx.helper.make_node("Gemm", ["A", "B", "C"], ["AB"]),
                onnx.helper.make_node("Transpose", ["AB"], ["Y"], perm=[1, 0]),
            ],
            [_value("A", [2, 3]), _value("B", [3, 4]), _value("C", [4])],
            [_value("Y", [4, 2])],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_gemm_output_transpose_rejects_shared_gemm(self):
        model = _model(
            [
                onnx.helper.make_node("Gemm", ["A", "B"], ["AB"]),
                onnx.helper.make_node("Transpose", ["AB"], ["Y"], perm=[1, 0]),
            ],
            [_value("A", [2, 3]), _value("B", [3, 4])],
            [_value("AB", [2, 4]), _value("Y", [4, 2])],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_gemm_sum_bias_both_orders(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                sum_inputs = ["C", "AB"] if reverse else ["AB", "C"]
                model = _model(
                    [
                        onnx.helper.make_node("Gemm", ["A", "B"], ["AB"], alpha=0.5, transB=1),
                        onnx.helper.make_node("Sum", sum_inputs, ["Y"]),
                    ],
                    [_value("A", [2, 3]), _value("B", [4, 3]), _value("C", [1, 4])],
                    [_value("Y", [2, 4])],
                )
                rewritten = ir.from_proto(ir.to_proto(model))
                self.assertEqual(
                    _matrix_optimizations.rules.apply_to_model(rewritten),
                    1,
                )
                self.assertEqual([node.op_type for node in rewritten.graph], ["Gemm"])
                inputs = (
                    self.rng.normal(size=(2, 3)).astype(np.float32),
                    self.rng.normal(size=(4, 3)).astype(np.float32),
                    self.rng.normal(size=(1, 4)).astype(np.float32),
                )
                testing.assert_numerically_equal(model, rewritten, inputs)

    def test_gemm_sum_bias_resets_unused_beta(self):
        for beta in (0.0, 0.5):
            with self.subTest(beta=beta):
                model = _model(
                    [
                        onnx.helper.make_node("Gemm", ["A", "B"], ["AB"], beta=beta),
                        onnx.helper.make_node("Sum", ["AB", "C"], ["Y"]),
                    ],
                    [_value("A", [2, 3]), _value("B", [3, 4]), _value("C", [4])],
                    [_value("Y", [2, 4])],
                )
                rewritten = ir.from_proto(ir.to_proto(model))
                self.assertEqual(_matrix_optimizations.rules.apply_to_model(rewritten), 1)
                gemm = next(iter(rewritten.graph))
                self.assertEqual(gemm.attributes.get_float("beta"), 1.0)
                inputs = (
                    self.rng.normal(size=(2, 3)).astype(np.float32),
                    self.rng.normal(size=(3, 4)).astype(np.float32),
                    self.rng.normal(size=(4,)).astype(np.float32),
                )
                testing.assert_numerically_equal(model, rewritten, inputs)

    def test_gemm_sum_rejects_existing_bias_and_output_growth(self):
        existing_bias = _model(
            [
                onnx.helper.make_node("Gemm", ["A", "B", "C1"], ["AB"]),
                onnx.helper.make_node("Sum", ["AB", "C2"], ["Y"]),
            ],
            [
                _value("A", [2, 3]),
                _value("B", [3, 4]),
                _value("C1", [4]),
                _value("C2", [4]),
            ],
            [_value("Y", [2, 4])],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(existing_bias), 0)

        grows_output = _model(
            [
                onnx.helper.make_node("Gemm", ["A", "B"], ["AB"]),
                onnx.helper.make_node("Sum", ["AB", "C"], ["Y"]),
            ],
            [_value("A", [1, 3]), _value("B", [3, 4]), _value("C", [2, 4])],
            [_value("Y", [2, 4])],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(grows_output), 0)

    def test_gemm_sum_rejects_shared_gemm(self):
        model = _model(
            [
                onnx.helper.make_node("Gemm", ["A", "B"], ["AB"]),
                onnx.helper.make_node("Sum", ["AB", "C"], ["Y"]),
                onnx.helper.make_node("Identity", ["AB"], ["Shared"]),
            ],
            [_value("A", [2, 3]), _value("B", [3, 4]), _value("C", [4])],
            [_value("Y", [2, 4]), _value("Shared", [2, 4])],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_reference_transpose_attribute_is_rejected_without_throwing(self):
        x = ir.Value(name="x", shape=ir.Shape([2, 3]))
        transpose = ir.Node(
            "",
            "Transpose",
            [x],
            {"perm": ir.RefAttr("perm", "function_perm", ir.AttributeType.INTS)},
            num_outputs=1,
        )
        self.assertFalse(_matrix_optimizations._matrix_transpose(transpose, x))

    def test_matmul_unit_reduction_to_mul(self):
        model = _model(
            [onnx.helper.make_node("MatMul", ["A", "B"], ["Y"])],
            [
                _value("A", [2, 3, 1], onnx.TensorProto.INT64),
                _value("B", [1, 1, 4], onnx.TensorProto.INT64),
            ],
            [_value("Y", [2, 3, 4], onnx.TensorProto.INT64)],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(rewritten), 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Mul"])
        inputs = (
            self.rng.integers(-20, 20, size=(2, 3, 1), dtype=np.int64),
            self.rng.integers(-20, 20, size=(1, 1, 4), dtype=np.int64),
        )
        testing.assert_numerically_equal(model, rewritten, inputs)

    def test_matmul_unit_reduction_rejects_float_signed_zero_counterexample(self):
        model = _model(
            [onnx.helper.make_node("MatMul", ["A", "B"], ["Y"])],
            [_value("A", [1, 1]), _value("B", [1, 1])],
            [_value("Y", [1, 1])],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)
        matmul = np.matmul(np.array([[-0.0]], np.float32), np.array([[1.0]], np.float32))
        multiply = np.array([[-0.0]], np.float32) * np.array([[1.0]], np.float32)
        self.assertFalse(np.signbit(matmul).item())
        self.assertTrue(np.signbit(multiply).item())

    def test_matmul_unit_reduction_rejects_vector(self):
        model = _model(
            [onnx.helper.make_node("MatMul", ["A", "B"], ["Y"])],
            [_value("A", [1]), _value("B", [1, 4])],
            [_value("Y", [4])],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_expand_broadcast_matmul_removes_materialization(self):
        shape_a = onnx.numpy_helper.from_array(np.array([2, 3, 4], np.int64), "shape_a")
        shape_b = onnx.numpy_helper.from_array(np.array([2, 4, 5], np.int64), "shape_b")
        model = _model(
            [
                onnx.helper.make_node("Expand", ["A", "shape_a"], ["AE"]),
                onnx.helper.make_node("Expand", ["B", "shape_b"], ["BE"]),
                onnx.helper.make_node("MatMul", ["AE", "BE"], ["Y"]),
            ],
            [_value("A", [1, 3, 4]), _value("B", [2, 4, 5])],
            [_value("Y", [2, 3, 5])],
            [shape_a, shape_b],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(rewritten), 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["MatMul"])
        inputs = (
            self.rng.normal(size=(1, 3, 4)).astype(np.float32),
            self.rng.normal(size=(2, 4, 5)).astype(np.float32),
        )
        testing.assert_numerically_equal(model, rewritten, inputs)

    def test_expand_broadcast_matmul_preserves_shared_expand(self):
        shape = onnx.numpy_helper.from_array(np.array([2, 3, 4], np.int64), "shape")
        model = _model(
            [
                onnx.helper.make_node("Expand", ["A", "shape"], ["AE"]),
                onnx.helper.make_node("MatMul", ["AE", "B"], ["Y"]),
                onnx.helper.make_node("Identity", ["AE"], ["Shared"]),
            ],
            [_value("A", [1, 3, 4]), _value("B", [2, 4, 5])],
            [_value("Y", [2, 3, 5]), _value("Shared", [2, 3, 4])],
            [shape],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 1)
        self.assertEqual(
            [node.op_type for node in model.graph], ["Expand", "MatMul", "Identity"]
        )
        matmul = list(model.graph)[1]
        self.assertEqual(matmul.inputs[0].name, "A")

    def test_expand_broadcast_matmul_rejects_matrix_dimension_expansion(self):
        shape = onnx.numpy_helper.from_array(np.array([2, 3, 4], np.int64), "shape")
        model = _model(
            [
                onnx.helper.make_node("Expand", ["A", "shape"], ["AE"]),
                onnx.helper.make_node("MatMul", ["AE", "B"], ["Y"]),
            ],
            [_value("A", [2, 3, 1]), _value("B", [2, 4, 5])],
            [_value("Y", [2, 3, 5])],
            [shape],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_two_pre_matmul_reshapes_removed_when_layout_is_redundant(self):
        shape_a = onnx.numpy_helper.from_array(np.array([2, 3, 4], np.int64), "shape_a")
        shape_b = onnx.numpy_helper.from_array(np.array([1, 4, 5], np.int64), "shape_b")
        model = _model(
            [
                onnx.helper.make_node("Reshape", ["A", "shape_a"], ["AR"]),
                onnx.helper.make_node("Reshape", ["B", "shape_b"], ["BR"]),
                onnx.helper.make_node("MatMul", ["AR", "BR"], ["Y"]),
            ],
            [_value("A", [2, 3, 4]), _value("B", [1, 4, 5])],
            [_value("Y", [2, 3, 5])],
            [shape_a, shape_b],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(rewritten), 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["MatMul"])

    def test_existing_broadcast_to_matmul_covers_pre_and_post_reshapes(self):
        shape_a = onnx.numpy_helper.from_array(np.array([2, 3, 4], np.int64), "shape_a")
        shape_c = onnx.numpy_helper.from_array(np.array([2, 3, 5], np.int64), "shape_c")
        model = _model(
            [
                onnx.helper.make_node("Reshape", ["A", "shape_a"], ["AR"]),
                onnx.helper.make_node("MatMul", ["AR", "B"], ["M"]),
                onnx.helper.make_node("Reshape", ["M", "shape_c"], ["Y"]),
            ],
            [_value("A", [2, 3, 4]), _value("B", [1, 4, 5])],
            [_value("Y", [2, 3, 5])],
            [shape_a, shape_c],
        )
        self.assertEqual(_broadcast_to_matmul.rules.apply_to_model(model), 1)
        self.assertEqual([node.op_type for node in model.graph], ["MatMul"])

    def test_max_x_positive_zero_to_relu(self):
        zero = onnx.numpy_helper.from_array(np.array(0, np.float32), "zero")
        model = _model(
            [onnx.helper.make_node("Max", ["X", "zero"], ["Y"])],
            [_value("X", [4])],
            [_value("Y", [4])],
            [zero],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(rewritten), 1)
        values = np.array([-0.0, np.nan, -2.0, 3.0], np.float32)
        testing.assert_numerically_equal(model, rewritten, (values,))
        self.assertEqual([node.op_type for node in rewritten.graph], ["Relu"])

    def test_max_reversed_or_negative_zero_not_rewritten(self):
        for inputs, zero_value in (
            (["zero", "X"], np.float32(0.0)),
            (["X", "zero"], np.float32(-0.0)),
        ):
            with self.subTest(inputs=inputs, zero=zero_value):
                zero = onnx.numpy_helper.from_array(np.array(zero_value), "zero")
                model = _model(
                    [onnx.helper.make_node("Max", inputs, ["Y"])],
                    [_value("X", [1])],
                    [_value("Y", [1])],
                    [zero],
                )
                self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_max_to_relu_rejects_nonscalar_zero_and_unsupported_dtype(self):
        vector_zero = onnx.numpy_helper.from_array(np.array([0], np.float32), "zero")
        rank_growth = _model(
            [onnx.helper.make_node("Max", ["X", "zero"], ["Y"])],
            [_value("X", [])],
            [_value("Y", [1])],
            [vector_zero],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(rank_growth), 0)

        uint_zero = onnx.numpy_helper.from_array(np.array(0, np.uint32), "zero")
        unsupported_dtype = _model(
            [onnx.helper.make_node("Max", ["X", "zero"], ["Y"])],
            [_value("X", [2], onnx.TensorProto.UINT32)],
            [_value("Y", [2], onnx.TensorProto.UINT32)],
            [uint_zero],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(unsupported_dtype), 0)

    def test_max_to_relu_rejects_bfloat16_negative_zero(self):
        zero = onnx.helper.make_tensor("zero", onnx.TensorProto.BFLOAT16, [], [-0.0])
        model = _model(
            [onnx.helper.make_node("Max", ["X", "zero"], ["Y"])],
            [_value("X", [2], onnx.TensorProto.BFLOAT16)],
            [_value("Y", [2], onnx.TensorProto.BFLOAT16)],
            [zero],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_where_to_leaky_relu(self):
        zero = onnx.numpy_helper.from_array(np.array(0, np.float32), "zero")
        slope = onnx.numpy_helper.from_array(np.array(0.25, np.float32), "slope")
        model = _model(
            [
                onnx.helper.make_node("Greater", ["X", "zero"], ["positive"]),
                onnx.helper.make_node("Mul", ["X", "slope"], ["scaled"]),
                onnx.helper.make_node("Where", ["positive", "X", "scaled"], ["Y"]),
            ],
            [_value("X", [4])],
            [_value("Y", [4])],
            [zero, slope],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(rewritten), 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["LeakyRelu"])
        values = np.array([-0.0, 0.0, -2.0, 3.0], np.float32)
        testing.assert_numerically_equal(model, rewritten, (values,))

    def test_where_to_leaky_relu_rejects_inexact_or_negative_slope(self):
        for alpha in (1.0 + 2.0**-30, -0.5):
            with self.subTest(alpha=alpha):
                zero = onnx.numpy_helper.from_array(np.array(0, np.float64), "zero")
                slope = onnx.numpy_helper.from_array(np.array(alpha, np.float64), "slope")
                model = _model(
                    [
                        onnx.helper.make_node("Greater", ["X", "zero"], ["positive"]),
                        onnx.helper.make_node("Mul", ["X", "slope"], ["scaled"]),
                        onnx.helper.make_node("Where", ["positive", "X", "scaled"], ["Y"]),
                    ],
                    [_value("X", [1], onnx.TensorProto.DOUBLE)],
                    [_value("Y", [1], onnx.TensorProto.DOUBLE)],
                    [zero, slope],
                )
                self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    @unittest.skipUnless(onnx.defs.has("LinearAttention"), "Requires ONNX opset 27 schemas.")
    def test_linear_attention_t1_equal_heads(self):
        inputs = {
            "query": self.rng.normal(size=(1, 1, 4)).astype(np.float32),
            "key": self.rng.normal(size=(1, 1, 4)).astype(np.float32),
            "value": self.rng.normal(size=(1, 1, 6)).astype(np.float32),
            "past_state": self.rng.normal(size=(1, 2, 2, 3)).astype(np.float32),
        }
        for scale in (0.5, 0.3, -0.7, 1e-20, 1e20):
            with self.subTest(scale=scale):
                model = _linear_attention_decomposition(scale=scale)
                rewritten = ir.from_proto(ir.to_proto(model))
                self.assertEqual(_matrix_optimizations.rules.apply_to_model(rewritten), 1)
                self.assertEqual(
                    [node.op_type for node in rewritten.graph], ["LinearAttention"]
                )
                expected = ReferenceEvaluator(ir.to_proto(model)).run(None, inputs)
                actual = ReferenceEvaluator(ir.to_proto(rewritten)).run(None, inputs)
                for expected_value, actual_value in zip(expected, actual):
                    np.testing.assert_array_equal(actual_value, expected_value)
                    self.assertEqual(actual_value.tobytes(), expected_value.tobytes())

    def test_linear_attention_rejects_zero_scale_and_gqa(self):
        zero_scale = _linear_attention_decomposition(scale=0.0)
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(zero_scale), 0)

        broadcast_gqa = _linear_attention_decomposition(q_heads=2, kv_heads=1)
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(broadcast_gqa), 0)

        old_opset = _linear_attention_decomposition()
        old_opset.graph.opset_imports[""] = 26
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(old_opset), 0)

    def test_switch_reshape_activation_before_layout(self):
        shape = onnx.numpy_helper.from_array(np.array([1, 2, 4], np.int64), "shape")
        model = _model(
            [
                onnx.helper.make_node("MatMul", ["A", "B"], ["AB"]),
                onnx.helper.make_node("Reshape", ["AB", "shape"], ["R"]),
                onnx.helper.make_node("LeakyRelu", ["R"], ["Y"], alpha=0.25),
            ],
            [_value("A", [2, 3]), _value("B", [3, 4])],
            [_value("Y", [1, 2, 4])],
            [shape],
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(rewritten), 1)
        self.assertEqual(
            [node.op_type for node in rewritten.graph],
            ["MatMul", "LeakyRelu", "Reshape"],
        )
        inputs = (
            self.rng.normal(size=(2, 3)).astype(np.float32),
            self.rng.normal(size=(3, 4)).astype(np.float32),
        )
        testing.assert_numerically_equal(model, rewritten, inputs)

    def test_switch_transpose_activation_rejects_shared_layout(self):
        model = _model(
            [
                onnx.helper.make_node("MatMul", ["A", "B"], ["AB"]),
                onnx.helper.make_node("Transpose", ["AB"], ["T"], perm=[1, 0]),
                onnx.helper.make_node("Relu", ["T"], ["Y"]),
                onnx.helper.make_node("Identity", ["T"], ["Shared"]),
            ],
            [_value("A", [2, 3]), _value("B", [3, 4])],
            [_value("Y", [4, 2]), _value("Shared", [4, 2])],
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    @unittest.skipUnless(onnx.defs.has("GroupedMatMul"), "Requires ONNX opset 29 schemas.")
    def test_grouped_matmul_with_bias_and_negative_indices(self):
        axes12 = onnx.numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes12")
        axis2 = onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2")
        model = _model(
            [
                onnx.helper.make_node("Unsqueeze", ["X", "axes12"], ["X4"]),
                onnx.helper.make_node("Gather", ["W", "I"], ["WI"], axis=0),
                onnx.helper.make_node("MatMul", ["X4", "WI"], ["MM"]),
                onnx.helper.make_node("Squeeze", ["MM", "axis2"], ["S"]),
                onnx.helper.make_node("Gather", ["B", "I"], ["BI"], axis=0),
                onnx.helper.make_node("Add", ["S", "BI"], ["Y"]),
            ],
            [
                _value("X", [3, 2]),
                _value("W", [4, 2, 5]),
                _value("I", [3, 2], onnx.TensorProto.INT64),
                _value("B", [4, 5]),
            ],
            [_value("Y", [3, 2, 5])],
            [axes12, axis2],
            opset=29,
        )
        rewritten = ir.from_proto(ir.to_proto(model))
        count = _matrix_optimizations.rules.apply_to_model(rewritten)

        self.assertEqual(count, 2)
        self.assertEqual(
            [node.op_type for node in rewritten.graph],
            ["Constant", "Constant", "Less", "Add", "Where", "GroupedMatMul"],
        )
        inputs = (
            self.rng.normal(size=(3, 2)).astype(np.float32),
            self.rng.normal(size=(4, 2, 5)).astype(np.float32),
            np.array([[0, -1], [1, 2], [-2, 3]], dtype=np.int64),
            self.rng.normal(size=(4, 5)).astype(np.float32),
        )
        feeds = dict(zip(("X", "W", "I", "B"), inputs))
        expected = ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)[0]
        actual = ReferenceEvaluator(ir.to_proto(rewritten)).run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        onnx.checker.check_model(ir.to_proto(rewritten), full_check=True)

    def test_grouped_matmul_preserves_shared_product_and_overridable_axes(self):
        for shared_product in (False, True):
            with self.subTest(shared_product=shared_product):
                model = _model(
                    [
                        onnx.helper.make_node(
                            "Gather", ["weights", "indices"], ["selected"], axis=0
                        ),
                        onnx.helper.make_node("Unsqueeze", ["x", "ua"], ["rows"]),
                        onnx.helper.make_node("MatMul", ["rows", "selected"], ["product"]),
                        onnx.helper.make_node("Squeeze", ["product", "sa"], ["y"]),
                    ],
                    [
                        _value("x", [2, 3]),
                        _value("weights", [2, 3, 4]),
                        _value("indices", [2, 2], onnx.TensorProto.INT64),
                    ],
                    [_value("y", [2, 2, 4])],
                    [
                        onnx.numpy_helper.from_array(np.array([1, 2], np.int64), "ua"),
                        onnx.numpy_helper.from_array(np.array([2], np.int64), "sa"),
                    ],
                    opset=29,
                )
                if shared_product:
                    product = next(n for n in model.graph if n.op_type == "MatMul").outputs[0]
                    model.graph.outputs.append(product)
                else:
                    model.graph.inputs.append(model.graph.initializers["ua"])
                self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    @unittest.skipUnless(onnx.defs.has("GroupedMatMul"), "Requires ONNX opset 29 schemas.")
    def test_grouped_bias_requires_equivalent_negative_index_normalization(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 29]>
            graph (float[2,3] x, float[2,3,4] weights,
                   int64[2,2] indices, float[2,4] bias)
                => (float[2,2,4] y)
            <int64 zero={0}, int64 wrong_groups={1}> {
                negative = Less(indices,zero)
                shifted = Add(indices,wrong_groups)
                normalized = Where(negative,shifted,indices)
                product = GroupedMatMul(x,weights,normalized)
                selected_bias = Gather<axis=0>(bias,indices)
                y = Add(product,selected_bias)
            }
            """
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)

    def test_grouped_matmul_requires_opset_29(self):
        axes12 = onnx.numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes12")
        axis2 = onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2")
        model = _model(
            [
                onnx.helper.make_node("Unsqueeze", ["X", "axes12"], ["X4"]),
                onnx.helper.make_node("Gather", ["W", "I"], ["WI"], axis=0),
                onnx.helper.make_node("MatMul", ["X4", "WI"], ["MM"]),
                onnx.helper.make_node("Squeeze", ["MM", "axis2"], ["Y"]),
            ],
            [
                _value("X", [3, 2]),
                _value("W", [4, 2, 5]),
                _value("I", [3, 2], onnx.TensorProto.INT64),
            ],
            [_value("Y", [3, 2, 5])],
            [axes12, axis2],
            opset=20,
        )
        self.assertEqual(_matrix_optimizations.rules.apply_to_model(model), 0)


if __name__ == "__main__":
    unittest.main()
