# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import onnx
import onnx.reference
import parameterized

import onnxscript
import onnxscript.onnx_types as ot
from onnxscript import ir
from onnxscript.onnx_opset import opset18
from onnxscript.rewriter import MatchingTracer, testing
from onnxscript.rewriter import pattern as orp
from onnxscript.rewriter.rules.common import _basic_rules

FLOAT = onnx.TensorProto.FLOAT


@onnxscript.script()
def cast_identity_model(x: ot.FLOAT["a", "b", "c"]) -> ot.FLOAT["a", "b", "c"]:  # noqa: F821, UP037
    y = opset18.Cast(x, to=onnx.TensorProto.FLOAT)
    return y


def _make_model(*args, **kwargs) -> ir.Model:
    return ir.serde.deserialize_model(onnx.helper.make_model(*args, **kwargs))


def clone_model(model: ir.Model) -> ir.Model:
    return ir.from_proto(ir.to_proto(model))


class BasicRulesTest(unittest.TestCase):
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
        ref = onnx.reference.ReferenceEvaluator(model)
        opt = onnx.reference.ReferenceEvaluator(optimized_model)
        expected = ref.run(None, feeds)
        got = opt.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    @parameterized.parameterized.expand(
        [
            (
                "no_op_transpose",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 1, 2]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [None, None, None])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [None, None, None])],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
            ),
            (
                "canceled_out_transposes",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Transpose", ["X"], ["xt"], perm=[1, 0]),
                            onnx.helper.make_node("Transpose", ["xt"], ["Y"], perm=[1, 0]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [None, None])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [None, None])],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
            ),
        ]
    )
    def test_basic_optimization_rules_identity(self, _: str, model: ir.Model):
        rule_set = _basic_rules.basic_optimization_rules()
        model_proto = ir.serde.serialize_model(model)
        rule_set.apply_to_model(model)
        rewritten_model = ir.serde.serialize_model(model)

        self.assertEqual(["Identity"], [n.op_type for n in model.graph])
        self._check_model(model_proto, rewritten_model)

    @parameterized.parameterized.expand(
        [
            (
                "consecutive_transposes",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Transpose", ["X"], ["xt"], perm=[1, 2, 0]),
                            onnx.helper.make_node("Transpose", ["xt"], ["Y"], perm=[1, 2, 0]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [None, None, None])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [None, None, None])],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
            ),
        ]
    )
    def test_basic_optimization_rules_transpose_transpose(self, _: str, model: ir.Model):
        rule_set = _basic_rules.basic_optimization_rules()
        model_proto = ir.serde.serialize_model(model)
        rule_set.apply_to_model(model)
        rewritten_model = ir.serde.serialize_model(model)
        self.assertEqual(["Transpose"], [n.op_type for n in model.graph])
        self._check_model(model_proto, rewritten_model)

    def _double_cast_model(self, ostype1, ostype2, ostype3):
        dtype2 = ostype2.dtype
        dtype3 = ostype3.dtype

        @onnxscript.script()
        def cast_cast_model(x):
            intermediate = opset18.Cast(x, to=dtype2)
            y = opset18.Cast(intermediate, to=dtype3)
            return y

        return cast_cast_model.to_model_proto(
            input_types=[ostype1[10]], output_types=[ostype3[10]]
        )

    @parameterized.parameterized.expand(
        [
            ("float16_float_float16", ot.FLOAT16, ot.FLOAT, ot.FLOAT16),
        ]
    )
    def test_cast_cast_rule(self, _: str, type1, type2, type3):
        rule = _basic_rules.cast_cast_rule
        model_proto = self._double_cast_model(type1, type2, type3)
        model = ir.serde.deserialize_model(model_proto)
        rule.apply_to_model(model)
        _rewritten_model = ir.serde.serialize_model(model)

        self.assertEqual(["Cast"], [n.op_type for n in model.graph])
        # TODO: (random) fp16 inputs
        # self._check_model(model_proto, rewritten_model, atol=1e-2)

    @parameterized.parameterized.expand(
        [
            (
                "cast_identity",
                ir.serde.deserialize_model(cast_identity_model.to_model_proto()),
            ),
        ]
    )
    def test_cast_identity_rule(self, _: str, model: ir.Model):
        rule_set = _basic_rules.basic_optimization_rules()
        model_proto = ir.serde.serialize_model(model)
        rule_set.apply_to_model(model)
        rewritten_model = ir.serde.serialize_model(model)

        self.assertEqual(["Identity"], [n.op_type for n in model.graph])
        self._check_model(model_proto, rewritten_model)

    @parameterized.parameterized.expand(
        [
            (
                "normal_case",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Expand", ["X", "shape"], ["Y"]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [3, 4, 5])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [3, 4, 5])],
                        [
                            onnx.numpy_helper.from_array(
                                np.array([3, 4, 5], dtype=np.int64), name="shape"
                            )
                        ],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
                ("Identity",),
            ),
            (
                "input_no_shape",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Identity", ["X"], ["Y"]),
                            onnx.helper.make_node("Expand", ["Y", "shape"], ["Z"]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [3, 4, 5])],
                        [onnx.helper.make_tensor_value_info("Z", FLOAT, [3, 4, 5])],
                        [
                            onnx.numpy_helper.from_array(
                                np.array([3, 4, 5], dtype=np.int64), name="shape"
                            )
                        ],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
                ("Identity", "Expand"),
            ),
        ]
    )
    def test_expand_identity_rule(
        self, _: str, model: ir.Model, expected_nodes: tuple[str, ...]
    ):
        rule_set = _basic_rules.basic_optimization_rules()
        model_proto = ir.serde.serialize_model(model)
        rule_set.apply_to_model(model)
        rewritten_model = ir.serde.serialize_model(model)

        self.assertEqual(tuple(n.op_type for n in model.graph), expected_nodes)
        self._check_model(model_proto, rewritten_model)

    @parameterized.parameterized.expand(
        [
            (
                "double_unsqueezes_1",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Unsqueeze", ["X", "axes1"], ["Xu"]),
                            onnx.helper.make_node("Unsqueeze", ["Xu", "axes2"], ["Y"]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [3])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [1, 3, 1])],
                        [
                            onnx.numpy_helper.from_array(
                                np.array([1], dtype=np.int64), name="axes1"
                            ),
                            onnx.numpy_helper.from_array(
                                np.array([0], dtype=np.int64), name="axes2"
                            ),
                        ],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
            ),
            (
                "double_unsqueezes_2",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Unsqueeze", ["X", "axes1"], ["Xu"]),
                            onnx.helper.make_node("Unsqueeze", ["Xu", "axes2"], ["Y"]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [3])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [1, 3, 1])],
                        [
                            onnx.numpy_helper.from_array(
                                np.array([0], dtype=np.int64), name="axes1"
                            ),
                            onnx.numpy_helper.from_array(
                                np.array([1], dtype=np.int64), name="axes2"
                            ),
                        ],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
            ),
            (
                "double_unsqueezes_3",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Unsqueeze", ["X", "axes1"], ["Xu"]),
                            onnx.helper.make_node("Unsqueeze", ["Xu", "axes2"], ["Y"]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [3])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [1, 3, 1])],
                        [
                            onnx.numpy_helper.from_array(
                                np.array(0, dtype=np.int64), name="axes1"
                            ),
                            onnx.numpy_helper.from_array(
                                np.array(1, dtype=np.int64), name="axes2"
                            ),
                        ],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
            ),
        ]
    )
    def test_unsqueeze_unsqueeze_rule(self, _: str, model: ir.Model):
        rule_set = _basic_rules.basic_optimization_rules()
        model_proto = ir.serde.serialize_model(model)
        rule_set.apply_to_model(model)
        rewritten_model = ir.serde.serialize_model(model)

        self.assertEqual(["Constant", "Unsqueeze"], [n.op_type for n in model.graph])
        self._check_model(model_proto, rewritten_model)

    @classmethod
    def _slices_split_models(cls):
        models = [
            _make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node(
                            "Slice", ["X", "zero", "half", "axis"], ["spl1"]
                        ),
                        onnx.helper.make_node(
                            "Slice", ["X", "half", "last", "axis"], ["spl2"]
                        ),
                    ],
                    "name",
                    [onnx.helper.make_tensor_value_info("X", FLOAT, [3, 4, 6])],
                    [
                        onnx.helper.make_tensor_value_info("spl1", FLOAT, [3, 4, 3]),
                        onnx.helper.make_tensor_value_info("spl2", FLOAT, [3, 4, 3]),
                    ],
                    [
                        onnx.numpy_helper.from_array(
                            np.array([0], dtype=np.int64), name="zero"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([3], dtype=np.int64), name="half"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([6], dtype=np.int64), name="last"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([2], dtype=np.int64), name="axis"
                        ),
                    ],
                ),
                opset_imports=[onnx.helper.make_opsetid("", 18)],
            ),
        ]
        return models

    @unittest.skipIf(True, reason="see https://github.com/microsoft/onnxscript/issues/1642")
    def test_slices_split_rule(self):
        for model_proto in self._slices_split_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = _basic_rules.basic_optimization_rules()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Split"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)

    def test_squeeze_reshape_1d_rule(self):
        rule = _basic_rules.squeeze_reshape_1d_rule

        def check(model_script, expected_count) -> None:
            model_proto = model_script.to_model_proto()
            ir_model = ir.serde.deserialize_model(model_proto)
            count = rule.apply_to_model(ir_model)
            self.assertEqual(count, expected_count)
            if count > 0:
                self.assertEqual([x.op_type for x in ir_model.graph], ["Identity"])
            rewritten_proto = ir.serde.serialize_model(ir_model)
            self._check_model(model_proto, rewritten_proto)

        op = onnxscript.opset17

        # input of shape [12]
        @onnxscript.script()
        def model1(X: ot.FLOAT[12]):
            return op.Reshape(op.Squeeze(X), [-1])

        check(model1, 1)

        # input of shape [1]
        @onnxscript.script()
        def model2(X: ot.FLOAT[1]):
            return op.Reshape(op.Squeeze(X), [-1])

        check(model2, 1)

        # input of shape [1, 1]
        # This should NOT be optimized to Identity
        @onnxscript.script()
        def model3(X: ot.FLOAT[1, 1]):
            return op.Reshape(op.Squeeze(X), [-1])

        check(model3, 0)


class ReshapeReshapeTest(unittest.TestCase):
    @staticmethod
    def create_model(
        input_shape, shape1, shape2, allowzero1=0, allowzero2=0, infer_shape=False
    ):
        def _convert_shape(shape, name):
            if isinstance(shape, np.ndarray):
                shape = tape.initializer(ir.Tensor(shape, name=name))
            elif isinstance(shape, (list, tuple)):
                shape = ir.val(name, ir.Shape(shape), ir.TensorType(ir.DataType.INT64))
                tape.graph_like.inputs.append(shape)
            else:
                raise TypeError(f"Unsupported type {type(shape)} for shape.")
            return shape

        x = ir.val("X", ir.Shape(input_shape), ir.TensorType(ir.DataType.FLOAT))
        y = ir.val("Y", type=ir.TensorType(ir.DataType.FLOAT))
        tape = ir.tape.Tape(ir.Graph([x], [y], nodes=[], opset_imports={"": 20}))

        # Build the graph.
        reshape = tape.op(
            "Reshape",
            inputs=[x, _convert_shape(shape1, "shape_")],
            attributes={"allowzero": allowzero1},
        )
        tape.op(
            "Reshape",
            inputs=[reshape, _convert_shape(shape2, "shape")],
            attributes={"allowzero": allowzero2},
            output=y,
        )
        model = ir.Model(tape.graph_like, ir_version=10)

        # Infer shapes.
        if infer_shape:
            model = ir.passes.common.ShapeInferencePass()(model).model
        return model

    @parameterized.parameterized.expand(
        [
            ((3, 4, 5), [4, 5, 3], [5, 4, 3]),
            ((3, 4, 5), [4, 5, 3], [5, 4, 3]),
            ((3, 4, 8), [2, 0, 3, -1], [0, 3, 2, 8]),
            ((3, 4, 8), [3, 4, -1], [-1, 12], 1),
            ((3, 4, 2), [0, 4, -1], [12, -1], 0, 1),
            ((3, 0, 8), [4, 2, 0, 0], [3, 0], 1, 1),
        ]
    )
    def test_reshape_reshape_rule(
        self, input_shape, shape1, shape2, allowzero1=0, allowzero2=0
    ):
        model = self.create_model(
            input_shape,
            np.array(shape1, dtype="int64"),
            np.array(shape2, dtype="int64"),
            allowzero1=allowzero1,
            allowzero2=allowzero2,
        )
        updated_model = clone_model(model)

        # check rewrite approach.
        count = _basic_rules.reshape_reshape_rule.apply_to_model(updated_model)
        self.assertEqual(count, 1)
        self.assertEqual(["Reshape"], [n.op_type for n in updated_model.graph])

        # Check inference.
        inputs = np.random.default_rng(10).random(input_shape, dtype="float32")
        testing.assert_numerically_equal(model, updated_model, (inputs,), atol=0, rtol=0)

    @parameterized.parameterized.expand([([3, 2, 3, 3, 3], 1), ([0, -1, 3, 2], 0)])
    def test_reshape_dynamic_reshape_rule(self, shape1, allowzero1=0):
        input_shape = (3, 6, 9)
        shape1 = np.array(shape1, dtype="int64")
        # Build the model with unknown shape1.
        model = self.create_model(
            input_shape,
            (shape1.size,),
            np.array((1, 6, 27), dtype="int64"),
            allowzero1=allowzero1,
        )
        updated_model = clone_model(model)

        # check rewrite approach.
        count = _basic_rules.reshape_reshape_rule.apply_to_model(updated_model)
        self.assertEqual(count, 1)
        self.assertEqual(["Reshape"], [n.op_type for n in updated_model.graph])

        # Check inference.
        feeds = {
            "X": np.random.default_rng(2).random(input_shape, dtype="float32"),
            "shape_": shape1,
        }
        testing.assert_numerically_equal(model, updated_model, feeds, atol=0, rtol=0)

    @parameterized.parameterized.expand(
        [((3, 6, 9), [0, 3, 2, -1]), ((0, 6, 2), [0, 0, 3], 1)]
    )
    def test_reshape_reshape_dynamic_rule(self, input_shape, shape2, allowzero2=0):
        # Note that shape inference is required for this test to be valid.
        shape2 = np.array(shape2, dtype="int64")
        model = self.create_model(
            input_shape,
            np.array((3, 2, -1), dtype="int64"),
            shape2,
            allowzero2=allowzero2,
            infer_shape=True,
        )
        updated_model = clone_model(model)

        # check rewrite approach.
        count = _basic_rules.reshape_reshape_rule.apply_to_model(updated_model)
        self.assertEqual(count, 1)
        self.assertEqual(["Reshape"], [n.op_type for n in updated_model.graph])

        # Check inference.
        inputs = np.random.default_rng(7).random(input_shape, dtype="float32")
        testing.assert_numerically_equal(model, updated_model, (inputs,), atol=0, rtol=0)

    @parameterized.parameterized.expand(
        [
            ((3,), "is not a constant"),
            (np.array([0, -1], dtype="int64"), "both 0 and -1 dimensions"),
            (np.array([0, 0, 3], dtype="int64"), "more than one 0 dimension"),
        ]
    )
    def test_unsupported_reshape_reshape(self, shape2, error_msg):
        model = self.create_model((1, 2, 3), np.array([1, 6], dtype="int64"), shape2)

        # Check rewrite approach.
        tracer = MatchingTracer()
        count = _basic_rules.reshape_reshape_rule.apply_to_model(model, tracer=tracer)
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[_basic_rules.reshape_reshape_rule][0]
        self.assertEqual(tracer_match.status.value, orp.MatchStatus.CONDITION_FAILED)
        self.assertRegex(tracer_match.match_result.reason, error_msg)


class Flatten2ReshapeTest(unittest.TestCase):
    @staticmethod
    def create_model(input_shape, axis=1):
        x = ir.val("X", ir.Shape(input_shape), ir.TensorType(ir.DataType.FLOAT))
        y = ir.val("Y", type=ir.TensorType(ir.DataType.FLOAT))
        tape = ir.tape.Tape(ir.Graph([x], [y], nodes=[], opset_imports={"": 20}))

        # Build the graph.
        tape.op("Flatten", inputs=[x], attributes={"axis": axis}, output=y)
        model = ir.Model(tape.graph_like, ir_version=10)
        return model

    @parameterized.parameterized.expand(list(range(-5, 6)))
    def test_flatten_to_reshape_rule(self, axis):
        input_shape = (1, 4, 8, 7, 5)
        model = self.create_model(input_shape=input_shape, axis=axis)
        updated_model = clone_model(model)

        # check rewrite approach.
        count = _basic_rules.flatten_to_reshape_rule.apply_to_model(updated_model)
        self.assertEqual(count, 1)
        self.assertEqual(["Reshape"], [n.op_type for n in updated_model.graph])

        # Check inference.
        inputs = np.random.default_rng(13).random(input_shape, dtype="float32")
        testing.assert_numerically_equal(model, updated_model, (inputs,), atol=0, rtol=0)

    @parameterized.parameterized.expand(list(range(-4, 5)))
    def test_flatten_to_reshape_dynamic_input(self, axis):
        model = self.create_model(input_shape=("N", "C1", "C2", "C3"), axis=axis)
        # Rule is supported in all cases if the output shape is known for non-special cases.
        input_shape = (1, 2, 3, 4)
        if axis not in {-3, 0, 1, 4}:
            out_shape = ir.Shape((np.prod(input_shape[:axis]), np.prod(input_shape[axis:])))
            model.graph.outputs[0].shape = out_shape
        updated_model = clone_model(model)

        # check rewrite approach.
        count = _basic_rules.flatten_to_reshape_rule.apply_to_model(updated_model)
        self.assertEqual(count, 1)
        self.assertEqual(["Reshape"], [n.op_type for n in updated_model.graph])

        # Check inference.
        inputs = np.random.default_rng(17).random(input_shape, dtype="float32")
        testing.assert_numerically_equal(model, updated_model, (inputs,), atol=0, rtol=0)

    def test_unsupported_flatten_to_reshape(self):
        model = self.create_model(input_shape=("N", "C1", "C2"), axis=2)

        # Check rewrite approach.
        tracer = MatchingTracer()
        count = _basic_rules.flatten_to_reshape_rule.apply_to_model(model, tracer=tracer)
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[_basic_rules.flatten_to_reshape_rule][0]
        self.assertEqual(tracer_match.status.value, orp.MatchStatus.CONDITION_FAILED)
        self.assertRegex(tracer_match.match_result.reason, "Impossible to compute new shape")


if __name__ == "__main__":
    unittest.main(verbosity=2)
