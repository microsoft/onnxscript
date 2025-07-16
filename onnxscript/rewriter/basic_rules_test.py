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
import onnxscript.rewriter.basic_rules as basic_rules
from onnxscript import ir
from onnxscript.onnx_opset import opset18

FLOAT = onnx.TensorProto.FLOAT


@onnxscript.script()
def cast_identity_model(x: ot.FLOAT["a", "b", "c"]) -> ot.FLOAT["a", "b", "c"]:  # noqa: F821, UP037
    y = opset18.Cast(x, to=onnx.TensorProto.FLOAT)
    return y


def _make_model(*args, **kwargs) -> ir.Model:
    return ir.serde.deserialize_model(onnx.helper.make_model(*args, **kwargs))


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
        rule_set = basic_rules.basic_optimization_rules()
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
        rule_set = basic_rules.basic_optimization_rules()
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
        rule = basic_rules.cast_cast_rule
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
        rule_set = basic_rules.basic_optimization_rules()
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
        rule_set = basic_rules.basic_optimization_rules()
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
        rule_set = basic_rules.basic_optimization_rules()
        model_proto = ir.serde.serialize_model(model)
        rule_set.apply_to_model(model)
        rewritten_model = ir.serde.serialize_model(model)

        self.assertEqual(["Constant", "Unsqueeze"], [n.op_type for n in model.graph])
        self._check_model(model_proto, rewritten_model)

    @parameterized.parameterized.expand(
        [
            (
                "double_reshape_1",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Reshape", ["X", "shape_"], ["Xu"]),
                            onnx.helper.make_node("Reshape", ["Xu", "shape"], ["Y"]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [3, 4, 5])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [5, 4, 3])],
                        [
                            onnx.numpy_helper.from_array(
                                np.array([4, 5, 3], dtype=np.int64), name="shape_"
                            ),
                            onnx.numpy_helper.from_array(
                                np.array([5, 4, 3], dtype=np.int64), name="shape"
                            ),
                        ],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
            ),
            (
                "double_reshape_2",
                _make_model(
                    onnx.helper.make_graph(
                        [
                            onnx.helper.make_node("Reshape", ["X", "shape_"], ["Xu"]),
                            onnx.helper.make_node("Reshape", ["Xu", "shape"], ["Y"]),
                        ],
                        "name",
                        [onnx.helper.make_tensor_value_info("X", FLOAT, [3, 4, 5])],
                        [onnx.helper.make_tensor_value_info("Y", FLOAT, [5, 4, 3])],
                        [
                            onnx.numpy_helper.from_array(
                                np.array([-1], dtype=np.int64), name="shape_"
                            ),
                            onnx.numpy_helper.from_array(
                                np.array([5, 4, 3], dtype=np.int64), name="shape"
                            ),
                        ],
                    ),
                    opset_imports=[onnx.helper.make_opsetid("", 18)],
                ),
            ),
        ]
    )
    def test_reshape_reshape_rule(self, _: str, model: ir.Model):
        rule_set = basic_rules.basic_optimization_rules()
        model_proto = ir.serde.serialize_model(model)
        rule_set.apply_to_model(model)
        rewritten_model = ir.serde.serialize_model(model)

        self.assertEqual(["Reshape"], [n.op_type for n in model.graph])
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
            rule_set = basic_rules.basic_optimization_rules()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Split"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)

    def test_squeeze_reshape_1d_rule(self):
        rule = basic_rules.squeeze_reshape_1d_rule

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
