# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import onnx
import onnx.reference

import onnxscript
import onnxscript.onnx_types as ot
import onnxscript.rewriter.llama_rule_sets as llama_rule_sets
from onnxscript import ir
from onnxscript.onnx_opset import opset18

FLOAT = onnx.TensorProto.FLOAT


class LlamaRuleSetsTest(unittest.TestCase):
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

    @classmethod
    def _identity_models(cls):
        models = [
            onnx.helper.make_model(
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
            onnx.helper.make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node("Mul", ["X", "one"], ["Y"]),
                    ],
                    "name",
                    [onnx.helper.make_tensor_value_info("X", FLOAT, [None])],
                    [onnx.helper.make_tensor_value_info("Y", FLOAT, [None])],
                    [
                        onnx.numpy_helper.from_array(
                            np.array([1], dtype=np.float32), name="one"
                        )
                    ],
                ),
                opset_imports=[onnx.helper.make_opsetid("", 18)],
            ),
            onnx.helper.make_model(
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
        ]
        return models

    def test_llama_p0_rule_set_identity(self):
        for model_proto in self._identity_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = llama_rule_sets.llama_p0_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Identity"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)

    @classmethod
    def _transpose_transpose_models(cls):
        models = [
            onnx.helper.make_model(
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
        ]
        return models

    def test_llama_p0_rule_set_transpose_transpose(self):
        for model_proto in self._transpose_transpose_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = llama_rule_sets.llama_p0_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Transpose"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)

    @classmethod
    def _cast_cast_models(cls):
        models = [
            onnx.helper.make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node(
                            "Cast", ["X"], ["Xc"], to=onnx.TensorProto.FLOAT16
                        ),
                        onnx.helper.make_node(
                            "Cast", ["Xc"], ["Y"], to=onnx.TensorProto.INT32
                        ),
                    ],
                    "name",
                    [onnx.helper.make_tensor_value_info("X", FLOAT, [None, None, None])],
                    [
                        onnx.helper.make_tensor_value_info(
                            "Y", onnx.TensorProto.INT32, [None, None, None]
                        )
                    ],
                ),
                opset_imports=[onnx.helper.make_opsetid("", 18)],
            ),
        ]
        return models

    def test_llama_p0_rule_set_cast_cast(self):
        for model_proto in self._cast_cast_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = llama_rule_sets.llama_p0_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Cast"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)

    @classmethod
    def _cast_identity_models(cls):
        @onnxscript.script()
        def model(x: ot.FLOAT["a", "b", "c"]) -> ot.FLOAT["a", "b", "c"]:  # noqa: F821, UP037
            y = opset18.Cast(x, to=onnx.TensorProto.FLOAT)
            return y

        return [model.to_model_proto()]

    def test_llama_p0_rule_set_cast_identity(self):
        for model_proto in self._cast_identity_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = llama_rule_sets.llama_p0_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Identity"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)

    @classmethod
    def _expand_identity_models(cls):
        models = [
            onnx.helper.make_model(
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
        ]
        return models

    def test_llama_p0_rule_set_expand_identity(self):
        for model_proto in self._expand_identity_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = llama_rule_sets.llama_p0_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Identity"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)

    @classmethod
    def _unsqueeze_unsqueeze_models(cls):
        models = [
            onnx.helper.make_model(
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
            onnx.helper.make_model(
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
        ]
        return models

    def test_llama_p0_rule_set_unsqueeze_unsqueeze(self):
        for model_proto in self._unsqueeze_unsqueeze_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = llama_rule_sets.llama_p0_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(
                ["Constant", "Unsqueeze"], [n.op_type for n in rewritten_model.graph.node]
            )
            self._check_model(model_proto, rewritten_model)

    @classmethod
    def _reshape_reshape_models(cls):
        models = [
            onnx.helper.make_model(
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
            onnx.helper.make_model(
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
        ]
        return models

    def test_llama_p0_rule_set_reshape_reshape(self):
        for model_proto in self._reshape_reshape_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = llama_rule_sets.llama_p0_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Reshape"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)

    @classmethod
    def _slides_split_models(cls):
        models = [
            onnx.helper.make_model(
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

    def test_llama_p0_rule_set_slice_split(self):
        for model_proto in self._slides_split_models():
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = llama_rule_sets.llama_p0_rule_set()
            rule_set.apply_to_model(ir_model)
            rewritten_model = ir.serde.serialize_model(ir_model)

            self.assertEqual(["Slice"], [n.op_type for n in rewritten_model.graph.node])
            self._check_model(model_proto, rewritten_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
