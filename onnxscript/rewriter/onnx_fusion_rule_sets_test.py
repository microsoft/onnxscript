# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import onnx
import onnx.reference

from onnxscript import ir
from onnxscript.rewriter import onnx_fusion_rule_sets

FLOAT = onnx.TensorProto.FLOAT
FLOAT16 = onnx.TensorProto.FLOAT16
INT64 = onnx.TensorProto.INT64


class OnnxFusionRuleTest(unittest.TestCase):
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
                feeds[i.name] = np.arange(0, shape[0]).astype(np.int64)
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
        use_ort: bool = False,
    ):
        # import onnxruntime does not because of the subfolder named onnxruntime
        import os
        import sys

        sys_path = sys.path
        sys.path = [os.path.join(onnx.__file__, "..", "..")] + [
            p for p in sys_path if "onnxruntime" in p
        ]
        import onnxruntime

        sys.path = sys_path
        if not feeds:
            feeds = self._get_random_inputs(model)

        if use_ort:
            cls = lambda onx: onnxruntime.InferenceSession(  # noqa: E731
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        else:
            cls = lambda onx: onnx.reference.ReferenceEvaluator(  # noqa: E731
                onx, verbose=0
            )
        ref = cls(model)
        opt = cls(optimized_model)
        expected = ref.run(None, feeds)
        got = opt.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    @classmethod
    def _softmax_cross_entropy_loss_models(cls):
        models = [
            onnx.helper.make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node("Equal", ["I", "B"], ["eq1"]),
                        onnx.helper.make_node("Not", ["eq1"], ["neq1"]),
                        onnx.helper.make_node("Where", ["neq1", "I", "zeroi"], ["ind"]),
                        onnx.helper.make_node("Unsqueeze", ["ind", "one"], ["flat_ind"]),
                        onnx.helper.make_node("LogSoftmax", ["X"], ["logX"], axis=1),
                        onnx.helper.make_node(
                            "GatherElements", ["logX", "flat_ind"], ["gx"], axis=1
                        ),
                        onnx.helper.make_node("Squeeze", ["gx", "one"], ["flat_gx"]),
                        onnx.helper.make_node("Neg", ["flat_gx"], ["neg_gx"]),
                        onnx.helper.make_node("Where", ["neq1", "neg_gx", "zerof"], ["w2"]),
                        onnx.helper.make_node("Cast", ["w2"], ["w2f"], to=FLOAT),
                        onnx.helper.make_node("Cast", ["neq1"], ["neq1f"], to=FLOAT),
                        onnx.helper.make_node(
                            "ReduceSum", ["w2f"], ["red1"], keepdims=0, noop_with_empty_axes=0
                        ),
                        onnx.helper.make_node(
                            "ReduceSum",
                            ["neq1f"],
                            ["red2"],
                            keepdims=0,
                            noop_with_empty_axes=0,
                        ),
                        onnx.helper.make_node("Cast", ["red1"], ["red1_16"], to=FLOAT16),
                        onnx.helper.make_node("Cast", ["red2"], ["red2_16"], to=FLOAT16),
                        onnx.helper.make_node("Div", ["red1_16", "red2_16"], ["Y"]),
                    ],
                    "name",
                    [
                        onnx.helper.make_tensor_value_info("X", FLOAT16, [None, None]),
                        onnx.helper.make_tensor_value_info("I", INT64, [None]),
                    ],
                    [onnx.helper.make_tensor_value_info("Y", FLOAT16, [None])],
                    [
                        onnx.numpy_helper.from_array(
                            np.array([-100], dtype=np.int64), name="B"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([1], dtype=np.int64), name="one"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([0], dtype=np.float16), name="zerof"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([0], dtype=np.int64), name="zeroi"
                        ),
                    ],
                ),
                opset_imports=[onnx.helper.make_opsetid("", 18)],
            ),
            onnx.helper.make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node("Equal", ["I", "B"], ["eq1"]),
                        onnx.helper.make_node("Not", ["eq1"], ["neq1"]),
                        onnx.helper.make_node("Equal", ["I", "B"], ["eq2"]),
                        onnx.helper.make_node("Not", ["eq2"], ["neq2"]),
                        onnx.helper.make_node("Equal", ["I", "B"], ["eq3"]),
                        onnx.helper.make_node("Not", ["eq3"], ["neq3"]),
                        onnx.helper.make_node("Where", ["neq1", "I", "zeroi"], ["ind"]),
                        onnx.helper.make_node("Unsqueeze", ["ind", "one"], ["flat_ind"]),
                        onnx.helper.make_node("LogSoftmax", ["X"], ["logX"], axis=1),
                        onnx.helper.make_node("Cast", ["flat_ind"], ["flat_ind_c"], to=INT64),
                        onnx.helper.make_node(
                            "GatherElements", ["logX", "flat_ind_c"], ["gx"], axis=1
                        ),
                        onnx.helper.make_node("Squeeze", ["gx", "one"], ["flat_gx"]),
                        onnx.helper.make_node("Neg", ["flat_gx"], ["neg_gx"]),
                        onnx.helper.make_node("Where", ["neq2", "neg_gx", "zerof"], ["w2"]),
                        onnx.helper.make_node("Cast", ["w2"], ["w2f"], to=FLOAT),
                        onnx.helper.make_node("Cast", ["neq3"], ["neq1f"], to=INT64),
                        onnx.helper.make_node("ReduceSum", ["w2f"], ["red1"], keepdims=0),
                        onnx.helper.make_node(
                            "ReduceSum",
                            ["neq1f"],
                            ["red2"],
                            keepdims=0,
                        ),
                        onnx.helper.make_node("Cast", ["red1"], ["red1_16"], to=FLOAT16),
                        onnx.helper.make_node("Cast", ["red2"], ["red2_16"], to=FLOAT16),
                        onnx.helper.make_node("Div", ["red1_16", "red2_16"], ["Y"]),
                    ],
                    "name",
                    [
                        onnx.helper.make_tensor_value_info("X", FLOAT16, [None, None]),
                        onnx.helper.make_tensor_value_info("I", INT64, [None]),
                    ],
                    [onnx.helper.make_tensor_value_info("Y", FLOAT16, [None])],
                    [
                        onnx.numpy_helper.from_array(
                            np.array([-100], dtype=np.int64), name="B"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([1], dtype=np.int64), name="one"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([0], dtype=np.float16), name="zerof"
                        ),
                        onnx.numpy_helper.from_array(
                            np.array([0], dtype=np.int64), name="zeroi"
                        ),
                    ],
                ),
                opset_imports=[onnx.helper.make_opsetid("", 18)],
            ),
        ]
        return models

    def test_onnx_fusion_rule_set_softmax_cross_entropy_loss(self):
        for model_proto in self._softmax_cross_entropy_loss_models():
            onnx.checker.check_model(model_proto)
            ir_model = ir.serde.deserialize_model(model_proto)
            rule_set = onnx_fusion_rule_sets.onnx_fusion_rule_set()
            rule_set.apply_to_model(ir_model, verbose=0)
            rewritten_model = ir.serde.serialize_model(ir_model)
            self.assertEqual(
                ["SoftmaxCrossEntropyLoss"], [n.op_type for n in rewritten_model.graph.node]
            )
            self._check_model(model_proto, rewritten_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
