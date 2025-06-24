# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Mapping, Sequence

import numpy as np
import onnx_ir as ir
import parameterized
from onnx_ir.passes.common import onnx_checker, shape_inference

from onnxscript.rewriter import pattern as orp
from onnxscript.rewriter import testing
from onnxscript.rewriter.fuse_pad_into_conv import (
    fuse_pad_into_conv,
    fuse_pad_into_conv_rule_set,
    normalize_pad_format_conv,
)


def _clone_model(model: ir.Model) -> ir.Model:
    return ir.from_proto(ir.to_proto(model))


class FusePadConvBaseTest(unittest.TestCase):
    @property
    def rng(self):
        return np.random.default_rng(20250522)

    def get_conv_weights(self, shape: Sequence[int], tape: ir.tape.Tape = None):
        w = ir.tensor(self.rng.uniform(-0.5, 0.5, shape).astype("float32"), name="W")
        if tape is not None:
            w = tape.initializer(w)
        return w

    def build_model(
        self,
        op_type: str,
        input_shape: ir.Shape,
        weight_shape: Sequence[int],
        pad_inputs: Sequence[ir.TensorProtocol | ir.Value | None],
        pad_attributes: Mapping[str, ir.Attr] | None = None,
        conv_attributes: Mapping[str, ir.Attr] | None = None,
    ) -> ir.Model:
        tape = ir.tape.Tape()
        inputs = []
        output_shape = ir.Shape((input_shape[0],) + ("?",) * (len(input_shape) - 1))

        # Convert pad_inputs to initializers (if needed)
        pad_inputs = list(pad_inputs)
        for idx, x in enumerate(pad_inputs):
            if isinstance(x, ir.TensorProtocol):
                pad_inputs[idx] = tape.initializer(x)
            elif isinstance(x, ir.Value):
                inputs.append(x)
            elif isinstance(x, float):
                pad_inputs[idx] = tape.op("Constant", inputs=[], attributes={"value_float": x})
            elif x is not None:
                raise ValueError(f"Unsupported type for pad input ({x}): {type(x)}.")

        # Register operations in the tape
        idtype = ir.DataType.UINT8 if op_type == "ConvInteger" else ir.DataType.FLOAT
        x = ir.Input("X", shape=input_shape, type=ir.TensorType(idtype))
        y = tape.op("Pad", inputs=[x, *pad_inputs], attributes=pad_attributes)
        y = tape.op(
            op_type,
            inputs=[y, self.get_conv_weights(weight_shape, tape)],
            attributes=conv_attributes,
            output=ir.Input("Y", shape=output_shape, type=ir.TensorType(x.dtype)),
        )
        if op_type == "ConvInteger":
            y.dtype = ir.DataType.INT32

        # Build the model
        ir_model = ir.Model(
            ir.Graph(
                inputs=[x, *inputs],
                outputs=[y],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": 20},
                name="model",
            ),
            ir_version=10,
        )
        onnx_checker.CheckerPass(True)(ir_model)
        ir_model = shape_inference.infer_shapes(ir_model)
        return ir_model


class FusePadConvTest(FusePadConvBaseTest):
    @parameterized.parameterized.expand(
        [
            (pad_pads, const_value, axes, conv_pads, conv_auto_pad)
            for pad_pads, axes, conv_pads, conv_auto_pad in [
                ([0, 0, 2, 2, 0, 0, 2, 2], None, None, None),
                ([0, 2, 2, 0, 2, 2], ir.tensor([1, -2, -1], name="axes"), [2, 0, 2, 0], None),
                ([1, 1, 1, 1], ir.tensor([-2, 3], name="axes"), [0, 1, 0, 1], None),
                ([1, 3, 1, 3], ir.tensor([3, 2], name="axes"), None, "VALID"),
            ]
            for const_value in [None, 0.0]
        ]
    )
    def test_fuse_pad_into_conv(self, pad_pads, const_value, axes, conv_pads, conv_auto_pad):
        pad_inputs = [ir.tensor(pad_pads, name="pads")]
        if const_value is not None or axes is not None:
            pad_inputs.append(const_value)
        if axes is not None:
            pad_inputs.append(axes)
        base_model = self.build_model(
            op_type="Conv",
            input_shape=ir.Shape(("N", 32, 14, 16)),
            weight_shape=(10, 32, 3, 3),
            pad_inputs=pad_inputs,
            conv_attributes={"pads": conv_pads, "auto_pad": conv_auto_pad},
        )
        updated_model = _clone_model(base_model)

        # Apply rule
        count = fuse_pad_into_conv_rule_set().apply_to_model(updated_model)

        # Check that Pad was fused
        self.assertEqual(count, 1 if conv_auto_pad is None else 2)
        self.assertEqual(updated_model.graph.num_nodes(), 1)
        onnx_checker.CheckerPass(True)(updated_model)

        # Check inference
        inputs = self.rng.random((1, 32, 14, 16), dtype="float32")
        testing.assert_numerically_equal(base_model, updated_model, (inputs,), atol=0, rtol=0)

    @parameterized.parameterized.expand(
        [
            (
                "constant",
                ir.tensor([1] * 10, name="pads"),
                ir.tensor([0.0], name="const_value"),
                None,
                "NOTSET",
                "must be zero in non-spatial dimensions",
            ),
            (
                "constant",
                ir.tensor([0, 0, 0, 0], name="pads"),
                ir.tensor([1.0], name="const_value"),
                ir.tensor([0, -1], name="axes"),
                "NOTSET",
                "must be equal to 0.",
            ),
            (
                "edge",
                ir.tensor([0, 0, 0, 0], name="pads"),
                ir.tensor([0.0], name="const_value"),
                ir.tensor([0, -1], name="axes"),
                "NOTSET",
                "mode must be 'constant'.",
            ),
            (
                "constant",
                ir.Value(
                    name="pads", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.INT64)
                ),
                None,
                ir.tensor([0, -1], name="axes"),
                "NOTSET",
                "pads is not a constant/initializer.",
            ),
            (
                "constant",
                ir.tensor([0] * 10, name="pads"),
                ir.Value(
                    name="cval", shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.FLOAT)
                ),
                None,
                "NOTSET",
                "cval is not a constant",
            ),
            (
                "constant",
                ir.tensor([0, 0, 0, 0], name="pads"),
                None,
                ir.Value(
                    name="axes", shape=ir.Shape([2]), type=ir.TensorType(ir.DataType.INT64)
                ),
                "NOTSET",
                "axes is not a constant",
            ),
            (
                "constant",
                ir.tensor([0, 0, 0, 0], name="pads"),
                ir.tensor([0.0], name="const_value"),
                ir.tensor([0, -1], name="axes"),
                "VALID",
                "auto_pad must be 'NOTSET'.",
            ),
        ]
    )
    def test_unsupported_fuse_pad_into_conv(
        self, mode, pads, const_value, axes, auto_pad, err_msg
    ):
        base_model = self.build_model(
            op_type="Conv",
            input_shape=ir.Shape(("N", 32, 14, 16, 12)),
            weight_shape=(10, 32, 3, 4, 5),
            pad_inputs=[pads, const_value, axes],
            pad_attributes={"mode": mode},
            conv_attributes={"auto_pad": auto_pad},
        )

        # Apply rule and check it was not applied
        tracer = orp.MatchingTracer()
        count = fuse_pad_into_conv.apply_to_model(base_model, tracer=tracer)
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[fuse_pad_into_conv][0]
        self.assertEqual(tracer_match.status.value, orp.MatchStatus.CONDITION_FAILED)
        self.assertRegex(tracer_match.match_result.reason, err_msg)


class FusePadConvIntegerTest(FusePadConvBaseTest):
    def get_conv_weights(self, shape: Sequence[int], tape: ir.tape.Tape = None):
        w = ir.tensor(self.rng.integers(0, 256, shape).astype("uint8"), name="W")
        if tape is not None:
            w = tape.initializer(w)
        return w

    @parameterized.parameterized.expand(
        [
            (pad_pads, const_value, axes, conv_pads, conv_auto_pad)
            for pad_pads, axes, conv_pads, conv_auto_pad in [
                ([0, 0, 3, 2, 0, 0, 1, 4], None, [1, 1, 1, 1], None),
                ([2, 2, 0, 2, 2, 0], ir.tensor([-2, -1, 1], name="axes"), None, None),
                ([1, 2, 2, 1], ir.tensor([-1, 2], name="axes"), [0, 1, 0, 1], None),
                ([3, 3], ir.tensor([2], name="axes"), None, "SAME_UPPER"),
            ]
            for const_value in [None, ir.tensor(np.array([0], "uint8"), name="const_value")]
        ]
    )
    def test_fuse_pad_into_conv_integer(
        self, pad_pads, const_value, axes, conv_pads, conv_auto_pad
    ):
        pad_inputs = [ir.tensor(pad_pads, name="pads")]
        if const_value is not None or axes is not None:
            pad_inputs.append(const_value)
        if axes is not None:
            pad_inputs.append(axes)
        base_model = self.build_model(
            op_type="ConvInteger",
            input_shape=ir.Shape(("N", 24, 19, 23)),
            weight_shape=(8, 24, 3, 3),
            pad_inputs=pad_inputs,
            conv_attributes={"pads": conv_pads, "auto_pad": conv_auto_pad},
        )
        updated_model = _clone_model(base_model)

        # Apply rule
        count = fuse_pad_into_conv_rule_set().apply_to_model(updated_model)

        # Check that Pad was fused
        self.assertEqual(count, 1 if conv_auto_pad is None else 2)
        self.assertEqual(updated_model.graph.num_nodes(), 1)
        onnx_checker.CheckerPass(True)(updated_model)

        # Check inference
        inputs = self.rng.integers(0, 255, (1, 24, 19, 23), dtype="uint8")
        testing.assert_numerically_equal(base_model, updated_model, (inputs,), atol=0, rtol=0)


class NormalizePadFormatTest(FusePadConvBaseTest):
    @parameterized.parameterized.expand(
        [
            (strides, kernel_shape, auto_pad)
            for strides, kernel_shape in [((2, 3), (1, 4)), ((2, 1), (5, 2))]
            for auto_pad in ["SAME_UPPER", "SAME_LOWER", "VALID"]
        ]
    )
    def test_normalize_pad_format(self, strides, kernel_shape, auto_pad):
        pad_inputs = [
            ir.tensor([1, 1, 1, 1], name="pads"),
            None,
            ir.tensor([2, 3], name="axes"),
        ]
        base_model = self.build_model(
            op_type="Conv",
            input_shape=ir.Shape(("N", 32, 22, 27)),
            weight_shape=(32, 32, *kernel_shape),
            pad_inputs=pad_inputs,
            conv_attributes={
                "strides": strides,
                "auto_pad": auto_pad,
                "kernel_shape": kernel_shape,
            },
        )
        updated_model = _clone_model(base_model)

        # Apply rule
        count = fuse_pad_into_conv_rule_set().apply_to_model(updated_model)

        # Check that Pad was fused
        self.assertEqual(count, 2)
        self.assertEqual(updated_model.graph.num_nodes(), 1)
        onnx_checker.CheckerPass(True)(updated_model)

        # Check inference
        inputs = self.rng.random((1, 32, 22, 27), dtype="float32")
        testing.assert_numerically_equal(base_model, updated_model, (inputs,), atol=0, rtol=0)

    def test_unsupported_normalize_pad_format(self):
        base_model = self.build_model(
            op_type="Conv",
            input_shape=ir.Shape(("N", 32, 14)),
            weight_shape=(32, 11, 4),
            pad_inputs=[ir.tensor([0, 0, 0, 0, 0, 0], name="pads")],
            conv_attributes={"auto_pad": "VALID"},
        )
        # Drop convolutional input shape
        base_model.graph[0].outputs[0].shape = None
        onnx_checker.CheckerPass(True)(base_model)

        # Apply rule and check it was not applied
        tracer = orp.MatchingTracer()
        count = normalize_pad_format_conv.apply_to_model(base_model, tracer=tracer)
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[normalize_pad_format_conv][0]
        self.assertEqual(tracer_match.status.value, orp.MatchStatus.CONDITION_FAILED)
        self.assertRegex(tracer_match.match_result.reason, "Input shapes are not defined")


if __name__ == "__main__":
    unittest.main()
