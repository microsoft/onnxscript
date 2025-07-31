# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx_ir.passes.common import onnx_checker, shape_inference

from onnxscript import optimizer
from onnxscript.rewriter import testing
from onnxscript.rewriter.fuse_hardswish import fuse_hardswish_rules


class FuseHardSwishTest(unittest.TestCase):
    @property
    def rng(self):
        return np.random.default_rng(20250621)

    def clone_model(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    def run_test(
        self,
        base_model: ir.Model,
        expected_op_types: list[str],
        dtype: str = "float",
    ):
        onnx_checker.CheckerPass(True)(base_model)
        base_model = shape_inference.infer_shapes(base_model)
        updated_model = self.clone_model(base_model)
        _ = fuse_hardswish_rules().apply_to_model(updated_model)

        # Polish model to remove unused constants
        updated_model = optimizer.optimize(updated_model)

        # Check expected op_types
        self.assertEqual([node.op_type for node in updated_model.graph], expected_op_types)

        # Check inference
        inputs = (self.rng.integers(low=-10, high=10, size=(2 * 32), dtype=np.int32),)
        if dtype == "float":
            inputs = (inputs[0].astype(np.float32),)

        testing.assert_numerically_equal(
            base_model,
            updated_model,
            inputs,
            ort_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )

        # Validate serialized model
        output_model_proto = ir.to_proto(updated_model)
        onnx.checker.check_model(output_model_proto, full_check=True)

    def test_hardsigmoid_fusion(self):
        model_text = """
            <ir_version: 8, opset_import: ["" : 18]>
            hardsigmoid (float[N] x) => (float[N] y) {
                three = Constant <value = float[1] {3.0}> ()
                six = Constant <value = float[1] {6.0}> ()
                zero = Constant <value = float[1] {0.0}> ()
                x_plus_3 = Add(x, three)
                clipped = Clip(x_plus_3, zero, six)
                y = Div(clipped, six)
            }
        """
        model = ir.from_proto(onnx.parser.parse_model(model_text))
        self.run_test(model, ["HardSigmoid"])

    def test_hardswish_fusion(self):
        model_text = """
            <ir_version: 8, opset_import: ["" : 18]>
            hardswish (float[N] x) => (float[N] y) {
                three = Constant <value = float[1] {3.0}> ()
                six = Constant <value = float[1] {6.0}> ()
                zero = Constant <value = float[1] {0.0}> ()
                x_plus_3 = Add(x, three)
                clipped = Clip(x_plus_3, zero, six)
                mul_x = Mul(clipped, x)
                y = Div(mul_x, six)
            }
        """
        model = ir.from_proto(onnx.parser.parse_model(model_text))
        self.run_test(model, ["HardSwish"])

    def test_hardswish_fusion_mul_last(self):
        model_text = """
            <ir_version: 8, opset_import: ["" : 18]>
            hardswish (float[N] x) => (float[N] y) {
                three = Constant <value = float[1] {3.0}> ()
                six = Constant <value = float[1] {6.0}> ()
                zero = Constant <value = float[1] {0.0}> ()
                x_plus_3 = Add(x, three)
                clipped = Clip(x_plus_3, zero, six)
                div_x = Div(clipped, six)
                y = Mul(div_x, x)
            }
        """
        model = ir.from_proto(onnx.parser.parse_model(model_text))
        self.run_test(model, ["HardSwish"])

    def test_hardswish_fusion_from_sigmoid(self):
        model_text = """
            <ir_version: 8, opset_import: ["" : 18]>
            hardswish (float[N] x) => (float[N] y) {
                hardsigmoid_out = HardSigmoid<alpha=0.16666666, beta=0.5>(x)
                y = Mul(hardsigmoid_out, x)
            }
        """
        model = ir.from_proto(onnx.parser.parse_model(model_text))
        self.run_test(model, ["HardSwish"])


if __name__ == "__main__":
    unittest.main()
