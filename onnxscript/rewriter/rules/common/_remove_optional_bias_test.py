# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.passes.common import onnx_checker

from onnxscript.rewriter import MatchingTracer, MatchStatus, RewriteRule, testing
from onnxscript.rewriter.rules.common import _remove_optional_bias
from onnxscript.rewriter.rules.common._remove_optional_bias import (
    remove_optional_bias_from_conv_rule,
    remove_optional_bias_from_conv_transpose_rule,
    remove_optional_bias_from_gemm_rule,
    remove_optional_bias_from_qlinear_conv_rule,
)


class _RemoveOptionalBiasTestBase(unittest.TestCase):
    @property
    def rng(self):
        return np.random.default_rng(20251016)

    def clone_model(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    def _get_test_model(
        self,
        op_type: str,
        input_shape: ir.Shape,
        weight_shape: ir.Shape,
        zero_bias: bool,
        attributes=None,
    ):
        tape = ir.tape.Tape()
        bias_shape = weight_shape[1] if op_type == "ConvTranspose" else weight_shape[0]
        output_shape = ir.Shape(("?",) * input_shape.rank())

        x = ir.val("X", shape=input_shape, type=ir.TensorType(ir.DataType.FLOAT))

        w = tape.initializer(
            ir.tensor(self.rng.uniform(-0.5, 0.5, weight_shape).astype(np.float32), name="W")
        )

        if zero_bias:
            bias = np.zeros(bias_shape, dtype=np.float32)
        else:
            bias = self.rng.uniform(-0.5, 0.5, bias_shape).astype(np.float32)

        b = tape.initializer(ir.tensor(bias, name="B"))
        y = tape.op(
            op_type,
            inputs=[x, w, b],
            attributes=attributes,
            output=ir.val("Y", shape=output_shape, type=ir.TensorType(ir.DataType.FLOAT)),
        )

        # Build the model
        ir_model = ir.Model(
            ir.Graph(
                inputs=[x],
                outputs=[y],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": 20},
                name="test_model",
            ),
            ir_version=10,
        )
        onnx_checker.CheckerPass(True)(ir_model)
        return ir_model

    def run_test(
        self,
        base_model: ir.Model,
        input_shape: tuple,
        input_dtype=np.float32,
    ):
        updated_model = self.clone_model(base_model)
        count = _remove_optional_bias.rules.apply_to_model(updated_model)

        # Check rule is applied
        self.assertEqual(count, 1)

        # Check number of inputs is reduced
        self.assertEqual(
            len(updated_model.graph[0].inputs), len(base_model.graph[0].inputs) - 1
        )

        # Prepare inputs
        inputs = (self.rng.random(input_shape).astype(input_dtype),)

        # Check inference
        testing.assert_numerically_equal(base_model, updated_model, inputs)

        # Validate serialized model
        output_model_proto = ir.serde.serialize_model(updated_model)
        onnx.checker.check_model(output_model_proto, full_check=True)

    def run_failed_condition_test(
        self,
        base_model: ir.Model,
        rewrite_rule: RewriteRule,
        expected_message: str,
    ):
        onnx_checker.CheckerPass(True)(base_model)

        updated_model = self.clone_model(base_model)
        tracer = MatchingTracer()
        count = rewrite_rule.apply_to_model(updated_model, tracer=tracer)

        # Check that the model is unchanged
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[rewrite_rule][0]
        self.assertEqual(tracer_match.status.value, MatchStatus.CONDITION_FAILED)
        self.assertRegex(tracer_match.match_result.reason, expected_message)


class RemoveOptionalBiasGemmTest(_RemoveOptionalBiasTestBase):
    def test_successful_remove_optional_bias_gemm(self):
        input_shape = (512, 256)
        base_model = self._get_test_model(
            op_type="Gemm",
            input_shape=ir.Shape(input_shape),
            weight_shape=ir.Shape((64, 256)),
            zero_bias=True,
            attributes={"transB": 1},
        )
        self.run_test(base_model, input_shape)

    def test_fail_remove_optional_bias_gemm(self):
        input_shape = (512, 256)
        base_model = self._get_test_model(
            op_type="Gemm",
            input_shape=ir.Shape(input_shape),
            weight_shape=ir.Shape((64, 256)),
            zero_bias=False,
            attributes={"transB": 1},
        )
        self.run_failed_condition_test(
            base_model, remove_optional_bias_from_gemm_rule, "Bias is not all zeros."
        )


class RemoveOptionalBiasGonvTest(_RemoveOptionalBiasTestBase):
    def test_successful_remove_optional_bias_conv(self):
        input_shape = (1, 3, 32, 32)
        base_model = self._get_test_model(
            op_type="Conv",
            input_shape=ir.Shape(input_shape),
            weight_shape=ir.Shape((16, 3, 3, 3)),
            zero_bias=True,
            attributes={"strides": (2, 2)},
        )
        self.run_test(base_model, input_shape)

    def test_fail_remove_optional_bias_conv(self):
        input_shape = (1, 3, 32, 32)
        base_model = self._get_test_model(
            op_type="Conv",
            input_shape=ir.Shape(input_shape),
            weight_shape=ir.Shape((16, 3, 3, 3)),
            zero_bias=False,
        )
        self.run_failed_condition_test(
            base_model, remove_optional_bias_from_conv_rule, "Bias is not all zeros."
        )


class RemoveOptionalBiasGonvTransposeTest(_RemoveOptionalBiasTestBase):
    def test_successful_remove_optional_bias_conv_transpose(self):
        input_shape = (1, 3, 32, 32)
        base_model = self._get_test_model(
            op_type="ConvTranspose",
            input_shape=ir.Shape(input_shape),
            weight_shape=ir.Shape((3, 16, 3, 3)),
            zero_bias=True,
        )
        self.run_test(base_model, input_shape)

    def test_fail_remove_optional_bias_conv_transpose(self):
        input_shape = (1, 3, 32, 32)
        base_model = self._get_test_model(
            op_type="ConvTranspose",
            input_shape=ir.Shape(input_shape),
            weight_shape=ir.Shape((3, 16, 3, 3)),
            zero_bias=False,
        )
        self.run_failed_condition_test(
            base_model, remove_optional_bias_from_conv_transpose_rule, "Bias is not all zeros."
        )


class RemoveOptionalBiasQLinearConvTest(_RemoveOptionalBiasTestBase):
    def _get_test_model(self, zero_bias):
        if zero_bias:
            bias = np.zeros((16,), dtype=np.int32)
        else:
            bias = self.rng.uniform(-5, 5, (16,)).astype(np.int32)

        w = ir.tensor(self.rng.uniform(-5, 5, (16, 3, 3, 3)).astype(np.uint8), name="W")
        b = ir.tensor(bias, name="B")

        model = ir.from_onnx_text(
            """
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (uint8[N, 3, 32, 32] X) => (uint8 [N, ?, ?, ?] Y)
            <uint8[16, 3, 3, 3] W, int32[16] B, float x_scale = {1.5}, uint8 x_zero_point = {123},
            float w_scale = {1.5}, uint8 w_zero_point = {123},
            float y_scale = {1.5}, uint8 y_zero_point = {123}>
            {
                Y = QLinearConv(X, x_scale, x_zero_point, W, w_scale, w_zero_point, y_scale, y_zero_point, B)
            }
        """,
            initializers=[w, b],
        )
        onnx_checker.CheckerPass(True)(model)
        return model

    def test_successful_remove_optional_bias_qlinear_conv(self):
        input_shape = (1, 3, 32, 32)
        base_model = self._get_test_model(zero_bias=True)
        self.run_test(base_model, input_shape, np.uint8)

    def test_fail_remove_optional_bias_qlinear_conv(self):
        base_model = self._get_test_model(zero_bias=False)
        self.run_failed_condition_test(
            base_model, remove_optional_bias_from_qlinear_conv_rule, "Bias is not all zeros."
        )


if __name__ == "__main__":
    unittest.main()
