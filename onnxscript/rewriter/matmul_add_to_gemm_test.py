# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from typing import Sequence

import numpy as np
import onnx
from onnx_ir.passes.common import onnx_checker, shape_inference
from parameterized import parameterized

from onnxscript import ir
from onnxscript.rewriter import MatchingTracer, MatchStatus, matmul_add_to_gemm, testing
from onnxscript.rewriter.matmul_add_to_gemm import matmul_add_to_gemm_rule


class _MatMulAddToGemmTestBase(unittest.TestCase):
    @property
    def rng(self):
        return np.random.default_rng(20250607)

    def clone_model(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    def get_test_model(
        self,
        input_shape: ir.Shape,
        weight_shape: ir.Shape,
        transA: bool = False,
        transB: bool = False,
        permA: Sequence[int] = [1, 0],
        permB: Sequence[int] = [1, 0],
        weight_as_inputs: bool = False,
        bias_as_inputs: bool = False,
    ):
        """Returns the following model:

            Y = Add(MatMul(Transpose(X), Transpose(W)), B)

        Where:
        - Transpose(X) is applied only if `transA=True`
        - Transpose(W) is applied only if `transB=True`
        - W and B can be graph inputs or initializers
        """
        tape = ir.tape.Tape()
        inputs = []
        bias_shape = weight_shape[0] if transB else weight_shape[-1]
        output_shape = ir.Shape(("?",) * input_shape.rank())

        x = ir.Input("X", shape=input_shape, type=ir.TensorType(ir.DataType.FLOAT))

        if weight_as_inputs:
            w = ir.Input("W", shape=weight_shape, type=ir.TensorType(ir.DataType.FLOAT))
            inputs.append(w)
        else:
            w = ir.tensor(
                self.rng.uniform(-0.5, 0.5, weight_shape).astype("float32"), name="W"
            )
            w = tape.initializer(w)

        if bias_as_inputs:
            b = ir.Input(
                "B", shape=ir.Shape([bias_shape]), type=ir.TensorType(ir.DataType.FLOAT)
            )
            inputs.append(b)
        else:
            b = ir.tensor(self.rng.uniform(-0.5, 0.5, bias_shape).astype("float32"), name="B")
            b = tape.initializer(b)

        x_t, w_t = None, None
        if transA:
            x_t = tape.op("Transpose", inputs=[x], attributes={"perm": permA})

        if transB:
            w_t = tape.op("Transpose", inputs=[w], attributes={"perm": permB})

        y = tape.op("MatMul", inputs=[x_t if transA else x, w_t if transB else w])
        y = tape.op(
            "Add",
            inputs=[y, b],
            output=ir.Input("Y", shape=output_shape, type=ir.TensorType(ir.DataType.FLOAT)),
        )

        # Build the model
        ir_model = ir.Model(
            ir.Graph(
                inputs=[x, *inputs],
                outputs=[y],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": 20},
                name="test_model",
            ),
            ir_version=10,
        )
        onnx_checker.CheckerPass(True)(ir_model)
        ir_model = shape_inference.infer_shapes(ir_model)
        return ir_model

    def check_matmul_add_to_gemm_incompatible_shapes(self, **kwargs):
        base_model = self.get_test_model(**kwargs)

        updated_model = self.clone_model(base_model)
        tracer = MatchingTracer()
        count = matmul_add_to_gemm_rule.apply_to_model(updated_model, tracer=tracer)

        # Check that the model is unchanged
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[matmul_add_to_gemm_rule][0]
        self.assertEqual(tracer_match.status.value, MatchStatus.CONDITION_FAILED)
        self.assertRegex(
            tracer_match.match_result.reason, "Rank of input_a and input_b must be 2"
        )


class MatMulAddToGemmTest(_MatMulAddToGemmTestBase):
    @parameterized.expand(
        [
            ("initializers", False, False),
            ("inputs", True, True),
        ]
    )
    def test_matmul_add_to_gemm(self, _, weight_as_inputs, bias_as_inputs):
        base_model = self.get_test_model(
            input_shape=ir.Shape((512, 256)),
            weight_shape=ir.Shape((256, 64)),
            weight_as_inputs=weight_as_inputs,
            bias_as_inputs=bias_as_inputs,
        )
        updated_model = self.clone_model(base_model)
        count = matmul_add_to_gemm.gemm_rule_set().apply_to_model(updated_model)

        # Check MatMul + Add are fused into Gemm
        self.assertEqual(count, 1)
        self.assertEqual(len(updated_model.graph), 1)

        # Prepare inputs
        if weight_as_inputs and bias_as_inputs:
            inputs = (
                self.rng.random((512, 256), dtype=np.float32),
                self.rng.random((256, 64), dtype=np.float32),
                self.rng.random((64), dtype=np.float32),
            )
        else:
            inputs = (self.rng.random((512, 256), dtype=np.float32),)

        # Check inference
        testing.assert_numerically_equal(base_model, updated_model, inputs)

        # Validate serialized model
        output_model_proto = ir.serde.serialize_model(updated_model)
        onnx.checker.check_model(output_model_proto, full_check=True)

    def test_matmul_add_to_gemm_incompatible_shapes(self):
        kwargs = {
            "input_shape": ir.Shape((1, 256, 512)),
            "weight_shape": ir.Shape((1, 512, 64)),
        }
        return super().check_matmul_add_to_gemm_incompatible_shapes(**kwargs)


class TransAMatMulAddToGemmTest(_MatMulAddToGemmTestBase):
    @parameterized.expand(
        [
            ("initializers", False, False),
            ("inputs", True, True),
        ]
    )
    def test_transpose_a_matmul_add_to_gemm(self, _, weight_as_inputs, bias_as_inputs):
        base_model = self.get_test_model(
            input_shape=ir.Shape((256, 512)),
            weight_shape=ir.Shape((256, 64)),
            weight_as_inputs=weight_as_inputs,
            bias_as_inputs=bias_as_inputs,
            transA=True,
        )
        updated_model = self.clone_model(base_model)
        count = matmul_add_to_gemm.gemm_rule_set().apply_to_model(updated_model)

        # Check MatMul(Transpose, W) + Add are fused into Gemm
        self.assertEqual(count, 1)
        self.assertEqual(len(updated_model.graph), 1)

        # Prepare inputs
        if weight_as_inputs and bias_as_inputs:
            inputs = (
                self.rng.random((256, 512), dtype=np.float32),
                self.rng.random((256, 64), dtype=np.float32),
                self.rng.random((64,), dtype=np.float32),
            )
        else:
            inputs = (self.rng.random((256, 512), dtype=np.float32),)

        # Check inference
        testing.assert_numerically_equal(base_model, updated_model, inputs)

        # Validate serialized model
        output_model_proto = ir.serde.serialize_model(updated_model)
        onnx.checker.check_model(output_model_proto, full_check=True)

    def test_transpose_a_matmul_add_to_gemm_incompatible_shapes(self):
        kwargs = {
            "input_shape": ir.Shape((1, 256, 512)),
            "weight_shape": ir.Shape((1, 256, 64)),
            "transA": True,
            "permA": [0, 2, 1],
        }
        return super().check_matmul_add_to_gemm_incompatible_shapes(**kwargs)


class TransBMatMulAddToGemmTest(_MatMulAddToGemmTestBase):
    @parameterized.expand(
        [
            ("initializers", False, False),
            ("inputs", True, True),
        ]
    )
    def test_transpose_b_matmul_add_to_gemm(self, _, weight_as_inputs, bias_as_inputs):
        base_model = self.get_test_model(
            input_shape=ir.Shape((512, 256)),
            weight_shape=ir.Shape((64, 256)),
            weight_as_inputs=weight_as_inputs,
            bias_as_inputs=bias_as_inputs,
            transB=True,
        )
        updated_model = self.clone_model(base_model)
        count = matmul_add_to_gemm.gemm_rule_set().apply_to_model(updated_model)

        # Check MatMul(X, Transpose) + Add are fused into Gemm
        self.assertEqual(count, 1)
        self.assertEqual(len(updated_model.graph), 1)

        # Prepare inputs
        if weight_as_inputs and bias_as_inputs:
            inputs = (
                self.rng.random((512, 256), dtype=np.float32),
                self.rng.random((64, 256), dtype=np.float32),
                self.rng.random((64,), dtype=np.float32),
            )
        else:
            inputs = (self.rng.random((512, 256), dtype=np.float32),)

        # Check inference
        testing.assert_numerically_equal(base_model, updated_model, inputs)

        # Validate serialized model
        output_model_proto = ir.serde.serialize_model(updated_model)
        onnx.checker.check_model(output_model_proto, full_check=True)

    def test_transpose_b_matmul_add_to_gemm_incompatible_shapes(self):
        kwargs = {
            "input_shape": ir.Shape((1, 512, 256)),
            "weight_shape": ir.Shape((1, 64, 256)),
            "transB": True,
            "permB": [0, 2, 1],
        }
        return super().check_matmul_add_to_gemm_incompatible_shapes(**kwargs)


class TransABMatMulAddToGemmTest(_MatMulAddToGemmTestBase):
    @parameterized.expand(
        [
            ("initializers", False, False),
            ("inputs", True, True),
        ]
    )
    def test_transpose_ab_matmul_add_to_gemm(self, _, weight_as_inputs, bias_as_inputs):
        base_model = self.get_test_model(
            input_shape=ir.Shape((256, 512)),
            weight_shape=ir.Shape((64, 256)),
            weight_as_inputs=weight_as_inputs,
            bias_as_inputs=bias_as_inputs,
            transA=True,
            transB=True,
        )
        updated_model = self.clone_model(base_model)
        count = matmul_add_to_gemm.gemm_rule_set().apply_to_model(updated_model)

        # Check MatMul(Transpose, Transpose) + Add are fused into Gemm
        self.assertEqual(count, 1)
        self.assertEqual(len(updated_model.graph), 1)

        # Prepare inputs
        if weight_as_inputs and bias_as_inputs:
            inputs = (
                self.rng.random((256, 512), dtype=np.float32),
                self.rng.random((64, 256), dtype=np.float32),
                self.rng.random((64), dtype=np.float32),
            )
        else:
            inputs = (self.rng.random((256, 512), dtype=np.float32),)

        # Check inference
        testing.assert_numerically_equal(base_model, updated_model, inputs)

        # Validate serialized model
        output_model_proto = ir.serde.serialize_model(updated_model)
        onnx.checker.check_model(output_model_proto, full_check=True)

    def test_transpose_ab_matmul_add_to_gemm_incompatible_shapes(self):
        kwargs = {
            "input_shape": ir.Shape((1, 256, 512)),
            "weight_shape": ir.Shape((1, 64, 256)),
            "transA": True,
            "transB": True,
            "permA": [0, 2, 1],
            "permB": [0, 2, 1],
        }
        return super().check_matmul_add_to_gemm_incompatible_shapes(**kwargs)


if __name__ == "__main__":
    unittest.main()
