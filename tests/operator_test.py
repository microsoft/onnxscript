# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import onnx.helper

import onnxscript.testing
from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


class TestConverter(unittest.TestCase):
    def test_plus_op(self):
        """Test that + is translated to Add op."""

        @script(default_opset=op)
        def plus1(x, y):
            return x + y

        @script()
        def plus2(x, y):
            return op.Add(x, y)

        onnxscript.testing.assert_isomorphic_function(plus1, plus2)

    def test_const_promotion(self):
        """Test promotion of constant literals to TensorProto."""

        @script()
        def explicit_plus1(A: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
            one = op.Constant(value=onnx.helper.make_tensor("one", 1, [], [1.0]))
            one_cast = op.CastLike(one, A)
            return op.Add(A, one_cast)

        @script()
        def implicit_plus1(A: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
            return op.Add(A, 1.0)

        onnxscript.testing.assert_isomorphic_function(explicit_plus1, implicit_plus1)

    def test_pow_function(self):
        @script(default_opset=op)
        def pow_function(X, Y):
            return op.Pow(X, Y)
    
        X = op.Constant(value=onnx.helper.make_tensor("X", onnx.TensorProto.FLOAT, [3], [2.0, 3.0, 4.0]))
        Y = op.Constant(value=onnx.helper.make_tensor("Y", onnx.TensorProto.FLOAT, [3], [1.0, 2.0, 3.0]))
        result = pow_function(X, Y)
        self.assertIsNotNone(result)


    def test_optional_with_input(self):
        @script(default_opset=op)
        def optional_with_input(input):
            return op.Optional(input)
    
        input = op.Constant(value=onnx.helper.make_tensor("input", onnx.TensorProto.FLOAT, [1], [1.0]))
        result = optional_with_input(input)
        self.assertIsNotNone(result)


    def test_cast_like_float_to_float(self):
        @script(default_opset=op)
        def cast_like(input, target_type):
            return op.CastLike(input, target_type)
    
        input = op.Constant(value=onnx.helper.make_tensor("input", onnx.TensorProto.FLOAT, [3], [1.0, 2.0, 3.0]))
        target_type = op.Constant(value=onnx.helper.make_tensor("target_type", onnx.TensorProto.FLOAT, [3], [0.0, 0.0, 0.0]))
        result = cast_like(input, target_type)
        self.assertIsNotNone(result)


    def test_bernoulli_with_seed_reproducibility(self):
        @script(default_opset=op)
        def bernoulli_with_seed(input):
            return op.Bernoulli(input, seed=42.0)
    
        input = op.Constant(value=onnx.helper.make_tensor("input", onnx.TensorProto.FLOAT, [3], [0.5, 0.5, 0.5]))
        result = bernoulli_with_seed(input)
        self.assertIsNotNone(result)


    def test_batch_normalization_inference_simple(self):
        @script(default_opset=op)
        def batch_norm_inference(X, scale, B, input_mean, input_var):
            return op.BatchNormalization(X, scale, B, input_mean, input_var)
    
        X = op.Constant(value=onnx.helper.make_tensor("X", onnx.TensorProto.FLOAT, [1, 2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
        scale = op.Constant(value=onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [2], [1.0, 1.0]))
        B = op.Constant(value=onnx.helper.make_tensor("B", onnx.TensorProto.FLOAT, [2], [0.0, 0.0]))
        input_mean = op.Constant(value=onnx.helper.make_tensor("input_mean", onnx.TensorProto.FLOAT, [2], [0.0, 0.0]))
        input_var = op.Constant(value=onnx.helper.make_tensor("input_var", onnx.TensorProto.FLOAT, [2], [1.0, 1.0]))
    
        result = batch_norm_inference(X, scale, B, input_mean, input_var)
        self.assertIsNotNone(result)



if __name__ == "__main__":
    unittest.main()
