# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import onnx_ir as ir
import parameterized

import onnxscript
import onnxscript.rewriter.ort_fusions._test_utils as test_utils
from onnxscript import FLOAT, OnnxFunction, script
from onnxscript import opset17 as op
from onnxscript.optimizer import optimize, remove_unused_nodes
from onnxscript.rewriter.layer_normalization import fuse_layer_normalization
import onnxscript.rewriter.testing


@script()
def _test_layer_norm_without_bias(x: FLOAT[2, 4, 8], scale: FLOAT[8]) -> FLOAT[2, 4, 8]:
    """LayerNorm pattern without bias."""
    # Compute mean: Mean = ReduceMean(X, axes=normalized_axes)
    mean = op.ReduceMean(x, [-1], keepdims=1)
    
    # Compute deviation: D = Sub(X, Mean)
    deviation = op.Sub(x, mean)
    
    # Compute squared deviation: DD = Mul(D, D)
    deviation_squared = op.Mul(deviation, deviation)
    
    # Compute variance: Var = ReduceMean(DD, axes=normalized_axes)
    variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)
    
    # Add epsilon: VarEps = Add(Var, epsilon)
    epsilon = op.Constant(value_float=1e-5)
    variance_plus_epsilon = op.Add(variance, epsilon)
    
    # Compute standard deviation: StdDev = Sqrt(VarEps)
    std_dev = op.Sqrt(variance_plus_epsilon)
    
    # Compute reciprocal: InvStdDev = Reciprocal(StdDev)
    inv_std_dev = op.Reciprocal(std_dev)
    
    # Normalize: Normalized = Mul(D, InvStdDev)
    normalized = op.Mul(deviation, inv_std_dev)
    
    # Scale: NormalizedScaled = Mul(Normalized, Scale)
    normalized_scaled = op.Mul(normalized, scale)
    
    return normalized_scaled


@script()
def _test_layer_norm_with_bias(x: FLOAT[2, 4, 8], scale: FLOAT[8], bias: FLOAT[8]) -> FLOAT[2, 4, 8]:
    """LayerNorm pattern with bias."""
    # Compute mean: Mean = ReduceMean(X, axes=normalized_axes)
    mean = op.ReduceMean(x, [-1], keepdims=1)
    
    # Compute deviation: D = Sub(X, Mean)
    deviation = op.Sub(x, mean)
    
    # Compute squared deviation: DD = Mul(D, D)
    deviation_squared = op.Mul(deviation, deviation)
    
    # Compute variance: Var = ReduceMean(DD, axes=normalized_axes)
    variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)
    
    # Add epsilon: VarEps = Add(Var, epsilon)
    epsilon = op.Constant(value_float=1e-5)
    variance_plus_epsilon = op.Add(variance, epsilon)
    
    # Compute standard deviation: StdDev = Sqrt(VarEps)
    std_dev = op.Sqrt(variance_plus_epsilon)
    
    # Compute reciprocal: InvStdDev = Reciprocal(StdDev)
    inv_std_dev = op.Reciprocal(std_dev)
    
    # Normalize: Normalized = Mul(D, InvStdDev)
    normalized = op.Mul(deviation, inv_std_dev)
    
    # Scale: NormalizedScaled = Mul(Normalized, Scale)
    normalized_scaled = op.Mul(normalized, scale)
    
    # Add bias: Y = Add(NormalizedScaled, B)
    result = op.Add(normalized_scaled, bias)
    
    return result


class LayerNormFusionTest(unittest.TestCase):
    def _check(
        self,
        test_data_constructor: OnnxFunction,
        expected_graph_len: int,
        expected_op_type: str,
        has_bias: bool = False,
    ):
        """Helper method to run a fusion test scenario."""
        model_proto = test_data_constructor.to_model_proto()
        # Create test inputs
        input_data = onnxscript.rewriter.testing.generate_random_inputs(model)

        model = ir.serde.deserialize_model(model_proto)
        fuse_layer_normalization(model)

        # Run original model
        original_output = test_utils.ort_run("Original", model, input_data)

        # Apply fusion
        fuse_layer_normalization(model)
        remove_unused_nodes(model)

        # Verify fusion occurred
        self.assertEqual(len(model.graph), expected_graph_len)
        self.assertEqual(model.graph.node(0).op_type, expected_op_type)

        # Run optimized model and verify outputs match
        optimized_output = test_utils.ort_run("Optimized", model, input_data)
        test_utils.assert_allclose(original_output, optimized_output, rtol=1e-4, atol=1e-4)

    def test_layer_norm_fusion_without_bias(self):
        """Test LayerNorm fusion without bias."""
        self._check(_test_layer_norm_without_bias, 1, "LayerNormalization", has_bias=False)

    def test_layer_norm_fusion_with_bias(self):
        """Test LayerNorm fusion with bias."""
        self._check(_test_layer_norm_with_bias, 1, "LayerNormalization", has_bias=True)


if __name__ == "__main__":
    unittest.main()