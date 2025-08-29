# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import onnx_ir as ir

import onnxscript
import onnxscript.optimizer
import onnxscript.rewriter.testing
from onnxscript import FLOAT, OnnxFunction, script
from onnxscript import opset18 as op
from onnxscript.rewriter.rules.fusion._layer_norm import fuse_layer_normalization


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
def _test_layer_norm_with_bias(
    x: FLOAT[2, 4, 8], scale: FLOAT[8], bias: FLOAT[8]
) -> FLOAT[2, 4, 8]:
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
    def _check(self, test_script: OnnxFunction):
        """Helper method to run a fusion test scenario."""
        model_proto = test_script.to_model_proto()
        # Create test inputs
        input_data = onnxscript.rewriter.testing.generate_random_inputs(model_proto)

        model = ir.serde.deserialize_model(model_proto)
        fuse_layer_normalization(model)

        onnxscript.optimizer.remove_unused_nodes(model)

        # Check that a LayerNormalization node was created
        self.assertEqual(["LayerNormalization"], [n.op_type for n in model.graph])

        fused_model_proto = ir.serde.serialize_model(model)

        onnxscript.rewriter.testing.assert_numerically_equal(
            model_proto, fused_model_proto, input_data
        )

    def test_layer_norm_fusion_without_bias(self):
        """Test LayerNorm fusion without bias."""
        self._check(_test_layer_norm_without_bias)

    def test_layer_norm_fusion_with_bias(self):
        """Test LayerNorm fusion with bias."""
        self._check(_test_layer_norm_with_bias)


if __name__ == "__main__":
    unittest.main()
