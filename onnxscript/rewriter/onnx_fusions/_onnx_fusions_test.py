# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
import onnxscript


class OnnxFusionsTest(unittest.TestCase):
    def test_rms_normalization_fusion(self):
        opset23 = onnxscript.values.Opset("", 23)
        @onnxscript.script()
        def rms_norm_script(embedding):
            two = opset23.Constant(value_float=2.0)
            pow_1 = opset23.Pow(embedding, two)
            mean = opset23.ReduceMean(pow_1, [-1], keepdims=1, noop_with_empty_axes=0)
            epsilon = opset23.Constant(value_float=1e-05)
            add_1 = opset23.Add(mean, epsilon)
            val_244 = opset23.Sqrt(add_1)
            rsqrt = opset23.Reciprocal(val_244)
            mul_3 = opset23.Mul(embedding, rsqrt)
        rms_norm_model = rms_norm_script.to_model_proto()


if __name__ == "__main__":
    unittest.main()