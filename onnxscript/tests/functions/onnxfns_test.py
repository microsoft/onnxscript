# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

from onnxscript.tests.common import onnx_script_test_case
from onnxscript.tests.models import onnxfns1


class TestOnnxFns(onnx_script_test_case.OnnxScriptTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rtol = 1e-05

    def test_onnxfns_relu(self):
        self.run_onnx_test(onnxfns1.Relu)

    def test_onnxfns_selu(self):
        default_alpha = 1.67326319217681884765625
        default_gamma = 1.05070102214813232421875

        self.run_onnx_test(onnxfns1.Selu, alpha=default_alpha, gamma=default_gamma)

    def test_onnxfns_elu(self):
        default_alpha = 1.0
        self.run_onnx_test(onnxfns1.Elu, alpha=default_alpha)

    def test_onnxfns_thresholded_relu(self):
        default_alpha = 1.0
        self.run_onnx_test(onnxfns1.ThresholdedRelu, alpha=default_alpha)

    def test_onnxfns_leaky_relu(self):
        default_alpha = 0.01
        self.run_onnx_test(onnxfns1.LeakyRelu, alpha=default_alpha)

    def test_onnxfns_prelu(self):
        self.run_onnx_test(onnxfns1.PRelu)

    def test_onnxfns_hard_sigmoid(self):
        default_alpha = 0.2
        default_beta = 0.5
        self.run_onnx_test(onnxfns1.HardSigmoid, alpha=default_alpha, beta=default_beta)

    def test_onnxfns_hard_shrink(self):
        default_bias = 0.0
        default_lambd = 0.5
        self.run_onnx_test(onnxfns1.Shrink, bias=default_bias, lambd=default_lambd)

    def test_onnxfns_hard_softplus(self):
        self.run_onnx_test(onnxfns1.Softplus)

    def test_onnxfns_hard_softsign(self):
        self.run_onnx_test(onnxfns1.Softsign)

    def test_onnxfns_hard_clip(self):
        self.run_onnx_test(
            onnxfns1.Clip,
            skip_eager_test=True,
            skip_test_names=[
                "test_clip_default_int8_min",
                "test_clip_default_int8_max",
                "test_clip_default_int8_inbounds",
            ],
        )


if __name__ == "__main__":
    unittest.main()
