import unittest
from onnxscript.test.functions.onnx_script_test_case import OnnxScriptTestCase
from onnxscript.utils import assign_eager_mode_evaluator_to_module
from onnxscript.test.models import onnxfns1

assign_eager_mode_evaluator_to_module(onnxfns1, "", 15)


class TestOnnxFns(OnnxScriptTestCase):
    def setUp(self):
        super().setUp()
        self.rtol = 1e-05

    def test_onnxfns_relu(self):
        self.run_onnx_test(onnxfns1.Relu)

    def test_onnxfns_selu(self):
        default_alpha = 1.67326319217681884765625
        default_gamma = 1.05070102214813232421875

        self.run_onnx_test(
            onnxfns1.Selu,
            alpha=default_alpha,
            gamma=default_gamma)

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
        self.run_onnx_test(
            onnxfns1.HardSigmoid,
            alpha=default_alpha,
            beta=default_beta)

    def test_onnxfns_hard_shrink(self):
        default_bias = 0.0
        default_lambd = 0.5
        self.run_onnx_test(
            onnxfns1.Shrink,
            bias=default_bias,
            lambd=default_lambd)

    def test_onnxfns_hard_softplus(self):
        self.run_onnx_test(onnxfns1.Softplus)

    def test_onnxfns_hard_softsign(self):
        self.run_onnx_test(onnxfns1.Softsign)

    # TODO: Clip has optional input min and max.
    # need to find out how to pass default min and max
    # to the test case executor.
    # def test_onnxfns_hard_clip(self):
    #     self.run_onnx_test(onnxfns1.Clip)


if __name__ == '__main__':
    unittest.main()
