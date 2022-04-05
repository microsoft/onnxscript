import unittest
from onnxscript.test.functions.onnx_script_test_case import OnnxScriptTestCase
from onnxscript.utils import assign_eager_mode_evaluator_to_module
from onnxscript.test.models import onnxfns1

assign_eager_mode_evaluator_to_module(onnxfns1, "", 15)


class TestOnnxFns(OnnxScriptTestCase):
    def setUp(self):
        super().setUp()
        self.rtol = 1e-05

    # def test_onnxfns_relu(self):
    #     self.run_onnx_test(onnxfns1.Relu)

    # FAIL : Fatal error: Selu is not a registered function/op
    # def test_onnxfns_selu(self):
    #     self.run_onnx_test(onnxfns1.Selu)

    # FAIL : Fatal error: Elu is not a registered function/op
    # def test_onnxfns_elu(self):
    #     self.run_onnx_test(onnxfns1.Elu)

    # FAIL : Fatal error: ThresholdedRelu is not a registered function/op
    # def test_onnxfns_thresholded_relu(self):
    #     self.run_onnx_test(onnxfns1.ThresholdedRelu)

    def test_onnxfns_relu(self):
        self.run_onnx_test(onnxfns1.Relu)

    # #  onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Type Error: Type (tensor(float)) of output arg (output_0) of node () does not match expected type (tensor(bool)).
    # def test_onnxfns_prelu(self):
    #     self.run_onnx_test(onnxfns1.PRelu)

    # FAIL : Fatal error: HardSigmoid is not a registered function/op
    # def test_onnxfns_hard_sigmoid(self):
    #     self.run_onnx_test(onnxfns1.HardSigmoid)

    # Fatal error: Shrink is not a registered function/op
    # def test_onnxfns_hard_shrink(self):
    #     self.run_onnx_test(onnxfns1.Shrink)

    # np.testing.assert_allclose failed
    # def test_onnxfns_hard_softplus(self):
    #     self.run_onnx_test(onnxfns1.Softplus)

    # np.testing.assert_allclose failed
    # def test_onnxfns_hard_softsign(self):
    #     self.run_onnx_test(onnxfns1.Softsign)

    # FAIL : Type Error: Type (tensor(float)) of output arg (output_0) of node () does not match expected type (tensor(bool)).
    # def test_onnxfns_hard_clip(self):
    #     self.run_onnx_test(onnxfns1.Clip)

if __name__ == '__main__':
    unittest.main()
