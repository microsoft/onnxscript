import unittest

import onnx
import pytest

from onnxscript.tests.common import onnx_script_test_case
from onnxscript.tests.models import onnxfns1A


class TestOnnxFns(onnx_script_test_case.OnnxScriptTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rtol = 1e-05

    def test_onnxfns_relu(self):
        self.run_onnx_test(onnxfns1A.Relu)

    @unittest.skipIf(
        not hasattr(onnx.FunctionProto, "attribute_proto"),
        reason="ONNX 1.13 does not support default values",
    )
    def test_onnxfns_selu(self):
        self.run_onnx_test(onnxfns1A.Selu)

    @unittest.skipIf(
        not hasattr(onnx.FunctionProto, "attribute_proto"),
        reason="current onnx does not support default values",
    )
    def test_onnxfns_elu(self):
        self.run_onnx_test(onnxfns1A.Elu)

    def test_onnxfns_elu05(self):
        self.run_onnx_test(onnxfns1A.Elu05)

    @unittest.skipIf(
        not hasattr(onnx.FunctionProto, "attribute_proto"),
        reason="ONNX 1.13 does not support default values",
    )
    def test_onnxfns_thresholded_relu(self):
        self.run_onnx_test(onnxfns1A.ThresholdedRelu)

    @unittest.skipIf(
        not hasattr(onnx.FunctionProto, "attribute_proto"),
        reason="ONNX 1.13 does not support default values",
    )
    def test_onnxfns_leaky_relu(self):
        self.run_onnx_test(onnxfns1A.LeakyRelu)

    def test_onnxfns_prelu(self):
        self.run_onnx_test(onnxfns1A.PRelu)

    @unittest.skipIf(
        not hasattr(onnx.FunctionProto, "attribute_proto"),
        reason="current onnx does not support default values",
    )
    def test_onnxfns_hard_sigmoid(self):
        self.run_onnx_test(onnxfns1A.HardSigmoid)

    @unittest.skipIf(
        not hasattr(onnx.FunctionProto, "attribute_proto"),
        reason="current onnx does not support default values",
    )
    def test_onnxfns_shrink(self):
        self.run_onnx_test(onnxfns1A.Shrink)

    def test_onnxfns_hard_softplus(self):
        self.run_onnx_test(onnxfns1A.Softplus)

    def test_onnxfns_hard_softsign(self):
        self.run_onnx_test(onnxfns1A.Softsign)

    @pytest.mark.xfail(
        reason="Clip has optional input min and max. Need to find out how to pass default min and max to the test case executor."
    )
    def test_onnxfns_hard_clip(self):
        self.run_onnx_test(onnxfns1A.Clip)


if __name__ == "__main__":
    unittest.main(verbosity=2)
