import unittest
from onnxscript.test.functions.onnx_script_test_case import OnnxScriptTestCase
from onnxscript.test.models import onnxfns1


class TestOnnxFns(OnnxScriptTestCase):
    def setUp(self):
        super().setUp()
        self.rtol = 1e-05

    def test_onnxfns(self):
        self.run_onnx_test(onnxfns1.Relu)


if __name__ == '__main__':
    unittest.main()
