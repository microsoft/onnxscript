import unittest
from onnx_script_test_case import OnnxScriptTestCase
from onnxscript.test.models import onnxfns


class TestOnnxFns(OnnxScriptTestCase):
    def test_onnxfns(self):
        self.run_onnx_test(onnxfns.Relu, None)


if __name__ == '__main__':
    unittest.main()
