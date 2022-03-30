import unittest
from onnxscript.test.functions.onnx_script_test_case import OnnxScriptTestCase
from onnxscript.test.models import onnxfns1
from onnxscript.eager_mode_evaluator import EagerModeEvaluator
onnxfns1.op = EagerModeEvaluator()


class TestOnnxFns(OnnxScriptTestCase):
    def test_onnxfns(self):
        self.run_onnx_test(onnxfns1.Relu)


if __name__ == '__main__':
    unittest.main()
