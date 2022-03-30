import unittest
from onnxscript.test.functions.onnx_script_test_case import OnnxScriptTestCase
from onnxscript.utils import assign_eager_mode_evaluator_to_module
from onnxscript.test.models import onnxfns1

assign_eager_mode_evaluator_to_module(onnxfns1, "", 15)


class TestOnnxFns(OnnxScriptTestCase):
    def test_onnxfns(self):
        self.run_onnx_test(onnxfns1.Relu)


if __name__ == '__main__':
    unittest.main()
