import numpy as np
import unittest
import test_lib
from onnxscript.test.models import if_statement

class TestOnnxIf(unittest.TestCase):
    def test_if(self):
        n = 8
        np.random.seed(0)
        a = np.random.rand(n).astype('float32').T
        b = np.random.rand(n).astype('float32').T

        # FIXME(liqunfu): expected are from ort evaluation. needs numpy oxs to provide expected instead.
        expected = np.array([
            0.5488135, 0.71518934, 0.60276335, 0.5448832,
            0.4236548, 0.6458941, 0.4375872, 0.891773], dtype=np.float32)

        cases = [test_lib.FunctionTestParams(if_statement.maxsum, [a, b], [expected])]
        test = test_lib.OnnxScriptTestCase()
        for case in cases:
            # test._run_converter_test(case)
            test._run_eager_test(case)

if __name__ == '__main__':
    unittest.main()