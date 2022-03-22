import numpy as np
import unittest
import test_lib
import gemmgelu

class TestGemmGelu(unittest.TestCase):
    def test_gemmgelu(self):
        np.random.seed(0)
        m = 2
        k = 4
        n = 8
        a = np.random.rand(k, m).astype('float32').T
        w = np.random.rand(n, k).astype('float32').T
        b = np.random.rand(n,).astype('float32').T

        # FIXME(liqunfu): expected are from ort evaluation. needs numpy oxs to provide expected instead.
        expected = np.array([
            [1.6088762, 1.2583977, 1.868434, 1.530172, 1.5025945, 1.5770031, 0.93028706, 1.4389044],
            [2.2128997, 1.3670988, 2.4269097, 2.1586964, 1.9926084, 2.0960782, 1.2971772, 2.0846245]], dtype=np.float32)

        cases = [test_lib.FunctionTestParams(gemmgelu.gemmgelu, [a, w, b], [expected])]
        test = test_lib.OnnxScriptTestCase()
        for case in cases:
            test._run_converter_test(case)
            test._run_eager_test(case)

if __name__ == '__main__':
    unittest.main()