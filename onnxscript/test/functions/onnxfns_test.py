# import numpy as np
from ast import Assert
import unittest
import test_lib
from onnxscript.test.models import onnxfns

# eager_test_only = False
# converter_test_only = False
# relu_onnx_test_case = test_lib.create_test_case(onnxfns.Relu, None, eager_test_only, converter_test_only)

# selu_onnx_test_case = test_lib.create_test_case(
#     onnxfns.Selu,
#     [
#         {'alpha': 2.0, 'gamma': 3.0},
#         {'alpha': 2.0, 'gamma': 3.0},
#         {'alpha': 2.0, 'gamma': 3.0},
#     ],
#     eager_test_only, converter_test_only)

# MaxsumTestCase = test_lib.create_test_case(onnxfns.Elu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.ThresholdedRelu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.LeakyRelu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.PRelu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.HardSigmoid, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Shrink, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Softplus, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Softsign, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Clip, cases)

class TestOnnxFns(unittest.TestCase):
    def test_onnxfns(self):
        test_lib.run_onnx_test(onnxfns.Relu, None)

if __name__ == '__main__':
    unittest.main()