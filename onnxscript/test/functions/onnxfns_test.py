import numpy as np
import unittest
import test_lib
from onnxscript.test.models import onnxfns
import onnx.backend.test.case.node as node_test

n = 4096
a = np.random.rand(n).astype('float32').T
b = np.random.rand(n).astype('float32').T

def create_test_case(function):
    io = []
    for case in node_test.collect_testcases_by_operator(function.__name__):
        io.extend([test_lib.Params(ds[0], ds[1]) for ds in case.data_sets])
    return test_lib.create_test_case(function, io)

relu_onnx_test_case = create_test_case(onnxfns.Relu)

# MaxsumTestCase = test_lib.create_test_case(onnxfns.Selu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Elu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.ThresholdedRelu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.LeakyRelu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.PRelu, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.HardSigmoid, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Shrink, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Softplus, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Softsign, cases)
# MaxsumTestCase = test_lib.create_test_case(onnxfns.Clip, cases)

if __name__ == '__main__':
    unittest.main()