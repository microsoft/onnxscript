import numpy as np
import unittest
import test_lib
import gemmgelu

m = 2048
k = 16
n = 4096
a = np.random.rand(k, m).astype('float32').T
w = np.random.rand(n, k).astype('float32').T
b = np.random.rand(n,).astype('float32').T
cases = (test_lib.InputOutput({'A': a, 'W': w, 'Bias': b}, []),)
    

GemmGeluTestCase = test_lib.create_test_case(gemmgelu.gemmgelu, cases)

if __name__ == '__main__':
    unittest.main()