import numpy as np
import unittest
import test_lib
from onnxscript.test.models import if_statement

n = 4096
a = np.random.rand(n).astype('float32').T
b = np.random.rand(n).astype('float32').T
cases = (test_lib.InputOutput({'A': a, 'B': b,}, []),)
    
MaxsumTestCase = test_lib.create_test_case(if_statement.maxsum, cases)
MaxsumTestCase2 = test_lib.create_test_case(if_statement.maxsum2, cases)
MaxsumTestCase3 = test_lib.create_test_case(if_statement.maxsum3, cases)

if __name__ == '__main__':
    unittest.main()