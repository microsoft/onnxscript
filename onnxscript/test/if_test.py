
import unittest
from .testutils import TestBase
from onnxscript import script
from onnxscript.onnx import opset15 as op

class IfOpTester(TestBase):

    def test_no_else(self):
        '''Basic test for if-then without else.'''
        # TODO: pass default opset as parameter to @script
        @script()
        def if1(cond, x, y):
            result = op.Identity(y)
            if cond:
                result = op.Identity(x)
            return result

        @script()
        def if2(cond, x, y):
            result = op.Identity(y)
            if cond:
                result = op.Identity(x)
            else:
                result = op.Identity(result)
            return result

        self.assertSame(if1, if2)


if __name__ == '__main__':
    unittest.main()