# SPDX-License-Identifier: Apache-2.0

import unittest
from .testutils import TestBase
from onnx import FunctionProto
from onnxscript import OnnxFunction, script
from .checker import isomorphic
from onnxscript.onnx import opset15 as op


class TestConverter(TestBase):

    def test_plus_op(self):
        '''Test that + is translated to Add op.'''
        # TODO: pass default opset as parameter to @script
        @script()
        def plus1(x, y):
            return x + y

        @script()
        def plus2(x, y):
            return op.Add(x, y)

        self.assertSame(plus1, plus2)


if __name__ == '__main__':
    unittest.main()
