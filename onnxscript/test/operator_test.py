# SPDX-License-Identifier: Apache-2.0

import unittest
from onnx import FunctionProto
from onnxscript import OnnxFunction, script
from .checker import isomorphic
from onnxscript.onnx import opset15 as op


def function_proto(f):
    if isinstance(f, FunctionProto):
        return f
    if isinstance(f, OnnxFunction):
        return f.to_function_proto()
    raise TypeError(f"Cannot convert {type(f)} to FunctionProto")


class TestConverter(unittest.TestCase):
    def assertSame(self, fn1, fn2):
        self.assertTrue(isomorphic(function_proto(fn1), function_proto(fn1)))

    def test_plus_op(self):
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
