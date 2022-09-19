# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

from onnx import helper

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT
from onnxscript.test.testutils import TestBase


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

    def test_const_promotion(self):
        '''Test promotion of constant literals to TensorProto.'''

        @script()
        def explicit_plus1(A: FLOAT["N"]) -> FLOAT["N"]:   # noqa: F821
            one = op.Constant(value=helper.make_tensor("one", 1, [], [1.0]))
            return op.Add(A, one)

        @script()
        def implicit_plus1(A: FLOAT["N"]) -> FLOAT["N"]:   # noqa: F821
            return op.Add(A, 1.0)

        self.assertSame(explicit_plus1, implicit_plus1)


if __name__ == '__main__':
    unittest.main()
