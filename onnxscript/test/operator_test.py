# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest
from onnxscript.test.testutils import TestBase
from onnxscript import script
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
