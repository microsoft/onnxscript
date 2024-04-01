# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import onnxscript.testing
from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.tests.common import testutils


class IfOpTest(testutils.TestBase):
    def test_no_else(self):
        """Basic test for if-then without else."""

        # TODO: pass default opset as parameter to @script
        @script()
        def if1(cond, x, y):
            result = op.Identity(y)
            if cond:
                result = op.Identity(x)
            return result

        # if1 should be treated as equivalent to the code if2 below
        @script()
        def if2(cond, x, y):
            result = op.Identity(y)
            if cond:
                result = op.Identity(x)
            else:
                result = op.Identity(result)
            return result

        # if2 should be treated as equivalent to the code if3 below (SSA renaming)
        @script()
        def if3(cond, x, y):
            result1 = op.Identity(y)
            if cond:
                result2 = op.Identity(x)
            else:
                result2 = op.Identity(result1)
            return result2

        onnxscript.testing.assert_isomorphic_function(if1, if2)
        onnxscript.testing.assert_isomorphic_function(if2, if3)


if __name__ == "__main__":
    unittest.main()
