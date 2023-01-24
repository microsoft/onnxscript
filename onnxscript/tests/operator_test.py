# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import onnx.helper

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.tests.common import testutils


class TestConverter(testutils.TestBase):
    def test_plus_op(self):
        """Test that + is translated to Add op."""

        @script(default_opset=op)
        def plus1(x, y):
            return x + y

        @script()
        def plus2(x, y):
            return op.Add(x, y)

        self.assertSame(plus1, plus2)

    def test_const_promotion(self):
        """Test promotion of constant literals to TensorProto."""

        @script()
        def explicit_plus1(A: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
            one = op.Constant(value=onnx.helper.make_tensor("one", 1, [], [1.0]))
            one_cast = op.CastLike(one, A)
            return op.Add(A, one_cast)

        @script()
        def implicit_plus1(A: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
            return op.Add(A, 1.0)

        self.assertSame(explicit_plus1, implicit_plus1)

    def test_bool_ops_binary(self):
        @script(default_opset=op)
        def py_two_or(a: BOOL, b: BOOL) -> BOOL:
            return a or b

        @script()
        def onnx_two_or(a: BOOL, b: BOOL) -> BOOL:
            return op.Or(a, b)

        self.assertSame(py_two_or, onnx_two_or)

        @script(default_opset=op)
        def py_two_and(a: BOOL, b: BOOL) -> BOOL:
            return a and b

        @script()
        def onnx_two_and(a: BOOL, b: BOOL) -> BOOL:
            return op.And(a, b)

        self.assertSame(py_two_and, onnx_two_and)

    def test_bool_ops_three(self):
        @script(default_opset=op)
        def py_three_or(a: BOOL, b: BOOL, c: BOOL) -> BOOL:
            return a or b or c

        @script()
        def onnx_three_or(a: BOOL, b: BOOL, c: BOOL) -> BOOL:
            return op.Or(op.Or(a, b), c)

        self.assertSame(py_three_or, onnx_three_or)

    def test_bool_ops_four(self):
        @script(default_opset=op)
        def py_four_and(a: BOOL, b: BOOL, c: BOOL, d: BOOL) -> BOOL:
            return a and b and c and d

        @script()
        def onnx_four_and(a: BOOL, b: BOOL, c: BOOL, d: BOOL) -> BOOL:
            return op.And(op.And(op.And(a, b), c), d)

        self.assertSame(py_four_and, onnx_four_and)


if __name__ == "__main__":
    unittest.main()
