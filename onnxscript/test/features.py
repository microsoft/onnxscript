# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest
from converter_test import TestConverter
from onnxscript import script
from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT

class TestConverter(TestConverter):
    def check(self, fun):
        self.validate_save(fun)

    def test_type_annotation(self):
        '''Test type annotations.'''

        @script()
        def static_shape(A: FLOAT[100], B: FLOAT[100]) -> FLOAT[100]:
            return op.Add (A, B)
        
        self.check(static_shape)

        @script()
        def symbolic_shape(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
            return op.Add (A, B)

        self.check(symbolic_shape)

        @script()
        def tensor_scalar(A: FLOAT["N"], B: FLOAT) -> FLOAT["N"]:
            return op.Add (A, B)

        self.check(tensor_scalar)

        @script()
        def unknown_rank(A: FLOAT[...], B: FLOAT[...]) -> FLOAT[...]:
            return op.Add (A, B)

        self.check(unknown_rank)

        @script()
        def invalid(A: FLOAT[100, 50], B: FLOAT[50, 100]) -> FLOAT[100, 50]:
            return op.Add (A, B)

        self.check(invalid)


if __name__ == '__main__':
    unittest.main()