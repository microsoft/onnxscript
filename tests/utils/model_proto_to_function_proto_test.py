# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnxruntime as ort

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT
from onnxscript.utils.model_proto_to_function_proto import (
    convert_model_proto_to_function_proto,
)
from onnxscript.values import Opset


class TestModelProtoToFunctionProto(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh custom opset for each test
        self.local = Opset("local", 1)

        # Define test functions
        @script(self.local, default_opset=op)
        def diff_square(x, y):
            diff = x - y
            return diff * diff

        @script(self.local)
        def sum_func(z):
            return op.ReduceSum(z, keepdims=1)

        @script()
        def l2norm_with_functions(x: FLOAT["N"], y: FLOAT["N"]) -> FLOAT[1]:  # noqa: F821
            return op.Sqrt(sum_func(diff_square(x, y)))

        self.diff_square = diff_square
        self.sum_func = sum_func
        self.l2norm_with_functions = l2norm_with_functions

    def test_multiple_functions_in_model_proto(self):
        """Test that multiple functions can be included in a single model proto."""
        # Add sum function to opset
        sum_model = self.sum_func.to_model_proto()
        sum_function_proto = convert_model_proto_to_function_proto(
            sum_model, "local", "sum_func"
        )

        model = self.l2norm_with_functions.to_model_proto(
            functions=[sum_function_proto, self.diff_square]
        )

        # Test execution
        session = ort.InferenceSession(model.SerializeToString())
        result = session.run(
            None,
            {
                "x": np.array([1.0, 2.0, 3.0]).astype(np.float32),
                "y": np.array([4.0, 5.0, 6.0]).astype(np.float32),
            },
        )

        # Verify result
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(np.sqrt(27.0), result[0][0], places=5)  # L2 norm of [3, 3, 3]
