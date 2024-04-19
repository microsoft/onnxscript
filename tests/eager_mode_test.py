# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import parameterized

import onnxscript
import onnxscript.evaluator
import onnxscript.tensor
from onnxscript import opset17 as op
from onnxscript import script


@parameterized.parameterized_class(
    (
        "name",
        "evaluator",
    ),
    [
        (
            "reference_runtime",
            onnxscript.evaluator.OnnxReferenceRuntimeEvaluator(),
        ),
        (
            "onnxruntime",
            onnxscript.evaluator.ORTEvaluator(),
        ),
    ],
)
class EagerModeTest(unittest.TestCase):
    evaluator: onnxscript.evaluator.Evaluator

    def setUp(self):
        self.default_evaluator = onnxscript.evaluator.default()
        onnxscript.evaluator.set_default(self.evaluator)

    def tearDown(self):
        onnxscript.evaluator.set_default(self.default_evaluator)

    def test_sequence_input(self):
        @script()
        def Concat(seq):
            return op.ConcatFromSequence(seq, axis=0)

        np_array = np.array([1, 2, 3], dtype=np.float32)
        output1 = Concat([np_array, np_array])
        self.assertIsInstance(output1, np.ndarray)

        os_tensor = onnxscript.tensor.Tensor(np_array)
        output2 = Concat([os_tensor, os_tensor])
        self.assertIsInstance(output2, onnxscript.tensor.Tensor)


@script()
def add_with_alpha(this, other, alpha: float = 1.0):
    alpha = op.CastLike(alpha, other)
    other = op.Mul(other, alpha)
    return op.Add(this, other)


@parameterized.parameterized_class(
    (
        "name",
        "evaluator",
    ),
    [
        (
            "reference_runtime",
            onnxscript.evaluator.OnnxReferenceRuntimeEvaluator(),
        ),
        (
            "onnxruntime",
            onnxscript.evaluator.ORTEvaluator(),
        ),
    ],
)
class TestEagerModeArguments(unittest.TestCase):
    evaluator: onnxscript.evaluator.Evaluator

    def setUp(self):
        self.default_evaluator = onnxscript.evaluator.default()
        onnxscript.evaluator.set_default(self.evaluator)

    def tearDown(self):
        onnxscript.evaluator.set_default(self.default_evaluator)

    def test_op_some_input_by_kwargs(self):
        self.assertEqual(op.Add(1, B=2), 3)

    def test_op_all_input_by_kwargs(self):
        self.assertEqual(op.Add(A=1, B=2), 3)

    def test_op_attribute_by_positional_args(self):
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        axes = np.array([0], dtype=np.int64)
        self.assertEqual(op.ReduceSum(data, axes, keepdims=True), 21)

    def test_op_input_and_attribute_by_kwargs_out_of_order(self):
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        axes = np.array([0], dtype=np.int64)
        self.assertEqual(op.ReduceSum(keepdims=True, axes=axes, data=data), 21)

    def test_function_some_input_by_kwargs(self):
        self.assertEqual(add_with_alpha(1.0, other=2.0), 3.0)

    def test_function_all_input_by_kwargs(self):
        self.assertEqual(add_with_alpha(this=1.0, other=2.0), 3.0)

    def test_function_attribute_by_positional_args(self):
        self.assertEqual(add_with_alpha(1.0, 2.0, 3.0), 7.0)

    def test_function_input_and_attribute_by_kwargs_out_of_order(self):
        self.assertEqual(add_with_alpha(alpha=3.0, other=2.0, this=1.0), 7.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
