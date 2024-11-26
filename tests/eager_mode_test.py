# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

    def test_adapt_to_user_mode_mixed_tuple(self):
        tensor_tuple = (onnxscript.tensor.Tensor(np.array([1, 2, 3])), np.array([4, 5, 6]))
        result = onnxscript.evaluator._adapt_to_user_mode(tensor_tuple)
        self.assertIsInstance(result, tuple)
        for item in result:
            self.assertIsInstance(item, np.ndarray)


    def test_base_evaluator_adapt_attributes_with_callable(self):
        class DummySchema:
            attributes = {'attr': None}
    
        def dummy_function():
            pass
    
        class DummyEvaluator(onnxscript.evaluator.BaseEvaluator):
            def _eval(self, schema, inputs, attributes, closure):
                pass
    
        evaluator = DummyEvaluator()
        with self.assertRaises(TypeError):
            evaluator.adapt_attributes(DummySchema(), {'attr': dummy_function})


    def test_adapt_to_user_mode_tuple_of_tensors(self):
        tensor_tuple = (onnxscript.tensor.Tensor(np.array([1, 2, 3])), onnxscript.tensor.Tensor(np.array([4, 5, 6])))
        result = onnxscript.evaluator._adapt_to_user_mode(tensor_tuple)
        self.assertIsInstance(result, tuple)
        for item in result:
            self.assertIsInstance(item, np.ndarray)


    def test_adapt_to_eager_mode_nested_list_of_integers(self):
        nested_list = [[1, 2], [3, 4]]
        result, has_array = onnxscript.evaluator._adapt_to_eager_mode(nested_list)
        self.assertFalse(has_array)
        for sublist in result:
            for item in sublist:
                self.assertIsInstance(item, onnxscript.tensor.Tensor)


    def test_adapt_to_user_mode_list_of_tensors(self):
        tensor_list = [onnxscript.tensor.Tensor(np.array([1, 2, 3])), onnxscript.tensor.Tensor(np.array([4, 5, 6]))]
        result = onnxscript.evaluator._adapt_to_user_mode(tensor_list)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, np.ndarray)


    def test_unwrap_tensors_in_kwargs(self):
        kwargs = {'a': onnxscript.tensor.Tensor(np.array([1, 2])), 'b': 3}
        result = onnxscript.evaluator._unwrap_tensors_in_kwargs(kwargs)
        self.assertIsInstance(result['a'], np.ndarray)
        self.assertEqual(result['b'], 3)



if __name__ == "__main__":
    unittest.main(verbosity=2)
