# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np

from onnxscript import evaluator, graph, script
from onnxscript.onnx_opset import opset17 as op
from onnxscript.onnx_types import FLOAT

from onnxscript import tensor
import onnx

class EvaluatorTest(unittest.TestCase):
    def test_evaluator(self):
        @script()
        def seq_map(x: FLOAT["N"]):  # noqa: F821
            seq1 = op.SequenceConstruct(x, x + 1, x + 2)

            @graph()
            def square(y: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
                return op.Mul(y, y)

            seq2 = op.SequenceMap(seq1, body=square)
            return seq2

        x = np.array([0.0, 1.0], dtype=np.float32)
        expected = [t * t for t in (x, x + 1, x + 2)]

        # Test using (current) default evaluator
        output = seq_map(x)
        np.testing.assert_equal(output, expected)

        # Test using ort-mixed-evaluator
        output = seq_map[evaluator.ort_mixed_evaluator](x)
        np.testing.assert_equal(output, expected)

        # Test using ort-evaluator
        output = seq_map[evaluator.ort_evaluator](x)
        np.testing.assert_equal(output, expected)


class ORTEvaluatorTest(unittest.TestCase):
    def test_it_ignores_unknown_function_kwargs_when_option_set_to_true(self):
        @script()
        def test_function(x, y: float = 1.0):
            return op.Add(x, y)

        x = np.array(0.0, dtype=np.float32)
        expected = np.array(1.0, dtype=np.float32)
        with evaluator.default_as(evaluator.ORTEvaluator(ignore_unknown_function_kwargs=True)):
            output = test_function(x, unknown=42)  # pylint: disable=unexpected-keyword-arg

        np.testing.assert_equal(output, expected)

    def test_it_raise_on_unknown_function_kwargs_by_default(self):
        @script()
        def test_function(x, y: float = 1.0):
            return op.Add(x, y)

        x = np.array(0.0, dtype=np.float32)
        with evaluator.default_as(evaluator.ORTEvaluator()):  # noqa: SIM117
            with self.assertRaises(TypeError):
                _ = test_function(x, unknown=42)  # pylint: disable=unexpected-keyword-arg



    def test_adapt_to_eager_mode_list_of_numpy_arrays(self):
        inputs = [np.array([1, 2]), np.array([3, 4])]
        expected = [tensor.Tensor(np.array([1, 2])), tensor.Tensor(np.array([3, 4]))]
        result, has_array = evaluator._adapt_to_eager_mode(inputs)
        for res, exp in zip(result, expected):
            np.testing.assert_array_equal(res.value, exp.value)
        self.assertTrue(has_array)


    def test_compute_num_outputs_scan(self):
        schema = onnx.defs.get_schema("Scan", 9)
        args = [np.array([1, 2, 3, 4])]
        kwargs = {'body': onnx.helper.make_graph([], "body", [], [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])])}
        expected_outputs = 1
        result = evaluator.compute_num_outputs(schema, args, kwargs)
        self.assertEqual(result, expected_outputs)


    def test_compute_num_outputs_variable_outputs(self):
        schema = onnx.defs.get_schema("Split", 13)
        args = [np.array([1, 2, 3, 4]), np.array([2, 2])]
        kwargs = {}
        expected_outputs = 2
        result = evaluator.compute_num_outputs(schema, args, kwargs)
        self.assertEqual(result, expected_outputs)


    def test_adapt_to_user_mode_single_numpy_array(self):
        input_array = np.array([1, 2, 3])
        expected = np.array([1, 2, 3])
        result = evaluator._adapt_to_user_mode(input_array)
        np.testing.assert_array_equal(result, expected)


    def test_adapt_to_eager_mode_single_none(self):
        input_none = None
        expected = None
        result, has_array = evaluator._adapt_to_eager_mode(input_none)
        self.assertEqual(result, expected)
        self.assertFalse(has_array)


    def test_adapt_to_eager_mode_single_scalar(self):
        input_scalar = 5
        expected = tensor.Tensor(np.array(input_scalar, dtype=np.int64))
        result, has_array = evaluator._adapt_to_eager_mode(input_scalar)
        self.assertEqual(result, expected)
        self.assertFalse(has_array)


    def test_adapt_to_user_mode_tuple_of_tensors(self):
        input_tensors = (tensor.Tensor(np.array([1, 2, 3])), tensor.Tensor(np.array([4, 5, 6])))
        expected = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        result = evaluator._adapt_to_user_mode(input_tensors)
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])


    def test_unwrap_tensors_in_kwargs_mixed(self):
        kwargs = {'a': tensor.Tensor(np.array([1, 2, 3])), 'b': np.array([4, 5, 6])}
        expected = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
        result = evaluator._unwrap_tensors_in_kwargs(kwargs)
        np.testing.assert_array_equal(result['a'], expected['a'])
        np.testing.assert_array_equal(result['b'], expected['b'])


    def test_compute_num_outputs_split_no_num_outputs(self):
        schema = onnx.defs.get_schema("Split", 13)
        args = [np.array([1, 2, 3, 4])]
        kwargs = {}
        with self.assertRaises(evaluator.EagerModeError):
            evaluator.compute_num_outputs(schema, args, kwargs)

if __name__ == "__main__":
    unittest.main()
