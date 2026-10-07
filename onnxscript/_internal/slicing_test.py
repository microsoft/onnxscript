# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Parity tests for scalar indexing combined with slices."""

import itertools
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import parameterized

from onnxscript import INT64, evaluator, opset15, script


@script(default_opset=opset15)
def last_row_slice(x: INT64["N", "M"]) -> INT64["K"]:  # noqa: F821
    return x[-1, :2]


@script(default_opset=opset15)
def last_column_slice(x: INT64["N", "M"]) -> INT64["K"]:  # noqa: F821
    return x[:2, -1]


@script(default_opset=opset15)
def last_element(x: INT64["N", "M"]) -> INT64:  # noqa: F821
    return x[-1, -1]


@script(default_opset=opset15)
def penultimate_row_last_column(x: INT64["N", "M"]) -> INT64:  # noqa: F821
    return x[-2, -1]


@script(default_opset=opset15)
def reversed_last_row(x: INT64["N", "M"]) -> INT64["M"]:  # noqa: F821
    return x[-1, ::-1]


@script(default_opset=opset15)
def first_row_slice(x: INT64["N", "M"]) -> INT64["K"]:  # noqa: F821
    return x[0, :2]


class TestScalarSlicing(unittest.TestCase):
    @parameterized.parameterized.expand(
        itertools.product(
            [
                (last_row_slice, (-1, slice(None, 2))),
                (last_column_slice, (slice(None, 2), -1)),
                (last_element, (-1, -1)),
                (penultimate_row_last_column, (-2, -1)),
                (reversed_last_row, (-1, slice(None, None, -1))),
                (first_row_slice, (0, slice(None, 2))),
            ],
            [(4, 3), (2, 5)],
            ["eager", "onnxruntime"],
        )
    )
    def test_matches_numpy(self, function_and_index, shape, mode):
        function, index = function_and_index
        data = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        if mode == "eager":
            with evaluator.default_as(evaluator.OnnxReferenceRuntimeEvaluator()):
                actual = function(data)
        else:
            model = function.to_model_proto(ir_version=9)
            onnx.checker.check_model(model)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            actual = session.run(None, {"x": data})[0]
        np.testing.assert_array_equal(actual, data[index])
        self.assertEqual(actual.shape, data[index].shape)


if __name__ == "__main__":
    unittest.main()
