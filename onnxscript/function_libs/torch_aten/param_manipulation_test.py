import collections
import unittest

import parameterized
import numpy as np

from onnxscript import tensor, INT64

from onnxscript.function_libs.torch_aten.param_manipulation import (
    ParamSchema,
    separate_input_attributes_from_arguments,
)


class TestParamManipulation(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "all_positional",
                (tensor.Tensor(np.array((), dtype=np.int64)), 42, 0.0),
                {},
                0.0,
            ),
            (
                "positional_with_default",
                (tensor.Tensor(np.array((), dtype=np.int64)), 42),
                {},
                100.0,
            ),
            (
                "positional_with_default_and_kwargs",
                (tensor.Tensor(np.array((), dtype=np.int64)),),
                {"b": 42},
                100.0,
            ),
            (
                "positional_with_kwargs",
                (tensor.Tensor(np.array((), dtype=np.int64)), 42),
                {"c": 0.0},
                0.0,
            ),
            (
                "positional_input_with_kwargs_attribute",
                (tensor.Tensor(np.array((), dtype=np.int64)),),
                {"b": 42, "c": 0.0},
                0.0,
            ),
            (
                "all_kwargs",
                (),
                {"a": tensor.Tensor(np.array((), dtype=np.int64)), "b": 42, "c": 0.0},
                0.0,
            ),
            (
                "all_kwargs_with_default",
                (),
                {"a": tensor.Tensor(np.array((), dtype=np.int64)), "b": 42},
                100.0,
            ),
            (
                "extra_positional",  # Probably warn about this
                (tensor.Tensor(np.array((), dtype=np.int64)), 42, 0.0, -1),
                {},
                0.0,
            ),
            (
                "extra_keyword",  # Probably warn about this
                (tensor.Tensor(np.array((), dtype=np.int64)), 42, 0.0),
                {"unknown": -1},
                0.0,
            ),
            (
                "extra_positional_and_keyword",  # Probably warn about this
                (tensor.Tensor(np.array((), dtype=np.int64)), 42, 0.0, -1),
                {"unknown": -1},
                0.0,
            ),
        ]
    )
    def test_separate_input_attributes_from_arguments_correct_on(
        self, _, args, kwargs, expected_c
    ):
        param_schemas = (
            ParamSchema(name="a", type=INT64, is_input=True),
            ParamSchema(name="b", type=int, is_input=False),
            ParamSchema(name="c", type=float, default=100.0, is_input=False),
        )

        expected_inputs = [tensor.Tensor(np.array((), dtype=np.int64))]
        expected_attributes = collections.OrderedDict(
            [
                ("b", 42),
                ("c", expected_c),
            ]
        )

        inputs, attributes = separate_input_attributes_from_arguments(
            param_schemas, args, kwargs
        )

        self.assertEqual(inputs, expected_inputs)
        self.assertEqual(attributes, expected_attributes)


if __name__ == "__main__":
    unittest.main()
