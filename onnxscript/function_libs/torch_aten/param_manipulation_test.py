import collections
import unittest

import parameterized

from onnxscript import INT64
from onnxscript.function_libs.torch_aten.param_manipulation import (
    ParamSchema,
    separate_input_attributes_from_arguments,
)

TEST_INPUT = "TEST_INPUT"


class TestParamManipulation(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "all_positional",
                (TEST_INPUT, 42, 0.0),
                {},
                0.0,
            ),
            (
                "positional_with_default",
                (TEST_INPUT, 42),
                {},
                100.0,
            ),
            (
                "positional_with_default_and_kwargs",
                (TEST_INPUT,),
                {"b": 42},
                100.0,
            ),
            (
                "positional_with_kwargs",
                (TEST_INPUT, 42),
                {"c": 0.0},
                0.0,
            ),
            (
                "positional_input_with_kwargs_attribute",
                (TEST_INPUT,),
                {"b": 42, "c": 0.0},
                0.0,
            ),
            (
                "all_kwargs",
                (),
                {"a": TEST_INPUT, "b": 42, "c": 0.0},
                0.0,
            ),
            (
                "all_kwargs_with_default",
                (),
                {"a": TEST_INPUT, "b": 42},
                100.0,
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

        expected_inputs = [TEST_INPUT]
        expected_attributes = collections.OrderedDict(
            [
                ("b", 42),
                ("c", expected_c),
            ]
        )

        inputs, attributes = separate_input_attributes_from_arguments(
            param_schemas, args, kwargs
        )

        print("\ninputs: ", inputs)
        print("\nexpected_inputs: ", expected_inputs)

        self.assertEqual(len(inputs), len(expected_inputs))
        for input_, expected_input in zip(inputs, expected_inputs):
            self.assertIs(input_, expected_input)
        self.assertEqual(attributes, expected_attributes)

    @parameterized.parameterized.expand(
        [
            (
                "extra_positional",
                (TEST_INPUT, 42, 0.0, -1),
                {},
            ),
            (
                "extra_keyword",
                (TEST_INPUT, 42, 0.0),
                {"unknown": -1},
            ),
            (
                "extra_positional_and_keyword",
                (TEST_INPUT, 42, 0.0, -1),
                {"unknown": -1},
            ),
        ]
    )
    def test_separate_input_attributes_from_arguments_raises_on_extra_args(
        self, _, args, kwargs
    ):
        param_schemas = (
            ParamSchema(name="a", type=INT64, is_input=True),
            ParamSchema(name="b", type=int, is_input=False),
            ParamSchema(name="c", type=float, default=100.0, is_input=False),
        )

        with self.assertRaises(TypeError):
            _, _ = separate_input_attributes_from_arguments(param_schemas, args, kwargs)


if __name__ == "__main__":
    unittest.main()
