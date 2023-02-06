# mypy: disable-error-code=misc

import collections
import unittest

import parameterized

from onnxscript import INT64, values
from onnxscript._internal import param_manipulation

TEST_INPUT = "TEST_INPUT"


class TestSeparateInputAttributesFromArguments(unittest.TestCase):
    """Unit tests for `param_manipulation.separate_input_attributes_from_arguments`."""

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
    def test_it_is_correct_on(self, _, args, kwargs, expected_c):
        param_schemas = (
            values.ParamSchema(name="a", type=INT64, is_input=True),
            values.ParamSchema(name="b", type=int, is_input=False),
            values.ParamSchema(name="c", type=float, default=100.0, is_input=False),
        )

        expected_inputs = [TEST_INPUT]
        expected_attributes = collections.OrderedDict(
            [
                ("b", 42),
                ("c", expected_c),
            ]
        )

        inputs, attributes = param_manipulation.separate_input_attributes_from_arguments(
            param_schemas, args, kwargs
        )

        self.assertEqual(len(inputs), len(expected_inputs))
        for input_, expected_input in zip(inputs, expected_inputs):
            self.assertIs(input_, expected_input)
        self.assertEqual(attributes, expected_attributes)

    @parameterized.parameterized.expand(
        [
            (
                "extra_keyword",
                (TEST_INPUT, 42, 0.0),
                {"unknown": -1},
            ),
        ]
    )
    def test_it_raises_on_extra_args(self, _, args, kwargs):
        param_schemas = (
            values.ParamSchema(name="a", type=INT64, is_input=True),
            values.ParamSchema(name="b", type=int, is_input=False),
            values.ParamSchema(name="c", type=float, default=100.0, is_input=False),
        )

        with self.assertRaises(TypeError):
            _, _ = param_manipulation.separate_input_attributes_from_arguments(
                param_schemas, args, kwargs
            )

    @parameterized.parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_it_raises_on_extra_kwargs_when_not_allow_extra_kwargs(
        self,
        fill_defaults: bool,
    ):
        param_schemas = (
            values.ParamSchema(name="a", type=INT64, is_input=True),
            values.ParamSchema(name="b", type=int, is_input=False),
            values.ParamSchema(name="c", type=float, default=100.0, is_input=False),
        )

        with self.assertRaises(TypeError):
            _, _ = param_manipulation.separate_input_attributes_from_arguments(
                param_schemas,
                (TEST_INPUT, 42),
                {"c": 1.0, "extra": 42},
                fill_defaults=fill_defaults,
                allow_extra_kwargs=False,
            )

    @parameterized.parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_it_does_not_fill_default_when_fill_defaults_is_false(
        self, allow_extra_kwargs: bool
    ):
        param_schemas = (
            values.ParamSchema(name="a", type=INT64, is_input=True),
            values.ParamSchema(name="b", type=int, is_input=False),
            values.ParamSchema(name="c", type=float, default=100.0, is_input=False),
        )

        inputs, attributes = param_manipulation.separate_input_attributes_from_arguments(
            param_schemas,
            (TEST_INPUT, 42),
            {},
            fill_defaults=False,
            allow_extra_kwargs=allow_extra_kwargs,
        )

        self.assertEqual(inputs, [TEST_INPUT])
        self.assertEqual(attributes, collections.OrderedDict([("b", 42)]))

    @parameterized.parameterized.expand(
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]
    )
    def test_it_raises_on_insufficient_args(
        self, fill_defaults: bool, allow_extra_kwargs: bool
    ):
        param_schemas = (
            values.ParamSchema(name="a", type=INT64, is_input=True),
            values.ParamSchema(name="b", type=int, is_input=False),
            values.ParamSchema(name="c", type=float, default=100.0, is_input=False),
        )

        with self.assertRaises(TypeError):
            _, _ = param_manipulation.separate_input_attributes_from_arguments(
                param_schemas,
                (TEST_INPUT,),
                {},
                fill_defaults=fill_defaults,
                allow_extra_kwargs=allow_extra_kwargs,
            )


if __name__ == "__main__":
    unittest.main()
