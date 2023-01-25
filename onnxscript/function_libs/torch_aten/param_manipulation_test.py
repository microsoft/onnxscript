import collections
from typing import Any
import unittest

import numpy as np

from onnxscript import tensor, INT64

from onnxscript.function_libs.torch_aten.param_manipulation import (
    ParamSchema,
    separate_input_attributes_from_arguments,
)


class TestParamManipulation(unittest.TestCase):
    def test_separate_input_attributes_from_arguments_should_separate_positional_arguments(
        self,
    ):
        param_schemas = (
            ParamSchema(name="a", type=INT64, is_input=True),
            ParamSchema(name="b", type=int, is_input=False),
            ParamSchema(name="c", type=float, default=1.0, is_input=False),
        )

        args = (tensor.Tensor(np.array((), dtype=np.int64)), 42, 0.0)
        kwargs: dict[str, Any] = {}

        expected_inputs = collections.OrderedDict(
            [("a", tensor.Tensor(np.array((), dtype=np.int64)))]
        )
        expected_attributes = collections.OrderedDict(
            [
                ("b", 42),
                ("c", 0.0),
            ]
        )

        inputs, attributes = separate_input_attributes_from_arguments(
            param_schemas, args, kwargs
        )

        self.assertEqual(inputs, expected_inputs)
        self.assertEqual(attributes, expected_attributes)
