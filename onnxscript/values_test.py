import unittest

import onnxscript
from onnxscript import values


class TracedOnnxFunctionTest(unittest.TestCase):
    def test_init(self):
        def function(input1, input2, attr1: int, attr2: int = 1):
            return input1 + input2 + attr1 + attr2

        opset = values.Opset("test", 1)
        traced_function = values.TracedOnnxFunction(opset, function)
        self.assertEqual(traced_function.opset, opset)
        self.assertEqual(traced_function.name, function.__name__)
        self.assertEqual(traced_function.func, function)

    def test_param_schemas_in_correct_order_with_mixed_inputs_and_attrs(self):
        opset = values.Opset("test", 1)

        @onnxscript.script(default_opset=opset)
        def function(input1, input2, attr1: int, attr2: float, input3, attr3: str = "default"):
            return opset.CustomOp(input1 + input2, input3, attr1, attr2, attr3)

        param_schemas = function.param_schemas()
        expected_ordered_param_names = [
            "input1",
            "input2",
            "attr1",
            "attr2",
            "input3",
            "attr3",
        ]
        self.assertEqual(len(param_schemas), len(expected_ordered_param_names))
        for i, param_schema in enumerate(param_schemas):
            self.assertEqual(param_schema.name, expected_ordered_param_names[i])
