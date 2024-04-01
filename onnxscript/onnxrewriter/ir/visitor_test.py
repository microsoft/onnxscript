import unittest

import onnx

from onnxrewriter.ir import visitor


class FunctionCallsiteProtoTransformerTest(unittest.TestCase):
    def test_function_optional_input_is_recorded_by_shape_env(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z) {
                z = custom.function(x)
            }
            <
                domain: "custom",
                opset_import: ["" : 18]
            >
            function (x, optional_y, optional_z) => (return_val)
            {
                return_val = custom.custom_op (x, optional_y, optional_z)
            }
            """
        )

        model_visitor = visitor.FunctionCallsiteProtoTransformer()
        model_visitor.visit_model(model)
        self.assertIsNotNone(
            model_visitor.function_shape_env.lookup(model.functions[0], "optional_y")
        )
        self.assertIsNotNone(
            model_visitor.function_shape_env.lookup(model.functions[0], "optional_z")
        )


if __name__ == "__main__":
    unittest.main()
