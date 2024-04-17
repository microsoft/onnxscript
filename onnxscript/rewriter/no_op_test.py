import unittest

import onnx.parser
import parameterized

from onnxscript import ir
from onnxscript.rewriter import no_op


class NoOpTest(unittest.TestCase):
    def _check(self, model_text: str) -> None:
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)
        count = no_op.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(model.graph.nodes[-1].op_type, "Identity")

    @parameterized.parameterized.expand(
        [
            ("float one input", "float[M]", "value_float=1.0", "one, input"),
            ("int one input", "int32[M]", "value_int=1", "one, input"),
            ("float input one", "float[M]", "value_float=1.0", "input, one"),
            ("int input one", "int32[M]", "value_int=1", "input, one"),
        ]
    )
    def test_mul_one_should_become_no_op(self, _, dtype, constant_value, input_order):
        self._check(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph ({dtype} input) => ({dtype} output)
            {{
                one = Constant<{constant_value}>()
                output = Mul({input_order})
            }}
        """
        )

    @parameterized.parameterized.expand(
        [
            ("float one input", "float[M]", "float one = {1.0}", "one, input"),
            ("int one input", "int32[M]", "int32 one = {1}", "one, input"),
            ("float input one", "float[M]", "float one = {1.0}", "input, one"),
            ("int input one", "int32[M]", "int32 one = {1}", "input, one"),
        ]
    )
    def test_mul_one_should_become_no_op_initializer(
        self, _, dtype, constant_value, input_order
    ):
        self._check(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph ({dtype} input) => ({dtype} output)
            <{constant_value}>
            {{
                output = Mul({input_order})
            }}
        """
        )

    @parameterized.parameterized.expand(
        [
            ("float zero input", "float[M]", "value_float=0.0", "zero, input"),
            ("int zero input", "int32[M]", "value_int=0", "zero, input"),
            ("float input zero", "float[M]", "value_float=0.0", "input, zero"),
            ("int input zero", "int32[M]", "value_int=0", "input, zero"),
        ]
    )
    def test_add_zero_should_become_no_op(self, _, dtype, constant_value, input_order):
        self._check(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph ({dtype} input) => ({dtype} output)
            {{
                zero = Constant<{constant_value}>()
                output = Add({input_order})
            }}
        """
        )

    @parameterized.parameterized.expand(
        [
            ("float input zero", "float[M]", "float zero = {0.0}", "input, zero"),
            ("int input zero", "int32[M]", "int32 zero = {0}", "input, zero"),
            ("float input zero", "float[M]", "float zero = {0.0}", "input, zero"),
            ("int input zero", "int32[M]", "int32 zero = {0}", "input, zero"),
        ]
    )
    def test_add_zero_should_become_no_op_initializer(
        self, _, dtype, constant_value, input_order
    ):
        self._check(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph ({dtype} input) => ({dtype} output)
            <{constant_value}>
            {{
                output = Add({input_order})
            }}
        """
        )

    @parameterized.parameterized.expand(
        [
            ("float input zero", "float[M]", "value_float=0.0", "input, zero"),
            ("int input zero", "int32[M]", "value_int=0", "input, zero"),
        ]
    )
    def test_sub_zero_should_become_no_op(self, _, dtype, constant_value, input_order):
        self._check(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph ({dtype} input) => ({dtype} output)
            {{
                zero = Constant<{constant_value}>()
                output = Sub({input_order})
            }}
        """
        )

    @parameterized.parameterized.expand(
        [
            ("float input zero", "float[M]", "float zero = {0.0}", "input, zero"),
            ("int input zero", "int32[M]", "int32 zero = {0}", "input, zero"),
        ]
    )
    def test_sub_zero_should_become_no_op_initializer(
        self, _, dtype, constant_value, input_order
    ):
        self._check(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph ({dtype} input) => ({dtype} output)
            <{constant_value}>
            {{
                output = Sub({input_order})
            }}
        """
        )

    @parameterized.parameterized.expand(
        [
            ("float input one", "float[M]", "value_float=1.0", "input, one"),
            ("int input one", "int32[M]", "value_int=1", "input, one"),
        ]
    )
    def test_div_one_should_become_no_op(self, _, dtype, constant_value, input_order):
        self._check(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph ({dtype} input) => ({dtype} output)
            {{
                one = Constant<{constant_value}>()
                output = Div({input_order})
            }}
        """
        )

    @parameterized.parameterized.expand(
        [
            ("float input one", "float[M]", "float one = {1.0}", "input, one"),
            ("int input one", "int32[M]", "int32 one = {1}", "input, one"),
        ]
    )
    def test_div_one_should_become_no_op_with_initializer(
        self, _, dtype, constant_value, input_order
    ):
        self._check(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph ({dtype} input) => ({dtype} output)
            <{constant_value}>
            {{
                output = Div({input_order})
            }}
        """
        )


if __name__ == "__main__":
    unittest.main()
