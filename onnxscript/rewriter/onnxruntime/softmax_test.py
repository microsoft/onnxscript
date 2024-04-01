import unittest

import onnx.parser
import parameterized

from onnxrewriter.ir import irbuilder
from onnxrewriter.rewriter.onnxruntime import softmax


class SoftmaxUpcastRemovalTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("Softmax<axis=-1>",),
            ("Softmax",),
        ]
    )
    def test_softmax_upcast_to_fp32_is_removed_when_input_and_final_output_is_fp16(
        self, softmax_op_str
    ):
        model = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float16[N] x) => (float16[N] z)
            {{
                x_fp32 = Cast<to=1>(x)
                z_fp32 = {softmax_op_str}(x_fp32)
                z = Cast<to=10>(z_fp32)
            }}
            """
        )
        ir = irbuilder.build_ir(model)
        count = softmax.rules.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertNotIn("Cast", {node.op_type for node in ir.graph.nodes})

    @parameterized.parameterized.expand(
        [
            ("Softmax<axis=-1>",),
            ("Softmax",),
        ]
    )
    def test_softmax_upcast_to_fp32_is_not_removed_when_input_is_not_fp16(
        self, softmax_op_str
    ):
        model = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int32[N] x) => (float16[N] z)
            {{
                x_fp32 = Cast<to=1>(x)
                z_fp32 = {softmax_op_str}(x_fp32)
                z = Cast<to=10>(z_fp32)
            }}
            """
        )
        ir = irbuilder.build_ir(model)
        count = softmax.rules.apply_to_model(ir)
        self.assertEqual(count, 0)
        self.assertEqual(
            len([node.op_type for node in ir.graph.nodes if node.op_type == "Cast"]), 2
        )

    @parameterized.parameterized.expand(
        [
            ("Softmax<axis=-1>",),
            ("Softmax",),
        ]
    )
    def test_softmax_upcast_to_fp32_is_not_removed_when_final_output_is_not_fp16(
        self, softmax_op_str
    ):
        model = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float16[N] x) => (double[N] z)
            {{
                x_fp32 = Cast<to=1>(x)
                z_fp32 = {softmax_op_str}(x_fp32)
                z = Cast<to=11>(z_fp32)
            }}
            """
        )
        ir = irbuilder.build_ir(model)
        count = softmax.rules.apply_to_model(ir)
        self.assertEqual(count, 0)
        self.assertEqual(
            len([node.op_type for node in ir.graph.nodes if node.op_type == "Cast"]), 2
        )


if __name__ == "__main__":
    unittest.main()
