from onnxscript import ir
import unittest
from onnxscript.rewriter.onnxruntime.bfloat16_utils import bfloat16_converter

class Bfloat16ConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = ir.Input(name="v0", shape=ir.Shape([2, 3, 4]))
        self.v0.dtype = ir.DataType.BFLOAT16
        self.v1 = ir.Input(name="v1", shape=ir.Shape([2, 3, 4]))
        self.v1.dtype = ir.DataType.BFLOAT16
        self.v2 = ir.Input(name="v2", shape=ir.Shape([2, 3, 4]))
        self.v2.dtype = ir.DataType.BFLOAT16

        self.add_node = ir.Node("", "Add", inputs=(self.v0, self.v1), num_outputs=1)
        self.add_node.outputs[0].dtype = ir.DataType.BFLOAT16
        self.mul_node = ir.Node("", "Mul", inputs=(self.add_node.outputs[0], self.v2), num_outputs=1)
        self.mul_node.outputs[0].dtype = ir.DataType.BFLOAT16
        self.graph = ir.Graph(
            inputs=(self.v0, self.v1, self.v2),
            outputs=(self.add_node.outputs[0], self.mul_node.outputs[0]),
            nodes=(self.add_node, self.mul_node),
            opset_imports={"": 1},
        )
        self.original_model = ir.Model(
            graph=self.graph,
            ir_version=8,
            producer_name="bfloat16_conversion_test"
        )
        self.converted_model = bfloat16_converter.dtype_adapter_for_bfloat16_model(self.original_model)

    def test_input_and_output_are_float16(self):
        for input in self.converted_model.graph.inputs:
            self.assertEqual(input.dtype, ir.DataType.FLOAT16)
        for output in self.converted_model.graph.outputs:
            self.assertEqual(output.dtype, ir.DataType.FLOAT16)

    def test_cast_nodes_are_inserted(self):
        cast_node_count = 0
        for node in self.converted_model.graph:
            if node.op_type == "Cast":
                cast_node_count += 1
        self.assertEqual(cast_node_count, 5)

        for input in self.converted_model.graph.inputs:
            for input_user, _ in input.uses():
                self.assertEqual(input_user.op_type, "Cast")
                self.assertEqual(input_user.outputs[0].dtype, ir.DataType.BFLOAT16)
        for output in self.converted_model.graph.outputs:
            self.assertEqual(output.producer().op_type, "Cast")
            self.assertEqual(output.producer().inputs[0].dtype, ir.DataType.BFLOAT16)


if __name__ == "__main__":
    unittest.main()
