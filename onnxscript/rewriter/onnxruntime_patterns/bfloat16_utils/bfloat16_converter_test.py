# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx
import onnx.checker
import onnx.shape_inference
import onnxruntime

from onnxscript import ir
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
        self.mul_node = ir.Node(
            "", "Mul", inputs=(self.add_node.outputs[0], self.v2), num_outputs=1
        )
        self.mul_node.outputs[0].dtype = ir.DataType.BFLOAT16
        self.graph = ir.Graph(
            name="bfloat16_conversion_test",
            inputs=(self.v0, self.v1, self.v2),
            outputs=(self.add_node.outputs[0], self.mul_node.outputs[0]),
            nodes=(self.add_node, self.mul_node),
            opset_imports={"": 18},
        )
        self.original_output_names = [output.name for output in self.graph.outputs]
        self.model = ir.Model(
            graph=self.graph,
            ir_version=8,
            producer_name="bfloat16_conversion_test",
        )
        bfloat16_converter.dtype_adapter_for_bfloat16_model(self.model)

    def test_input_and_output_are_float16(self):
        for input in self.model.graph.inputs:
            self.assertEqual(input.dtype, ir.DataType.FLOAT16)
        for output in self.model.graph.outputs:
            self.assertEqual(output.dtype, ir.DataType.FLOAT16)

    def test_cast_nodes_are_inserted(self):
        cast_node_count = 0
        for node in self.model.graph:
            if node.op_type == "Cast":
                cast_node_count += 1
        self.assertEqual(cast_node_count, 5)

        for input in self.model.graph.inputs:
            for input_user, _ in input.uses():
                self.assertEqual(input_user.op_type, "Cast")
                self.assertEqual(input_user.outputs[0].dtype, ir.DataType.BFLOAT16)
        for output in self.model.graph.outputs:
            self.assertEqual(output.producer().op_type, "Cast")
            self.assertEqual(output.producer().inputs[0].dtype, ir.DataType.BFLOAT16)

    def test_graph_output_name_is_preserved(self):
        self.assertEqual(
            [output.name for output in self.model.graph.outputs],
            self.original_output_names,
        )

    def test_bfloat16_converted_model_runtime(self):
        model_proto = ir.serde.serialize_model(self.model)
        model_proto_filled_shape_type = onnx.shape_inference.infer_shapes(
            model_proto, check_type=True, strict_mode=True, data_prop=True
        )
        onnx.checker.check_model(model_proto_filled_shape_type, full_check=True)
        try:
            ort_session = onnxruntime.InferenceSession(
                model_proto_filled_shape_type.SerializeToString(),
                providers=["CPUExecutionProvider"],
            )
            v0 = np.random.randn(2, 3, 4).astype(np.float16)
            v1 = np.random.randn(2, 3, 4).astype(np.float16)
            v2 = np.random.randn(2, 3, 4).astype(np.float16)
            ort_inputs = {"v0": v0, "v1": v1, "v2": v2}
            ort_outputs = ort_session.run(None, ort_inputs)
            expected_output = (v0 + v1) * v2
            np.testing.assert_allclose(ort_outputs[0], expected_output, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(ort_outputs[1], expected_output, rtol=1e-2, atol=1e-2)
        except Exception as e:
            self.assertIn(
                "[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Add(14)",
                str(e),
            )


if __name__ == "__main__":
    unittest.main()
