# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pathlib
import unittest

import onnx

from onnxscript import ir


class GraphViewTest(unittest.TestCase):
    def test_it_can_be_serialized_as_graph_proto(self):
        data_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "testdata/e2e_models/torchscript_model/torchscript_model.onnx"
        )
        model_proto = onnx.load(data_path)
        model = ir.serde.deserialize_model(model_proto)

        # Obtain a subset of nodes that belong to the first layer
        nodes = [
            node
            for node in model.graph
            if node.name is not None and node.name.startswith("/model/layers.0")
        ]

        inputs = set()
        outputs = set()
        for node in nodes:
            inputs.update(node.inputs)
            outputs.update(node.outputs)

        graph_inputs = sorted(inputs - outputs, key=lambda val: val.name)
        graph_outputs = sorted(outputs - inputs, key=lambda val: val.name)

        graph_view = ir.GraphView(graph_inputs, graph_outputs, nodes=nodes)
        model = ir.Model(graph_view, ir_version=8)
        _ = ir.serde.serialize_model(model)
        # It should succeed


    def test_deserialize_string_tensor_with_external_data_location(self):
        tensor_proto = onnx.TensorProto()
        tensor_proto.data_type = onnx.TensorProto.STRING
        tensor_proto.data_location = onnx.TensorProto.EXTERNAL
        tensor_proto.string_data.extend([b"external_data"])
        tensor_proto.dims.extend([1])
        
        tensor = ir.serde.deserialize_tensor(tensor_proto)
        
        self.assertIsInstance(tensor, ir.ExternalTensor)
        self.assertEqual(tensor.shape.dims, (1,))


    def test_serialize_string_tensor(self):
        string_tensor = ir.StringTensor([b"foo", b"bar"], shape=ir.Shape([2]))
        
        tensor_proto = ir.serde.serialize_tensor(string_tensor)
        
        self.assertEqual(tensor_proto.data_type, onnx.TensorProto.STRING)
        self.assertEqual(tensor_proto.string_data, [b"foo", b"bar"])


    def test_deserialize_string_tensor(self):
        tensor_proto = onnx.TensorProto()
        tensor_proto.data_type = onnx.TensorProto.STRING
        tensor_proto.string_data.extend([b"hello", b"world"])
        tensor_proto.dims.extend([2])
        
        tensor = ir.serde.deserialize_tensor(tensor_proto)
        
        self.assertIsInstance(tensor, ir.StringTensor)
        self.assertEqual(tensor.numpy().tolist(), [b"hello", b"world"])


    def test_deserialize_unsorted_graph(self):
        graph_proto = onnx.GraphProto()
        node_proto_1 = onnx.NodeProto()
        node_proto_1.name = "node1"
        node_proto_1.op_type = "Add"
        node_proto_1.input.extend(["input1", "input2"])
        node_proto_1.output.extend(["output1"])
        
        node_proto_2 = onnx.NodeProto()
        node_proto_2.name = "node2"
        node_proto_2.op_type = "Mul"
        node_proto_2.input.extend(["output1", "input3"])
        node_proto_2.output.extend(["output2"])
        
        graph_proto.node.extend([node_proto_2, node_proto_1])
        
        graph = ir.serde.deserialize_graph(graph_proto)
        self.assertEqual(len(graph), 2)
        self.assertEqual(graph[0].name, "node2")
        self.assertEqual(graph[1].name, "node1")


    def test_serialize_function_no_inputs(self):
        function_proto = onnx.FunctionProto()
        function_proto.name = "test_function"
        function_proto.domain = "test_domain"
        function_proto.opset_import.add(domain="", version=13)
        
        model_proto = onnx.ModelProto()
        model_proto.ir_version = 7
        model_proto.functions.extend([function_proto])
        
        model = ir.serde.deserialize_model(model_proto)
        serialized_model = ir.serde.serialize_model(model)
        
        self.assertEqual(len(serialized_model.functions), 1)
        self.assertEqual(serialized_model.functions[0].name, "test_function")


    def test_serialize_sparse_tensor_not_implemented_error(self):
        attribute_proto = onnx.AttributeProto()
        with self.assertRaises(NotImplementedError):
            ir.serde._fill_in_value_for_attribute(attribute_proto, ir.AttributeType.SPARSE_TENSOR, None)


    def test_to_proto_not_implemented_error(self):
        class UnsupportedIRObject:
            pass
    
        unsupported_ir_object = UnsupportedIRObject()
        with self.assertRaises(NotImplementedError):
            ir.serde.to_proto(unsupported_ir_object)


    def test_from_proto_not_implemented_error(self):
        class UnsupportedProto:
            pass
    
        unsupported_proto = UnsupportedProto()
        with self.assertRaises(NotImplementedError):
            ir.serde.from_proto(unsupported_proto)


if __name__ == "__main__":
    unittest.main()
