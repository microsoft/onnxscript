# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import google.protobuf.text_format
import ml_dtypes
import numpy as np
import onnx
import parameterized

from onnxscript import ir
from onnxscript._internal import version_utils
from onnxscript.ir import serde


class ConvenienceFunctionsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("model", onnx.ModelProto()),
            ("graph", onnx.GraphProto()),
            ("node", onnx.NodeProto(input=["X"], output=["Y"])),
            (
                "tensor",
                onnx.helper.make_tensor("test_tensor", onnx.TensorProto.FLOAT, [1], [1.0]),
            ),
            ("value_info", onnx.ValueInfoProto()),
            ("type", onnx.TypeProto()),
            ("attribute", onnx.AttributeProto()),
        ]
    )
    def test_from_proto(self, _: str, proto):
        serde.from_proto(proto)

    @parameterized.parameterized.expand(
        [
            ("model", ir.Model(ir.Graph([], [], nodes=[]), ir_version=1)),
            ("graph", ir.Graph([], [], nodes=[])),
            (
                "node",
                ir.Node("", "Op", inputs=[], outputs=[ir.Value(name="value")]),
            ),
            (
                "tensor",
                serde.TensorProtoTensor(
                    onnx.helper.make_tensor("test_tensor", onnx.TensorProto.FLOAT, [1], [1.0])
                ),
            ),
            ("value", ir.Value(name="value")),
            ("type", ir.SequenceType(ir.OptionalType(ir.TensorType(ir.DataType.COMPLEX128)))),
            ("attribute", ir.Attr("attribute", ir.AttributeType.FLOAT, 1)),
            ("ref_attribute", ir.RefAttr("ref_attr", "attr", ir.AttributeType.FLOAT)),
            ("graph_view", ir.GraphView([], [], nodes=[])),
        ]
    )
    def test_to_proto(self, _: str, ir_object):
        serde.to_proto(ir_object)


class TensorProtoTensorTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("FLOAT", onnx.TensorProto.FLOAT),
            ("BOOL", onnx.TensorProto.BOOL),
            ("FLOAT16", onnx.TensorProto.FLOAT16),
            ("DOUBLE", onnx.TensorProto.DOUBLE),
        ]
    )
    def test_tensor_proto_tensor(self, _: str, dtype: int):
        tensor_proto = onnx.helper.make_tensor(
            "test_tensor", dtype, [1, 9], [-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]
        )
        tensor = serde.TensorProtoTensor(tensor_proto)
        expected_array = onnx.numpy_helper.to_array(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        if dtype == onnx.TensorProto.BOOL and version_utils.numpy_older_than("1.25"):
            self.skipTest("numpy<1.25 does not support bool dtype in from_dlpack")
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @unittest.skipIf(
        version_utils.onnx_older_than("1.17"),
        "numpy_helper.to_array was not correctly implemented in onnx<1.17",
    )
    def test_tensor_proto_tensor_bfloat16(self):
        expected_array = np.array(
            [[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]], dtype=ml_dtypes.bfloat16
        )
        tensor_proto = onnx.helper.make_tensor(
            "test_tensor",
            onnx.TensorProto.BFLOAT16,
            [1, 9],
            np.array([[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]]),
        )
        tensor = serde.TensorProtoTensor(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(
            array_from_raw_data.view(ml_dtypes.bfloat16), expected_array
        )
        # Test dlpack
        with self.assertRaises(BufferError):
            # NumPy does not support bfloat16 in from_dlpack
            np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            (
                "FLOAT8E4M3FN",
                onnx.TensorProto.FLOAT8E4M3FN,
                ml_dtypes.float8_e4m3fn,
            ),
            (
                "FLOAT8E4M3FNUZ",
                onnx.TensorProto.FLOAT8E4M3FNUZ,
                ml_dtypes.float8_e4m3fnuz,
            ),
            (
                "FLOAT8E5M2",
                onnx.TensorProto.FLOAT8E5M2,
                ml_dtypes.float8_e5m2,
            ),
            (
                "FLOAT8E5M2FNUZ",
                onnx.TensorProto.FLOAT8E5M2FNUZ,
                ml_dtypes.float8_e5m2fnuz,
            ),
        ]
    )
    def test_tensor_proto_tensor_float8(self, _: str, dtype: int, np_dtype):
        expected_array = np.array([[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 40.0, 2.0]])
        tensor_proto = onnx.helper.make_tensor("test_tensor", dtype, [1, 9], expected_array)
        tensor = serde.TensorProtoTensor(tensor_proto)
        np.testing.assert_array_equal(
            tensor.numpy().view(np_dtype).astype(np.float32), expected_array
        )
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = (
            serde.TensorProtoTensor(tensor_proto_from_raw_data)
            .numpy()
            .view(np_dtype)
            .astype(np.float32)
        )
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        with self.assertRaises(BufferError):
            # DL Pack does not support float8
            np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            ("INT8", onnx.TensorProto.INT8),
            ("INT16", onnx.TensorProto.INT16),
            ("INT32", onnx.TensorProto.INT32),
            ("INT64", onnx.TensorProto.INT64),
            ("INT4", onnx.TensorProto.INT4),
        ]
    )
    def test_tensor_proto_tensor_int(self, _: str, dtype: int):
        tensor_proto = onnx.helper.make_tensor("test_tensor", dtype, [1, 4], [-1, 0, 1, 8])
        tensor = serde.TensorProtoTensor(tensor_proto)
        expected_array = onnx.numpy_helper.to_array(
            tensor_proto
        )  # [-1, 0, 1, 7], 8 is clamped to 7
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        if dtype == onnx.TensorProto.INT4:
            return  # DL Pack does not support int4
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            ("UINT8", onnx.TensorProto.UINT8),
            ("UINT16", onnx.TensorProto.UINT16),
            ("UINT32", onnx.TensorProto.UINT32),
            ("UINT64", onnx.TensorProto.UINT64),
            ("UINT4", onnx.TensorProto.UINT4),
        ]
    )
    def test_tensor_proto_tensor_uint(self, _: str, dtype: int):
        tensor_proto = onnx.helper.make_tensor("test_tensor", dtype, [1, 3], [0, 1, 8])
        tensor = serde.TensorProtoTensor(tensor_proto)
        expected_array = onnx.numpy_helper.to_array(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        if dtype == onnx.TensorProto.UINT4:
            return  # DL Pack does not support uint4
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    @parameterized.parameterized.expand(
        [
            ("COMPLEX64", onnx.TensorProto.COMPLEX64, np.complex64),
            ("COMPLEX128", onnx.TensorProto.COMPLEX128, np.complex128),
        ]
    )
    def test_tensor_proto_tensor_complex(self, _: str, dtype: int, np_dtype: np.dtype):
        expected_array = np.array([[0.0 + 1j, 0.2 - 1j, 0.3]], dtype=np_dtype)
        tensor_proto = onnx.helper.make_tensor(
            "test_tensor", dtype, [1, 3], [0.0 + 1j, 0.2 - 1j, 0.3]
        )
        tensor = serde.TensorProtoTensor(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    def test_tensor_proto_tensor_empty_tensor(self):
        tensor_proto = onnx.helper.make_tensor("test_tensor", onnx.TensorProto.FLOAT, [0], [])
        tensor = serde.TensorProtoTensor(tensor_proto)
        expected_array = onnx.numpy_helper.to_array(tensor_proto)
        np.testing.assert_array_equal(tensor.numpy(), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())


class DeserializeGraphTest(unittest.TestCase):
    def test_deserialize_graph_handles_unsorted_graph(self):
        node_0 = ir.Node(
            "",
            "Op_0",
            inputs=[ir.Input("input_0"), ir.Input("input_1")],
            num_outputs=2,
            name="node_0",
        )
        node_1 = ir.Node(
            "",
            "Op_1",
            inputs=[node_0.outputs[0]],
            num_outputs=1,
            name="node_1",
        )
        graph = ir.Graph(
            inputs=node_0.inputs,  # type: ignore
            outputs=[node_1.outputs[0]],
            # Unsorted nodes
            nodes=[node_1, node_0],
            name="test_graph",
        )
        graph_proto = serde.serialize_graph(graph)
        deserialized_graph = serde.deserialize_graph(graph_proto)
        self.assertEqual(deserialized_graph[0].op_type, "Op_1")
        self.assertEqual(deserialized_graph[1].op_type, "Op_0")

    def test_deserialize_graph_handles_invalid_output(self):
        # The graph has an output that is not connected to any node, and it does not
        # have shape/type information.
        graph_with_invalid_output = ir.Graph(
            inputs=[],
            outputs=[ir.Value(name="invalid_output")],
            nodes=[],
            name="graph_with_invalid_output",
        )
        graph_proto = serde.serialize_graph(graph_with_invalid_output)
        deserialized_graph = serde.deserialize_graph(graph_proto)
        self.assertEqual(len(deserialized_graph.outputs), 1)
        self.assertEqual(deserialized_graph.outputs[0].name, "invalid_output")
        self.assertEqual(deserialized_graph.outputs[0].type, None)
        self.assertEqual(deserialized_graph.outputs[0].shape, None)
        self.assertEqual(deserialized_graph.outputs[0].dtype, None)


class QuantizationAnnotationTest(unittest.TestCase):
    """Test that quantization annotations are correctly serialized and deserialized."""

    def setUp(self):
        model_text = """\
ir_version: 8
producer_name: "pytorch"
producer_version: "2.1.1"
graph {
  input {
    name: "input"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  output {
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  node {
    input: "input"
    output: "intermediate_value"
    op_type: "TestOp1"
    domain: "test_domain"
  }
  node {
    input: "intermediate_value"
    output: "output"
    op_type: "TestOp2"
    domain: "test_domain"
  }
  quantization_annotation {
    tensor_name: "input"
    quant_parameter_tensor_names {
      key: "custom_key"
      value: "arbitrary_value_input"
    }
  }
  quantization_annotation {
    tensor_name: "intermediate_value"
    quant_parameter_tensor_names {
      key: "custom_key"
      value: "arbitrary_value_intermediate"
    }
  }
  quantization_annotation {
    tensor_name: "output"
    quant_parameter_tensor_names {
      key: "custom_key"
      value: "arbitrary_value_output"
    }
  }
}"""
        self.model = onnx.ModelProto()
        google.protobuf.text_format.Parse(model_text, self.model)

    def test_deserialize_quantization_annotation(self):
        model = serde.deserialize_model(self.model)
        self.assertEqual(
            model.graph.inputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_input"},
        )
        self.assertEqual(
            model.graph.node(0).outputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_intermediate"},
        )
        self.assertEqual(
            model.graph.outputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_output"},
        )

    def test_serde_roundtrip(self):
        model = serde.deserialize_model(self.model)
        serialized_model = serde.serialize_model(model)
        deserialized_model = serde.deserialize_model(serialized_model)
        self.assertEqual(
            deserialized_model.graph.inputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_input"},
        )
        self.assertEqual(
            deserialized_model.graph.node(0).outputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_intermediate"},
        )
        self.assertEqual(
            deserialized_model.graph.outputs[0].meta["quant_parameter_tensor_names"],
            {"custom_key": "arbitrary_value_output"},
        )


if __name__ == "__main__":
    unittest.main()
