import unittest
from typing import Callable

import numpy as np
import onnx
import parameterized

from onnxscript import ir
from onnxscript.ir import serde


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
        np.testing.assert_array_equal(np.from_dlpack(tensor), tensor.numpy())

    def test_tensor_proto_tensor_bfloat16(self):
        expected_array = np.array([[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]])
        tensor_proto = onnx.helper.make_tensor(
            "test_tensor", onnx.TensorProto.BFLOAT16, [1, 9], expected_array
        )
        tensor = serde.TensorProtoTensor(tensor_proto)
        np.testing.assert_array_equal(
            onnx.numpy_helper.bfloat16_to_float32(tensor.numpy()), expected_array
        )
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

    @parameterized.parameterized.expand(
        [
            (
                "FLOAT8E4M3FN",
                onnx.TensorProto.FLOAT8E4M3FN,
                lambda x: onnx.numpy_helper.float8e4m3_to_float32(x, fn=True),
            ),
            (
                "FLOAT8E4M3FNUZ",
                onnx.TensorProto.FLOAT8E4M3FNUZ,
                lambda x: onnx.numpy_helper.float8e4m3_to_float32(x, fn=True, uz=True),
            ),
            (
                "FLOAT8E5M2",
                onnx.TensorProto.FLOAT8E5M2,
                onnx.numpy_helper.float8e5m2_to_float32,
            ),
            (
                "FLOAT8E5M2FNUZ",
                onnx.TensorProto.FLOAT8E5M2FNUZ,
                lambda x: onnx.numpy_helper.float8e5m2_to_float32(x, fn=True, uz=True),
            ),
        ]
    )
    def test_tensor_proto_tensor_float8(self, _: str, dtype: int, to_float32_func: Callable):
        expected_array = np.array([[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 40.0, 2.0]])
        tensor_proto = onnx.helper.make_tensor("test_tensor", dtype, [1, 9], expected_array)
        tensor = serde.TensorProtoTensor(tensor_proto)
        np.testing.assert_array_equal(to_float32_func(tensor.numpy()), expected_array)
        raw_data = tensor.tobytes()
        tensor_proto_from_raw_data = onnx.TensorProto(
            dims=tensor_proto.dims,
            data_type=tensor_proto.data_type,
            raw_data=raw_data,
        )
        if dtype in (onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FNUZ):
            # TODO: Remove the fix when ONNX 1.17 releases
            self.skipTest("ONNX to_array fails: https://github.com/onnx/onnx/pull/6124")
        array_from_raw_data = onnx.numpy_helper.to_array(tensor_proto_from_raw_data)
        np.testing.assert_array_equal(array_from_raw_data, expected_array)
        # Test dlpack
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


if __name__ == "__main__":
    unittest.main()
