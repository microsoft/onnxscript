import unittest

import numpy as np
import onnx
import parameterized

from onnxscript.ir import serde


class TensorProtoTensorTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("FLOAT", onnx.TensorProto.FLOAT),
            ("BOOL", onnx.TensorProto.BOOL),
            ("FLOAT16", onnx.TensorProto.FLOAT16),
            ("DOUBLE", onnx.TensorProto.DOUBLE),
            ("BFLOAT16", onnx.TensorProto.BFLOAT16),
            ("FLOAT8E4M3FN", onnx.TensorProto.FLOAT8E4M3FN),
            ("FLOAT8E4M3FNUZ", onnx.TensorProto.FLOAT8E4M3FNUZ),
            ("FLOAT8E5M2", onnx.TensorProto.FLOAT8E5M2),
            ("FLOAT8E5M2FNUZ", onnx.TensorProto.FLOAT8E5M2FNUZ),
        ]
    )
    def test_tensor_proto_tensor(self, _: str, dtype: int):
        if dtype in (onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FNUZ):
            # TODO: Remove the fix when ONNX 1.17 releases
            self.skipTest("ONNX to_array fails: https://github.com/onnx/onnx/pull/6124")
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

    @parameterized.parameterized.expand(
        [
            ("COMPLEX64", onnx.TensorProto.COMPLEX64),
            ("COMPLEX128", onnx.TensorProto.COMPLEX128),
        ]
    )
    @unittest.skip(
        # TODO: Remove the fix when ONNX 1.17 releases
        "numpy_helper.to_array fails on complex numbers: https://github.com/onnx/onnx/pull/6124"
    )
    def test_tensor_proto_tensor_complex(self, _: str, dtype: int):
        tensor_proto = onnx.helper.make_tensor(
            "test_tensor", dtype, [1, 3], [0.0 + 1j, 0.2 - 1j, 0.3]
        )
        expected_array = onnx.numpy_helper.to_array(tensor_proto)
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
