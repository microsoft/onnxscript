# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx

from onnxscript.ir import _enums


class DataTypeTest(unittest.TestCase):
    def test_enums_are_the_same_as_spec(self):
        self.assertEqual(_enums.DataType.FLOAT, onnx.TensorProto.FLOAT)
        self.assertEqual(_enums.DataType.UINT8, onnx.TensorProto.UINT8)
        self.assertEqual(_enums.DataType.INT8, onnx.TensorProto.INT8)
        self.assertEqual(_enums.DataType.UINT16, onnx.TensorProto.UINT16)
        self.assertEqual(_enums.DataType.INT16, onnx.TensorProto.INT16)
        self.assertEqual(_enums.DataType.INT32, onnx.TensorProto.INT32)
        self.assertEqual(_enums.DataType.INT64, onnx.TensorProto.INT64)
        self.assertEqual(_enums.DataType.STRING, onnx.TensorProto.STRING)
        self.assertEqual(_enums.DataType.BOOL, onnx.TensorProto.BOOL)
        self.assertEqual(_enums.DataType.FLOAT16, onnx.TensorProto.FLOAT16)
        self.assertEqual(_enums.DataType.DOUBLE, onnx.TensorProto.DOUBLE)
        self.assertEqual(_enums.DataType.UINT32, onnx.TensorProto.UINT32)
        self.assertEqual(_enums.DataType.UINT64, onnx.TensorProto.UINT64)
        self.assertEqual(_enums.DataType.COMPLEX64, onnx.TensorProto.COMPLEX64)
        self.assertEqual(_enums.DataType.COMPLEX128, onnx.TensorProto.COMPLEX128)
        self.assertEqual(_enums.DataType.BFLOAT16, onnx.TensorProto.BFLOAT16)
        self.assertEqual(_enums.DataType.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FN)
        self.assertEqual(_enums.DataType.FLOAT8E4M3FNUZ, onnx.TensorProto.FLOAT8E4M3FNUZ)
        self.assertEqual(_enums.DataType.FLOAT8E5M2, onnx.TensorProto.FLOAT8E5M2)
        self.assertEqual(_enums.DataType.FLOAT8E5M2FNUZ, onnx.TensorProto.FLOAT8E5M2FNUZ)
        self.assertEqual(_enums.DataType.UINT4, onnx.TensorProto.UINT4)
        self.assertEqual(_enums.DataType.INT4, onnx.TensorProto.INT4)
        if hasattr(onnx.TensorProto, "FLOAT4E2M1"):
            self.assertEqual(_enums.DataType.FLOAT4E2M1, onnx.TensorProto.FLOAT4E2M1)
        self.assertEqual(_enums.DataType.UNDEFINED, onnx.TensorProto.UNDEFINED)

    def test_from_numpy_takes_np_dtype_and_returns_data_type(self):
        array = np.array([], dtype=np.float64)
        self.assertEqual(_enums.DataType.from_numpy(array.dtype), _enums.DataType.DOUBLE)

    def test_numpy_returns_np_dtype(self):
        self.assertEqual(_enums.DataType.DOUBLE.numpy(), np.dtype(np.float64))

    def test_itemsize_returns_size_of_data_type_in_bytes(self):
        self.assertEqual(_enums.DataType.DOUBLE.itemsize, 8)
        self.assertEqual(_enums.DataType.INT4.itemsize, 0.5)

    def test_repr_and_str_return_name(self):
        self.assertEqual(str(_enums.DataType.DOUBLE), "DOUBLE")
        self.assertEqual(repr(_enums.DataType.DOUBLE), "DOUBLE")


class AttributeTypeTest(unittest.TestCase):
    def test_enums_are_the_same_as_spec(self):
        self.assertEqual(_enums.AttributeType.FLOAT, onnx.AttributeProto.FLOAT)
        self.assertEqual(_enums.AttributeType.INT, onnx.AttributeProto.INT)
        self.assertEqual(_enums.AttributeType.STRING, onnx.AttributeProto.STRING)
        self.assertEqual(_enums.AttributeType.TENSOR, onnx.AttributeProto.TENSOR)
        self.assertEqual(_enums.AttributeType.GRAPH, onnx.AttributeProto.GRAPH)
        self.assertEqual(_enums.AttributeType.FLOATS, onnx.AttributeProto.FLOATS)
        self.assertEqual(_enums.AttributeType.INTS, onnx.AttributeProto.INTS)
        self.assertEqual(_enums.AttributeType.STRINGS, onnx.AttributeProto.STRINGS)
        self.assertEqual(_enums.AttributeType.TENSORS, onnx.AttributeProto.TENSORS)
        self.assertEqual(_enums.AttributeType.GRAPHS, onnx.AttributeProto.GRAPHS)
        self.assertEqual(_enums.AttributeType.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSOR)
        self.assertEqual(
            _enums.AttributeType.SPARSE_TENSORS, onnx.AttributeProto.SPARSE_TENSORS
        )
        self.assertEqual(_enums.AttributeType.TYPE_PROTO, onnx.AttributeProto.TYPE_PROTO)
        self.assertEqual(_enums.AttributeType.TYPE_PROTOS, onnx.AttributeProto.TYPE_PROTOS)
        self.assertEqual(_enums.AttributeType.UNDEFINED, onnx.AttributeProto.UNDEFINED)


if __name__ == "__main__":
    unittest.main()
