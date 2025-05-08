# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint: disable=protected-access
import unittest

import ml_dtypes
import numpy as np
import onnx
import onnx._custom_element_types
import parameterized

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

    @parameterized.parameterized.expand(
        [
            ("string", np.array("some_string").dtype, _enums.DataType.STRING),
            ("float64", np.dtype(np.float64), _enums.DataType.DOUBLE),
            ("float32", np.dtype(np.float32), _enums.DataType.FLOAT),
            ("float16", np.dtype(np.float16), _enums.DataType.FLOAT16),
            ("int32", np.dtype(np.int32), _enums.DataType.INT32),
            ("int16", np.dtype(np.int16), _enums.DataType.INT16),
            ("int8", np.dtype(np.int8), _enums.DataType.INT8),
            ("int64", np.dtype(np.int64), _enums.DataType.INT64),
            ("uint8", np.dtype(np.uint8), _enums.DataType.UINT8),
            ("uint16", np.dtype(np.uint16), _enums.DataType.UINT16),
            ("uint32", np.dtype(np.uint32), _enums.DataType.UINT32),
            ("uint64", np.dtype(np.uint64), _enums.DataType.UINT64),
            ("bool", np.dtype(np.bool_), _enums.DataType.BOOL),
            ("complex64", np.dtype(np.complex64), _enums.DataType.COMPLEX64),
            ("complex128", np.dtype(np.complex128), _enums.DataType.COMPLEX128),
            ("bfloat16", np.dtype(ml_dtypes.bfloat16), _enums.DataType.BFLOAT16),
            ("float8e4m3fn", np.dtype(ml_dtypes.float8_e4m3fn), _enums.DataType.FLOAT8E4M3FN),
            (
                "float8e4m3fnuz",
                np.dtype(ml_dtypes.float8_e4m3fnuz),
                _enums.DataType.FLOAT8E4M3FNUZ,
            ),
            ("float8e5m2", np.dtype(ml_dtypes.float8_e5m2), _enums.DataType.FLOAT8E5M2),
            (
                "float8e5m2fnuz",
                np.dtype(ml_dtypes.float8_e5m2fnuz),
                _enums.DataType.FLOAT8E5M2FNUZ,
            ),
            ("uint4", np.dtype(ml_dtypes.uint4), _enums.DataType.UINT4),
            ("int4", np.dtype(ml_dtypes.int4), _enums.DataType.INT4),
            ("float4e2m1", np.dtype(ml_dtypes.float4_e2m1fn), _enums.DataType.FLOAT4E2M1),
            (
                "onnx_ref_bfloat16",
                onnx._custom_element_types.bfloat16,
                _enums.DataType.BFLOAT16,
            ),
            (
                "onnx_ref_float8e4m3fn",
                onnx._custom_element_types.float8e4m3fn,
                _enums.DataType.FLOAT8E4M3FN,
            ),
            (
                "onnx_ref_float8e4m3fnuz",
                onnx._custom_element_types.float8e4m3fnuz,
                _enums.DataType.FLOAT8E4M3FNUZ,
            ),
            (
                "onnx_ref_float8e5m2",
                onnx._custom_element_types.float8e5m2,
                _enums.DataType.FLOAT8E5M2,
            ),
            (
                "onnx_ref_float8e5m2fnuz",
                onnx._custom_element_types.float8e5m2fnuz,
                _enums.DataType.FLOAT8E5M2FNUZ,
            ),
            (
                "onnx_ref_uint4",
                onnx._custom_element_types.uint4,
                _enums.DataType.UINT4,
            ),
            ("onnx_ref_int4", onnx._custom_element_types.int4, _enums.DataType.INT4),
        ]
    )
    def test_from_numpy_takes_np_dtype_and_returns_data_type(
        self, _: str, np_dtype: np.dtype, onnx_type: _enums.DataType
    ):
        self.assertEqual(_enums.DataType.from_numpy(np_dtype), onnx_type)

    def test_numpy_returns_np_dtype(self):
        self.assertEqual(_enums.DataType.DOUBLE.numpy(), np.dtype(np.float64))

    def test_itemsize_returns_size_of_data_type_in_bytes(self):
        self.assertEqual(_enums.DataType.DOUBLE.itemsize, 8)
        self.assertEqual(_enums.DataType.INT4.itemsize, 0.5)

    def test_repr_and_str_return_name(self):
        self.assertEqual(str(_enums.DataType.DOUBLE), "DOUBLE")
        self.assertEqual(repr(_enums.DataType.DOUBLE), "DOUBLE")

    def test_short_name_conversion(self):
        for dtype in _enums.DataType:
            short_name = dtype.short_name()
            self.assertEqual(_enums.DataType.from_short_name(short_name), dtype)

    def test_access_by_name(self):
        self.assertEqual(_enums.DataType["FLOAT"], _enums.DataType.FLOAT)
        self.assertEqual(_enums.DataType["UINT8"], _enums.DataType.UINT8)
        self.assertEqual(_enums.DataType["INT8"], _enums.DataType.INT8)
        self.assertEqual(_enums.DataType["UINT16"], _enums.DataType.UINT16)
        self.assertEqual(_enums.DataType["INT16"], _enums.DataType.INT16)
        self.assertEqual(_enums.DataType["INT32"], _enums.DataType.INT32)
        self.assertEqual(_enums.DataType["INT64"], _enums.DataType.INT64)
        self.assertEqual(_enums.DataType["STRING"], _enums.DataType.STRING)
        self.assertEqual(_enums.DataType["BOOL"], _enums.DataType.BOOL)
        self.assertEqual(_enums.DataType["FLOAT16"], _enums.DataType.FLOAT16)
        self.assertEqual(_enums.DataType["DOUBLE"], _enums.DataType.DOUBLE)
        self.assertEqual(_enums.DataType["UINT32"], _enums.DataType.UINT32)
        self.assertEqual(_enums.DataType["UINT64"], _enums.DataType.UINT64)
        self.assertEqual(_enums.DataType["COMPLEX64"], _enums.DataType.COMPLEX64)
        self.assertEqual(_enums.DataType["COMPLEX128"], _enums.DataType.COMPLEX128)
        self.assertEqual(_enums.DataType["BFLOAT16"], _enums.DataType.BFLOAT16)
        self.assertEqual(_enums.DataType["FLOAT8E4M3FN"], _enums.DataType.FLOAT8E4M3FN)
        self.assertEqual(_enums.DataType["FLOAT8E4M3FNUZ"], _enums.DataType.FLOAT8E4M3FNUZ)
        self.assertEqual(_enums.DataType["FLOAT8E5M2"], _enums.DataType.FLOAT8E5M2)
        self.assertEqual(_enums.DataType["FLOAT8E5M2FNUZ"], _enums.DataType.FLOAT8E5M2FNUZ)
        self.assertEqual(_enums.DataType["UINT4"], _enums.DataType.UINT4)
        self.assertEqual(_enums.DataType["INT4"], _enums.DataType.INT4)
        self.assertEqual(_enums.DataType["FLOAT4E2M1"], _enums.DataType.FLOAT4E2M1)
        self.assertEqual(_enums.DataType["UNDEFINED"], _enums.DataType.UNDEFINED)


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
