# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest
from typing import Any

import ml_dtypes
import numpy as np
import onnx
import onnx.external_data_helper
import parameterized
import torch

from onnxscript import ir
from onnxscript.ir import _core


class TensorTest(unittest.TestCase):
    def test_initialize(self):
        tensor = _core.Tensor(
            np.random.rand(1, 2).astype(np.float32),
            dtype=ir.DataType.FLOAT,
            shape=_core.Shape((1, 2)),
            name="test",
        )
        self.assertEqual(tensor.name, "test")
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        self.assertEqual(tensor.shape, _core.Shape((1, 2)))
        np.testing.assert_array_equal(tensor, tensor)

    def test_init_raises_when_value_is_not_array(self):
        with self.assertRaises(TypeError):
            _core.Tensor(42)

    def test_init_requires_type_when_value_is_not_np_array(self):
        torch_tensor = torch.tensor(42)
        with self.assertRaises(ValueError):
            _core.Tensor(torch_tensor)

    @parameterized.parameterized.expand(
        [
            ("bfloat16", np.uint16, ir.DataType.BFLOAT16),
            (
                "float8e4m3fn",
                np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)})),
                ir.DataType.FLOAT8E4M3FN,
            ),
            ("float8e4m3fnuz", np.uint8, ir.DataType.FLOAT8E4M3FNUZ),
            ("float8e5m2", np.uint8, ir.DataType.FLOAT8E5M2),
            ("float8e5m2fnuz", np.uint8, ir.DataType.FLOAT8E5M2FNUZ),
            ("int4", np.int8, ir.DataType.INT4),
            ("int4_uint8", np.uint8, ir.DataType.INT4),
            ("uint4", np.uint8, ir.DataType.UINT4),
            ("float4e2m1", np.uint8, ir.DataType.FLOAT4E2M1),
        ]
    )
    def test_init_with_non_native_numpy_dtype(self, _: str, np_dtype, dtype: ir.DataType):
        array = np.array([0b1, 0b11], dtype=np_dtype)
        tensor = _core.Tensor(array, dtype=dtype)
        self.assertEqual(tensor.dtype, dtype)
        np.testing.assert_array_equal(tensor, array.view(dtype.numpy()))

    def test_initialize_with_just_np_array(self):
        array = np.random.rand(1, 2)
        tensor = _core.Tensor(array)
        np.testing.assert_array_equal(tensor, array)

    def test_initialize_raises_when_numpy_dtype_doesnt_match(self):
        array = np.random.rand(1, 2).astype(np.float32)
        with self.assertRaises(TypeError):
            _core.Tensor(array, dtype=ir.DataType.INT64)

    def test_initialize_supports_custom_dtype(self):
        custom_dtype = np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)}))
        array = np.random.rand(1, 2).astype(custom_dtype)
        _core.Tensor(array, dtype=ir.DataType.FLOAT8E4M3FN)

    def test_initialize_raises_when_numpy_dtype_doesnt_match_custom_dtype(self):
        custom_dtype = np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)}))
        array = np.random.rand(1, 2).astype(custom_dtype)
        with self.assertRaises(TypeError):
            _core.Tensor(array, dtype=ir.DataType.BFLOAT16)

    def test_initialize_with_torch_tensor(self):
        array = np.random.rand(1, 2).astype(np.int64)
        np_tensor = _core.Tensor(array)
        torch_tensor = _core.Tensor(torch.tensor(array), dtype=ir.DataType.INT64)
        np.testing.assert_array_equal(torch_tensor, array)
        np.testing.assert_array_equal(torch_tensor, np_tensor)

    def test_dlpack_np_to_torch(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        torch_tensor = torch.from_dlpack(tensor)
        np.testing.assert_array_equal(torch_tensor, array)

    def test_dlpack_torch_to_np(self):
        torch_tensor = torch.rand(1, 2)
        tensor = _core.Tensor(torch_tensor, dtype=ir.DataType.FLOAT)
        array = np.from_dlpack(tensor)
        np.testing.assert_array_equal(array, torch_tensor)

    def test_repr(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertIsInstance(repr(tensor), str)

    def test_dtype_returns_data_type_enum(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)

    def test_shape(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertEqual(tensor.shape, _core.Shape((1, 2)))

    def test_numpy_returns_np_array(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        np.testing.assert_equal(tensor.numpy(), array)

    def test_numpy_returns_data_when_dtype_is_not_supported(self):
        array = np.array([1], dtype=np.uint8)
        tensor = _core.Tensor(array, dtype=ir.DataType.INT4)
        np.testing.assert_equal(tensor.numpy(), array)

    def test_tobytes(self):
        array = np.random.rand(1, 2).astype(np.float32)
        torch_tensor = torch.tensor(array)
        tensor = _core.Tensor(torch_tensor, dtype=ir.DataType.FLOAT)
        self.assertEqual(tensor.tobytes(), array.tobytes())

    def test_tobytes_returns_packed_data_for_int4(self):
        array = np.array([-8, -1, 0, 1, 2, 7, 1], dtype=np.int8)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.INT4)
        self.assertEqual(tensor.tobytes(), b"\xf8\x10r\x01")

    def test_tobytes_returns_packed_data_for_int4_ml_dtypes(self):
        array = np.array([-8, -1, 0, 1, 2, 7, 1], dtype=ml_dtypes.int4)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.INT4)
        self.assertEqual(tensor.tobytes(), b"\xf8\x10r\x01")

    def test_tobytes_returns_packed_data_for_uint4(self):
        array = np.array([0, 1, 2, 7, 15], dtype=np.uint8)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.UINT4)
        self.assertEqual(tensor.tobytes(), b"\x10r\x0f")

    def test_tobytes_returns_packed_data_for_uint4_ml_dtypes(self):
        array = np.array([0, 1, 2, 7, 15], dtype=ml_dtypes.uint4)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.UINT4)
        self.assertEqual(tensor.tobytes(), b"\x10r\x0f")

    def test_tobytes_returns_packed_data_for_float4e2m1(self):
        array = np.array([0, 1, 2, 7, 15], dtype=np.uint8)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.FLOAT4E2M1)
        self.assertEqual(tensor.tobytes(), b"\x10r\x0f")

    def test_tobytes_returns_packed_data_for_float4e2m1_ml_dtypes(self):
        array = np.array([0, 1, 2, 7, 15], dtype=np.uint8)
        # Test odd sized array
        assert len(array) % 2 == 1
        tensor = _core.Tensor(array, dtype=ir.DataType.FLOAT4E2M1)
        self.assertEqual(tensor.tobytes(), b"\x10r\x0f")

    def test_metadata(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        tensor.meta["test"] = 1
        self.assertEqual(tensor.meta["test"], 1)
        tensor.metadata_props["test"] = "any string"
        self.assertEqual(tensor.metadata_props["test"], "any string")


def _to_external_tensor(tensor_proto, dir: str, filename: str):
    onnx.external_data_helper.set_external_data(tensor_proto, location=filename)
    path = pathlib.Path(dir) / filename
    with open(path, "wb") as f:
        f.write(tensor_proto.raw_data)
    tensor_proto.ClearField("raw_data")
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL


class ExternalTensorTest(unittest.TestCase):
    """Test the memory mapped external tensor class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.external_data_name = "test_model.bin"
        self.base_path = self.temp_dir.name
        self.data = np.random.rand(2, 42).astype(np.float32)
        self.data_float16 = np.random.rand(2, 42).astype(np.float16)
        self.model = self._simple_model_with_external(
            self.base_path, self.external_data_name, self.data
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _simple_model_with_external(
        self, base_path: str, external_data_name: str, data: np.ndarray
    ) -> onnx.ModelProto:
        input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None])
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None])
        raw_data = data.tobytes()
        tensor = onnx.helper.make_tensor(
            "input", onnx.TensorProto.FLOAT, data.shape, raw_data, raw=True
        )
        raw_data2 = self.data_float16.tobytes()
        tensor2 = onnx.helper.make_tensor(
            "input2", onnx.TensorProto.FLOAT16, data.shape, raw_data2, raw=True
        )
        onnx.external_data_helper.set_external_data(
            tensor, external_data_name, offset=0, length=len(raw_data)
        )
        onnx.external_data_helper.set_external_data(
            tensor2, external_data_name, offset=len(raw_data), length=len(raw_data2)
        )

        node = onnx.helper.make_node("Identity", inputs=["input"], outputs=["output"])
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [node], "test_graph", [input], [output], initializer=[tensor, tensor2]
            )
        )
        tensor.ClearField("raw_data")
        tensor2.ClearField("raw_data")
        # Save the data to disk
        with open(pathlib.Path(base_path) / external_data_name, "wb") as f:
            f.write(raw_data)
            f.write(raw_data2)
        return model

    def test_initialize(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        np.testing.assert_equal(tensor, self.data)
        # Ensure repeated reads are consistent
        np.testing.assert_equal(tensor, self.data)

    def test_release_does_not_invalidate_tensor(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        self.assertEqual(tensor.tobytes(), self.data.tobytes())
        # Release tensor
        tensor.release()
        self.assertEqual(tensor.raw, None)
        # Tensor can be re-loaded after release
        self.assertEqual(tensor.tobytes(), self.data.tobytes())

    def test_initialize_with_relative_path(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            name="input",
            shape=_core.Shape(external_tensor.dims),
            base_dir=pathlib.Path(self.base_path),
        )
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        np.testing.assert_equal(tensor, self.data)
        # Ensure repeated reads are consistent
        np.testing.assert_equal(tensor, self.data)

    def test_totypes_returns_correct_data_in(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        external_tensor2 = self.model.graph.initializer[1]
        external_info2 = onnx.external_data_helper.ExternalDataInfo(external_tensor2)
        tensor2 = _core.ExternalTensor(
            external_info2.location,
            offset=external_info2.offset,
            length=external_info2.length,
            dtype=ir.DataType.FLOAT16,
            base_dir=self.base_path,
            name="input",
            shape=_core.Shape(external_tensor2.dims),
        )
        self.assertEqual(tensor.tobytes(), self.data.tobytes())
        self.assertEqual(tensor2.tobytes(), self.data_float16.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(tensor.tobytes(), self.data.tobytes())
        self.assertEqual(tensor2.tobytes(), self.data_float16.tobytes())

    @parameterized.parameterized.expand(
        [
            ("FLOAT", ir.DataType.FLOAT),
            ("BOOL", ir.DataType.BOOL),
            ("FLOAT16", ir.DataType.FLOAT16),
            ("DOUBLE", ir.DataType.DOUBLE),
        ]
    )
    def test_external_tensor(self, _: str, dtype: ir.DataType):
        expected_array = np.array(
            [[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]]
        ).astype(dtype.numpy())
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array, dtype=dtype))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    def test_external_tensor_bfloat16(self):
        expected_array = np.array(
            [[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 42.0, 2.0]]
        ).astype(ml_dtypes.bfloat16)
        tensor_proto = ir.serde.serialize_tensor(
            ir.Tensor(expected_array.view(np.uint16), dtype=ir.DataType.BFLOAT16)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(
                tensor.numpy().view(ml_dtypes.bfloat16), expected_array
            )
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    @parameterized.parameterized.expand(
        [
            (
                "FLOAT8E4M3FN",
                ir.DataType.FLOAT8E4M3FN,
                ml_dtypes.float8_e4m3fn,
            ),
            (
                "FLOAT8E4M3FNUZ",
                ir.DataType.FLOAT8E4M3FNUZ,
                ml_dtypes.float8_e4m3fnuz,
            ),
            (
                "FLOAT8E5M2",
                ir.DataType.FLOAT8E5M2,
                ml_dtypes.float8_e5m2,
            ),
            (
                "FLOAT8E5M2FNUZ",
                ir.DataType.FLOAT8E5M2FNUZ,
                ml_dtypes.float8_e5m2fnuz,
            ),
        ]
    )
    def test_external_tensor_float8(self, _: str, dtype: ir.DataType, np_dtype):
        expected_array = np.array(
            [[-3.0, -1.0, -0.5, -0.0, +0.0, 0.5, 1.0, 40.0, 2.0]]
        ).astype(np_dtype)
        tensor_proto = ir.serde.serialize_tensor(
            ir.Tensor(expected_array.view(np.uint8), dtype=dtype)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy().view(np_dtype), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    @parameterized.parameterized.expand(
        [
            ("INT8", ir.DataType.INT8),
            ("INT16", ir.DataType.INT16),
            ("INT32", ir.DataType.INT32),
            ("INT64", ir.DataType.INT64),
            ("INT4", ir.DataType.INT4),
        ]
    )
    def test_external_tensor_int(self, _: str, dtype: ir.DataType):
        expected_array = np.array([[-8, 0, 1, 7]]).astype(dtype.numpy())
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array, dtype=dtype))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    @parameterized.parameterized.expand(
        [
            ("UINT8", ir.DataType.UINT8),
            ("UINT16", ir.DataType.UINT16),
            ("UINT32", ir.DataType.UINT32),
            ("UINT64", ir.DataType.UINT64),
            ("UINT4", ir.DataType.UINT4),
        ]
    )
    def test_external_tensor_uint(self, _: str, dtype: ir.DataType):
        expected_array = np.array([[0, 1, 15]]).astype(dtype.numpy())
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array, dtype=dtype))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    @parameterized.parameterized.expand(
        [
            ("COMPLEX64", np.complex64),
            ("COMPLEX128", np.complex128),
        ]
    )
    def test_external_tensor_complex(self, _: str, np_dtype: np.dtype):
        expected_array = np.array([[0.0 + 1j, 0.2 - 1j, 0.3]], dtype=np_dtype)
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    def test_external_tensor_float4e2m1(self):
        expected_array = np.array([0, 1, 2, 7, 15]).view(ml_dtypes.float4_e2m1fn)
        tensor_proto = ir.serde.serialize_tensor(
            ir.Tensor(expected_array, dtype=ir.DataType.FLOAT4E2M1)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor

    def test_external_tensor_empty_tensor(self):
        expected_array = np.array([], dtype=np.float32)
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor


class SymbolicDimTest(unittest.TestCase):
    def test_init_raises_when_value_is_int(self):
        # Static dimensions should be python integers
        with self.assertRaises(TypeError):
            _core.SymbolicDim(42)

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_equality_with_other_dimensions(self, _: str, value: Any):
        dim1 = _core.SymbolicDim(value)
        dim2 = _core.SymbolicDim(value)
        self.assertEqual(dim1, dim2)

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_equality_with_python_values(self, _: str, value: Any):
        dim = _core.SymbolicDim(value)
        self.assertEqual(dim, value)
        self.assertIn(value, [dim])
        self.assertIn(dim, [value])

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_it_is_hashable(self, _: str, value: Any):
        dim = _core.SymbolicDim(value)
        self.assertEqual(hash(dim), hash(value))
        self.assertIn(dim, {dim})
        self.assertIn(dim, {value})


class ShapeTest(unittest.TestCase):
    def test_init_raises_when_denotations_and_dims_have_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "denotations"):
            _core.Shape([42], ["DATA_CHANNEL", "BATCH"])

    def test_int_dimensions_are_python_ints(self):
        shape = _core.Shape([42])
        self.assertIsInstance(shape[0], int)

    def test_str_dimensions_are_symbolic_dims(self):
        shape = _core.Shape(["any string"])
        self.assertIsInstance(shape[0], _core.SymbolicDim)

    def test_none_dimensions_are_symbolic_dims(self):
        shape = _core.Shape([None])
        self.assertIsInstance(shape[0], _core.SymbolicDim)

    def test_init_raises_when_dims_is_not_a_list(self):
        with self.assertRaises(TypeError):
            _core.Shape(42)

    def test_init_converts_np_shape_to_tuple(self):
        dims = np.array([42, 42])
        shape = _core.Shape(dims)
        self.assertEqual(shape.dims, tuple(dims))

    def test_init_converts_np_int_to_python_int(self):
        dims = [np.int32(42)]
        shape = _core.Shape(dims)
        self.assertIsInstance(shape[0], int)
        self.assertNotIsInstance(shape[0], np.int32)
        self.assertIsInstance(shape.dims[0], int)

    @parameterized.parameterized.expand(
        [
            ("empty", (), ()),
            ("1d", (42,), (42,)),
            ("int", (42, 42), (42, 42)),
            ("str", ("any string", "any string"), ("any string", "any string")),
            ("None", (None, None), (None, None)),
        ]
    )
    def test_eq_with_other_shapes(
        self, _: str, dims_1: tuple[Any, ...], dims_2: tuple[Any, ...]
    ):
        shape_1 = _core.Shape(dims_1)
        shape_2 = _core.Shape(dims_2)
        self.assertEqual(shape_1, shape_2)

    @parameterized.parameterized.expand(
        [
            ("empty", ()),
            ("1d", (42,)),
            ("int", (42, 42)),
            ("str", ("any string", "any string")),
            ("None", (None, None)),
        ]
    )
    def test_eq_with_tuple(self, _: str, dims: tuple[Any, ...]):
        shape = _core.Shape(dims)
        self.assertEqual(shape, dims)

    @parameterized.parameterized.expand(
        [
            ("empty", []),
            (
                "1d",
                [
                    42,
                ],
            ),
            ("int", [42, 42]),
            ("str", ["any string", "any string"]),
            ("None", [None, None]),
        ]
    )
    def test_eq_with_list(self, _: str, dims: list[Any]):
        shape = _core.Shape(dims)
        self.assertEqual(shape, dims)

    def test_eq_with_np_shape(self):
        dims = (42,)
        array = np.zeros(dims)
        shape = _core.Shape(dims)
        self.assertEqual(shape, array.shape)

    @parameterized.parameterized.expand(
        [
            ("empty", (), (1,)),
            ("d", (42,), (0,)),
            ("rank", (42, 42), (42, 42, 42)),
            ("str", ("any string",), (42,)),
            ("None", (None, None), (None, 42)),
        ]
    )
    def test_ne_with_other_shapes(
        self, _: str, dims_1: tuple[Any, ...], dims_2: tuple[Any, ...]
    ):
        shape_1 = _core.Shape(dims_1)
        shape_2 = _core.Shape(dims_2)
        self.assertNotEqual(shape_1, shape_2)

    def test_ne_with_random_object(self):
        shape = _core.Shape((42,))
        self.assertNotEqual(shape, 42)

    def test_setitem_raises_when_shape_is_frozen(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",), frozen=True)
        with self.assertRaisesRegex(TypeError, "frozen"):
            shape[0] = 1

    def test_getitem(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",))
        self.assertEqual(shape[0], 42)

    def test_getitem_accepts_a_slice(self):
        shape = _core.Shape([1, 2, 3, 4])
        self.assertEqual(shape[1:3], (2, 3))

    @parameterized.parameterized.expand(
        [
            ("int", 42),
            ("str", "any string"),
            ("None", None),
            ("SymbolicDim", _core.SymbolicDim("any string")),
        ]
    )
    def test_setitem(self, _: str, value):
        shape = _core.Shape([0])
        shape[0] = value
        dim = shape[0]
        if isinstance(dim, _core.SymbolicDim):
            self.assertEqual(dim.value, value)
        else:
            self.assertEqual(dim, value)

    def test_len(self):
        shape = _core.Shape([42, "any string"])
        self.assertEqual(len(shape), 2)

    def test_get_denotation(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",))
        self.assertEqual(shape.get_denotation(0), "DATA_CHANNEL")

    def test_set_denotation(self):
        shape = _core.Shape([42, 0], ["DATA_CHANNEL", "BATCH"])
        shape.set_denotation(1, "UPDATED")
        self.assertEqual(shape.get_denotation(1), "UPDATED")

    def test_set_denotation_is_still_possible_when_shape_is_frozen(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",), frozen=True)
        shape.set_denotation(0, "UPDATED")
        self.assertEqual(shape.get_denotation(0), "UPDATED")

    def test_is_static(self):
        dim_from_numpy = np.array([42]).shape[0]
        np_int = np.int32(42)
        shape = _core.Shape([42, "any string", dim_from_numpy, np_int])
        self.assertTrue(shape.is_static(0))
        self.assertFalse(shape.is_static(1))
        self.assertTrue(shape.is_static(2))
        self.assertTrue(shape.is_static(3))
        self.assertFalse(shape.is_static())

    def test_is_static_raises_when_index_out_of_range(self):
        shape = _core.Shape([42])
        with self.assertRaises(IndexError):
            shape.is_static(1)

    def test_is_static_on_whole_shape(self):
        shape = _core.Shape([42, "any string"])
        self.assertFalse(shape.is_static())
        shape = _core.Shape([42, 42])
        self.assertTrue(shape.is_static())

    def test_is_static_on_empty_shape(self):
        shape = _core.Shape(())
        self.assertTrue(shape.is_static())

    def test_is_dynamic(self):
        dim_from_numpy = np.array([42]).shape[0]
        np_int = np.int32(42)
        shape = _core.Shape([42, "any string", dim_from_numpy, np_int])
        self.assertFalse(shape.is_dynamic(0))
        self.assertTrue(shape.is_dynamic(1))
        self.assertFalse(shape.is_dynamic(2))
        self.assertFalse(shape.is_dynamic(3))
        self.assertTrue(shape.is_dynamic())

    def test_is_dynamic_raises_when_index_out_of_range(self):
        shape = _core.Shape([42])
        with self.assertRaises(IndexError):
            shape.is_dynamic(1)

    def test_is_dynamic_on_whole_shape(self):
        shape = _core.Shape([42, "any string"])
        self.assertTrue(shape.is_dynamic())
        shape = _core.Shape([42, 42])
        self.assertFalse(shape.is_dynamic())

    def test_is_dynamic_on_empty_shape(self):
        shape = _core.Shape(())
        self.assertFalse(shape.is_dynamic())


class ValueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = _core.Value(name="v0")
        self.v1 = _core.Value(name="v1")
        self.node = _core.Node(
            "test", "TestOp", inputs=(self.v0, self.v1, self.v1), num_outputs=2
        )

    def test_initialize(self):
        _ = _core.Value()

    def test_it_is_hashable(self):
        value = _core.Value()
        self.assertIsInstance(hash(value), int)
        self.assertIn(value, {value})

    def test_meta(self):
        value = _core.Value()
        value.meta["test"] = 1
        self.assertEqual(value.meta["test"], 1)
        value.metadata_props["test"] = "any string"
        self.assertEqual(value.metadata_props["test"], "any string")

    def test_producer(self):
        self.assertEqual(self.v0.producer(), None)
        self.assertEqual(self.v1.producer(), None)
        self.assertEqual(self.node.outputs[0].producer(), self.node)
        self.assertEqual(self.node.outputs[1].producer(), self.node)

    def test_consumers(self):
        self.assertEqual(self.v0.consumers(), (self.node,))
        self.assertEqual(self.v1.consumers(), (self.node,))
        self.assertEqual(self.node.outputs[0].consumers(), ())
        self.assertEqual(self.node.outputs[1].consumers(), ())

    # TODO(justinchuby): Test all methods


class NodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = _core.Value(name="v0")
        self.v1 = _core.Value(name="v1")
        self.node = _core.Node(
            "test", "TestOp", inputs=(self.v0, self.v1, self.v1), num_outputs=3
        )
        self.node_a = _core.Node("test", "TestOpA", inputs=[self.node.outputs[0]])
        self.node_b = _core.Node("test", "TestOpB", inputs=self.node.outputs)

    def test_it_is_hashable(self):
        self.assertIsInstance(hash(self.node), int)
        self.assertIn(self.node, {self.node})

    def test_init_with_values(self):
        self.assertEqual(self.node.domain, "test")
        self.assertEqual(self.node.op_type, "TestOp")
        self.assertEqual(self.node.inputs, (self.v0, self.v1, self.v1))
        self.assertEqual(len(self.node.outputs), 3)
        self.assertEqual(self.node.attributes, {})

    def test_init_with_preinitialized_outputs(self):
        out_1 = _core.Value(
            name="out_1",
            shape=_core.Shape([1]),
            type=_core.TensorType(ir.DataType.BFLOAT16),
        )
        out_2 = _core.Value(
            name="out_2",
            shape=_core.Shape([2]),
            type=_core.TensorType(ir.DataType.INT4),
        )
        node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1), outputs=[out_1, out_2])
        self.assertEqual(node.outputs[0].name, "out_1")
        self.assertEqual(node.outputs[0].shape, _core.Shape([1]))
        self.assertEqual(node.outputs[0].dtype, ir.DataType.BFLOAT16)
        self.assertEqual(node.outputs[1].name, "out_2")
        self.assertEqual(node.outputs[1].shape, _core.Shape([2]))
        self.assertEqual(node.outputs[1].dtype, ir.DataType.INT4)
        self.assertIs(node.outputs[0], out_1)
        self.assertIs(node.outputs[1], out_2)
        self.assertIs(node.outputs[0].producer(), node)
        self.assertIs(node.outputs[1].producer(), node)
        self.assertIs(node.outputs[0].index(), 0)
        self.assertIs(node.outputs[1].index(), 1)

    def test_init_raises_when_num_outputs_does_not_match_outputs(self):
        with self.assertRaisesRegex(ValueError, "outputs"):
            _core.Node("test", "TestOp", inputs=(self.v0, self.v1), num_outputs=2, outputs=[])

    def test_init_with_zero_num_outputs(self):
        node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1), num_outputs=0)
        self.assertEqual(node.outputs, ())

    def test_init_with_empty_outputs(self):
        node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1), outputs=[])
        self.assertEqual(node.outputs, ())

    def test_init_produces_one_output_with_unspecified_output_argument(self):
        node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1))
        self.assertEqual(len(node.outputs), 1)

    def test_metadata(self):
        self.node.meta["test"] = 1
        self.assertEqual(self.node.meta["test"], 1)
        self.node.metadata_props["test"] = "any string"
        self.assertEqual(self.node.metadata_props["test"], "any string")

    def test_it_is_added_to_a_graph_if_specified(self):
        graph = _core.Graph(
            (self.v0, self.v1),  # type: ignore
            self.node.outputs,
            nodes=(self.node,),
        )
        self.assertIn(self.node, graph)

    def test_predecessors(self):
        self.assertEqual(self.node.predecessors(), ())
        self.assertEqual(self.node_a.predecessors(), (self.node,))
        self.assertEqual(self.node_b.predecessors(), (self.node,))

    def test_predecessors_are_unique(self):
        # node_b has three inputs from node, but only one predecessor
        self.assertEqual(self.node_b.predecessors(), (self.node,))

    def test_successors(self):
        self.assertEqual(self.node.successors(), (self.node_a, self.node_b))
        self.assertEqual(self.node_a.successors(), ())
        self.assertEqual(self.node_b.successors(), ())

    def test_successors_are_unique(self):
        self.assertEqual(self.node.successors(), (self.node_a, self.node_b))

    # TODO(justinchuby): Test all methods


class GraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = _core.Value(name="v0")
        self.v1 = _core.Value(name="v1")
        self.node = _core.Node(
            "", "Add", inputs=(self.v0, self.v1), num_outputs=1, name="node_add"
        )
        self.graph = _core.Graph(
            (self.v0, self.v1),
            self.node.outputs,
            nodes=(self.node,),
            opset_imports={"": 1},
        )

    def test_initialize(self):
        self.assertEqual(self.graph.inputs, [self.v0, self.v1])
        self.assertEqual(self.graph.outputs, [*self.node.outputs])
        self.assertEqual(self.graph.opset_imports, {"": 1})
        self.assertEqual(self.graph.initializers, {})
        self.assertIsNone(self.graph.doc_string)

    def test_it_is_hashable(self):
        self.assertIsInstance(hash(self.graph), int)
        self.assertIn(self.graph, {self.graph})

    def test_it_is_iterable_of_nodes(self):
        self.assertEqual(list(self.graph), [self.node])

    def test_node_returns_node_by_name(self):
        self.assertIs(self.graph.node("node_add"), self.node)

    def test_node_returns_node_by_index(self):
        self.assertIs(self.graph.node(0), self.node)

    def test_node_raises_when_node_does_not_exist(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            self.graph.node("non_existent")

    def test_node_raises_when_index_out_of_range(self):
        with self.assertRaises(IndexError):
            self.graph.node(1)

    def test_num_nodes_returns_the_count_of_nodes(self):
        self.assertEqual(self.graph.num_nodes(), 1)
        self.assertEqual(self.graph.num_nodes(), len(self.graph))

    def test_metadata(self):
        self.graph.meta["test"] = 1
        self.assertEqual(self.graph.meta["test"], 1)
        self.graph.metadata_props["test"] = "any string"
        self.assertEqual(self.graph.metadata_props["test"], "any string")

    def test_remove_removes_node_from_graph(self):
        self.graph.remove(self.node)
        self.assertEqual(list(self.graph), [])
        self.assertIsNone(self.node.graph)

    def test_remove_does_not_change_input_users(self):
        self.graph.remove(self.node)
        self.assertEqual(tuple(self.v0.uses()), ((self.node, 0),))
        self.assertEqual(tuple(self.v1.uses()), ((self.node, 1),))

    def test_remove_does_not_change_graph_in_out(self):
        self.graph.remove(self.node)
        self.assertEqual(self.graph.inputs, [self.v0, self.v1])
        self.assertEqual(self.graph.outputs, list(self.node.outputs))

    def test_remove_raises_when_node_does_not_belong_to_graph(self):
        node = _core.Node("", "Add", inputs=(self.v0, self.v1), num_outputs=1)
        with self.assertRaisesRegex(ValueError, "graph"):
            self.graph.remove(node)

    def test_remove_safe_raises_when_node_output_is_graph_output(self):
        with self.assertRaisesRegex(ValueError, "output"):
            self.graph.remove(self.node, safe=True)

    def test_remove_safe_raises_when_node_has_users(self):
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        add_node = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1)
        identity_node = _core.Node("", "Identity", inputs=add_node.outputs, num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            identity_node.outputs,
            nodes=(add_node, identity_node),
            opset_imports={"": 1},
        )
        with self.assertRaisesRegex(ValueError, "used by other nodes"):
            graph.remove(add_node, safe=True)

    def test_remove_safe_removes_uses_of_removed_nodes(self):
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        add_node = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1)
        identity_node = _core.Node("", "Identity", inputs=add_node.outputs, num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            identity_node.outputs,
            nodes=(add_node, identity_node),
            opset_imports={"": 1},
        )
        # Remove add_node and check that it is no longer a consumer of v0 and v1
        sub_node = _core.Node("", "Sub", inputs=(v0, v1), num_outputs=1)
        identity_node.replace_input_with(0, sub_node.outputs[0])
        graph.insert_before(identity_node, sub_node)
        graph.remove(add_node, safe=True)
        self.assertEqual(tuple(v0.uses()), ((sub_node, 0),))
        self.assertEqual(tuple(v1.uses()), ((sub_node, 1),))
        self.assertEqual(tuple(graph), (sub_node, identity_node))
        self.assertEqual(add_node.inputs, (None, None))

    def test_register_initializer(self):
        self.v1.const_value = ir.tensor([1, 2, 3])
        self.graph.register_initializer(self.v1)
        self.assertEqual(self.graph.initializers, {self.v1.name: self.v1})

    def test_register_initializer_raises_when_value_is_not_constant(self):
        with self.assertRaises(ValueError):
            self.graph.register_initializer(self.v0)

    def test_register_initializer_raises_when_a_different_value_is_already_registered(self):
        self.v1.const_value = ir.tensor([1, 2, 3])
        self.graph.register_initializer(self.v1)
        # This is fine
        self.graph.register_initializer(self.v1)
        self.v0.name = "v1"
        with self.assertRaisesRegex(ValueError, "already registered"):
            # Registering a different value with the same name should raise
            self.graph.register_initializer(self.v0)

    def test_register_initializer_raises_when_value_does_not_have_a_name(self):
        self.v1.name = None
        with self.assertRaises(ValueError):
            self.graph.register_initializer(self.v1)

    # TODO(justinchuby): Test graph mutation methods

    # Test topological sort.
    # Graph structure:
    #   nodes: [node, ...]
    #   edges: [(predecessor_node, successor_node), ...]
    #   subgraphs: {node: [subgraph, ...]}

    def test_topological_sort_empty_graph(self):
        graph = _core.Graph(
            inputs=(),
            outputs=(),
            nodes=(),
        )
        graph.sort()
        self.assertEqual(tuple(graph), ())

    def test_topological_sort_linear_dependencies(self):
        # nodes=[1,2,3], edges=[(1,2),(2,3)]
        v0 = _core.Value(name="v0")
        node1 = _core.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(node1.outputs[0],), num_outputs=1)
        node3 = _core.Node("", "Node3", inputs=(node2.outputs[0],), num_outputs=1)
        graph = _core.Graph(
            (v0,),
            node3.outputs,
            nodes=(node3, node2, node1),
        )
        graph.sort()
        sorted_nodes = tuple(graph)
        expected_order = (node1, node2, node3)
        self.assertEqual(sorted_nodes, expected_order)

    def test_topological_sort_independent_subgraphs(self):
        # nodes=[1,2,3,4], edges=[(1,3),(2,4)]
        v0 = _core.Value(name="v0")
        v1 = _core.Value(name="v1")
        node1 = _core.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(v1,), num_outputs=1)
        node3 = _core.Node("", "Node3", inputs=(node1.outputs[0],), num_outputs=1)
        node4 = _core.Node("", "Node4", inputs=(node2.outputs[0],), num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            (node3.outputs[0], node4.outputs[0]),
            nodes=(node4, node3, node2, node1),
        )
        graph.sort()
        sorted_nodes = tuple(graph)
        expected_order = (node2, node4, node1, node3)
        self.assertEqual(sorted_nodes, expected_order)

    def test_topological_sort_shared_successor(self):
        # nodes=[1,2,3], edges=[(1,3),(2,3)]
        v0 = _core.Value(name="v0")
        node1 = _core.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(v0,), num_outputs=1)
        node3 = _core.Node(
            "", "Node3", inputs=(node1.outputs[0], node2.outputs[0]), num_outputs=1
        )
        graph = _core.Graph(
            (v0,),
            (node3.outputs[0],),
            nodes=(node3, node2, node1),
        )
        graph.sort()
        sorted_nodes = tuple(graph)
        expected_order = (node2, node1, node3)
        self.assertEqual(sorted_nodes, expected_order)

    def _create_shared_predecessor_nodes(
        self,
    ) -> tuple[_core.Value, tuple[_core.Node, _core.Node, _core.Node]]:
        # nodes=[0,1,2], edges=[(0,1),(0,2)]
        v0 = _core.Value(name="v0")
        node0 = _core.Node("", "Node0", inputs=(v0,), num_outputs=1)
        node1 = _core.Node("", "Node1", inputs=(node0.outputs[0],), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(node0.outputs[0],), num_outputs=1)
        return v0, (node0, node1, node2)

    @parameterized.parameterized.expand(
        [
            ("012", (0, 1, 2), (0, 1, 2)),
            ("021", (0, 2, 1), (0, 2, 1)),
            ("102", (1, 0, 2), (0, 1, 2)),
            ("120", (1, 2, 0), (0, 1, 2)),
            ("201", (2, 0, 1), (0, 2, 1)),
            ("210", (2, 1, 0), (0, 2, 1)),
        ]
    )
    def test_topological_sort_shared_predecessor(
        self, _: str, initial_order: tuple[int], expected_order: tuple[int]
    ):
        v0, nodes = self._create_shared_predecessor_nodes()
        graph = _core.Graph((v0,), (), nodes=[nodes[i] for i in initial_order])
        graph.sort()
        sorted_nodes = list(graph)
        self.assertEqual(sorted_nodes, [nodes[i] for i in expected_order])

    def test_topological_sort_cycle_detection(self):
        # nodes=[1,2,3], edges=[(1,2),(2,3),(3,2)]
        v0 = _core.Value(name="v0")
        node1 = _core.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = _core.Node("", "Node2", inputs=(node1.outputs[0], v0), num_outputs=1)
        node3 = _core.Node("", "Node3", inputs=(node2.outputs[0],), num_outputs=1)
        node2.replace_input_with(1, node3.outputs[0])
        graph = _core.Graph(
            (v0,),
            (node3.outputs[0],),
            nodes=(node1, node2, node3),
        )
        with self.assertRaises(ValueError):
            graph.sort()

    def test_topological_sort_subgraph(self):
        # main_graph: nodes=[a,b,c,d,>,if], edges=[(a,>),(b,>),(>,if)], subgraphs={if:[then_graph,else_graph]}
        # then_graph: nodes=[sub], edges=[(c,sub),(d,sub)]
        # else_graph: nodes=[add], edges=[(c,add),(d,add)]
        v0 = _core.Value(name="va")
        v1 = _core.Value(name="vb")
        v2 = _core.Value(name="vc")
        v3 = _core.Value(name="vd")
        node0 = _core.Node("", "a", inputs=(v0,), num_outputs=1)
        node1 = _core.Node("", "b", inputs=(v1,), num_outputs=1)
        node2 = _core.Node("", "c", inputs=(v2,), num_outputs=1)
        node3 = _core.Node("", "d", inputs=(v3,), num_outputs=1)
        node4 = _core.Node(
            "", "sub", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1
        )
        node5 = _core.Node(
            "", "add", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1
        )
        node6 = _core.Node("", ">", inputs=(node0.outputs[0], node1.outputs[0]), num_outputs=1)
        then_graph = _core.Graph(
            inputs=(node2.outputs[0], node3.outputs[0]),
            outputs=(node4.outputs[0],),
            nodes=(node4,),
            name="then_graph",
        )
        else_graph = _core.Graph(
            inputs=(node2.outputs[0], node3.outputs[0]),
            outputs=(node5.outputs[0],),
            nodes=(node5,),
            name="else_graph",
        )
        node7 = _core.Node(
            "",
            "if",
            inputs=(node6.outputs[0],),
            num_outputs=1,
            attributes=[
                ir.AttrGraph("then_branch", then_graph),
                ir.AttrGraph("else_branch", else_graph),
            ],
        )
        main_graph_rev = _core.Graph(
            inputs=(v0, v1, v2, v3),
            outputs=(node7.outputs[0],),
            nodes=(node7, node6, node3, node2, node1, node0),  # if, >, d, c, b, a
            name="main_graph_rev",
        )
        main_graph_rev.sort()
        self.assertEqual(
            tuple(node.op_type for node in tuple(main_graph_rev)),
            ("d", "c", "b", "a", ">", "if"),
        )


class ModelTest(unittest.TestCase):
    def test_graphs_returns_all_subgraphs(self):
        # main_graph: nodes=[a,b,c,d,>,if], edges=[(a,>),(b,>),(>,if)], subgraphs={if:[then_graph,else_graph]}
        # then_graph: nodes=[sub], edges=[(c,sub),(d,sub)]
        # else_graph: nodes=[add], edges=[(c,add),(d,add)]
        v0 = _core.Value(name="va")
        v1 = _core.Value(name="vb")
        v2 = _core.Value(name="vc")
        v3 = _core.Value(name="vd")
        node0 = _core.Node("", "a", inputs=(v0,), num_outputs=1)
        node1 = _core.Node("", "b", inputs=(v1,), num_outputs=1)
        node2 = _core.Node("", "c", inputs=(v2,), num_outputs=1)
        node3 = _core.Node("", "d", inputs=(v3,), num_outputs=1)
        node4 = _core.Node(
            "", "sub", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1
        )
        node5 = _core.Node(
            "", "add", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1
        )
        node6 = _core.Node("", ">", inputs=(node0.outputs[0], node1.outputs[0]), num_outputs=1)
        then_graph = _core.Graph(
            inputs=(node2.outputs[0], node3.outputs[0]),
            outputs=(node4.outputs[0],),
            nodes=(node4,),
            name="then_graph",
        )
        else_graph = _core.Graph(
            inputs=(node2.outputs[0], node3.outputs[0]),
            outputs=(node5.outputs[0],),
            nodes=(node5,),
            name="else_graph",
        )
        node7 = _core.Node(
            "",
            "if",
            inputs=(node6.outputs[0],),
            num_outputs=1,
            attributes=[
                ir.AttrGraph("then_branch", then_graph),
                ir.AttrGraph("else_branch", else_graph),
            ],
        )
        main_graph = _core.Graph(
            inputs=(v0, v1, v2, v3),
            outputs=(node7.outputs[0],),
            nodes=(node0, node1, node2, node6, node7),
            name="main_graph",
        )
        model = _core.Model(main_graph, ir_version=10)
        self.assertEqual(
            tuple(model.graphs()),
            (main_graph, then_graph, else_graph),
        )


class TypeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("tensor", _core.TensorType(ir.DataType.FLOAT)),
            ("sequence", _core.SequenceType(_core.TensorType(ir.DataType.BOOL))),
            ("optional", _core.OptionalType(_core.TensorType(ir.DataType.FLOAT16))),
            (
                "sequence_optional",
                _core.SequenceType(_core.OptionalType(_core.TensorType(ir.DataType.INT8))),
            ),
            (
                "optional_sequence",
                _core.OptionalType(_core.SequenceType(_core.TensorType(ir.DataType.INT16))),
            ),
        ]
    )
    def test_type_is_hashable(self, _: str, type_: ir.TypeProtocol):
        self.assertIsInstance(hash(type_), int)
        self.assertIn(type_, {type_})  # type: ignore
        # Assert that a different type object can still be matched
        self.assertIn(copy.deepcopy(type_), {type_})  # type: ignore

    def test_type_is_comparable(self):
        self.assertEqual(
            _core.TensorType(ir.DataType.FLOAT), _core.TensorType(ir.DataType.FLOAT)
        )
        self.assertNotEqual(
            _core.TensorType(ir.DataType.FLOAT), _core.TensorType(ir.DataType.FLOAT16)
        )

    @parameterized.parameterized.expand(
        [
            ("tensor", _core.TensorType(ir.DataType.FLOAT)),
            ("sequence", _core.SequenceType(_core.TensorType(ir.DataType.BOOL))),
            ("optional", _core.OptionalType(_core.TensorType(ir.DataType.FLOAT16))),
            (
                "sequence_optional",
                _core.SequenceType(_core.OptionalType(_core.TensorType(ir.DataType.INT8))),
            ),
            (
                "optional_sequence",
                _core.OptionalType(_core.SequenceType(_core.TensorType(ir.DataType.INT16))),
            ),
        ]
    )
    def test_composite_type_is_comparable(self, _: str, type_: ir.TypeProtocol):
        self.assertEqual(type_, type_)
        # Equal even if deep-copied
        self.assertEqual(type_, copy.deepcopy(type_))


class AttrTest(unittest.TestCase):
    """Test the Attr class."""

    def test_init(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 42, doc_string="test string")
        self.assertEqual(attr.name, "test")
        self.assertEqual(attr.value, 42)
        self.assertEqual(attr.type, ir.AttributeType.INT)
        self.assertEqual(attr.doc_string, "test string")

    def test_as_float(self):
        attr = _core.Attr("test", ir.AttributeType.FLOAT, 42.0)
        self.assertEqual(attr.as_float(), 42.0)

        attr_int_value = _core.Attr("test", ir.AttributeType.FLOAT, 42)
        self.assertEqual(attr_int_value.as_float(), 42.0)

    def test_as_int(self):
        attr = _core.Attr("test", ir.AttributeType.INT, 0)
        self.assertEqual(attr.as_int(), 0)

    def test_as_string(self):
        attr = _core.Attr("test", ir.AttributeType.STRING, "test string")
        self.assertEqual(attr.as_string(), "test string")

    def test_as_tensor(self):
        attr = _core.Attr("test", ir.AttributeType.TENSOR, ir.tensor([42.0]))
        np.testing.assert_equal(attr.as_tensor().numpy(), np.array([42.0]))

    def test_as_graph(self):
        attr = _core.Attr("test", ir.AttributeType.GRAPH, _core.Graph((), (), nodes=()))
        self.assertIsInstance(attr.as_graph(), _core.Graph)

    def test_as_floats(self):
        attr = _core.Attr("test", ir.AttributeType.FLOATS, [42.0])
        self.assertEqual(attr.as_floats(), [42.0])

    def test_as_ints(self):
        attr = _core.Attr("test", ir.AttributeType.INTS, [42])
        self.assertEqual(attr.as_ints(), [42])

    def test_as_strings(self):
        attr = _core.Attr("test", ir.AttributeType.STRINGS, ["test string", ""])
        self.assertEqual(attr.as_strings(), ["test string", ""])

    def test_as_tensors(self):
        attr = _core.Attr("test", ir.AttributeType.TENSORS, [ir.tensor([42.0])])
        np.testing.assert_equal(attr.as_tensors()[0].numpy(), np.array([42.0]))

    def test_as_graphs(self):
        attr = _core.Attr("test", ir.AttributeType.GRAPHS, [_core.Graph((), (), nodes=())])
        self.assertIsInstance(attr.as_graphs()[0], _core.Graph)


class LazyTensorTest(unittest.TestCase):
    def test_lazy_tensor_initialization(self):
        def tensor_fn():
            return ir.tensor([1, 2, 3], dtype=ir.DataType.INT64)

        lazy_tensor = _core.LazyTensor(
            tensor_fn, dtype=ir.DataType.INT64, shape=ir.Shape((3,))
        )
        self.assertEqual(lazy_tensor.dtype, ir.DataType.INT64)
        self.assertEqual(lazy_tensor.shape, (3,))

    def test_lazy_tensor_numpy(self):
        def tensor_fn():
            return ir.tensor([1, 2, 3], dtype=ir.DataType.INT64)

        lazy_tensor = _core.LazyTensor(
            tensor_fn, dtype=ir.DataType.INT64, shape=ir.Shape((3,))
        )
        np.testing.assert_array_equal(lazy_tensor.numpy(), np.array([1, 2, 3]))

    def test_lazy_tensor_tobytes(self):
        def tensor_fn():
            return ir.tensor([1, 2, 3], dtype=ir.DataType.INT64)

        lazy_tensor = _core.LazyTensor(
            tensor_fn, dtype=ir.DataType.INT64, shape=ir.Shape((3,))
        )
        self.assertEqual(
            lazy_tensor.tobytes(),
            b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00",
        )


if __name__ == "__main__":
    unittest.main()
