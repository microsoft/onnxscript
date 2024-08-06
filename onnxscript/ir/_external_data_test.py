# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnx
import onnx.external_data_helper

from onnxscript import ir
from onnxscript.ir import _external_data


class ExternalDataTest(unittest.TestCase):
    def test_set_base_dir_sets_base_dir_for_all_external_tensors(self):
        attr_tensor = onnx.helper.make_tensor(
            name="test_constant",
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=b"\x01\x00\x00\x00",
            raw=True,
        )
        graph = onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node(
                    "Constant",
                    [],
                    ["test"],
                    value=attr_tensor,
                )
            ],
            name="test",
            inputs=[],
            outputs=[],
            initializer=[
                onnx.helper.make_tensor(
                    name="test_tensor",
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[1],
                    vals=b"\x01\x00\x00\x00",
                    raw=True,
                ),
            ],
        )
        model_proto = onnx.helper.make_model(graph)
        onnx.external_data_helper.convert_model_to_external_data(
            model_proto, location="tempdir", size_threshold=0, convert_attribute=True
        )
        model = ir.serde.deserialize_model(model_proto)
        expected_dir = "something_else"
        _external_data.set_base_dir(model.graph, expected_dir)

        initializer_tensor = model.graph.initializers["test_tensor"].const_value
        assert isinstance(initializer_tensor, ir.ExternalTensor)
        self.assertEqual(initializer_tensor.base_dir, expected_dir)
        attr_tensor = model.graph.node(0).attributes["value"].value
        self.assertEqual(attr_tensor.base_dir, expected_dir)


if __name__ == "__main__":
    unittest.main()
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
    ) -> _core.Model:

        tensor1 = _core.Tensor(
            np.random.rand(1, 2).astype(np.float32),
            dtype=ir.DataType.FLOAT,
            shape=_core.Shape((1, 2)),
            name="tensor1",
        )
        tensor2 = _core.Tensor(
            np.random.rand(1, 2).astype(np.float32),
            dtype=ir.DataType.FLOAT,
            shape=_core.Shape((1, 2)),
            name="tensor2",
        )

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
            initializers=[tensor1, tensor2],
            # Unsorted nodes
            nodes=[node_1, node_0],
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=8)
        return model
        '''
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
        #tensor.ClearField("raw_data")
        #tensor2.ClearField("raw_data")
        # Save the data to disk
        #with open(pathlib.Path(base_path) / external_data_name, "wb") as f:
        #    f.write(raw_data)
        #    f.write(raw_data2)
        return model
    '''

    def test_simple_model_with_external_tensors(self):
        ir._external_data.convert_model_to_external_data(self.model, self.base_path)
    '''
    def test_initialize(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            path=pathlib.Path(self.base_path) / external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        np.testing.assert_equal(tensor, self.data)
        # Ensure repeated reads are consistent
        np.testing.assert_equal(tensor, self.data)

    def test_totypes_returns_correct_data_in(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            path=pathlib.Path(self.base_path) / external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=ir.DataType.FLOAT,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        external_tensor2 = self.model.graph.initializer[1]
        external_info2 = onnx.external_data_helper.ExternalDataInfo(external_tensor2)
        tensor2 = _core.ExternalTensor(
            path=pathlib.Path(self.base_path) / external_info2.location,
            offset=external_info2.offset,
            length=external_info2.length,
            dtype=ir.DataType.FLOAT16,
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
        ir._external_data.convert_model_to_external_data(ir.Tensor(expected_array, dtype=dtype))
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
    '''
