# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import tempfile
import typing
import unittest

import numpy as np
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


class OffsetCalcTest(unittest.TestCase):
    """Test the offset calculation for the external tensor class."""

    def test_align_offset_false(self):
        # Tensor size > Align Threshold
        current_offset = 20000
        tensor_size = 1048
        new_offset = _external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset, tensor_size, align_offset=False
        )
        self.assertEqual(current_offset, new_offset)

    def test_align_with_small_align_threshold(self):
        # Tensor size < Align Threshold
        current_offset = 20000
        tensor_size = 1048
        new_offset = _external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset,
            tensor_size,
            align_threshold=1000,
        )
        self.assertNotEqual(current_offset, new_offset)

    def test_align_with_large_align_threshold(self):
        # Tensor size > Align Threshold
        current_offset = 20000
        tensor_size = 1048
        new_offset = _external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset,
            tensor_size,
        )
        self.assertEqual(current_offset, new_offset)

    def test_allocation_granularity_diff(self):
        # Tensor size > Align Threshold
        current_offset = 20000
        tensor_size = 1048577
        new_offset_1 = _external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset,
            tensor_size,
            allocation_granularity=4000,
        )
        new_offset_2 = _external_data._compute_new_offset(  # pylint: disable=protected-access
            current_offset,
            tensor_size,
        )
        self.assertNotEqual(current_offset, new_offset_1)
        self.assertNotEqual(current_offset, new_offset_2)
        self.assertNotEqual(new_offset_1, new_offset_2)


class OffloadExternalTensorTest(unittest.TestCase):
    """Test the memory mapped external tensor class."""

    def setUp(self):
        # File paths
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # pylint: disable=consider-using-with
        self.external_data_name = "external_tensors.bin"
        self.base_path = self.temp_dir.name
        self.ext_data_1 = "external_data_1.bin"
        self.ext_data_2 = "external_data_2.bin"
        # Data for the tensors
        self.data = np.random.rand(2, 42).astype(np.float32)
        self.data_other = np.random.rand(2, 42).astype(np.float32)
        self.data_float16 = np.random.rand(2, 42).astype(np.float16)
        self.data_ext1_1 = np.random.rand(1, 42).astype(np.float32)
        self.data_ext1_2 = np.random.rand(4, 42).astype(np.float16)
        self.data_ext2_1 = np.random.rand(5, 42).astype(np.float16)
        self.custom_data = np.random.rand(3, 42).astype(np.float32)
        # Model Assignments
        self.model = self._simple_model()
        self.model_with_external_data_same_path = self._model_with_external_data_same_path()
        self.model_with_external_data_diff_path = self._model_with_external_data_diff_path()
        self.model_with_custom_tensor_class = self._model_with_custom_tensor_class()
        self.model_with_mixed_external_data = self._model_with_mixed_external_data()

    def tearDown(self) -> None:
        # Handle exceptions for windows and python versions < 3.10
        try:
            self.temp_dir.cleanup()
        except PermissionError as e:
            print(f"PermissionError: {e}")
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"An unexpected error occurred: {e}")

    def _simple_model(self) -> ir.Model:
        tensor1 = ir.Tensor(
            self.data,
            dtype=ir.DataType.FLOAT,
            shape=ir.Shape(self.data.shape),
            name="tensor1",
        )
        tensor2 = ir.Tensor(
            self.data_float16,
            dtype=ir.DataType.FLOAT16,
            shape=ir.Shape(self.data_float16.shape),
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
            initializers=[
                ir.Value(name="tensor1", const_value=tensor1),
                ir.Value(name="tensor2", const_value=tensor2),
            ],
            # Unsorted nodes
            nodes=[node_1, node_0],
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=8)
        return model

    def _setup_custom_tensor_class(self, name, value):
        class CustomTensorType(ir.TensorProtocol):
            def __init__(
                self,
                value: np.ndarray,
            ):
                self.name = name
                self._raw = value
                if isinstance(value, np.ndarray):
                    self._dtype = ir._enums.DataType.from_numpy(value.dtype)
                self._shape = ir.Shape(getattr(value, "shape"), frozen=True)  # noqa: B009

            @property
            def dtype(self) -> ir._enums.DataType:
                """The data type of the tensor. Immutable."""
                return self._dtype

            @property
            def shape(self) -> ir.Shape:
                """The shape of the tensor. Immutable."""
                return self._shape

            @property
            def nbytes(self) -> int:
                return len(self.tobytes())

            def __array__(self, dtype: typing.Any = None) -> np.ndarray:
                if isinstance(self._raw, np.ndarray):
                    return self._raw
                else:
                    return TypeError

            def numpy(self) -> np.ndarray:
                return self._raw

            def tobytes(self) -> bytes:
                if isinstance(self._raw, np.ndarray):
                    return self._raw.tobytes()
                else:
                    return TypeError

        return CustomTensorType(value)

    def _model_with_external_data_same_path(self) -> ir.Model:
        model = self._simple_model()
        raw_data = self.data_other.tobytes()
        # Save the data to disk
        file_path = os.path.join(self.base_path, self.external_data_name)
        with open(file_path, "wb") as f:
            f.write(raw_data)
        tensor_same_file = ir.ExternalTensor(
            location=self.external_data_name,
            offset=0,
            length=len(raw_data),
            dtype=ir.DataType.FLOAT,
            name="tensor_same_file",
            shape=ir.Shape(self.data_other.shape),
            base_dir=self.base_path,
        )
        model.graph.initializers["tensor_same_file"] = ir.Value(
            name="tensor_same_file", const_value=tensor_same_file
        )
        return model

    def _model_with_external_data_diff_path(self) -> ir.Model:
        model = self._simple_model()
        # File 1
        file_path_1 = os.path.join(self.base_path, self.ext_data_1)
        with open(file_path_1, "wb") as f:
            f.write(self.data_ext1_1.tobytes())
            f.write(self.data_ext1_2.tobytes())
        tensor_ext1_1 = ir.ExternalTensor(
            location=self.ext_data_1,
            offset=0,
            length=len(self.data_ext1_1.tobytes()),
            dtype=ir.DataType.FLOAT,
            name="tensor_ext1_1",
            shape=ir.Shape(self.data_ext1_1.shape),
            base_dir=self.base_path,
        )
        tensor_ext1_2 = ir.ExternalTensor(
            location=self.ext_data_1,
            offset=len(self.data_ext1_1.tobytes()),
            length=len(self.data_ext1_2.tobytes()),
            dtype=ir.DataType.FLOAT16,
            name="tensor_ext1_2",
            shape=ir.Shape(self.data_ext1_2.shape),
            base_dir=self.base_path,
        )
        # File 2
        file_path_2 = os.path.join(self.base_path, self.ext_data_2)
        with open(file_path_2, "wb") as f:
            f.write(self.data_ext2_1.tobytes())
        tensor_ext2_1 = ir.ExternalTensor(
            location=self.ext_data_2,
            offset=0,
            length=len(self.data_ext2_1.tobytes()),
            dtype=ir.DataType.FLOAT16,
            name="tensor_ext2_1",
            shape=ir.Shape(self.data_ext2_1.shape),
            base_dir=self.base_path,
        )
        model.graph.initializers["tensor_ext1_1"] = ir.Value(
            name="tensor_ext1_1", const_value=tensor_ext1_1
        )
        model.graph.initializers["tensor_ext1_2"] = ir.Value(
            name="tensor_ext1_2", const_value=tensor_ext1_2
        )
        model.graph.initializers["tensor_ext2_1"] = ir.Value(
            name="tensor_ext2_1", const_value=tensor_ext2_1
        )
        return model

    def _model_with_custom_tensor_class(self) -> ir.Model:
        model = self._simple_model()
        custom_tensor = self._setup_custom_tensor_class("custom_tensor", self.custom_data)
        model.graph.initializers["custom_tensor"] = ir.Value(
            name="custom_tensor", const_value=custom_tensor
        )
        return model

    def _model_with_mixed_external_data(self) -> ir.Model:
        model = self._simple_model()
        model_same_path = self.model_with_external_data_same_path
        model_diff_path = self.model_with_external_data_diff_path
        model_custom_tensor = self.model_with_custom_tensor_class
        model.graph.initializers["tensor_same_file"] = model_same_path.graph.initializers[
            "tensor_same_file"
        ]
        model.graph.initializers["tensor_ext1_1"] = model_diff_path.graph.initializers[
            "tensor_ext1_1"
        ]
        model.graph.initializers["tensor_ext1_2"] = model_diff_path.graph.initializers[
            "tensor_ext1_2"
        ]
        model.graph.initializers["tensor_ext2_1"] = model_diff_path.graph.initializers[
            "tensor_ext2_1"
        ]
        model.graph.initializers["custom_tensor"] = model_custom_tensor.graph.initializers[
            "custom_tensor"
        ]
        return model

    def test_external_data_simple(self):
        model_with_external_data = _external_data.to_external_data(
            self.model, self.base_path, self.external_data_name
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())

    def test_same_path_external_data_written_to_memory(self):
        model_with_external_data = _external_data.to_external_data(
            self.model_with_external_data_same_path,
            self.base_path,
            self.external_data_name,
            load_external_to_memory=True,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "tensor_same_file"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())

    def test_same_path_external_data_written_to_disk(self):
        model_with_external_data = _external_data.to_external_data(
            self.model_with_external_data_same_path,
            self.base_path,
            self.external_data_name,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "tensor_same_file"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())

    def test_external_data_diff_paths(self):
        model_with_external_data = _external_data.to_external_data(
            self.model_with_external_data_diff_path,
            self.base_path,
            self.external_data_name,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "tensor_ext1_1"
        ].const_value
        external_tensor4 = model_with_external_data.graph.initializers[
            "tensor_ext1_2"
        ].const_value
        external_tensor5 = model_with_external_data.graph.initializers[
            "tensor_ext2_1"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext2_1.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext2_1.tobytes())

    def test_custom_tensor_in_initializers(self):
        model_with_external_data = _external_data.to_external_data(
            self.model_with_custom_tensor_class,
            self.base_path,
            self.external_data_name,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "custom_tensor"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.custom_data.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.custom_data.tobytes())

    def test_mixed_external_data_to_disk(self):
        model_with_external_data = _external_data.to_external_data(
            self.model_with_mixed_external_data,
            self.base_path,
            self.external_data_name,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "tensor_same_file"
        ].const_value
        external_tensor4 = model_with_external_data.graph.initializers[
            "custom_tensor"
        ].const_value
        external_tensor5 = model_with_external_data.graph.initializers[
            "tensor_ext1_1"
        ].const_value
        external_tensor6 = model_with_external_data.graph.initializers[
            "tensor_ext1_2"
        ].const_value
        external_tensor7 = model_with_external_data.graph.initializers[
            "tensor_ext2_1"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.custom_data.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor6.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor7.numpy().tobytes(), self.data_ext2_1.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.custom_data.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor6.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor7.numpy().tobytes(), self.data_ext2_1.tobytes())

    def test_mixed_external_data_to_memory(self):
        model_with_external_data = _external_data.to_external_data(
            self.model_with_mixed_external_data,
            self.base_path,
            self.external_data_name,
            load_external_to_memory=True,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers[
            "tensor_same_file"
        ].const_value
        external_tensor4 = model_with_external_data.graph.initializers[
            "custom_tensor"
        ].const_value
        external_tensor5 = model_with_external_data.graph.initializers[
            "tensor_ext1_1"
        ].const_value
        external_tensor6 = model_with_external_data.graph.initializers[
            "tensor_ext1_2"
        ].const_value
        external_tensor7 = model_with_external_data.graph.initializers[
            "tensor_ext2_1"
        ].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.custom_data.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor6.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor7.numpy().tobytes(), self.data_ext2_1.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        self.assertEqual(external_tensor4.numpy().tobytes(), self.custom_data.tobytes())
        self.assertEqual(external_tensor5.numpy().tobytes(), self.data_ext1_1.tobytes())
        self.assertEqual(external_tensor6.numpy().tobytes(), self.data_ext1_2.tobytes())
        self.assertEqual(external_tensor7.numpy().tobytes(), self.data_ext2_1.tobytes())

    def test_external_data_sorted(self):
        model_with_external_data = _external_data.to_external_data(
            self.model_with_mixed_external_data,
            self.base_path,
            self.external_data_name,
        )
        file_path = os.path.join(self.base_path, self.external_data_name)
        expected_tensor_order = [
            model_with_external_data.graph.initializers["tensor2"].const_value.tobytes(),
            model_with_external_data.graph.initializers["tensor_ext1_1"].const_value.tobytes(),
            model_with_external_data.graph.initializers["tensor1"].const_value.tobytes(),
            model_with_external_data.graph.initializers[
                "tensor_same_file"
            ].const_value.tobytes(),
            model_with_external_data.graph.initializers["tensor_ext1_2"].const_value.tobytes(),
            model_with_external_data.graph.initializers["tensor_ext2_1"].const_value.tobytes(),
            model_with_external_data.graph.initializers["custom_tensor"].const_value.tobytes(),
        ]
        sorted_tensor_order = [
            self.data_float16.tobytes(),
            self.data_ext1_1.tobytes(),
            self.data.tobytes(),
            self.data_other.tobytes(),
            self.data_ext1_2.tobytes(),
            self.data_ext2_1.tobytes(),
            self.custom_data.tobytes(),
        ]
        with open(file_path, "r+b") as data_file:
            current_offset = 0
            for i, tensor_bytes in enumerate(sorted_tensor_order):
                data_file.seek(current_offset)
                tensor_length = len(tensor_bytes)
                tensor_data = data_file.read(tensor_length)
                current_offset += tensor_length
                self.assertEqual(tensor_data, tensor_bytes)
                self.assertEqual(tensor_data, expected_tensor_order[i])


if __name__ == "__main__":
    unittest.main()
