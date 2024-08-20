# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pathlib
import tempfile
import unittest

import onnx
import onnx.external_data_helper

import numpy as np

from onnxscript import ir
from onnxscript.ir import _core, _external_data


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
        new_offset = _external_data.compute_new_offset(
            current_offset, tensor_size, align_offset=False
        )
        self.assertEqual(current_offset, new_offset)

    def test_align_with_small_align_threshold(self):
        # Tensor size < Align Threshold
        current_offset = 20000
        tensor_size = 1048
        new_offset = _external_data.compute_new_offset(
            current_offset,
            tensor_size,
            align_threshold=1000,
        )
        self.assertNotEqual(current_offset, new_offset)

    def test_align_with_large_align_threshold(self):
        # Tensor size > Align Threshold
        current_offset = 20000
        tensor_size = 1048
        new_offset = _external_data.compute_new_offset(
            current_offset,
            tensor_size,
        )
        self.assertEqual(current_offset, new_offset)

    def test_allocation_granularity_diff(self):
        # Tensor size > Align Threshold
        current_offset = 20000
        tensor_size = 1048577
        new_offset_1 = _external_data.compute_new_offset(
            current_offset,
            tensor_size,
            allocation_granularity=4000,
        )
        new_offset_2 = _external_data.compute_new_offset(
            current_offset,
            tensor_size,
        )
        self.assertNotEqual(current_offset, new_offset_1)
        self.assertNotEqual(current_offset, new_offset_2)
        self.assertNotEqual(new_offset_1, new_offset_2)


class ExternalTensorTest(unittest.TestCase):
    """Test the memory mapped external tensor class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.external_data_name = "external_tensors.bin"
        self.base_path = self.temp_dir.name
        self.data = np.random.rand(2, 42).astype(np.float32)
        self.data_other = np.random.rand(2, 42).astype(np.float32)
        self.data_float16 = np.random.rand(2, 42).astype(np.float16)
        self.model = self._simple_model_with_external()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _simple_model_with_external(self) -> _core.Model:
        tensor1 = _core.Tensor(
            self.data,
            dtype=ir.DataType.FLOAT,
            shape=_core.Shape(self.data.shape),
            name="tensor1",
        )
        tensor2 = _core.Tensor(
            self.data_float16,
            dtype=ir.DataType.FLOAT16,
            shape=_core.Shape(self.data_float16.shape),
            name="tensor2",
        )
        raw_data = self.data_other.tobytes()
        # Save the data to disk
        file_path = pathlib.Path(self.base_path) / self.external_data_name
        with open(file_path, "w+b") as f:
            f.write(raw_data)
        tensor3 = _core.ExternalTensor(
            path=file_path,
            offset=0,
            length=len(raw_data),
            dtype=ir.DataType.FLOAT,
            name="tensor3",
            shape=_core.Shape(self.data_other.shape),
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
                ir.Value(name="tensor3", const_value=tensor3),
            ],
            # Unsorted nodes
            nodes=[node_1, node_0],
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=8)
        return model

    def test_initialize(self):
        model_with_external_data = _external_data.to_external_data(
            self.model, self.base_path, file_path=self.external_data_name
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        self.assertEqual(external_tensor.dtype, ir.DataType.FLOAT)
        np.testing.assert_equal(external_tensor.numpy(), self.data)
        # Ensure repeated reads are consistent
        np.testing.assert_equal(external_tensor.numpy(), self.data)

    def test_totypes_returns_correct_data_in(self):
        model_with_external_data = _external_data.to_external_data(
            self.model, self.base_path, file_path=self.external_data_name
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())

    def test_non_empty_external_data(self):
        model_with_external_data = _external_data.to_external_data(
            self.model,
            self.base_path,
            file_path=self.external_data_name,
            load_external_to_memory=True,
        )
        external_tensor = model_with_external_data.graph.initializers["tensor1"].const_value
        external_tensor2 = model_with_external_data.graph.initializers["tensor2"].const_value
        external_tensor3 = model_with_external_data.graph.initializers["tensor3"].const_value

        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(external_tensor.numpy().tobytes(), self.data.tobytes())
        self.assertEqual(external_tensor2.numpy().tobytes(), self.data_float16.tobytes())
        self.assertEqual(external_tensor3.numpy().tobytes(), self.data_other.tobytes())


if __name__ == "__main__":
    unittest.main()
