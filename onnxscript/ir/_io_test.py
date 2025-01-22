# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the _io module."""

import os
import tempfile
import unittest

import numpy as np
import parameterized

from onnxscript import ir
from onnxscript.ir import _io


def _create_initializer(tensor: ir.TensorProtocol) -> ir.Value:
    return ir.Value(
        name=tensor.name,
        shape=tensor.shape,
        type=ir.TensorType(tensor.dtype),
        const_value=tensor,
    )


def _create_simple_model_with_initializers() -> ir.Model:
    tensor_0 = ir.tensor([0.0], dtype=ir.DataType.FLOAT, name="initializer_0")
    initializer = _create_initializer(tensor_0)
    tensor_1 = ir.tensor([1.0], dtype=ir.DataType.FLOAT)
    identity_node = ir.Node("", "Identity", inputs=(initializer,))
    identity_node.outputs[0].shape = ir.Shape([1])
    identity_node.outputs[0].dtype = ir.DataType.FLOAT
    identity_node.outputs[0].name = "identity_0"
    const_node = ir.Node(
        "",
        "Constant",
        inputs=(),
        outputs=(
            ir.Value(name="const_0", shape=tensor_1.shape, type=ir.TensorType(tensor_1.dtype)),
        ),
        attributes=ir.convenience.convert_attributes(dict(value=tensor_1)),
    )
    graph = ir.Graph(
        inputs=[initializer],
        outputs=[*identity_node.outputs, *const_node.outputs],
        nodes=[identity_node, const_node],
        initializers=[initializer],
        name="test_graph",
    )
    return ir.Model(graph, ir_version=10)


class IOFunctionsTest(unittest.TestCase):
    def test_load(self):
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(model, path)
            loaded_model = _io.load(path)
        self.assertEqual(loaded_model.ir_version, model.ir_version)
        self.assertEqual(loaded_model.graph.name, model.graph.name)
        self.assertEqual(len(loaded_model.graph.initializers), 1)
        self.assertEqual(len(loaded_model.graph), 2)
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["initializer_0"].const_value.numpy(),
            np.array([0.0]),
        )
        np.testing.assert_array_equal(
            loaded_model.graph.node(1).attributes["value"].as_tensor().numpy(), np.array([1.0])
        )
        self.assertEqual(loaded_model.graph.inputs[0].name, "initializer_0")
        self.assertEqual(loaded_model.graph.outputs[0].name, "identity_0")
        self.assertEqual(loaded_model.graph.outputs[1].name, "const_0")

    def test_save_with_external_data_does_not_modify_model(self):
        model = _create_simple_model_with_initializers()
        self.assertIsInstance(model.graph.initializers["initializer_0"].const_value, ir.Tensor)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = "model.data"
            _io.save(model, path, external_data=external_data_file)
            self.assertTrue(os.path.exists(path))
            external_data_path = os.path.join(tmpdir, external_data_file)
            self.assertTrue(os.path.exists(external_data_path))
            loaded_model = _io.load(path)

            # The loaded model contains external data
            initializer_tensor = loaded_model.graph.initializers["initializer_0"].const_value
            self.assertIsInstance(initializer_tensor, ir.ExternalTensor)
            # The attribute is not externalized
            const_attr_tensor = loaded_model.graph.node(1).attributes["value"].as_tensor()
            self.assertIsInstance(const_attr_tensor, ir.TensorProtoTensor)
            np.testing.assert_array_equal(initializer_tensor.numpy(), np.array([0.0]))
            np.testing.assert_array_equal(const_attr_tensor.numpy(), np.array([1.0]))

        # The original model is not changed and can be accessed even if the
        # external data file is deleted
        initializer_tensor = model.graph.initializers["initializer_0"].const_value
        self.assertIsInstance(initializer_tensor, ir.Tensor)
        const_attr_tensor = model.graph.node(1).attributes["value"].as_tensor()
        self.assertIsInstance(const_attr_tensor, ir.Tensor)
        np.testing.assert_array_equal(initializer_tensor.numpy(), np.array([0.0]))
        np.testing.assert_array_equal(const_attr_tensor.numpy(), np.array([1.0]))

    def test_save_raise_when_external_data_is_not_relative_path(self):
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = os.path.join(tmpdir, "model.data")
            with self.assertRaises(ValueError):
                _io.save(model, path, external_data=external_data_file)

    def test_save_with_external_data_invalidates_obsolete_external_tensors(
        self, _: str
    ):
        model = _create_simple_model_with_initializers()
        self.assertIsInstance(model.graph.initializers["initializer_0"].const_value, ir.Tensor)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = "model.data"
            _io.save(model, path, external_data=external_data_file)
            # The original model is modified
            initializer_tensor = model.graph.initializers["initializer_0"].const_value
            self.assertIsInstance(initializer_tensor, ir.ExternalTensor)

            # Now if we create a different initializer and save that model with the same external data file
            tensor_2 = ir.tensor([2.0], dtype=ir.DataType.FLOAT, name="initializer_2")
            initializer_2 = _create_initializer(tensor_2)
            model.graph.initializers["initializer_2"] = initializer_2
            with self.assertRaises(ValueError):
                # The existing model has to be modified to use in memory tensors
                # for the values to stay correct
                _io.save(model, path, external_data=external_data_file)


if __name__ == "__main__":
    unittest.main()
