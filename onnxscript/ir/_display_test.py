# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test display() methods in various classes."""

import contextlib
import unittest

import numpy as np

import onnxscript.ir as ir


import ml_dtypes
class DisplayTest(unittest.TestCase):
    def test_tensor_display_does_not_raise_on_nan_values(self):
        array_with_nan = np.array([np.inf, -np.inf, np.nan, 5, -10], dtype=np.float32)
        tensor = ir.Tensor(array_with_nan, dtype=ir.DataType.FLOAT)
        with contextlib.redirect_stdout(None):
            tensor.display()


    def test_graph_register_initializer(self):
        graph = ir.Graph(inputs=[], outputs=[], nodes=[])
        tensor_value = ir.Tensor(np.array([1, 2, 3], dtype=np.int32), dtype=ir.DataType.INT32)
        initializer = ir.Value(name="init_tensor", const_value=tensor_value)
        
        # Register initializer successfully
        graph.register_initializer(initializer)
        self.assertIn("init_tensor", graph.initializers)
        
        # Attempt to register an initializer without a name
        unnamed_initializer = ir.Value(const_value=tensor_value)
        with self.assertRaises(ValueError):
            graph.register_initializer(unnamed_initializer)
        
        # Attempt to register an initializer that is produced by a node
        node = ir.Node(domain="", op_type="Add", inputs=[], outputs=[initializer])
        with self.assertRaises(ValueError):
            graph.register_initializer(initializer)


    def test_graph_remove_node_still_in_use(self):
        graph = ir.Graph(inputs=[], outputs=[], nodes=[])
        node1 = ir.Node(domain="", op_type="Add", inputs=[], outputs=[ir.Value()])
        node2 = ir.Node(domain="", op_type="Add", inputs=[node1.outputs[0]], outputs=[ir.Value()])
        graph.append(node1)
        graph.append(node2)
        with self.assertRaises(ValueError):
            graph.remove(node1, safe=True)


    def test_shape_initialization_with_symbolic_dim(self):
        symbolic_dim = ir.SymbolicDim("N")
        shape = ir.Shape([symbolic_dim, 3, 5])
        self.assertEqual(shape.dims, (symbolic_dim, 3, 5))


    def test_external_tensor_with_relative_path(self):
        tensor = ir.ExternalTensor(
            location="relative/path/to/data",
            offset=0,
            length=100,
            dtype=ir.DataType.FLOAT,
            shape=ir.Shape([10, 10]),
            name="test_tensor"
        )
        self.assertEqual(tensor.location, "relative/path/to/data")


    def test_graph_append_node_from_another_graph(self):
        graph1 = ir.Graph(inputs=[], outputs=[], nodes=[])
        graph2 = ir.Graph(inputs=[], outputs=[], nodes=[])
        node = ir.Node(domain="", op_type="Add", inputs=[], outputs=[])
        graph1.append(node)
        with self.assertRaises(ValueError):
            graph2.append(node)


    def test_tensor_initialization_with_mismatched_dtype(self):
        array_int32 = np.array([1, 2, 3], dtype=np.int32)
        with self.assertRaises(TypeError):
            ir.Tensor(array_int32, dtype=ir.DataType.FLOAT)


    def test_tensor_initialization_with_non_standard_dtypes(self):
        array_bfloat16 = np.array([1.0, 2.0, 3.0], dtype=ml_dtypes.bfloat16)
        tensor_bfloat16 = ir.Tensor(array_bfloat16, dtype=ir.DataType.BFLOAT16)
        self.assertEqual(tensor_bfloat16.dtype, ir.DataType.BFLOAT16)
        
        array_float8 = np.array([1.0, 2.0, 3.0], dtype=ml_dtypes.float8_e4m3fn)
        tensor_float8 = ir.Tensor(array_float8, dtype=ir.DataType.FLOAT8E4M3FN)
        self.assertEqual(tensor_float8.dtype, ir.DataType.FLOAT8E4M3FN)


if __name__ == "__main__":
    unittest.main()
