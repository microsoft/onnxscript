import os
import tempfile
import unittest

from onnxscript import ir
from onnxscript.ir import _io


def _create_simple_model():
    tensor = ir.tensor([1.0], dtype=ir.DataType.FLOAT, name="X")
    node = ir.Node("Identity", inputs=[tensor], outputs=["Y"])
    graph = ir.graph([node], name="test_graph", outputs=[node.outputs[0]], initializers=[tensor])
    return ir.model(graph)

class TestIOFunctions(unittest.TestCase):
    def test_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            # Create a simple ONNX model
            tensor = ir.tensor([1.0], dtype=ir.DataType.FLOAT, name="X")
            node = ir.node("Identity", inputs=[tensor], outputs=["Y"])
            graph = ir.graph([node], name="test_graph", outputs=[node.outputs[0]], initializers=[tensor])
            model = ir.model(graph)
            # Save the model to a file
            with open(path, "wb") as f:
                f.write(model.SerializeToString())

            # Load the model using the _io.load function
            loaded_model = _io.load(path)

            # Check that the loaded model is correct
            self.assertEqual(loaded_model.graph.name, "test_graph")
            self.assertEqual(len(loaded_model.graph.initializers), 1)
            self.assertEqual(loaded_model.graph.initializers["X"].const_value, [1.0])

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_path = "external_data"

            # Create a simple ONNX model
            tensor = ir.tensor([1.0], dtype=ir.DataType.FLOAT, name="X")
            node = ir.node("Identity", inputs=[tensor], outputs=["Y"])
            graph = ir.graph([node], name="test_graph", outputs=[node.outputs[0]], initializers=[tensor])
            model = ir.model(graph)
            core_model = _core.Model(model)

            # Save the model using the _io.save function
            _io.save(core_model, path, external_data=external_data_path, modify_model=True)

            # Load the model back to verify it was saved correctly
            loaded_model = _io.load(path)

            # Check that the loaded model is correct
            self.assertEqual(loaded_model.graph.name, "test_graph")
            self.assertEqual(len(loaded_model.graph.initializers), 1)
            self.assertEqual(loaded_model.graph.initializers["X"].const_value, [1.0])

    def test_save_without_external_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")

            # Create a simple ONNX model
            tensor = ir.tensor([1.0], dtype=ir.DataType.FLOAT, name="X")
            node = ir.node("Identity", inputs=[tensor], outputs=["Y"])
            graph = ir.graph([node], name="test_graph", outputs=[node.outputs[0]], initializers=[tensor])
            model = ir.model(graph)
            core_model = _core.Model(model)

            # Save the model using the _io.save function without external data
            _io.save(core_model, path, modify_model=True)

            # Load the model back to verify it was saved correctly
            loaded_model = _io.load(path)

            # Check that the loaded model is correct
            self.assertEqual(loaded_model.graph.name, "test_graph")
            self.assertEqual(len(loaded_model.graph.initializers), 1)
            self.assertEqual(loaded_model.graph.initializers["X"].const_value, [1.0])

    def test_save_with_external_data_modify_model_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_path = "external_data"

            # Create a simple ONNX model
            tensor = ir.tensor([1.0], dtype=ir.DataType.FLOAT, name="X")
            node = ir.node("Identity", inputs=[tensor], outputs=["Y"])
            graph = ir.graph([node], name="test_graph", outputs=[node.outputs[0]], initializers=[tensor])
            model = ir.model(graph)
            core_model = _core.Model(model)

            # Save the model using the _io.save function with external data and modify_model=True
            _io.save(core_model, path, external_data=external_data_path, modify_model=True)

            # Load the model back to verify it was saved correctly
            loaded_model = _io.load(path)

            # Check that the loaded model is correct
            self.assertEqual(loaded_model.graph.name, "test_graph")
            self.assertEqual(len(loaded_model.graph.initializers), 1)
            self.assertEqual(loaded_model.graph.initializers["X"].const_value, [1.0])

    def test_save_with_external_data_modify_model_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_path = "external_data"

            # Create a simple ONNX model
            tensor = ir.tensor([1.0], dtype=ir.DataType.FLOAT, name="X")
            node = ir.node("Identity", inputs=[tensor], outputs=["Y"])
            graph = ir.graph([node], name="test_graph", outputs=[node.outputs[0]], initializers=[tensor])
            model = ir.model(graph)
            core_model = _core.Model(model)

            # Save the model using the _io.save function with external data and modify_model=False
            _io.save(core_model, path, external_data=external_data_path, modify_model=False)

            # Load the model back to verify it was saved correctly
            loaded_model = _io.load(path)

            # Check that the loaded model is correct
            self.assertEqual(loaded_model.graph.name, "test_graph")
            self.assertEqual(len(loaded_model.graph.initializers), 1)
            self.assertEqual(loaded_model.graph.initializers["X"].const_value, [1.0])

if __name__ == "__main__":
    unittest.main()
