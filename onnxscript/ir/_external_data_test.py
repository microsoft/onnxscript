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
        assert isinstance(
            initializer_tensor, ir.ExternalTensor
        )
        self.assertEqual(
            initializer_tensor.base_dir, expected_dir
        )
        attr_tensor = model.graph.node(0).attributes["value"].value
        self.assertEqual(attr_tensor.base_dir, expected_dir)


if __name__ == "__main__":
    unittest.main()
