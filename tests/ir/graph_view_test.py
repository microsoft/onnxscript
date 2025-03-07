# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pathlib
import unittest

import onnx

from onnxscript import ir


class GraphViewTest(unittest.TestCase):
    def test_it_can_be_serialized_as_graph_proto(self):
        data_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "testdata/e2e_models/torchscript_model/torchscript_model.onnx"
        )
        model_proto = onnx.load(data_path)
        model = ir.serde.deserialize_model(model_proto)

        # Obtain a subset of nodes that belong to the first layer
        nodes = [
            node
            for node in model.graph
            if node.name is not None and node.name.startswith("/model/layers.0")
        ]

        inputs = set()
        outputs = set()
        for node in nodes:
            inputs.update(node.inputs)
            outputs.update(node.outputs)

        graph_inputs = sorted(inputs - outputs, key=lambda val: val.name)
        graph_outputs = sorted(outputs - inputs, key=lambda val: val.name)

        graph_view = ir.GraphView(graph_inputs, graph_outputs, nodes=nodes)
        model = ir.Model(graph_view, ir_version=8)
        _ = ir.serde.serialize_model(model)
        # It should succeed


if __name__ == "__main__":
    unittest.main()
