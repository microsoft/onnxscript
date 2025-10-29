# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import onnx
import onnx_ir
from onnx import helper


def _initializers_to_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    new_nodes = []

    # Keep track of names to remove from inputs
    init_names = {init.name for init in graph.initializer}

    for init in graph.initializer:
        # Convert initializer to Constant node
        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[init.name],
            value=init,  # Directly use TensorProto
        )
        new_nodes.append(const_node)

    # Filter out initializer names from graph inputs
    filtered_inputs = [i for i in graph.input if i.name not in init_names]
    graph.ClearField("input")
    graph.input.extend(filtered_inputs)

    # Add new Constant nodes at the beginning
    all_nodes = new_nodes + list(graph.node)
    graph.ClearField("node")
    graph.node.extend(all_nodes)

    # Clear initializers (since we replaced them)
    graph.ClearField("initializer")

    return model


def convert_model_proto_to_function_proto(
    model: onnx.ModelProto, domain: str, name: str
) -> onnx.FunctionProto:
    """Converts an arbitrary ModelProto to a FunctionProto.

    Since function protos don't support initializers (or rather it does not make sense in the context of a function)
    we need to convert them to constants first.
    """
    model = _initializers_to_constants(model)
    model_ir = onnx_ir.serde.deserialize_model(model)
    function_ir = onnx_ir.Function(
        domain=domain, name=name, graph=model_ir.graph, attributes={}
    )
    return onnx_ir.to_proto(function_ir)
