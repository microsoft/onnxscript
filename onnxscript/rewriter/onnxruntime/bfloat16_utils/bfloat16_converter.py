# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging

from onnxscript import ir

logger = logging.getLogger(__name__)


def _convert_inputs_from_bfloat16_to_float16(value: ir.Input) -> None:
    if value.dtype != ir.DataType.BFLOAT16:
        return
    value.dtype = ir.DataType.FLOAT16
    _insert_cast_nodes_for_float16_to_bfloat16_to_inputs(value)


def _convert_outputs_from_bfloat16_to_float16(value: ir.Value) -> None:
    if value.dtype != ir.DataType.BFLOAT16:
        return
    _insert_cast_nodes_for_bfloat16_to_float16_to_outputs(value)


def _insert_cast_nodes_for_float16_to_bfloat16_to_inputs(value: ir.Input) -> None:
    user_nodes_and_indices = tuple(value.uses())

    attr = ir.AttrInt64(name="to", value=ir.DataType.BFLOAT16)
    cast = ir.Node(
        domain="",
        op_type="Cast",
        inputs=[value],
        num_outputs=1,
        attributes=[attr],
    )
    cast.outputs[0].dtype = ir.DataType.BFLOAT16
    cast.outputs[0].shape = value.shape

    for node, index in tuple(value.uses()):
        if node is cast:
            continue
        node.replace_input_with(index, cast.outputs[0])

    # NOTE: A safer way to insert the cast node is to prepend it to the first node
    # of the graph
    assert user_nodes_and_indices[0][0].graph is not None, "The node should belong to a graph"
    user_nodes_and_indices[0][0].graph[0].prepend(cast)


def _insert_cast_nodes_for_bfloat16_to_float16_to_outputs(value: ir.Value) -> None:
    node = value.producer()
    index = value.index()
    if node is None or index is None:
        logger.warning("Output value %s has no producer or index", value)
        return

    attr = ir.AttrInt64(name="to", value=ir.DataType.FLOAT16)
    cast = ir.Node(
        domain="",
        op_type="Cast",
        inputs=[node.outputs[index]],
        num_outputs=1,
        attributes=[attr],
    )
    cast.outputs[0].dtype = ir.DataType.FLOAT16
    cast.outputs[0].shape = node.outputs[index].shape
    node.append(cast)

    assert node.graph is not None, "Node graph should not be None"
    # Update graph/function outputs
    for idx, graph_or_function_output in enumerate(node.graph.outputs):
        if graph_or_function_output == node.outputs[index]:
            node.graph.outputs[idx] = cast.outputs[0]
    # Swap the output name of the node with the output name of the cast node to
    # preserve the output name in the graph
    node.outputs[index].name, cast.outputs[0].name = (
        cast.outputs[0].name,
        node.outputs[index].name,
    )


def dtype_adapter_for_bfloat16_model(model: ir.Model) -> None:
    """Adapt the model datatype if it's bfloat16.

    Because onnxruntime does not support bfloat16 as input/output datatype, we need to
    convert the bfloat16 datatype to float16. This function will convert the bfloat16
    datatype to float16 and insert Cast nodes to convert float16 to bfloat16.

    Model:
        inputs(float16) -> Cast(bfloat16) -> nodes(bfloat16) -> Cast(float16) -> outputs(float16)

    TODO: Delete this function after onnxruntime supports bfloat16.

    Args:
        model: The model to adapt.

    """
    for input in model.graph.inputs:
        _convert_inputs_from_bfloat16_to_float16(input)
    for output in model.graph.outputs:
        _convert_outputs_from_bfloat16_to_float16(output)
