import logging

from onnxscript import ir

logger = logging.getLogger(__name__)


def _convert_inputs_from_bfloat16_to_float16(value: ir.Input) -> None:
    if value.dtype != ir.DataType.BFLOAT16:
        return
    value.dtype = ir.DataType.FLOAT16
    value_users = tuple(value.uses())
    for node, index in value_users:
        _insert_cast_nodes_for_float16_to_bfloat16_to_inputs(node, index)


def _convert_outputs_from_bfloat16_to_float16(value: ir.Value) -> None:
    if value.dtype != ir.DataType.BFLOAT16:
        return
    node = value.producer()
    index = value.index()
    if node is None or index is None:
        logger.warning("Output value %s has no producer or index", value)
        return
    _insert_cast_nodes_for_bfloat16_to_float16_to_outputs(node, index)


def _insert_cast_nodes_for_float16_to_bfloat16_to_inputs(node: ir.Node, index: int) -> None:
    attr = ir.AttrInt64(name="to", value=ir.DataType.BFLOAT16)
    cast = ir.Node(
        domain="",
        op_type="Cast",
        inputs=[node.inputs[index]],
        num_outputs=1,
        attributes=[attr],
    )
    cast.outputs[0].dtype = ir.DataType.BFLOAT16  # type: ignore[union-attr]
    cast.outputs[0].shape = node.inputs[index].shape  # type: ignore[union-attr]
    node.prepend(cast)
    node.replace_input_with(index, cast.outputs[0])


def _insert_cast_nodes_for_bfloat16_to_float16_to_outputs(node: ir.Node, index: int) -> None:
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
