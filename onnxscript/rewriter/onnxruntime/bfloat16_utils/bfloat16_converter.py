from onnxscript import ir

def _convert_inputs_from_bfloat16_to_float16(value: ir.Input):
    if value.dtype == ir.DataType.BFLOAT16:
        value.dtype = ir.DataType.FLOAT16
        for node, index in value.uses():
            _insert_cast_nodes_for_float16_to_bfloat16(node)

def _convert_outputs_from_bfloat16_to_float16(value: ir.Output):
    if value.dtype == ir.DataType.BFLOAT16:
        value.dtype = ir.DataType.FLOAT16
        for node in value.producers():
            _insert_cast_nodes_for_float16_to_bfloat16(node)

def _insert_cast_nodes_for_float16_to_bfloat16(node: ir.Node):
    pass

def dtype_adapter_for_bfloat16_model(model: ir.Model) -> ir.Model:
    """Adapt the model datatype if it's bfloat16.

    Because onnxruntime does not support bfloat16 as input/output datatype, we need to
    convert the bfloat16 datatype to float16. This function will convert the bfloat16
    datatype to float16 and insert Cast nodes to convert float16 to bfloat16.

    TODO: Delete this function after onnxruntime supports bfloat16.

    Args:
        model: The model to adapt.

    Returns:
        The adapted model.
    """
    for input in model.graph.inputs:
        _convert_inputs_from_bfloat16_to_float16(input)
    for output in model.graph.outputs:
        _convert_outputs_from_bfloat16_to_float16(output)
    return model
