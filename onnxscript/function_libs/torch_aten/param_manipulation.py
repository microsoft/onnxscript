
def _split_args_kwargs_to_input_attr(
    onnx_func: onnxscript.OnnxFunction,
    args: Sequence[ValidArgumentType],
    kwargs: Dict[str, ValidArgumentType],
) -> Tuple[Sequence[ValidInputType], Dict[str, ValidArgumentType]]:
    """Split the args and kwargs supplied to `onnx_func` to onnx inputs and attributes.

    Args:
        onnx_func: The onnx function.
        args: The positional arguments supplied to `onnx_func`.
        kwargs: The keyword arguments supplied to `onnx_func`.

    Returns:
        A tuple of (onnx_inputs, onnx_attributes).
    """
    function_ir = onnx_func.function_ir
    # The first len(func_ir.inputs) arguments are onnx inputs
    onnx_inputs = args[: len(function_ir.inputs)]
    # The rest is onnx attributes
    attributes_in_args = args[len(function_ir.inputs) :]
    # Construct a dictionary of attributes with their names specified in the function
    # definition
    onnx_attributes = {}

    # (1) Some/All attributes are supplied as positional arguments
    attr_name_to_protos = collections.OrderedDict(
        (attr.name, attr) for attr in function_ir.attr_protos
    )

    assert len(function_ir.attr_protos) >= len(attributes_in_args)
    for attr_proto, attr_value in zip(attr_name_to_protos.values(), attributes_in_args):
        onnx_attributes[attr_proto.name] = attr_value

    # (2) Some/All attributes are supplied as kwargs
    for key, value in kwargs.items():
        # (3) Some arguments in kwargs are not defined in the onnx function
        if key not in attr_name_to_protos:
            warnings.warn(f"Attribute '{key}' is not defined in the function definition")
            continue

        onnx_attributes[key] = value

    # (4) Fill in the default values from the attr_proto if not supplied by caller
    for key, attr_proto in attr_name_to_protos.items():
        if key not in onnx_attributes:
            onnx_attributes[key] = attr_proto.value

    return onnx_inputs, onnx_attributes
