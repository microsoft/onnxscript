"""Function for manipulating input parameters of an Op or a OnnxFunction."""
from __future__ import annotations

import collections
from typing import Any, OrderedDict, Sequence

from onnxscript import values


def separate_input_attributes_from_arguments(
    param_schemas: Sequence[values.ParamSchema],
    args,
    kwargs,
    fill_defaults: bool = True,
    allow_extra_kwargs: bool = False,
) -> tuple[list[Any], OrderedDict[str, Any]]:
    """Separate Python args and kwargs into ONNX inputs and attributes.

    Args:
        param_schemas: The parameter schemas of an Op or a OnnxFunction.
        args: The Python positional arguments supplied by the caller.
        kwargs: The Python keyword arguments supplied by the caller.
        fill_defaults: Whether to fill the default values for attributes.

    Returns:
        A tuple of two elements:
        - A list of ONNX inputs.
        - An ordered dictionary of ONNX attribute names and values.
    """
    # args, kwargs and param_schemas should be all in order
    # user may not specify all attributes
    if len(args) + len(kwargs) > len(param_schemas):
        raise TypeError("Inputs are more than expected in schema")

    all_param_names = {param.name for param in param_schemas}
    extra_kwargs = set(kwargs).difference(all_param_names)
    if extra_kwargs and not allow_extra_kwargs:
        raise TypeError(f"Unexpected keyword arguments '{extra_kwargs}'")

    onnx_inputs = []
    onnx_attributes = collections.OrderedDict()

    for i, param in enumerate(param_schemas):
        if i < len(args):
            if param.is_input:
                onnx_inputs.append(args[i])
            else:
                onnx_attributes[param.name] = args[i]
        elif param.name in kwargs:
            if param.is_input:
                onnx_inputs.append(kwargs[param.name])
            else:
                onnx_attributes[param.name] = kwargs[param.name]
        elif (
            param.is_attribute
            and param.default is not values._EmptyDefault  # pylint: disable=protected-access
        ):
            # User did not provide the attribute
            if fill_defaults:
                onnx_attributes[param.name] = param.default
            else:
                continue
        else:
            raise TypeError(f"Required argument '{param}' was not provided")

    return onnx_inputs, onnx_attributes
