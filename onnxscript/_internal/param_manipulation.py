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
        allow_extra_kwargs: Whether to allow extra keyword arguments.
            When set to True, extra/unknown arguments will be ignored.

    Returns:
        A tuple of two elements:
        - A list of ONNX inputs.
        - An ordered dictionary of ONNX attribute names and values.

    Raises:
        TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
        TypeError: When a required input is not provided.
    """
    # args, kwargs and param_schemas should be all in order
    # user may not specify all inputs or attributes

    all_param_names = {param.name for param in param_schemas}
    extra_kwargs = set(kwargs).difference(all_param_names)
    if extra_kwargs and not allow_extra_kwargs:
        raise TypeError(f"Unexpected keyword arguments '{extra_kwargs}'")

    onnx_inputs = []
    onnx_attributes = collections.OrderedDict()

    for i, param in enumerate(param_schemas):
        if param.is_variadic_input:
            # Exhaust all remaining args
            onnx_inputs.extend(args[i:])
            args = []
            continue
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
        elif param.required:
            raise TypeError(f"Required input '{param}' was not provided")

    return onnx_inputs, onnx_attributes
