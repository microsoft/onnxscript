# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Function for manipulating input parameters of an Op or a OnnxFunction."""

from __future__ import annotations

import collections
from typing import Any, OrderedDict

from onnxscript import ir


def separate_input_attributes_from_arguments(
    op_signature: ir.schemas.OpSignature,
    args,
    kwargs,
    fill_defaults: bool = True,
    allow_extra_kwargs: bool = False,
) -> tuple[list[Any], OrderedDict[str, Any]]:
    """Separate Python args and kwargs into ONNX inputs and attributes.

    Args:
        op_signature: The operator signature containing parameter information.
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
    # args, kwargs and op_signature.params should be all in order
    # user may not specify all inputs or attributes

    all_param_names = {param.name for param in op_signature.params}
    extra_kwargs = set(kwargs).difference(all_param_names)
    if extra_kwargs and not allow_extra_kwargs:
        raise TypeError(f"Unexpected keyword arguments '{extra_kwargs}'")

    onnx_inputs = []
    onnx_attributes = collections.OrderedDict()

    for i, param in enumerate(op_signature.params):
        is_input = isinstance(param, ir.schemas.Parameter)
        is_variadic = isinstance(param, ir.schemas.Parameter) and param.variadic

        if is_variadic:
            # Exhaust all remaining args
            onnx_inputs.extend(args[i:])
            args = []
            continue
        if i < len(args):
            if is_input:
                onnx_inputs.append(args[i])
            else:
                onnx_attributes[param.name] = args[i]
        elif param.name in kwargs:
            if is_input:
                onnx_inputs.append(kwargs[param.name])
            else:
                onnx_attributes[param.name] = kwargs[param.name]
        elif isinstance(param, ir.schemas.AttributeParameter) and param.has_default():
            # User did not provide the attribute
            if fill_defaults:
                # Extract the value from the Attr object
                onnx_attributes[param.name] = param.default.value
        elif param.required:
            raise TypeError(f"Required input '{param}' was not provided")

    return onnx_inputs, onnx_attributes


def tag_arguments_with_signature(
    op_signature: ir.schemas.OpSignature,
    args,
    kwargs,
    fill_defaults: bool = True,
    allow_extra_kwargs: bool = False,
) -> tuple[
    list[tuple[Any, ir.schemas.Parameter | ir.schemas.AttributeParameter]],
    dict[str, tuple[Any, ir.schemas.Parameter | ir.schemas.AttributeParameter]],
]:
    """Tag Python args and kwargs with matching ONNX Parameter/AttributeParameter.

    Args:
        op_signature: The operator signature containing parameter information.
        args: The Python positional arguments supplied by the caller.
        kwargs: The Python keyword arguments supplied by the caller.
        fill_defaults: Whether to fill the default values for attributes.
        allow_extra_kwargs: Whether to allow extra keyword arguments.
            When set to True, extra/unknown arguments will be ignored.

    Returns:
        A tuple of two elements:
        - A list of tuple of Python positional argument and Parameter/AttributeParameter.
        - An ordered dictionary of Python keyword argument names and tuple of argument
            value and Parameter/AttributeParameter.

    Raises:
        TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
        TypeError: When a required input is not provided.
    """
    # args, kwargs and op_signature.params should be all in order
    # user may not specify all inputs or attributes

    all_param_names = {param.name for param in op_signature.params}
    extra_kwargs = set(kwargs).difference(all_param_names)
    if extra_kwargs and not allow_extra_kwargs:
        raise TypeError(f"Unexpected keyword arguments '{extra_kwargs}'")

    tagged_args: list[tuple[Any, ir.schemas.Parameter | ir.schemas.AttributeParameter]] = []
    tagged_kwargs: dict[
        str, tuple[Any, ir.schemas.Parameter | ir.schemas.AttributeParameter]
    ] = {}

    for i, param in enumerate(op_signature.params):
        is_variadic = isinstance(param, ir.schemas.Parameter) and param.variadic

        if is_variadic:
            # Exhaust all remaining args
            tagged_args.extend((arg, param) for arg in args[i:])
            args = []
            continue
        if i < len(args):
            tagged_args.append((args[i], param))
        elif param.name in kwargs:
            tagged_kwargs[param.name] = (kwargs[param.name], param)
        elif param.has_default():
            # User did not provide the input/attribute
            if fill_defaults:
                default_value = param.default
                # Extract value from Attr object if it's an AttributeParameter
                if isinstance(param, ir.schemas.AttributeParameter):
                    default_value = param.default.value
                tagged_kwargs[param.name] = (default_value, param)
        elif param.required:
            raise TypeError(f"Required input/attribute '{param}' was not provided")

    return tagged_args, tagged_kwargs


def turn_to_kwargs_to_avoid_ordering(
    op_signature: ir.schemas.OpSignature,
    inputs: list[Any],
    attributes: dict[str, Any],
) -> dict[str, Any]:
    """Return the inputs and attributes to the order of the function signature."""
    for idx, param in enumerate(op_signature.params):
        if param.name not in attributes:
            is_variadic = isinstance(param, ir.schemas.Parameter) and param.variadic
            if is_variadic:
                attributes[param.name] = inputs[idx:]
            elif inputs:
                attributes[param.name] = inputs.pop(0)
    return attributes
