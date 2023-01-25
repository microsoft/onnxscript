"""Function for manipulating input parameters of an Op or a OnnxFunction."""
from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, List, OrderedDict, Sequence, Tuple, Union

from onnxscript import values

# A special value to indicate that the default value is not specified
_EmptyDefault = object()


@dataclasses.dataclass(frozen=True)
class ParamSchema:
    """A schema for a parameter of an Op or a OnnxFunction.

    Attributes:
        name: The name of the parameter.
        type: The type of the parameter.
        default: The default value of the parameter.
        is_input: Whether the parameter is an ONNX input.
    """

    name: str
    type: type
    default: Any = _EmptyDefault
    is_input: bool = True

    def __repr__(self) -> str:
        param_kind = "INPUT" if self.is_input else "ATTRIBUTE"
        text = f"{self.name}<{param_kind}>: {self.type}"
        if self.default is not _EmptyDefault:
            text += f" = {self.default}"
        return text

    @property
    def is_attribute(self) -> bool:
        """Returns True if the parameter is an ONNX attribute."""
        return not self.is_input


def extract_param_schema_from_function(
    func: Union[values.OnnxFunction, Callable]
) -> List[ParamSchema]:
    if isinstance(func, values.OnnxFunction):
        func = func.function
    params_dict = dict(inspect.getfullargspec(func)._asdict())
    print(params_dict)
    input_length = len(params_dict["args"]) - len(params_dict["defaults"])
    inputs = params_dict["args"][:input_length]
    attributes = params_dict["args"][input_length:]
    # args with default value are attributes
    param_schema_list = []
    for arg_name in inputs:
        arg_annotation = params_dict["annotations"].get(arg_name)
        # FIXME: better way to check annotation existence?
        assert arg_annotation is not None
        param_schema = ParamSchema(
            name=arg_name, type=arg_annotation, default=_EmptyDefault, is_input=True
        )
        param_schema_list.append(param_schema)
    for arg_name, arg_default in zip(attributes, params_dict["defaults"]):
        arg_annotation = params_dict["annotations"].get(arg_name)
        # FIXME: better way to check annotation existence?
        assert arg_annotation is not None
        param_schema = ParamSchema(
            name=arg_name, type=arg_annotation, default=arg_default, is_input=False
        )
        param_schema_list.append(param_schema)
    return param_schema_list


def extract_param_schema_from_op(op: values.Op):
    # TODO(titaiwang): How do you tell the param is attr/input
    params_dict = dict(inspect.getfullargspec(op)._asdict())
    print(params_dict)


def separate_input_attributes_from_arguments(
    param_schemas: Sequence[ParamSchema],
    args,
    kwargs,
) -> Tuple[List[Any], OrderedDict[str, Any]]:
    """Separate Python args and kwargs into ONNX inputs and attributes.

    Args:
        param_schemas: The parameter schemas of an Op or a OnnxFunction.
        args: The Python positional arguments supplied by the caller.
        kwargs: The Python keyword arguments supplied by the caller.

    Returns:
        A tuple of two elements:
        - A list of ONNX inputs.
        - An ordered dictionary of ONNX attribute names and values.
    """
    # args, kwargs and param_schemas should be all in order
    # user might not specify all attributes
    assert len(args) + len(kwargs) <= len(param_schemas)
    onnx_inputs, onnx_attributes = [], OrderedDict()
    for idx, param in enumerate(param_schemas):
        if idx < len(args):
            if not param.is_attribute:
                onnx_inputs.append(args[idx])
            onnx_attributes[param.name] = args
        elif param.name in kwargs:
            onnx_attributes[param.name] = kwargs[param.name]
        else:
            onnx_attributes[param.name] = param.default

    return onnx_inputs, onnx_attributes
