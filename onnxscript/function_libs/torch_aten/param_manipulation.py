"""Function for manipulating input parameters of an Op or a OnnxFunction."""
from __future__ import annotations

import collections
import dataclasses
from typing import Any, List, OrderedDict, Sequence, Tuple
import warnings

import onnx

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
    type: Any = None  # Op input does not have a type, for now
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


def extract_param_schema_from_function(onnx_func: values.OnnxFunction) -> List[ParamSchema]:

    function_ir = onnx_func.function_ir
    # The first len(func_ir.inputs) arguments are onnx inputs
    inputs = function_ir.inputs
    # The rest is onnx attributes
    attributes = function_ir.attrs
    # Construct a dictionary of attributes with their names specified in the function
    # definition
    attr_name_to_protos = collections.OrderedDict(
        (attr.name, attr) for attr in function_ir.attr_protos
    )

    # args with default value are attributes
    param_schemas = []
    for arg in inputs:
        param_schema = ParamSchema(name=arg.name, type=arg.typeinfo, is_input=True)
        param_schemas.append(param_schema)

    for attr_name in attributes:
        # FIXME(justinchuby): Where can we find the type?
        param_schema = ParamSchema(name=attr_name, type=None, is_input=False)
        param_schemas.append(param_schema)

    for name, attr_value in attr_name_to_protos.items():
        param_schema = ParamSchema(
            name=name,
            type=_ATTRIBUTE_TYPE_TO_PYTHON_TYPE[attr_value.type],
            default=_get_attribute_value(attr_value.attr_proto),
            is_input=False,
        )
        param_schemas.append(param_schema)
    return param_schemas


_ATTRIBUTE_TYPE_TO_PYTHON_TYPE = {
    onnx.defs.OpSchema.AttrType.FLOAT: float,
    onnx.defs.OpSchema.AttrType.INT: int,
    onnx.defs.OpSchema.AttrType.STRING: str,
    onnx.defs.OpSchema.AttrType.TENSOR: None,
    onnx.defs.OpSchema.AttrType.GRAPH: None,
    onnx.defs.OpSchema.AttrType.SPARSE_TENSOR: None,
    onnx.defs.OpSchema.AttrType.TYPE_PROTO: None,
    onnx.defs.OpSchema.AttrType.FLOATS: Sequence[float],
    onnx.defs.OpSchema.AttrType.INTS: Sequence[int],
    onnx.defs.OpSchema.AttrType.STRINGS: Sequence[str],
    onnx.defs.OpSchema.AttrType.TENSORS: None,
    onnx.defs.OpSchema.AttrType.GRAPHS: None,
    onnx.defs.OpSchema.AttrType.SPARSE_TENSORS: None,
    onnx.defs.OpSchema.AttrType.TYPE_PROTOS: None,
}


def _get_attribute_value(attr_proto):
    if attr_proto.type == onnx.AttributeProto.UNDEFINED:
        return _EmptyDefault
    if attr_proto.type == onnx.AttributeProto.FLOAT:
        return attr_proto.f
    if attr_proto.type == onnx.AttributeProto.INT:
        return attr_proto.i
    if attr_proto.type == onnx.AttributeProto.STRING:
        return attr_proto.s
    if attr_proto.type == onnx.AttributeProto.FLOATS:
        return [float(v) for v in attr_proto.f]
    if attr_proto.type == onnx.AttributeProto.INTS:
        return [int(v) for v in attr_proto.i]
    raise TypeError(f"Unsupported attribute type: {attr_proto.type}")


def extract_param_schema_from_op_schema(op_schema: onnx.defs.OpSchema) -> List[ParamSchema]:
    param_schemas = []
    for input_ in op_schema.inputs:
        param_schema = ParamSchema(name=input_.name, is_input=True)
        param_schemas.append(param_schema)
    for attr_name, attribute in op_schema.attributes.items():
        default_attr_proto = attribute.default_value
        param_schema = ParamSchema(
            name=attr_name,
            type=_ATTRIBUTE_TYPE_TO_PYTHON_TYPE[attribute.type],
            default=_get_attribute_value(default_attr_proto),
            is_input=False,
        )
        param_schemas.append(param_schema)

    return param_schemas


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
    # TODO: Remove print statements
    print("args", args)
    print("kwargs", kwargs)
    print("param_schemas", param_schemas)
    if len(args) + len(kwargs) > len(param_schemas):
        # TODO(justinchuby): Log diagnostic information.
        warnings.warn(
            f"Inputs are more than expected in schema. Expected {len(param_schemas)} arguments, "
            f"but got {len(args)} positional arguments and {len(kwargs)} keyword arguments. "
            f"The extra inputs will be ignored. "
            f"args: {args}, kwargs: {kwargs}, param_schemas: {param_schemas}"
        )
    if len(args) > len(param_schemas):
        raise TypeError(
            f"Too many arguments are provided. Expected {len(param_schemas)} arguments, "
            f"but got {len(args)} positional arguments and {len(kwargs)} keyword arguments."
        )

    onnx_inputs = []
    onnx_attributes = OrderedDict()
    for i, param in enumerate(param_schemas):
        if i < len(args):
            if not param.is_attribute:
                onnx_inputs.append(args[i])
            else:
                onnx_attributes[param.name] = args[i]
        elif param.name in kwargs:
            if not param.is_attribute:
                onnx_inputs.append(kwargs[param.name])
            else:
                onnx_attributes[param.name] = kwargs[param.name]
        else:
            # input doesn't have default
            onnx_attributes[param.name] = param.default

    return onnx_inputs, onnx_attributes
