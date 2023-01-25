"""Function for manipulating input parameters of an Op or a OnnxFunction."""
from __future__ import annotations

import collections
import dataclasses
from typing import Any, List, OrderedDict, Sequence, Tuple

# A special value to indicate that the default value is not specified
EmptyDefault = object()


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
    default: Any = EmptyDefault
    is_input: bool = True

    def __repr__(self) -> str:
        param_kind = "INPUT" if self.is_input else "ATTRIBUTE"
        text = f"{self.name}<{param_kind}>: {self.type}"
        if self.default is not EmptyDefault:
            text += f" = {self.default}"
        return text

    @property
    def is_attribute(self) -> bool:
        """Returns True if the parameter is an ONNX attribute."""
        return not self.is_input


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
    raise NotImplementedError()
