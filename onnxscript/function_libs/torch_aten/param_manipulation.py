"""Function for manipulating input parameters of an Op or a OnnxFunction."""
from __future__ import annotations

import collections
import dataclasses
from typing import Any, OrderedDict, Sequence, Tuple

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

    def __repr__(self):
        param_kind = "INPUT" if self.is_input else "ATTRIBUTE"
        text = f"{self.name}<{param_kind}>: {self.type}"
        if self.default is not EmptyDefault:
            text += f" = {self.default}"
        return text

    @property
    def is_attribute(self):
        """Check if the parameter is an ONNX attribute."""
        return not self.is_input


def separate_input_attributes_from_arguments(
    param_schemas: Sequence[ParamSchema],
    args,
    kwargs,
) -> Tuple[OrderedDict[str, Any], OrderedDict[str, Any]]:
    pass
