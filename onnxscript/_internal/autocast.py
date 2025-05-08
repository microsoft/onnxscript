# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
import onnx
import onnx.helper  # noqa: TID251
from onnx.defs import OpSchema

from onnxscript import ir, tensor

if TYPE_CHECKING:
    from onnxscript import converter

# Conversions from python values to ONNX are used by both the script converter as well
# as the eager-mode runtime and both need to be consistent. The script converter converts
# python values into ONNX TensorProto, while the runtime converts python values into
# ONNXScript runtime's value-representation (based on Tensor).


# Utilities to convert a python value to TensorProto (for use by the script converter)


def pyvalue_to_onnx_tensor(tensor_name: str, pyvalue):
    return ir.serde.serialize_tensor(ir.tensor(pyvalue, name=tensor_name))


_REPEATED_ATTRIBUTE_TYPES = frozenset(
    {
        onnx.AttributeProto.FLOATS,
        onnx.AttributeProto.INTS,
        onnx.AttributeProto.STRINGS,
        onnx.AttributeProto.TENSORS,
        onnx.AttributeProto.GRAPHS,
        onnx.AttributeProto.SPARSE_TENSORS,
        onnx.AttributeProto.TYPE_PROTOS,
    }
)


def pyvalue_to_onnx_attribute(
    key: str,
    value: Any,
    name_generator: Callable[[], str],
    attr_type: Optional[onnx.AttributeProto.AttributeType] = None,
) -> onnx.AttributeProto:
    """Helper function to create an ONNX AttributeProto.

    This is a refinement of onnx.helper.make_attribute that works with ONNX Script
    conventions for allowed types for attribute-values. In particular, it allows
    * Empty lists as attribute values, provided the attribute type is specified
    and is a list type.
    * Scalar-values like 1.0 as well as lists like [1, -1] to be specified
    when the attribute type is TensorProto by automatically converting the value
    into a 0-D or 1-D tensor respectively.
    """
    if isinstance(value, list) and not value:
        # Empty list value:
        if attr_type is None:
            raise ValueError("Attribute type must be specified for empty list value.")
        if attr_type not in _REPEATED_ATTRIBUTE_TYPES:
            raise ValueError("Empty list value is only allowed for repeated attribute types.")
        return onnx.AttributeProto(name=key, type=attr_type)
    elif attr_type == onnx.AttributeProto.TENSOR and not isinstance(value, onnx.TensorProto):
        return onnx.AttributeProto(
            name=key, type=attr_type, t=pyvalue_to_onnx_tensor(name_generator(), value)
        )
    else:
        # When the value is a subgraph, ONNX IR will complain that some values are
        # not found from the scope.
        return onnx.helper.make_attribute(key, value)  # noqa: TID251


# Utilities to convert python values into onnxscript tensors.


def _promotable(x) -> bool:
    """Checks if a runtime parameter value needs to be promoted into an onnxscript value.
    This is the runtime-equivalent of the promotion of literal constants into ONNX values
    in the static converter.
    """
    if isinstance(x, (bool, int, float)):
        return True
    if isinstance(x, list) and x:
        # Note: This is meant to handle valid scenarios correctly. No attempt is
        # made yet to capture all invalid usages in runtime mode.
        return _promotable(x[0])
    return False


def _get_dtype(pyvalue):
    """Return np.dtype to use when converting a python value to an onnxscript tensor.
    Note that int constants are treated as int64, as that is the common type in ONNX
    for shape/index values.
    """
    if isinstance(pyvalue, bool):
        return np.bool_
    elif isinstance(pyvalue, int):
        return np.int64
    elif isinstance(pyvalue, float):
        return np.float32
    elif isinstance(pyvalue, list):
        if pyvalue:
            # TODO: What to do about lists with mixed value types, like [1, 2.0]?
            # Should at least produce an error/warning message.
            return _get_dtype(pyvalue[0])
        raise ValueError("Cannot determine target type for empty list")
    raise TypeError(f"Value of unexpected type {type(pyvalue)}")


def cast_pyvalue_to_os_tensor(pyvalue, dtype=None):
    """Promotes python values into onnxscript tensors.
    The optional argument dtype specifies the desired np.dtype of the tensor,
    used only when a non-standard onnxscript-value is promoted into one.
    """
    if _promotable(pyvalue):
        if dtype is None:
            dtype = _get_dtype(pyvalue)
        return tensor.Tensor(np.array(pyvalue, dtype=dtype))
    return pyvalue


def cast_inputs(
    get_type_info: Callable[[Any], Any],
    cast: Callable[[Any, Any], Any],
    op_schema: OpSchema | None,
    args,
) -> tuple[Any, ...]:
    """Uses schema specification to support a limited form of auto-casting.

    * Scalars are promoted to tensors.
    * Further. they are cast to the required type when used in ops with other
    tensor inputs that are required to be of same type.
    Thus, in "A+1" or "Add(A, 1)", the value 1 will be converted to the same
    type as A.

    This is used by the converter in a static-mode, as well as by the eager-mode
    execution in a dynamic-mode.
    """
    if op_schema is None:
        # Either an error or a custom op.
        # No checks/casts in this case.
        return tuple(cast(x, None) for x in args)

    expected_inputs = op_schema.inputs
    # We make two passes. In the first pass, we identify known type-bindings for
    # type-variables: eg., {'T1' : np.float32, 'T2' : np.int32}.
    # In the second pass, we use these bindings to cast scalar-values to
    # tensors of appropriate types. The two passes are needed to handle cases
    # like "Add(1, X)" where 1 must be cast to the same type as X.
    type_bindings: dict[Optional[str], np.dtype] = {}
    args_typevars: list[tuple[str, Optional[str]]] = []
    for i, x in enumerate(args):
        if i < len(expected_inputs):
            expected = expected_inputs[i]
        elif expected_inputs[-1].option == OpSchema.FormalParameterOption.Variadic:
            expected = expected_inputs[-1]
            if not expected.is_homogeneous:
                args_typevars.append((x, None))
                continue
        else:
            raise ValueError(
                f"Number of actual parameters {len(args)} "
                f"exceeds number of formal parameters {len(expected_inputs)}."
            )
        typevar = expected.type_str
        if "(" not in typevar:
            # typevar is an identifier, like "T"
            typeinfo = get_type_info(x)
            if typeinfo is not None:
                type_bindings[typevar] = typeinfo
        args_typevars.append((x, typevar))
    cast_args = [cast(x, type_bindings.get(typevar)) for x, typevar in args_typevars]
    return tuple(cast_args)


def dynamic_cast_inputs(op_schema: OpSchema, args):
    """Used for autocast during eager-mode execution."""

    def get_type_info(x):
        return x.dtype if isinstance(x, tensor.Tensor) else None

    return cast_inputs(get_type_info, cast_pyvalue_to_os_tensor, op_schema, args)


def static_cast_inputs(
    converter_: converter.Converter,
    op_schema: Optional[OpSchema],
    args: Sequence[Optional[converter.Variable]],
) -> tuple[str, ...]:
    """Used for autocast during script-translation.
    This is meant to transform expressions like "Add(X, 1)" to "Add(X, CastLike(1, X))"
    Polymorphic constants (like 0 and 1) are cast to the type of other operands as needed.
    """

    def get_type_info(x: Optional[converter.Variable]) -> Optional[converter.Variable]:
        """Returns x back if x can serve as the target-type for a cast (as the second
        argument of CastLike) and None otherwise. In the expression "Add(X, 1), 1 is
        castable, while X can serve as the target-type.
        """
        return None if x is None or x.is_castable else x

    def cast_like(
        x: Optional[converter.Variable], y: Optional[converter.Variable]
    ) -> Optional[str]:
        if x is None:
            return None
        if x.is_castable and y is not None:
            # Polymorphic constant x is cast to the type of y:
            x_cast = converter_.generate_unique_name(f"{x.name}_cast")
            converter_.emit([x_cast], "CastLike", [x.name, y.name])
            return x_cast
        return x.name

    return cast_inputs(get_type_info, cast_like, op_schema, args)
