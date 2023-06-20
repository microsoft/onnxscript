from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from onnx.defs import OpSchema

from onnxscript import tensor, values


def get_dtype(pyvalue):
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
            return get_dtype(pyvalue[0])
        raise ValueError("Cannot determine target type for empty list")
    raise TypeError(f"Value of unexpected type {type(pyvalue)}")


def cast_pyvalue_to_os_tensor(pyvalue, dtype=None):
    """Promotes python values into onnxscript tensors.
    The optional argument dtype specifies the desired np.dtype of the tensor,
    used only when a non-onnxscript-tensor is promoted into one.
    """
    if isinstance(pyvalue, tensor.Tensor):
        return pyvalue
    if isinstance(pyvalue, np.ndarray):
        if dtype is not None and pyvalue.dtype != dtype:
            pyvalue = pyvalue.astype(dtype)
        return tensor.Tensor(pyvalue)
    if dtype is None:
        dtype = get_dtype(pyvalue)
    if isinstance(pyvalue, (bool, int, float, list)):
        return tensor.Tensor(np.array(pyvalue, dtype=dtype))
    return pyvalue


def cast_inputs(
    get_type_info: Callable[[Any], Any],
    cast: Callable[[Any, Any], Any],
    op_schema: OpSchema,
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


def static_cast_inputs(converter, op_schema: Optional[OpSchema], args) -> tuple[str, ...]:
    """Used for autocast during script-translation."""

    def get_type_info(x):
        return x if not x.is_const() else None

    def cast(x, typeinfo) -> str:
        if x.is_const() and typeinfo is not None:
            # Scalar values are promoted to tensors of a type chosen as below:

            tmp = converter.generate_unique_name(f"{x.name}_cast")
            converter.emit(
                [tmp],
                values.Op(converter.default_opset, "CastLike"),
                [x.name, typeinfo],
                [],
            )
            return tmp
        return x.name

    return cast_inputs(get_type_info, cast, op_schema, args)
