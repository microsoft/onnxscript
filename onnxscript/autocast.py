from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from onnx.defs import OpSchema

from onnxscript import tensor, values


def cast_inputs(
    get_type_info: Callable[[Any], Any],
    cast: Callable[[Any, Any], Any],
    op_schema: OpSchema,
    *args,
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
            if not expected.isHomogeneous:  # type: ignore[attr-defined]
                args_typevars.append((x, None))
                continue
        else:
            raise ValueError(
                f"Number of actual parameters {len(args)} "
                f"exceeds number of formal parameters {len(expected_inputs)}."
            )
        typevar = expected.typeStr  # type: ignore[attr-defined]
        if "(" not in typevar:
            # typevar is an identifier, like "T"
            typeinfo = get_type_info(x)
            if typeinfo is not None:
                type_bindings[typevar] = typeinfo
        args_typevars.append((x, typevar))
    cast_args = [cast(x, type_bindings.get(typevar)) for x, typevar in args_typevars]
    return tuple(cast_args)


def dynamic_cast_inputs(op_schema: OpSchema, *args):
    """Used for autocast during eager-mode execution."""

    def get_type_info(x):
        return x.dtype if isinstance(x, tensor.Tensor) else None

    def cast(x, typeinfo) -> tensor.Tensor:
        if isinstance(x, (bool, int, float)):
            # Scalar values are promoted to tensors of a type chosen as below:
            if typeinfo is not None:
                dtype = typeinfo
            elif isinstance(x, bool):
                dtype = np.bool_
            elif isinstance(x, int):
                dtype = np.int64
            else:
                assert isinstance(x, float)
                dtype = np.float32
            return tensor.Tensor(np.array(x, dtype=dtype))
        return x

    return cast_inputs(get_type_info, cast, op_schema, *args)


def static_cast_inputs(converter, op_schema: Optional[OpSchema], *args):
    """Used for autocast during script-translation."""
    if op_schema is None:
        return args

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

    return cast_inputs(get_type_info, cast, op_schema, *args)
