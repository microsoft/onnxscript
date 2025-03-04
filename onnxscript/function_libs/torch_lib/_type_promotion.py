"""Type promotion functions for op implementations."""

from typing import Sequence
from onnxscript import ir

def _get_higher_dtype(a: ir.DataType, b: ir.DataType) -> ir.DataType:
    """Get the higher dtype of two dtypes."""
    # Reference: https://github.com/pytorch/pytorch/blob/bdd942efd76e74baa5dd0a262f7c843ddfe2e11b/torch/_prims_common/__init__.py#L1160
    if a == b:
        return a

    if a is None:
        return b

    if b is None:
        return a

    ordered_datatypes = (
        (ir.DataType.BOOL,),
        (ir.DataType.UINT8, ir.DataType.INT8),
        (ir.DataType.INT16,),
        (ir.DataType.INT32,),
        (ir.DataType.INT64,),
        (ir.DataType.FLOAT16, ir.DataType.BFLOAT16),
        (ir.DataType.FLOAT,),
        (ir.DataType.DOUBLE,),
        (ir.DataType.COMPLEX64,),
        (ir.DataType.COMPLEX128,),
    )

    for idx, dtypes in enumerate(ordered_datatypes):
        if a in dtypes and b in dtypes:
            return ordered_datatypes[idx + 1][0]
        if a in dtypes:
            return b
        if b in dtypes:
            return a

    raise ValueError(f"Unexpected data types: {a}, {b}")


def promote_types(op, values: Sequence[ir.Value]) -> Sequence[ir.Value]:
    """Promote the types of the given values."""
    if not values:
        return ()

    for value in values:
        if value.dtype is None:
            raise ValueError(f"Value {value} does not have dtype information and cannot be promoted.")

    promoted = values[0].dtype
    assert promoted is not None
    for value in values[1:]:
        dtype = value.dtype
        assert dtype is not None
        promoted = _get_higher_dtype(promoted, dtype)

    results = []
    for value in values:
        if value.dtype != promoted:
            new_val = op.Cast(value, to=promoted)
            new_val.dtype = promoted
            new_val.shape = value.shape
            results.append(new_val)
        else:
            results.append(value)

    return results
