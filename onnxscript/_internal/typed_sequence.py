# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Iterable


class _TypedSequence(list):
    """A list subclass that preserves ONNX dtype information.

    ONNX sequences need to carry element type information, but Python lists
    don't have this natively. This class wraps a list with dtype info, which is
    especially important for empty sequences where type cannot be inferred.

    Example::
        seq = _TypedSequence(onnx.TensorProto.FLOAT16)
        # seq is now an empty list that knows it should contain FLOAT16 tensors
    """

    __slots__ = ("_onnx_dtype",)

    def __init__(self, dtype: int, iterable: Iterable[Any] | None = None) -> None:
        super().__init__(iterable if iterable is not None else ())
        self._onnx_dtype = dtype

    @property
    def onnx_dtype(self) -> int:
        """Return the ONNX TensorProto data type for sequence elements."""
        return self._onnx_dtype

    def __repr__(self) -> str:
        return f"_TypedSequence(dtype={self._onnx_dtype}, {list(self)})"
