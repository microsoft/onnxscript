# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from onnx import TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnxscript import onnx_opset


class Tensor:
    """An implementation of ONNX Tensors, based on a wrapper around numpy arrays.
    Serves to define overloaded ops with an ONNX/ONNXScript semantics.
    """

    def __init__(self, nparray: Optional[np.ndarray], opset=None):
        if nparray is not None and not isinstance(nparray, np.ndarray):
            raise TypeError(
                f"Unexpected type {type(nparray)}. It must be a numpy array or None."
            )

        self._nparray = nparray
        self._opset: Any = opset or onnx_opset.default_opset

    @property
    def value(self) -> np.ndarray:
        if self._nparray is None:
            raise ValueError("Tensor does not have a value.")
        return self._nparray

    @property
    def rank(self) -> int:
        return len(self.value.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    @property
    def onnx_dtype(self) -> TensorProto.DataType:
        # FIXME: NP_TYPE_TO_TENSOR_TYPE is deprecated
        return NP_TYPE_TO_TENSOR_TYPE[self.dtype]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"

    def __bool__(self) -> bool:
        return bool(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __index__(self) -> int:
        return self.value.__index__()

    def __getitem__(self, index):
        op = self._opset
        if op.version < 13:
            raise RuntimeError("Indexing requires opset 13 or later.")
        if isinstance(index, int):
            # case A[i]: indexing
            # promote integer input to tensor
            i = Tensor(np.array(index))
            # use Gather to perform indexing
            return op.Gather(self, i, axis=0)
        if not isinstance(index, (slice, tuple)):
            raise TypeError(f"Unexpected type {type(index)} for index.")
        # case A[i:j] or A[i:j:k], A[i:k, j:l]
        if isinstance(index, slice):
            # Treat 1-dimensional slicing A[i:j] as generic n-dimensional case A[i:j,...]
            index = (index,)
        shape = self.shape
        indices_ = []
        to_squeeze = []
        for axis_, s in enumerate(index):
            if isinstance(s, slice):
                if s.step is None or s.step > 0:
                    indices_.append([s.start or 0, s.stop or shape[axis_], axis_, s.step or 1])
                else:
                    indices_.append([s.start or (shape[axis_] - 1), s.stop, axis_, s.step])
            elif isinstance(s, int):
                indices_.append([s, s + 1, axis_, 1])
                to_squeeze.append(axis_)
            else:
                raise TypeError(f"Unexpected type {type(s)}: slice or int expected.")
        indices = np.array(indices_, dtype=np.int64).T
        starts = Tensor(indices[0])
        ends = Tensor(indices[1])
        axis = Tensor(indices[2])
        steps = Tensor(indices[3])
        result = op.Slice(self, starts, ends, axis, steps)
        if to_squeeze:
            result = Tensor(np.squeeze(result.value, axis=tuple(to_squeeze)))
        return result

    def __mod__(self, other):
        if self.onnx_dtype in {
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        }:
            return self._opset.Mod(self, other, fmod=1)
        return self._opset.Mod(self, other)

    def __ne__(self, other):
        temp = self._opset.Equal(self, other)
        return self._opset.Not(temp)

    def __neg__(self):
        return self._opset.Neg(self)

    def __add__(self, other):
        return self._opset.Add(self, other)

    def __radd__(self, other):
        return self._opset.Add(other, self)

    def __and__(self, other):
        return self._opset.And(self, other)

    def __rand__(self, other):
        return self._opset.And(other, self)

    def __mul__(self, other):
        return self._opset.Mul(self, other)

    def __rmul__(self, other):
        return self._opset.Mul(other, self)

    def __matmul__(self, other):
        return self._opset.MatMul(self, other)

    def __or__(self, other):
        return self._opset.Or(self, other)

    def __pow__(self, other):
        return self._opset.Pow(self, other)

    def __sub__(self, other):
        return self._opset.Sub(self, other)

    def __truediv__(self, other):
        return self._opset.Div(self, other)

    def __lt__(self, other):
        return self._opset.Less(self, other)

    def __le__(self, other):
        return self._opset.LessOrEqual(self, other)

    def __eq__(self, other):
        return self._opset.Equal(self, other)

    def __ge__(self, other):
        return self._opset.GreaterOrEqual(self, other)

    def __gt__(self, other):
        return self._opset.Greater(self, other)
