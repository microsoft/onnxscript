# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import onnx.helper
from onnx import TensorProto

from onnxscript import onnx_opset
from onnxscript._internal import autocast


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
        # FIXME(justinhuby): Create a better way to determine the opset version
        self._opset: Any = opset or onnx_opset.opset18

    @property
    def value(self) -> np.ndarray:
        if self._nparray is None:
            raise ValueError("Tensor does not have a value.")
        return self._nparray

    @property
    def rank(self) -> int:
        return len(self.value.shape)

    @property
    def is_scalar(self) -> bool:
        return self.rank == 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    @property
    def onnx_dtype(self) -> int:
        return onnx.helper.np_dtype_to_tensor_dtype(self.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"

    def __bool__(self) -> bool:
        return bool(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __len__(self) -> int:
        return self.shape[0]

    def __index__(self) -> int:
        return self.value.__index__()

    def __getitem__(self, index):
        op = self._opset
        if op.version < 13:
            raise RuntimeError("Indexing requires opset 13 or later.")
        if not isinstance(index, tuple):
            # Normalize representation to a tuple.
            # A single index-value is equivalent to a tuple with a single element.
            index = (index,)
        if len(index) > self.rank:
            raise ValueError(
                f"Number of indices {len(index)} is greater than rank {self.rank}"
            )

        # Promote integer indices to tensors of rank 0
        index = [autocast.cast_pyvalue_to_os_tensor(x) for x in index]
        # Process all elements in index
        shape = self.shape
        sliced_indices = []
        scalar_indices = []
        to_squeeze = []
        non_scalar_indices = []
        for axis_, s in enumerate(index):
            if isinstance(s, slice):
                if s.start is None and s.stop is None and s.step is None:
                    continue
                if s.step is None or s.step > 0:
                    sliced_indices.append(
                        [
                            s.start or 0,
                            s.stop if s.stop is not None else shape[axis_],
                            axis_,
                            s.step or 1,
                        ]
                    )
                else:
                    sliced_indices.append(
                        [
                            s.start if s.start is not None else (shape[axis_] - 1),
                            s.stop if s.stop is not None else -(shape[axis_] + 1),
                            axis_,
                            s.step,
                        ]
                    )
            elif isinstance(s, Tensor):
                if s.is_scalar:
                    scalar_indices.append([s, s + 1, axis_, 1])
                    to_squeeze.append(axis_)
                else:
                    non_scalar_indices.append((axis_, s))
            else:
                raise TypeError(f"Unexpected type {type(s)}: slice or int expected.")

        # Non-scalar-indexing requires the use of ONNX Gather operation.
        # Slicing can be implemented efficiently using ONNX's Slice operation.
        # Scalar-indexing can be implemented using either Gather or with the Slice operation.
        # We map scalar-indexing into the Slice operation, except in the special case
        # of a single scalar-index (with no other sliced_index), which we map directly
        # to a Gather.

        if not (sliced_indices or scalar_indices or non_scalar_indices):
            # Edge case: no index specified. Eg. A[:, :]
            return op.Identity(self)
        if not sliced_indices and len(scalar_indices) == 1:
            # Special case of indexing along a single axis: A[i], A[:, i], A[:, :, i] etc.
            # promote integer input to tensor
            axis = to_squeeze[0]
            index_value = index[axis]
            # use Gather to perform indexing
            result = op.Gather(self, index_value, axis=axis)
        elif sliced_indices or scalar_indices:
            sliced_indices = sliced_indices + scalar_indices
            indices = np.array(sliced_indices, dtype=np.int64).T
            starts = Tensor(indices[0])
            ends = Tensor(indices[1])
            axes = Tensor(indices[2])
            steps = Tensor(indices[3])
            result = op.Slice(self, starts, ends, axes, steps)
            if to_squeeze:
                result = Tensor(np.squeeze(result.value, axis=tuple(to_squeeze)))
        else:
            result = self
        for axis, value in non_scalar_indices:
            result = op.Gather(result, value, axis=axis)

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

    def __rsub__(self, other):
        return self._opset.Sub(other, self)

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
