# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
from onnx import TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


class EagerArray:
    """
    Wraps arrays to intercept calls to operators and use onnxruntime
    to process the output.
    """

    def __init__(self, tensor, opset=None):
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"Unexpected type {type(tensor)}. It must be a numpy array.")
        self._tensor = tensor
        from onnxscript.onnx_opset import default_opset
        self._opset = opset or default_opset

    @property
    def value(self):
        return self._tensor

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def onnx_dtype(self):
        return NP_TYPE_TO_TENSOR_TYPE[self.dtype]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r})"

    def __bool__(self):
        return self.value.__bool__()

    def __int__(self):
        return self.value.__int__()

    def __getitem__(self, index):
        op = self._opset
        if op.version < 13:
            raise RuntimeError("Indexing requires opset 13 or later.")
        if isinstance(index, int):
            # case A[i]: indexing
            # promote integer input to tensor
            i = EagerArray(np.array(index))
            # use Gather to perform indexing
            return op.Gather(self, i, axis=0)
        if not isinstance(index, (slice, tuple)):
            raise TypeError(f"Unexpected type {type(index)} for index.")
        # case A[i:j] or A[i:j:k], A[i:k, j:l]
        if isinstance(index, slice):
            # Treat 1-dimensional slicing A[i:j] as generic n-dimensional case A[i:j,...]
            index = (index, )
        shape = self.shape
        indices = []
        to_squeeze = []
        for axis, s in enumerate(index):
            if isinstance(s, slice):
                if s.step is None or s.step > 0:
                    indices.append([s.start or 0, s.stop or shape[axis],
                                    axis, s.step or 1])
                else:
                    indices.append([s.start or (shape[axis] - 1), s.stop,
                                    axis, s.step])
            elif isinstance(s, int):
                indices.append([s, s + 1, axis, 1])
                to_squeeze.append(axis)
            else:
                raise TypeError(f"Unexpected type {type(s)}: slice or int expected.")
        indices = np.array(indices, dtype=np.int64).T
        starts = EagerArray(indices[0])
        ends = EagerArray(indices[1])
        axis = EagerArray(indices[2])
        steps = EagerArray(indices[3])
        result = op.Slice(self, starts, ends, axis, steps)
        if len(to_squeeze) > 0:
            result = EagerArray(np.squeeze(result.value, axis=tuple(to_squeeze)))
        return result

    def __mod__(self, other):
        if self.onnx_dtype in {TensorProto.FLOAT, TensorProto.DOUBLE,
                            TensorProto.FLOAT16, TensorProto.BFLOAT16}:
            return self._opset.Mod(self, other, fmod=1)
        else:
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
