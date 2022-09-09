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
        from onnxscript.onnx_opset import opset14
        self._opset = opset or opset14

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

    def __getitem__(a, index):
        op = a._opset
        if op.version < 13:
            raise RuntimeError("Indexing requires opset 13 or later.")
        if isinstance(index, int):
            # case A[i]: indexing
            # promote integer input to tensor
            i = EagerArray(np.array(index))
            # use Gather to perform indexing
            return op.Gather(a, i, axis=0)
        if not isinstance(index, (slice, tuple)):
            raise TypeError(f"Unexpected type {type(index)} for index.")
        # case A[i:j] or A[i:j:k], A[i:k, j:l]
        if isinstance(index, slice):
            # Treat 1-dimensional slicing A[i:j] as generic n-dimensional case A[i:j,...]
            index = (index, )
        shape = a.shape
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
        result = op.Slice(a, starts, ends, axis, steps)
        if len(to_squeeze) > 0:
            result = EagerArray(np.squeeze(result.value, axis=tuple(to_squeeze)))
        return result

    def __mod__(a, b):
        if a.onnx_dtype in {TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16, TensorProto.BFLOAT16}:
            return a._opset.Mod(a, b, fmod=1)
        else:
            return a._opset.Mod(a, b)

    def __ne__(a, b):
        temp = a._opset.Equal(a, b)
        return a._opset.Not(temp)

    def __neg__(a):
        return a._opset.Neg(a)

    def __add__(a, b):
        return a._opset.Add(a, b)

    def __radd__(a, b):
        return a._opset.Add(b, a)

    def __and__(a, b):
        return a._opset.And(a, b)

    def __rand__(a, b):
        return a._opset.And(b, a)

    def __mul__(a, b):
        return a._opset.Mul(a, b)

    def __rmul__(a, b):
        return a._opset.Mul(b, a)

    def __matmul__(a, b):
        return a._opset.MatMul(a, b)

    def __or__(a, b):
        return a._opset.Or(a, b)

    def __pow__(a, b):
        return a._opset.Pow(a, b)

    def __sub__(a, b):
        return a._opset.Sub(a, b)

    def __truediv__(a, b):
        return a._opset.Div(a, b)

    def __lt__(a, b):
        return a._opset.Less(a, b)

    def __le__(a, b):
        return a._opset.LessOrEqual(a, b)

    def __eq__(a, b):
        return a._opset.Equal(a, b)

    def __ge__(a, b):
        return a._opset.GreaterOrEqual(a, b)

    def __gt__(a, b):
        return a._opset.Greater(a, b)
