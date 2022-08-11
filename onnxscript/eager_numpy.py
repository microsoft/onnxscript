# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np


class NumpyArray:
    """
    Wraps numpy arrays to intercept calls to operators.
    """
    def __init__(self, tensor):
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"Unexpected type {type(tensor)}. It must be a numpy array.")
        self._tensor = tensor

    @property
    def value(self):
        return self._tensor

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r})"

    def __getitem__(self, index):
        raise NotImplementedError()

    def __bool__(self):
        return self.value.__bool__()

    def __int__(self):
        return self.value.__int__()

    def __mod__(self, b):
        if isinstance(b, float):
            return NumpyArray(np.fmod(self.value, b))
        return NumpyArray(self._tensor % b)

    def _unary_op(self, op):
        return NumpyArray(op(self.value))

    def __neg__(self):
        return self._unary_op(lambda x: -x)

    def _bin_op(self, b, op, check=True):
        if isinstance(b, NumpyArray):
            if check and self.dtype != b.dtype:
                raise TypeError(
                    f"Binary operation with different element type {self.dtype} "
                    f"and {b.dtype}.")
            v = op(self.value, b.value)
        else:
            v = op(self.value, b)
        # array(4., dtype=float32) - array(0., dtype=float32) -> 4.0 (not an array anymore)
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        return NumpyArray(v)

    def __add__(a, b):
        return a._bin_op(b, lambda x, y: x + y)

    def __radd__(a, b):
        return a._bin_op(b, lambda x, y: x + y)

    def __and__(a, b):
        return a._bin_op(b, lambda x, y: x & y)

    def __mul__(a, b):
        return a._bin_op(b, lambda x, y: x * y)

    def __matmul__(a, b):
        return a._bin_op(b, lambda x, y: x @ y)

    def __or__(a, b):
        return a._bin_op(b, lambda x, y: x | y)

    def __pow__(a, b):
        return a._bin_op(b, (lambda x, y: x ** y), check=False)

    def __sub__(a, b):
        return a._bin_op(b, lambda x, y: x - y)

    def __truediv__(a, b):
        return a._bin_op(b, lambda x, y: x / y)

    def __lt__(a, b):
        return a._bin_op(b, lambda x, y: x < y)

    def __le__(a, b):
        return a._bin_op(b, lambda x, y: x <= y)

    def __eq__(a, b):
        return a._bin_op(b, lambda x, y: x == y)

    def __ne__(a, b):
        return a._bin_op(b, lambda x, y: x != y)

    def __ge__(a, b):
        return a._bin_op(b, lambda x, y: x >= y)

    def __gt__(a, b):
        return a._bin_op(b, lambda x, y: x > y)
