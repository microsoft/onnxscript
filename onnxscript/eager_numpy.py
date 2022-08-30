# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
from onnx import numpy_helper, TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import Fail


class OrtFunction:
    """
    Creates and retain function based on onnxruntime.
    """
    _mapping_function_op = {
        '__add__': ('Add', None, 2),
        '__and__': ('And', None, 2),
        '__eq__': ('Equal', np.dtype('bool'), 2),
        '__ge__': ('GreaterOrEqual', np.dtype('bool'), 2),
        '__gt__': ('Greater', np.dtype('bool'), 2),
        '__le__': ('LessOrEqual', np.dtype('bool'), 2),
        '__lt__': ('Less', np.dtype('bool'), 2),
        '__matmul__': ('MatMul', None, 2),
        '__mul__': ('Mul', None, 2),
        '__neg__': ('Opp', None, 1),
        '__or__': ('Or', None, 2),
        '__pow__': ('Pow', None, 2),
        '__sub__': ('Sub', None, 2),
        '__truediv__': ('Div', None, 2),
    }

    def __init__(self):
        self._onnx_models = {}
        self._ort_sessions = {}
        self._functions = {}

    def __getitem__(self, name_dtype):
        dtype, name = name_dtype
        if name not in self._functions:
            self._functions[name] = {}
        typed_functions = self._functions[name]
        if dtype not in typed_functions:
            typed_functions[dtype] = self.create(name, dtype)
        return typed_functions[dtype]

    def create(self, op_name, dtype):
        try:
            onnx_op, out_dtype, n_inputs = OrtFunction._mapping_function_op[op_name]
        except KeyError:
            raise NotImplementedError(
                f"Unable to create onnx model for operator {op_name!r} and dtype={dtype!r}.")
        if n_inputs not in (1, 2):
            raise NotImplementedError(
                f"Unable to create onnx model for operator {op_name!r} and dtype={dtype!r}.")
        onnx_dtype = NP_TYPE_TO_TENSOR_TYPE[dtype]
        X = make_tensor_value_info('X', onnx_dtype, [])
        if n_inputs > 1:
            Y = make_tensor_value_info('Y', onnx_dtype, [])
        if out_dtype is None:
            Z = make_tensor_value_info('Z', onnx_dtype, [])
        else:
            out_onnx_dtype = NP_TYPE_TO_TENSOR_TYPE[out_dtype]
            Z = make_tensor_value_info('Z', out_onnx_dtype, [])
        node = make_node(onnx_op, ['X', 'Y'], ['Z'])
        graph = make_graph([node], 'lr', [X, Y], [Z])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        # unused but useful to debug
        if op_name not in self._onnx_models:
            self._onnx_models[op_name] = {}
            self._ort_sessions[op_name] = {}
        self._onnx_models[op_name][dtype] = onnx_model
        try:
            sess = InferenceSession(onnx_model.SerializeToString())
        except Fail as e:
            raise RuntimeError(
                f"Unable to create an InferenceSession for operator {op_name!r} "
                f"and dtype={dtype!r} with onnx model\n{onnx_model}") from e
        self._ort_sessions[op_name][dtype] = sess
        f = lambda x, y: sess.run(['Z'], {'X': x, 'Y': y})[0]
        return f


class EagerArray:
    """
    Wraps arrays to intercept calls to operators and use onnxruntime
    to process the output.
    """

    _cache = OrtFunction()

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
            return EagerArray(np.fmod(self.value, b))
        return EagerArray(self._tensor % b)

    def _ort_op(self, b, op, check=True):
        f = EagerArray._cache[self.dtype, op]
        if isinstance(b, EagerArray):
            if check and self.dtype != b.dtype:
                raise TypeError(
                    f"Binary operation with different element type {self.dtype} "
                    f"and {b.dtype}.")
            v = f(self.value, b.value)
        else:
            v = f(self.value, b)
        # array(4., dtype=float32) - array(0., dtype=float32) -> 4.0 (not an array anymore)
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        return EagerArray(v)

    def __neg__(self):
        return self._ort_op('__neg__')

    def __add__(a, b):
        return a._ort_op(b, '__add__')

    def __radd__(a, b):
        return a._ort_op(b, '__add__')

    def __and__(a, b):
        return a._ort_op(b, '__and__')

    def __rand__(a, b):
        return a._ort_op(b, '__and__')

    def __mul__(a, b):
        return a._ort_op(b, '__mul__')

    def __rmul__(a, b):
        return a._ort_op(b, '__mul__')

    def __matmul__(a, b):
        return a._ort_op(b, '__matmul__')

    def __or__(a, b):
        return a._ort_op(b, '__or__')

    def __pow__(a, b):
        return a._ort_op(b, '__pow__', check=False)

    def __sub__(a, b):
        return a._ort_op(b, '__sub__')

    def __truediv__(a, b):
        return a._ort_op(b, '__truediv__')

    def __lt__(a, b):
        return a._ort_op(b, '__lt__')

    def __le__(a, b):
        return a._ort_op(b, '__le__')

    def __eq__(a, b):
        return a._ort_op(b, '__eq__')

    def __ne__(a, b):
        return a._ort_op(b, '__ne__')

    def __ge__(a, b):
        return a._ort_op(b, '__ge__')

    def __gt__(a, b):
        return a._ort_op(b, '__gt__')
