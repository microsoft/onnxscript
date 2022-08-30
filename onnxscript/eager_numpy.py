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
        '__add__': ('Add', None),
        '__and__': ('And', None),
        '__eq__': ('Equal', np.dtype('bool')),
        '__ge__': ('GreaterOrEqual', np.dtype('bool')),
        '__gt__': ('Greater', np.dtype('bool')),
        '__le__': ('LessOrEqual', np.dtype('bool')),
        '__lt__': ('Less', np.dtype('bool')),
        '__matmul__': ('MatMul', None),
        '__mul__': ('Mul', None),
        '__neg__': ('Opp', None),
        '__or__': ('Or', None),
        '__pow__': ('Pow', None),
        '__sub__': ('Sub', None),
        '__truediv__': ('Div', None),
    }

    def __init__(self):
        self._onnx_models = {}
        self._ort_sessions = {}
        self._functions = {}

    def __getitem__(self, name_dtype):
        name = name_dtype[0]
        dtypes = name_dtype[1:]
        if name not in self._functions:
            self._functions[name] = {}
        typed_functions = self._functions[name]
        if dtypes not in typed_functions:
            typed_functions[dtypes] = self.create(name, dtypes)
        return typed_functions[dtypes]

    def create(self, op_name, dtypes):
        try:
            onnx_op, out_dtype = OrtFunction._mapping_function_op[op_name]
        except KeyError:
            raise NotImplementedError(
                f"Unable to create onnx model for operator {op_name!r} and dtype={dtypes!r}.")
        if len(dtypes) not in (1, 2):
            raise NotImplementedError(
                f"Unable to create onnx model for operator {op_name!r} and dtype={dtypes!r}.")
        onnx_dtypes = tuple(NP_TYPE_TO_TENSOR_TYPE[dtype] for dtype in dtypes)
        X = [make_tensor_value_info(f'X{i}', onnx_dtype, [])
             for i, onnx_dtype in enumerate(onnx_dtypes)]
        if out_dtype is None:
            Z = make_tensor_value_info('Z', onnx_dtypes[0], [])
        else:
            out_onnx_dtype = NP_TYPE_TO_TENSOR_TYPE[out_dtype]
            Z = make_tensor_value_info('Z', out_onnx_dtype, [])
        if op_name != 'Pow' and len(onnx_dtypes) == 2 and onnx_dtypes[0] != onnx_dtypes[1]:
            # need of CastLike because input type are different
            nodes = [
                make_node('CastLike', [X[1].name, X[0].name], ['c']),
                make_node(onnx_op, [X[0].name, 'c'], ['Z'])]
        else:
            nodes = [make_node(onnx_op, [x.name for x in X], ['Z'])]
        graph = make_graph(nodes, f"eager_{op_name}", X, [Z])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        # unused but useful to debug
        if op_name not in self._onnx_models:
            self._onnx_models[op_name] = {}
            self._ort_sessions[op_name] = {}
        self._onnx_models[op_name][dtypes] = onnx_model
        try:
            sess = InferenceSession(onnx_model.SerializeToString())
        except Fail as e:
            raise RuntimeError(
                f"Unable to create an InferenceSession for operator {op_name!r} "
                f"and dtype={dtypes!r} with onnx model\n{onnx_model}") from e
        self._ort_sessions[op_name][dtypes] = sess
        f = lambda *x: sess.run(['Z'], {f'X{i}': x for i, x in enumerate(x)})[0]
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
        if isinstance(b, (int, float)):
            b = np.array(b)
        f = EagerArray._cache[op, self.dtype, None if b is None else b.dtype]
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
        return self._ort_op(None, '__neg__')

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
