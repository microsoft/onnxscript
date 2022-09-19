# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
from onnx import TensorProto
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx.numpy_helper import from_array
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    RuntimeException,
)


class OrtFunction:
    """
    Implements onnx models for many unary and binary operators
    with onnxruntime.
    """

    _mapping_function_op = {
        "__add__": ("Add", None),
        "__and__": ("And", None),
        "__eq__": ("Equal", np.dtype("bool")),
        "__ge__": ("GreaterOrEqual", np.dtype("bool")),
        "__gt__": ("Greater", np.dtype("bool")),
        "__le__": ("LessOrEqual", np.dtype("bool")),
        "__lt__": ("Less", np.dtype("bool")),
        "__matmul__": ("MatMul", None),
        "__mod__": ("Mod", None),
        "__mul__": ("Mul", None),
        "__neg__": ("Opp", None),
        "__or__": ("Or", None),
        "__pow__": ("Pow", None),
        "__sub__": ("Sub", None),
        "__truediv__": ("Div", None),
    }

    def __init__(self):
        self._onnx_models = {}
        self._ort_sessions = {}
        self._functions = {}

    def __getitem__(self, name_dtype):
        name = name_dtype[0]
        size = len(name_dtype) // 2
        dtypes = name_dtype[1 : 1 + size]
        shapes = name_dtype[1 + size :]
        if len(shapes) != len(dtypes):
            raise NotImplementedError(
                f"Unable to create a session for name_dtype={name_dtype!r}."
            )
        key = name_dtype
        if key not in self._functions:
            self.create(key, name, dtypes, shapes)
        return self._functions[key]

    def create(self, key, op_name, dtypes, shapes):
        """
        Creates an onnx model for given operator and instantiates
        an InferenceSession.

        :param key: cache the onnx model and the session using this key
        :param op_name: operator name (main domain)
        :param dtypes: list of dtypes for every input
        :param shapes: list of number of dimensions for every input
        :return: a lambda function calling with the expected signature
            and calling onnxruntime underneath
        """
        if len(dtypes) != len(shapes):
            raise RuntimeError(
                f"dtypes={dtypes!r} and shapes={shapes!r} should have the same size."
            )
        if op_name == "__getitem__":
            return self._create_index(key, op_name, dtypes, shapes)
        if op_name == "Squeeze":
            return self._create_squeeze(key, op_name, dtypes, shapes)
        return self._create_matrix_op(key, op_name, dtypes, shapes)

    def _create_matrix_op(self, key, op_name, dtypes, shapes):
        """
        Creates an onnx model for a matrix operator and instantiates
        an InferenceSession.

        :param key: cache the onnx model and the session using this key
        :param op_name: operator name (main domain)
        :param dtypes: list of dtypes for every input
        :param shapes: list of number of dimensions for every input
        :return: a lambda function calling with the expected signature
            and calling onnxruntime underneath
        """
        try:
            onnx_op, out_dtype = OrtFunction._mapping_function_op[op_name]
        except KeyError:
            raise NotImplementedError(
                f"Unable to create onnx model for operator {op_name!r}, "
                f"dtypes={dtypes!r}, shapes={shapes!r}."
            )
        if len(dtypes) not in (1, 2):
            raise NotImplementedError(
                f"Unable to create onnx model for operator {op_name!r}, "
                f"dtypes={dtypes!r}, shapes={shapes!r}."
            )
        onnx_dtypes = [NP_TYPE_TO_TENSOR_TYPE[dtype] for dtype in dtypes]
        shapes = [[None] * s for s in shapes]
        X = [
            make_tensor_value_info(f"X{i}", onnx_dtype, shape)
            for i, (onnx_dtype, shape) in enumerate(zip(onnx_dtypes, shapes))
        ]
        if out_dtype is None:
            Z = make_tensor_value_info("Z", onnx_dtypes[0], [])
        else:
            out_onnx_dtype = NP_TYPE_TO_TENSOR_TYPE[out_dtype]
            Z = make_tensor_value_info("Z", out_onnx_dtype, [])

        # handles specific cases first such as pow, mod
        if onnx_op == "Mod" and onnx_dtypes[0] in {
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        }:
            nodes = [
                make_node("CastLike", [X[1].name, X[0].name], ["c"]),
                make_node(onnx_op, [X[0].name, "c"], ["Z"], fmod=1),
            ]
        elif (
            op_name != "Pow"
            and len(onnx_dtypes) == 2
            and onnx_dtypes[0] != onnx_dtypes[1]
        ):
            # need of CastLike because input type are different
            nodes = [
                make_node("CastLike", [X[1].name, X[0].name], ["c"]),
                make_node(onnx_op, [X[0].name, "c"], ["Z"]),
            ]
        else:
            nodes = [make_node(onnx_op, [x.name for x in X], ["Z"])]
        return self._create_end(key, dtypes, shapes, X, Z, nodes)

    def _create_end(
        self, key, dtypes, shapes, X, Z, nodes, inits=None, call_ort_function=None
    ):
        graph = make_graph(nodes, f"eager_{key[0]}", X, [Z], initializer=inits)
        onnx_model = make_model(graph)
        check_model(onnx_model)
        # unused but useful to debug
        self._onnx_models[key] = onnx_model
        try:
            sess = InferenceSession(onnx_model.SerializeToString())
        except Fail as e:
            raise RuntimeError(
                f"Unable to create an InferenceSession for operator {key[0]!r}, "
                f"dtype={dtypes!r}, shapes={shapes!r} with onnx model\n{onnx_model}"
            ) from e
        self._ort_sessions[key] = sess

        if call_ort_function is None:

            def call_ort(*x):
                try:
                    return sess.run(["Z"], {f"X{i}": x for i, x in enumerate(x)})[0]
                except RuntimeException as e:
                    raise RuntimeError(
                        f"Unable to run onnxruntime, op_name={key[0]!r}, "
                        f"dtypes={dtypes!r}, shapes={shapes!r}, with input types with "
                        f"onnx model\n{onnx_model}"
                    ) from e

            self._functions[key] = call_ort
            return call_ort

        def call_ort(*x):
            try:
                return call_ort_function(onnx_model, sess, *x)
            except (RuntimeException, InvalidArgument, Fail) as e:
                raise RuntimeError(
                    f"Unable to run onnxruntime, op_name={key[0]!r}, dtypes={dtypes!r}, "
                    f"shapes={shapes!r}, with onnx model\n{onnx_model}"
                ) from e

        self._functions[key] = call_ort
        return call_ort

    def _create_index(self, key, op_name, dtypes, shapes):
        """
        Creates an onnx model for indexing and instantiates
        an InferenceSession.

        :param key: cache the onnx model and the session using this key
        :param op_name: operator name (main domain)
        :param dtypes: list of dtypes for every input
        :param shapes: list of number of dimensions for every input
        :return: a lambda function calling with the expected signature
            and calling onnxruntime underneath

        This function assumes default opset >= 13. (Squeeze)
        """
        if onnx_opset_version() < 13:
            raise RuntimeError(
                "onnx package is too old. A new version must be "
                "installed to support opset 13."
            )
        if len(dtypes) != 2:
            raise NotImplementedError(
                f"Unable to create onnx model for operator {op_name!r}, "
                f"dtypes={dtypes!r}, shapes={shapes!r}."
            )

        onnx_dtypes = [NP_TYPE_TO_TENSOR_TYPE[dtypes[0]]]
        if dtypes[1] in (slice, tuple):
            onnx_dtypes.append(TensorProto.INT64)
        else:
            onnx_dtypes.append(NP_TYPE_TO_TENSOR_TYPE[dtypes[1]])
        shapes = [[None] * s for s in shapes]
        Z = make_tensor_value_info("Z", onnx_dtypes[0], [])

        if dtypes[1] in (np.int32, np.int64) and len(shapes[1]) == 0:
            # notation A[i]
            X = [
                make_tensor_value_info(f"X{i}", onnx_dtype, shape)
                for i, (onnx_dtype, shape) in enumerate(zip(onnx_dtypes, shapes))
            ]
            nodes = [make_node("Gather", [x.name for x in X], ["Z"], axis=0)]
            return self._create_end(key, dtypes, shapes, X, Z, nodes)

        if dtypes[1] in (slice, tuple):
            # notation A[i:j] or A[i:j:k], A[i:k, k:l]
            X = [
                make_tensor_value_info("X0", onnx_dtypes[0], shapes[0]),
                make_tensor_value_info("X1", TensorProto.INT64, [None, None]),
            ]
            inits = [
                from_array(np.array([0], dtype=np.int64), name="i0"),
                from_array(np.array([1], dtype=np.int64), name="i1"),
                from_array(np.array([2], dtype=np.int64), name="i2"),
                from_array(np.array([3], dtype=np.int64), name="i3"),
            ]
            nodes = [
                make_node("Gather", [X[1].name, "i0"], ["starts_"], axis=0),
                make_node("Gather", [X[1].name, "i1"], ["ends_"], axis=0),
                make_node("Gather", [X[1].name, "i2"], ["axis_"], axis=0),
                make_node("Gather", [X[1].name, "i3"], ["steps_"], axis=0),
                make_node("Squeeze", ["starts_", "i0"], ["starts"]),
                make_node("Squeeze", ["ends_", "i0"], ["ends"]),
                make_node("Squeeze", ["axis_", "i0"], ["axis"]),
                make_node("Squeeze", ["steps_", "i0"], ["steps"]),
                make_node(
                    "Slice", [X[0].name, "starts", "ends", "axis", "steps"], ["Z"]
                ),
            ]

            sess_squeeze = self["Squeeze", dtypes[0], len(shapes[0])]

            def call_ort(onnx_model, sess, *x):
                if isinstance(x[1], slice):
                    slices = (x[1],)
                elif not isinstance(x[1], tuple):
                    raise TypeError(
                        f"Unexpected type for x[1] ({type(x[1])}), it should a tuple "
                        f"of slices, dtypes={dtypes}, shapes={shapes}."
                    )
                elif any(map(lambda t: not isinstance(t, (slice, int)), x[1])):
                    raise TypeError(
                        f"Unexpected type for x[1] ({[type(t) for t in x[1]]}), "
                        f"it should be slices, dtypes={dtypes}, shapes={shapes}."
                    )
                else:
                    slices = x[1]
                indices = []
                to_squeeze = []
                for axis, s in enumerate(slices):
                    if isinstance(s, slice):
                        if s.step is None or s.step > 0:
                            indices.append(
                                [
                                    s.start or 0,
                                    s.stop or x[0].shape[0],
                                    axis,
                                    s.step or 1,
                                ]
                            )
                        else:
                            indices.append(
                                [s.start or (x[0].shape[0] - 1), s.stop, axis, s.step]
                            )
                    else:
                        # integer
                        indices.append([s, s + 1, axis, 1])
                        to_squeeze.append(axis)
                index = np.array(indices, dtype=np.int64).T
                res = sess.run(["Z"], {"X0": x[0], "X1": index})[0]
                if len(to_squeeze) == 0:
                    return res

                # if one index is an integer, the dimension needs to be squeezed
                for ax in reversed(to_squeeze):
                    res = sess_squeeze(res, np.array([ax], dtype=np.int64))
                return res

            return self._create_end(
                key,
                dtypes,
                shapes,
                X,
                Z,
                nodes,
                inits=inits,
                call_ort_function=call_ort,
            )

        # inits = [from_array(np.array([0], dtype=np.int64))]
        raise NotImplementedError(
            f"dtypes={dtypes!r}, shapes={shapes!r} is not supported yet."
        )

    def _create_squeeze(self, key, op_name, dtypes, shapes):
        """
        Creates an onnx model for squeezing an axis and instantiates
        an InferenceSession.

        :param key: cache the onnx model and the session using this key
        :param op_name: operator name (main domain)
        :param dtypes: list of dtypes for every input
        :param shapes: list of number of dimensions for every input
        :return: a lambda function calling with the expected signature
            and calling onnxruntime underneath

        This function assumes default opset >= 13. (Squeeze)
        """
        if onnx_opset_version() < 13:
            raise RuntimeError(
                "onnx package is too old. A new version must be "
                "installed to support opset 13."
            )
        if len(dtypes) != 1:
            raise NotImplementedError(
                f"Unable to create onnx model for operator {op_name!r}, "
                f"dtypes={dtypes!r}, shapes={shapes!r}."
            )

        onnx_dtypes = [NP_TYPE_TO_TENSOR_TYPE[dtypes[0]]]
        shapes = [[None] * shapes[0]]
        Z = make_tensor_value_info("Z", onnx_dtypes[0], [])

        X = [
            make_tensor_value_info("X", onnx_dtypes[0], shapes[0]),
            make_tensor_value_info("axis", TensorProto.INT64, [None]),
        ]
        nodes = [make_node("Squeeze", ["X", "axis"], ["Z"])]

        def call_ort(onnx_model, sess, *x):
            return sess.run(["Z"], {"X": x[0], "axis": x[1]})[0]

        return self._create_end(
            key, dtypes, shapes, X, Z, nodes, call_ort_function=call_ort
        )


class EagerArray:
    """
    Wraps arrays to intercept calls to operators and use onnxruntime
    to process the output.
    """

    _cache = OrtFunction()

    def __init__(self, tensor):
        if not isinstance(tensor, np.ndarray):
            raise TypeError(
                f"Unexpected type {type(tensor)}. It must be a numpy array."
            )
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

    def __bool__(self):
        return self.value.__bool__()

    def __int__(self):
        return self.value.__int__()

    def _ort_op(self, b, op, check=True):
        if isinstance(b, (int, float)):
            b = np.array(b)
        if b is None:
            f = EagerArray._cache[op, self.dtype, None, len(self.shape), 0]
        elif isinstance(b, (slice, tuple)):
            f = EagerArray._cache[op, self.dtype, type(b), len(self.shape), 1]
        else:
            f = EagerArray._cache[
                op, self.dtype, b.dtype, len(self.shape), len(b.shape)
            ]
        if isinstance(b, EagerArray):
            if check and self.dtype != b.dtype:
                raise TypeError(
                    f"Binary operation with different element type {self.dtype} "
                    f"and {b.dtype}."
                )
            v = f(self.value, b.value)
        else:
            v = f(self.value, b)
        # array(4., dtype=float32) - array(0., dtype=float32) -> 4.0 (not an array anymore)
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        return EagerArray(v)

    def __getitem__(self, index):
        return self._ort_op(index, "__getitem__")

    def __neg__(self):
        return self._ort_op(None, "__neg__")

    def __add__(a, b):
        return a._ort_op(b, "__add__")

    def __radd__(a, b):
        return a._ort_op(b, "__add__")

    def __and__(a, b):
        return a._ort_op(b, "__and__")

    def __rand__(a, b):
        return a._ort_op(b, "__and__")

    def __mod__(a, b):
        return a._ort_op(b, "__mod__")

    def __mul__(a, b):
        return a._ort_op(b, "__mul__")

    def __rmul__(a, b):
        return a._ort_op(b, "__mul__")

    def __matmul__(a, b):
        return a._ort_op(b, "__matmul__")

    def __or__(a, b):
        return a._ort_op(b, "__or__")

    def __pow__(a, b):
        return a._ort_op(b, "__pow__", check=False)

    def __sub__(a, b):
        return a._ort_op(b, "__sub__")

    def __truediv__(a, b):
        return a._ort_op(b, "__truediv__")

    def __lt__(a, b):
        return a._ort_op(b, "__lt__")

    def __le__(a, b):
        return a._ort_op(b, "__le__")

    def __eq__(a, b):
        return a._ort_op(b, "__eq__")

    def __ne__(a, b):
        return a._ort_op(b, "__ne__")

    def __ge__(a, b):
        return a._ort_op(b, "__ge__")

    def __gt__(a, b):
        return a._ort_op(b, "__gt__")
