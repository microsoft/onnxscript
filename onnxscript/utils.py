# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Any
import onnx
from onnx import TensorProto

# TODO: enable invocation of ORT kernels


class Model:
    def __init__(self, onnxfile) -> None:
        # delayed import to avoid having a strong dependency on onnxruntime
        from onnxruntime import InferenceSession
        self.session = InferenceSession(onnxfile)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        inputs = self.session.get_inputs()
        for i, arg in enumerate(args):
            kwds[inputs[i].name] = arg
        return self.session.run(None, kwds)


def make_ort_value(v):
    pass


def make_attrs(**kwds):
    pass


def call_ort(opname, *args, **kwds):
    model = Model("todo")
    ort_args = [make_ort_value(x) for x in args]
    return model(ort_args)


def convert_arrays_to_value_infos(names, arr_list):
    value_infos = []
    for name, arr in zip(names, arr_list):
        elem_type: TensorProto.DataType
        shape: tuple
        if isinstance(arr, np.ndarray):
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
            shape = arr.shape
        else:
            raise ValueError(f"cannot covert a {type(arr)} to value_info")

        value_info = onnx.helper.make_tensor_value_info(
            name=name,
            elem_type=elem_type,
            shape=shape)
        value_infos.append(value_info)
    return value_infos
