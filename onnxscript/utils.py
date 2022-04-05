# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import numbers
import numpy as np
from typing import Any, Sequence, Text
import onnx
from onnx import TensorProto, ValueInfoProto, \
    ModelProto, OperatorSetIdProto, FunctionProto
from .converter import Converter

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
        elif isinstance(arr, numbers.Number):
            nparray = np.array(arr)
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[nparray.dtype]
            shape = nparray.shape
        else:
            raise ValueError(f"cannot covert a {type(arr)} to value_info")

        value_info = onnx.helper.make_tensor_value_info(
            name=name,
            elem_type=elem_type,
            shape=shape)
        value_infos.append(value_info)
    return value_infos


def convert_python_function_to_function_proto(function, domain, opset_imports):
    converter = Converter()
    module = importlib.import_module(function.__module__)

    ir_functions = converter.convert(inspect.getsource(module))
    ir_functions = [
        x for x in ir_functions if x.name == function.__name__]
    if len(ir_functions) != 1:
        raise ValueError(f"Cannot find signle function of \
            '{function.__name__}' from module '{module.__name__}.py'")

    return ir_functions[0].to_function_proto_with_opset_imports(
        domain, opset_imports)


def make_model_from_function_proto(
        function_proto: FunctionProto,
        input_value_infos: Sequence[ValueInfoProto],
        output_value_infos: Sequence[ValueInfoProto],
        domain: Text,
        onnx_opset_imports: Sequence[OperatorSetIdProto],
        local_opset_import: OperatorSetIdProto,
        **attrs: Any
        ) -> ModelProto:
    """Creates a model containing a single call to a given
        function with input and output value_infos, etc.

    Arguments:
        FunctionProto (FunctionProto): function proto
            representing a single call
        input_value_infos (list of ValueInfoProto): function's input
        output_value_infos (list of ValueInfoProto): function's output
        domain (string): domain of the node for the function
        onnx_opset_imports (string, default None): opsets that are used by the function
        local_opset_import (string, default None): opset of the function
        **attrs (dict): the attributes of the node for the function
    Returns:
        ModelProto
    """

    input_names = [vi.name for vi in input_value_infos]
    output_names = [vi.name for vi in output_value_infos]
    node = onnx.helper.make_node(
        function_proto.name, input_names, output_names,
        domain=domain,
        **(attrs or {}))
    graph = onnx.helper.make_graph(
        [node], "node_graph",
        input_value_infos, output_value_infos)
    model = onnx.helper.make_model(
        graph,
        functions=[function_proto],
        producer_name='onnx-script',
        opset_imports=[*onnx_opset_imports, local_opset_import])
    return model


def assign_eager_mode_evaluator_to_module(module, domain, version):
    from onnxscript.eager_mode_evaluator import EagerModeEvaluator
    module.op = EagerModeEvaluator(domain, version)
