# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pprint
import numbers
import numpy as np
import onnx
from onnx import numpy_helper, AttributeProto, TypeProto
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidGraph, InvalidArgument
from .utils import values_to_value_infos
from .irbuilder import select_ir_version
from .eager_array import EagerArray


class EagerModeError(RuntimeError):
    pass


def convert_to_tensor(v, k):
    if isinstance(v, np.ndarray):
        return numpy_helper.from_array(v)
    if isinstance(v, list):
        return numpy_helper.from_array(np.array(v))
    if isinstance(v, numbers.Number):
        return numpy_helper.from_array(np.array([v]))
    if isinstance(v, onnx.TensorProto):
        return v
    raise ValueError(f"Attribute {k!r} must be convertable to TensorProto, got {type(v)}.")


def convert_attributes_to_tensors_with_schema(attribute_dict, schema_attribute_dict):
    # Constant and ConstantLike are the 2 ops in onnx
    # that take a tensor as attribute value.
    # onnx-script tends to use a literal number for attribute.
    # This methods is to make this scenario work.
    for k, v in attribute_dict.items():
        attribute_type = schema_attribute_dict[k].type
        if attribute_type == AttributeProto.TENSOR:
            attribute_dict[k] = convert_to_tensor(v, k)


def _rename_io(prefix, i, arg):
    if arg is None:
        return ""
    return f"{prefix}{i}"


def _compute_outputs(schema, *args, **kwargs):
    if schema.domain == '':
        if schema.name == 'BatchNormalization':
            if not kwargs.get('training_mode', 0):
                return ["output0"]
        if schema.name == 'LSTM':
            return ["output0", "output1", "output2"]
        if schema.name == 'Split':
            if len(args) == 1:
                raise EagerModeError(
                    "Operator Split: the number of expected outputs defines the split. "
                    "This information is unknown here.")
    return None


_cache_models = {}


def _cache_(model, providers):
    global _cache_models
    serialized = model.SerializeToString()
    key = serialized, tuple(providers)
    if key in _cache_models:
        return _cache_models[key]
    sess = InferenceSession(serialized, providers=providers)
    _cache_models[key] = sess
    return sess


def call_ort(schema, *args, **kwargs):

    inputs = []
    for i, arg in enumerate(args):
        if arg is None:
            inputs.append("")
            continue
        if not isinstance(arg, (EagerArray, list, int, float)):
            raise TypeError(f"Unexpected type {type(arg)} for input {i} "
                            f"and operator {schema.name!r}.")
        inputs.append(_rename_io("input", i, arg))

    convert_attributes_to_tensors_with_schema(kwargs, schema.attributes)

    # The number of outputs may be different based on the inputs.
    # The schema alone cannot be used in all cases (see BachNormalization).
    outputs = _compute_outputs(schema, *args, **kwargs)
    if outputs is None:
        outputs = ["output" + str(i) for i in range(len(schema.outputs))]

    node = onnx.helper.make_node(schema.name, inputs, outputs, **kwargs)
    input_value_infos = values_to_value_infos(inputs, list(args))
    output_value_infos = [onnx.helper.make_value_info(name, TypeProto()) for name in outputs]

    graph = onnx.helper.make_graph(
        [node], "node_graph", input_value_infos, output_value_infos)
    opset_id = onnx.helper.make_opsetid(schema.domain, schema.since_version)
    model = onnx.helper.make_model(graph, opset_imports=[opset_id],
                                   ir_version=select_ir_version(schema.since_version,
                                                                domain=schema.domain))
    try:
        sess = _cache_(model, ['CPUExecutionProvider'])
    except (Fail, InvalidGraph, InvalidArgument) as e:
        raise RuntimeError(
            "Unable to create onnxruntime InferenceSession with onnx "
            "model\n%s" % str(model)) from e

    session_run_input = {}
    tensor_class = None
    for name, arg in zip(inputs, args):
        if arg is None:
            continue
        if isinstance(arg, EagerArray):
            session_run_input[name] = arg.value
            tensor_class = EagerArray
        elif isinstance(arg, list):
            session_run_input[name] = arg
        elif isinstance(arg, (int, float)):
            session_run_input[name] = np.array(arg)
        else:
            raise TypeError(
                f"Unable to call onnxruntime with type {type(arg)} for input {name!r}).")

    try:
        got = sess.run(None, session_run_input)
    except (RuntimeError, Fail) as e:
        raise RuntimeError(
            f"Unable to execute model operator {schema.name!r} due to {e!r}"
            f"\ninput types:\n"
            f"{pprint.pformat({k: type(v) for k, v in zip(inputs, args)})}"
            f"\nmodified input types:\n"
            f"{pprint.pformat({k: type(v) for k, v in session_run_input.items()})}"
            f"\ninputs:\n{pprint.pformat(session_run_input)}\n{model}")

    if tensor_class is None:
        tensor_class = EagerArray
    new_got = []
    for i, g in enumerate(got):
        if isinstance(g, np.ndarray):
            new_got.append(tensor_class(g))
        elif isinstance(g, list):
            new_got.append(g)
        else:
            raise TypeError(
                f"Unexpected output type {type(g)} for output {i}).")
    return new_got[0] if len(new_got) == 1 else new_got
