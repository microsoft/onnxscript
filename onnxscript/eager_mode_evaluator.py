# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pprint

import numpy as np
import onnx
from onnx import TypeProto
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    InvalidGraph,
)

from onnxscript import irbuilder, tensor, utils


class EagerModeError(RuntimeError):
    pass


def _rename_io(prefix, i, arg):
    if arg is None:
        return ""
    return f"{prefix}{i}"


def compute_num_outputs(schema, *args, **kwargs):
    """Returns the number of outputs expected.
    TODO: Use ONNX type inference to replace the special-case handling below.
    """
    if schema.domain == "":
        if schema.name == "BatchNormalization":
            if not kwargs.get("training_mode", 0):
                return 1
        if schema.name == "LSTM":
            return 3
        if schema.name == "Split":
            if len(args) == 1:
                raise EagerModeError(
                    "Operator Split: the number of expected outputs defines the split. "
                    "This information is unknown here."
                )
        if schema.name == "Scan":
            scan_body = kwargs["body"]
            return len(scan_body.output)
        if schema.name == "Loop":
            loop_body = kwargs["body"]
            return len(loop_body.output) - 1
    return len(schema.outputs)


_cache_models = {}


def _cache_(model, providers):
    serialized = model.SerializeToString()
    key = serialized, tuple(providers)
    if key in _cache_models:
        return _cache_models[key]
    sess = InferenceSession(serialized, providers=providers)
    _cache_models[key] = sess
    return sess


def os_to_ort_value(v):
    """Converts an onnxscript encoding of an ONNX value into the encoding used by ORT."""
    if isinstance(v, tensor.Tensor):
        return v.value
    if isinstance(v, list):
        return v
    if v is None:
        # Treated as a static-optional value.
        # Dynamic optional None not yet supported.
        return v
    if isinstance(v, np.ndarray):
        return v
    raise TypeError(f"Unexpected ORT value type {type(v)}.")


def ort_to_os_value(v):
    """Converts an ORT encoding of an ONNX value into the encoding used by onnxscript."""
    if isinstance(v, np.ndarray):
        return tensor.Tensor(v)
    if isinstance(v, list):
        return v
    if v is None:
        raise TypeError("Dynamic optional values not yet supported.")
    raise TypeError(f"Unexpected ORT value type {type(v)}.")


def call_ort(schema, *args, **kwargs):
    # Convert input values to ORT representation-type:
    args = [os_to_ort_value(x) for x in args]

    # Construct ONNX model with a single op call:
    inputs = [_rename_io("input", i, arg) for i, arg in enumerate(args)]

    num_outputs = compute_num_outputs(schema, *args, **kwargs)
    outputs = [f"output{str(i)}" for i in range(num_outputs)]

    node = onnx.helper.make_node(schema.name, inputs, outputs, **kwargs)
    input_value_infos = utils.values_to_value_infos(inputs, list(args))
    output_value_infos = [onnx.helper.make_value_info(name, TypeProto()) for name in outputs]

    graph = onnx.helper.make_graph([node], "node_graph", input_value_infos, output_value_infos)
    opset_id = onnx.helper.make_opsetid(schema.domain, schema.since_version)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[opset_id],
        ir_version=irbuilder.select_ir_version(schema.since_version, domain=schema.domain),
    )
    try:
        sess = _cache_(model, ["CPUExecutionProvider"])
    except (Fail, InvalidGraph, InvalidArgument) as e:
        raise RuntimeError(
            f"Unable to create onnxruntime InferenceSession "
            f"with onnx model\n{utils.proto2text(model)}"
        ) from e

    session_run_input = {name: arg for name, arg in zip(inputs, args) if name != ""}

    try:
        result = sess.run(None, session_run_input)
    except (RuntimeError, Fail) as e:
        raise RuntimeError(
            f"Unable to execute model operator {schema.name!r} due to {e!r}"
            f"\ninput types:\n"
            f"{pprint.pformat({k: type(v) for k, v in zip(inputs, args)})}"
            f"\nmodified input types:\n"
            f"{pprint.pformat({k: type(v) for k, v in session_run_input.items()})}"
            f"\ninputs:\n{pprint.pformat(session_run_input)}\n{model}"
        ) from e

    # Map ORT output values to the onnxscript representation-type.
    cast_result = [ort_to_os_value(x) for x in result]
    return cast_result[0] if len(cast_result) == 1 else cast_result
