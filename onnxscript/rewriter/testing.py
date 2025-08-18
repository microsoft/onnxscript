# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from onnxscript import ir


def generate_random_inputs(model: onnx.ModelProto) -> dict[str, Any]:
    feeds: dict[str, Any] = {}
    for input in model.graph.input:
        input_type = input.type.tensor_type
        shape = tuple(input_type.shape.dim)
        if not all(hasattr(d, "dim_value") for d in shape):
            raise ValueError(f"Input {input.name} has dynamic shape dimensions.")
        shape = tuple(d.dim_value for d in shape)
        if input_type.elem_type == onnx.TensorProto.FLOAT:
            if shape:
                feeds[input.name] = np.random.randn(*shape).astype(np.float32)
            else:
                feeds[input.name] = np.random.randn(1).astype(np.float32)
        else:
            raise ValueError(f"Not implemented for input type {input_type.elem_type}")
    return feeds


def assert_numerically_equal(
    original_model_proto: onnx.ModelProto | ir.Model,
    rewritten_model_proto: onnx.ModelProto | ir.Model,
    args: tuple[Any, ...] | dict[str, Any],
    ort_optimization_level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    rtol: float = 1,
    atol: float = 1e-3,
):
    """Assert that the two models are numerically equal.

    Args:
        original_model_proto: The original model proto or ir.Model.
        rewritten_model_proto: The rewritten by the rules model proto or ir.Model.
        args: The positional arguments to pass to the model.
        ort_optimization_level: Onnxruntime optimization level.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """

    if isinstance(original_model_proto, ir.Model):
        original_model_proto = ir.serde.serialize_model(original_model_proto)
    if isinstance(rewritten_model_proto, ir.Model):
        rewritten_model_proto = ir.serde.serialize_model(rewritten_model_proto)

    if isinstance(args, dict):
        original_proto_ort_inputs = args
        the_rewritten_proto_ort_inputs = args
    else:
        original_proto_ort_inputs = {
            k.name: v for k, v in zip(original_model_proto.graph.input, args)
        }
        the_rewritten_proto_ort_inputs = {
            k.name: v for k, v in zip(rewritten_model_proto.graph.input, args)
        }

    original_proto_ort_inference_session = _ort_session_initializer(
        original_model_proto.SerializeToString(), ort_optimization_level
    )
    run_options = ort.RunOptions()
    run_options.log_severity_level = 3  # 3: Error
    original_outputs = original_proto_ort_inference_session.run(
        None, original_proto_ort_inputs, run_options=run_options
    )

    the_rewritten_proto_ort_inference_session = _ort_session_initializer(
        rewritten_model_proto.SerializeToString(), ort_optimization_level
    )
    the_rewritten_outputs = the_rewritten_proto_ort_inference_session.run(
        None, the_rewritten_proto_ort_inputs, run_options=run_options
    )

    np.testing.assert_allclose(
        original_outputs, the_rewritten_outputs, rtol=rtol, atol=atol, equal_nan=True
    )


def _ort_session_initializer(
    model: str | bytes, ort_optimization_level: ort.GraphOptimizationLevel
) -> ort.InferenceSession:
    """Initialize an ONNX Runtime inference session with the specified model."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # 3: Error
    session_options.graph_optimization_level = ort_optimization_level
    possible_providers = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    available_providers = set(ort.get_available_providers())
    providers = [
        provider for provider in possible_providers if provider in available_providers
    ]
    return ort.InferenceSession(model, providers=providers, sess_options=session_options)
