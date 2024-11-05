# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from onnxscript import ir


def assert_numerically_equal(
    original_model_proto: onnx.ModelProto | ir.Model,
    rewritten_model_proto: onnx.ModelProto | ir.Model,
    args: tuple[Any, ...],
    rtol: float = 1,
    atol: float = 1e-3,
):
    """Assert that the two models are numerically equal.

    Args:
        original_model_proto: The original model proto or ir.Model.
        rewritten_model_proto: The rewritten by the rules model proto or ir.Model.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        args: The positional arguments to pass to the model.
    """

    if isinstance(original_model_proto, ir.Model):
        original_model_proto = ir.serde.serialize_model(original_model_proto)
    if isinstance(rewritten_model_proto, ir.Model):
        rewritten_model_proto = ir.serde.serialize_model(rewritten_model_proto)

    original_proto_ort_inputs = {
        k.name: v for k, v in zip(original_model_proto.graph.input, args)
    }
    original_proto_ort_inference_session = _ort_session_initializer(
        original_model_proto.SerializeToString()
    )
    run_options = ort.RunOptions()
    run_options.log_severity_level = 3  # 3: Error
    original_outputs = original_proto_ort_inference_session.run(
        None, original_proto_ort_inputs, run_options=run_options
    )

    the_rewritten_proto_ort_inputs = {
        k.name: v for k, v in zip(rewritten_model_proto.graph.input, args)
    }
    the_rewritten_proto_ort_inference_session = _ort_session_initializer(
        rewritten_model_proto.SerializeToString()
    )
    the_rewritten_outputs = the_rewritten_proto_ort_inference_session.run(
        None, the_rewritten_proto_ort_inputs, run_options=run_options
    )

    np.testing.assert_allclose(
        original_outputs, the_rewritten_outputs, rtol=rtol, atol=atol, equal_nan=True
    )


def _ort_session_initializer(model: str | bytes) -> ort.InferenceSession:
    """Initialize an ONNX Runtime inference session with the specified model."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # 3: Error
    possible_providers = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    available_providers = set(ort.get_available_providers())
    providers = [
        provider for provider in possible_providers if provider in available_providers
    ]
    return ort.InferenceSession(model, providers=providers, sess_options=session_options)
