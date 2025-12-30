# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx_ir as ir

from onnxscript.rewriter.rules.fusion import _gqa, _rms_normalization, _rotary_embedding


def _get_onnx_opset_version(model: ir.Model) -> int | None:
    """Get the ONNX opset version imported by the model."""
    model_version1 = model.opset_imports.get("")
    model_version2 = model.opset_imports.get("ai.onnx")
    if model_version1 is not None and model_version2 is not None:
        if model_version1 != model_version2:
            raise ValueError(
                f"Model imports multiple onnx opsets: {model_version1} and {model_version2}."
            )
    return model_version1 or model_version2


def _opset_23_fuse(model: ir.Model, *, debug: bool = False) -> dict[str, int]:
    """Apply fusions targeting ONNX opset 23."""
    counts: dict[str, int] = {}
    counts["RMSNormalization"] = _rms_normalization.fuse_rms_normalization(model, debug=debug)
    counts["RotaryEmbedding"] = _rotary_embedding.fuse_rotary_embedding(model, debug=debug)
    counts["GQA"] = _gqa.fuse_gqa(model, debug=debug)
    return counts


def fuse(model: ir.Model, *, debug: bool = False) -> dict[str, int]:
    """Apply fusions targeting ONNX ops."""
    model_opset_version = _get_onnx_opset_version(model)
    if model_opset_version == 23:
        return _opset_23_fuse(model, debug=debug)
    return {}
