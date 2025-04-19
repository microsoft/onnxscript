# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Version conversion passes."""

from __future__ import annotations

__all__ = [
    "ConvertVersionPass",
]

import logging

import onnx

from onnxscript import ir
from onnxscript.ir.passes.common import _c_api_utils
from onnxscript.ir.passes.common import inliner as _inliner
from onnxscript.version_converter import _version_converter

logger = logging.getLogger(__name__)


class ConvertVersionPass(ir.passes.InPlacePass):
    """Convert the model to the specified ONNX opset version.

    This pass leverages the onnxscript version converter to convert the model. If
    the conversion is not supported, it falls back to the onnx C API to convert
    the model. This pass is in-place.

    The pass is an no-op if the c-api fails.

    Attributes:
        target_version: The target ONNX opset version to convert the model to.
        fallback: Whether to fallback to the onnx version converter if the
            target version is not supported. Default is True.
    """

    def __init__(self, target_version: int, fallback: bool = False) -> None:
        super().__init__()
        self.target_version = target_version
        self.fallback = fallback
        self.inliner = _inliner.InlinePass()

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        if "" in model.graph.opset_imports:
            onnx_opset_version = model.graph.opset_imports[""]
            if onnx_opset_version == self.target_version:
                # No need to convert the version
                return ir.passes.PassResult(model, False)

        # In functions, we can have attribute-parameters, which means we don't know the value of the attribute.
        # Hence, we inline all the functions.
        self.inliner(model)

        if _version_converter.version_supported(model, self.target_version):
            _version_converter.convert_version(
                model,
                target_version=self.target_version,
            )
            return ir.passes.PassResult(model, True)

        if not self.fallback:
            logger.info(
                "The model version conversion is not supported by the onnxscript version converter "
                "and fallback is disabled. The model was not modified"
                " (target version: %d). "
                "Set fallback=True to enable fallback to the onnx c-api version converter.",
                self.target_version,
            )
            return ir.passes.PassResult(model, False)

        # If the onnxscript version converter does not support the conversion,
        # we can use the onnx C API to convert the model
        def _partial_convert_version(proto: onnx.ModelProto) -> onnx.ModelProto:
            """Partial function to check the model."""
            return onnx.version_converter.convert_version(
                proto, target_version=self.target_version
            )

        try:
            converted_proto = _c_api_utils.call_onnx_api(
                func=_partial_convert_version, model=model
            )
        except Exception as e:
            logger.warning(
                "Failed to convert the model to the target version %d using the ONNX C API. "
                "The model was not modified",
                self.target_version,
                exc_info=e,
            )
            return ir.passes.PassResult(model, False)

        converted_model = ir.from_proto(converted_proto)

        # Recover the initializers in the converted model
        for input in converted_model.graph.inputs:
            if input.name in model.graph.initializers:
                input.const_value = model.graph.initializers[input.name].const_value
                converted_model.graph.register_initializer(input)
        user_inputs = converted_model.graph.inputs[: len(model.graph.inputs)]
        converted_model.graph.inputs.clear()
        converted_model.graph.inputs.extend(user_inputs)

        # Return the converted graph to the original model to keep the pass in-place
        model.graph = converted_model.graph
        return ir.passes.PassResult(model, True)
