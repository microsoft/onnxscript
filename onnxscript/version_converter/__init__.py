# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    "ConvertVersionPass",
    "convert_version",
]

import logging

import onnx

import onnxscript.ir.passes
import onnxscript.ir.passes.common
from onnxscript import ir
from onnxscript.ir.passes.common import _c_api_utils
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
            target version is not supported. Default is False.
    """

    def __init__(self, target_version: int, fallback: bool = False) -> None:
        super().__init__()
        self.target_version = target_version
        self.fallback = fallback
        self.convert_pass = ir.passes.Sequential(
            onnxscript.ir.passes.common.InlinePass(),
            _ConvertVersionPassRequiresInline(
                target_version=target_version,
                fallback=fallback,
            ),
            onnxscript.ir.passes.common.RemoveUnusedNodesPass(),
            onnxscript.ir.passes.common.RemoveUnusedFunctionsPass(),
            onnxscript.ir.passes.common.RemoveUnusedOpsetsPass(),
        )

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return self.convert_pass(model)


class _ConvertVersionPassRequiresInline(ir.passes.InPlacePass):
    """Convert the model to the specified ONNX opset version.

    This pass leverages the onnxscript version converter to convert the model. If
    the conversion is not supported, it falls back to the onnx C API to convert
    the model. This pass is in-place.

    The pass is an no-op if the c-api fails.

    Attributes:
        target_version: The target ONNX opset version to convert the model to.
        fallback: Whether to fallback to the onnx version converter if the
            target version is not supported.
    """

    def __init__(self, target_version: int, fallback: bool) -> None:
        super().__init__()
        self.target_version = target_version
        self.fallback = fallback

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        if model.functions:
            raise ValueError(
                "The model contains functions. The version conversion pass does not support "
                "functions. Please use `onnxscript.ir.passes.common.InlinePass` to inline the "
                f"functions before applying this pass ({self.__class__.__name__})."
            )
        if "" in model.graph.opset_imports:
            onnx_opset_version = model.graph.opset_imports[""]
            if onnx_opset_version == self.target_version:
                # No need to convert the version
                return ir.passes.PassResult(model, False)

        # When fallback is disabled, always use the onnxscript version converter;
        # When fallback is enabled, use the onnxscript version converter
        # if the target version is supported. Otherwise, use the onnx C API
        # to convert the model.
        if not self.fallback or _version_converter.version_supported(
            model, self.target_version
        ):
            _version_converter.convert_version(
                model,
                target_version=self.target_version,
            )
            return ir.passes.PassResult(model, True)

        if not self.fallback:
            logger.warning(
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
        except Exception as e:  # pylint: disable=broad-exception-caught
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


def convert_version(
    model: ir.Model | onnx.ModelProto, target_version: int, fallback=None
) -> None:
    """Convert the model to the specified ONNX opset version.

    Args:
        model: The model to convert.
        target_version: The target ONNX opset version.
        fallback: Whether to fallback to the onnx version converter if the
            target version is not supported. Default is False.
    """
    if isinstance(model, onnx.ModelProto):
        model_proto = model
        model = ir.from_proto(model)
    else:
        model_proto = None

    assert isinstance(model, ir.Model)
    ConvertVersionPass(target_version=target_version, fallback=fallback)(model)

    if model_proto is not None:
        # Update the model proto in-place
        model_proto.graph.Clear()
        del model_proto.functions
        model_proto.graph.CopyFrom(ir.to_proto(model.graph))
