# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    "ConvertVersionPass",
    "convert_version",
]

import logging

import onnx
import onnx_ir as ir
import onnx_ir.passes.common as common_passes

from onnxscript.version_converter import _c_api_utils, _version_converter

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
        self._inline_pass = ir.passes.Sequential(
            common_passes.InlinePass(),
            # NOTE: Old torch version might include legacy Rank onnx functions which is
            # not used in the converted model.
            common_passes.RemoveUnusedFunctionsPass(),
            common_passes.RemoveUnusedOpsetsPass(),
        )
        self._convert_pass = _ConvertVersionPassRequiresInline(
            target_version=target_version,
            fallback=fallback,
        )
        self._cleanup_passes = ir.passes.Sequential(
            common_passes.RemoveUnusedNodesPass(),
            common_passes.RemoveUnusedFunctionsPass(),
            common_passes.RemoveUnusedOpsetsPass(),
        )

    def _prepare_functions_for_inlining(self, model: ir.Model) -> bool:
        source_version = _version_converter._get_onnx_opset_version(model)
        if source_version is None:
            source_version = next(
                (
                    version
                    for function in model.functions.values()
                    if (version := _version_converter._get_onnx_opset_version(function))
                    is not None
                ),
                None,
            )
        if source_version is None:
            return False

        def preserve_versions(graph: ir.Graph | ir.Function, inherited: int) -> None:
            version = _version_converter._get_onnx_opset_version(graph) or inherited
            for node in graph:
                if node.domain in ("", "ai.onnx") and node.version is None:
                    node.version = version
                for attr in node.attributes.values():
                    if attr.is_ref():
                        continue
                    if attr.type == ir.AttributeType.GRAPH:
                        preserve_versions(attr.as_graph(), version)
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for subgraph in attr.as_graphs():
                            preserve_versions(subgraph, version)

        modified = False
        for function in model.functions.values():
            version = _version_converter._get_onnx_opset_version(function)
            if version is not None and version != source_version:
                # The inliner requires identical imports. Preserve the actual source
                # versions on nodes so adapters still run after attribute resolution.
                preserve_versions(function, version)
                _version_converter._set_onnx_opset_version(function, source_version)
                modified = True
        return modified

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Run the conversion pass outside of Sequential so that errors
        # (e.g. VersionConverterError) propagate directly without being
        # wrapped in PassError.
        prepared = self._prepare_functions_for_inlining(model)
        inline_result = self._inline_pass(model)
        result = self._convert_pass(inline_result.model)
        cleanup_result = self._cleanup_passes(result.model)
        return ir.passes.PassResult(
            cleanup_result.model,
            prepared or result.modified or cleanup_result.modified or inline_result.modified,
        )


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
        # When fallback is disabled, always use the onnxscript version converter;
        # When fallback is enabled, use the onnxscript version converter
        # if the target version is supported. Otherwise, use the onnx C API
        # to convert the model.
        if not self.fallback or _version_converter.version_supported(
            model, self.target_version
        ):
            modified = _version_converter.convert_version(
                model,
                target_version=self.target_version,
            )
            return ir.passes.PassResult(model, modified)

        if not self.fallback:
            logger.warning(
                "The model version conversion is not supported by the onnxscript version converter "
                "and fallback is disabled. The model was not modified"
                " (target version: %d). "
                "Set fallback=True to enable fallback to the onnx c-api version converter.",
                self.target_version,
            )
            return ir.passes.PassResult(model, False)
        else:
            logger.warning(
                "The model version conversion is not supported by the onnxscript version converter "
                "and fallback is enabled. The model will be converted using the onnx C API "
                "(target version: %d).",
                self.target_version,
            )

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
        del model_proto.functions[:]
        model_proto.graph.CopyFrom(ir.to_proto(model.graph))
        del model_proto.opset_import[:]
        for domain, version in model.opset_imports.items():
            model_proto.opset_import.add(domain=domain, version=version)
