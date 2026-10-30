# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convert the model to the specified ONNX opset version."""

from __future__ import annotations

import dataclasses
import functools
import logging
from typing import Callable, Sequence, Union

import onnx.defs
import onnx_ir as ir
import onnx_ir.convenience as ir_convenience
import onnx_ir.passes.common as ir_passes_common

import onnxscript.utils.metadata_merger as metadata_merger
from onnxscript._internal.tape_builder import BuilderBase, TapeBuilder

logger = logging.getLogger(__name__)


SUPPORTED_MAX_ONNX_OPSET = 28
SUPPORTED_MIN_ONNX_OPSET = 18


def _get_onnx_opset_version(model: ir.Model | ir.Graph | ir.Function) -> int | None:
    """Get the ONNX opset version imported by the model."""
    model_version1 = model.opset_imports.get("")
    model_version2 = model.opset_imports.get("ai.onnx")
    if model_version1 is not None and model_version2 is not None:
        if model_version1 != model_version2:
            raise ValueError(
                f"Model imports conflicting standard-domain opsets: {model_version1} "
                f"and {model_version2}. Use the same version for '' and 'ai.onnx' "
                "before converting."
            )
    return model_version1 if model_version1 is not None else model_version2


def _set_onnx_opset_version(
    model_or_function: ir.Model | ir.Graph | ir.Function, version: int
) -> None:
    """Set the ONNX opset version imported by the model or function."""
    if "ai.onnx" in model_or_function.opset_imports:
        del model_or_function.opset_imports["ai.onnx"]
    model_or_function.opset_imports[""] = version


class VersionConverterError(RuntimeError):
    """Raised when an node's version cannot be upgraded/downgraded successfully."""


@dataclasses.dataclass
class Replacement:
    """A replacement for a node in the graph."""

    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]


# A version-adapter function takes a node, a VCContext and returns
# a Replacement for the node or None (if no replacement is needed).

VCContext = BuilderBase
ReturnValue = Union[Sequence[ir.Value], ir.Value, None]
AdapterFunction = Callable[[ir.Node, VCContext], ReturnValue]


def version_supported(model: ir.Model, target_version: int) -> bool:
    """Check if the target version is supported by the current version."""
    if not SUPPORTED_MIN_ONNX_OPSET <= target_version <= SUPPORTED_MAX_ONNX_OPSET:
        return False
    current_version = _get_onnx_opset_version(model)
    return current_version is None or (
        SUPPORTED_MIN_ONNX_OPSET <= current_version <= target_version
    )


class AdapterRegistry:
    """A class that maintains a registry of adapters for ops."""

    def __init__(self):
        self.op_adapters: dict[tuple[str, str, int, bool], AdapterFunction] = {}

    def lookup_adapters(
        self,
        domain: str,
        opname: str,
        original_version: int,
        up_conversion: bool = True,
    ) -> AdapterFunction | None:
        adapter_func = self.op_adapters.get((domain, opname, original_version, up_conversion))
        if adapter_func is not None:
            return adapter_func
        return None

    def register(
        self, opname: str, domain: str = "", node_version=None, up_conversion=True
    ) -> Callable[[AdapterFunction], AdapterFunction]:
        """Register an adapter based on the domain, operator type, node version and whether to upgrade/downgrade node version"""

        def decorator(function: AdapterFunction) -> AdapterFunction:
            @functools.wraps(function)
            def wrapped_function(*args, **kwargs):
                return function(*args, **kwargs)

            self.op_adapters[(domain, opname, node_version, up_conversion)] = function
            return wrapped_function

        return decorator


registry: AdapterRegistry = AdapterRegistry()

register = registry.register


def _get_input(node: ir.Node, index: int) -> ir.Value | None:
    if index < len(node.inputs):
        return node.inputs[index]
    return None


def _constant_attribute(
    node: ir.Node, name: str, constant_name: str, attr_type: ir.AttributeType, default=None
) -> ir.Attr:
    attr = node.attributes.get(name)
    if attr is None:
        if default is None:
            raise VersionConverterError(
                f"Cannot convert {node.op_type} node {node.name!r}: missing required "
                f"attribute {name!r}. Supply the attribute before converting."
            )
        return ir.Attr(constant_name, attr_type, default)
    if attr.type != attr_type:
        raise VersionConverterError(
            f"Cannot convert {node.op_type} node {node.name!r}: attribute {name!r} "
            f"must have type {attr_type}, not {attr.type}. Correct the attribute type."
        )
    if attr.is_ref():
        return ir.RefAttr(constant_name, attr.ref_attr_name, attr_type)
    return ir.Attr(constant_name, attr_type, attr.value)


def _require_input(node: ir.Node, index: int) -> ir.Value:
    value = _get_input(node, index)
    if value is None:
        raise VersionConverterError(
            f"Cannot convert {node.op_type} node {node.name!r}: required input "
            f"{index} is missing. Supply all required inputs before converting."
        )
    return value


## Op-specific adapters

# Opset 19 -> 20


@register("DFT", node_version=19, up_conversion=True)
def dft_19_20(node: ir.Node, op):
    input = _require_input(node, 0)
    dft_length = _get_input(node, 1)
    # In opset 19 `axis` is an attribute defaulting to 1; in opset 20 it became an
    # input defaulting to -2. An omitted attribute must therefore be materialized
    # explicitly, or the converted node silently transforms a different axis.
    axis = _constant_attribute(node, "axis", "value_int", ir.AttributeType.INT, 1)
    axis_value = op.Constant(value_int=axis)
    attributes = {name: attr for name, attr in node.attributes.items() if name != "axis"}
    return op.DFT(input, dft_length, axis_value, **attributes)


@register("GridSample", node_version=19, up_conversion=True)
def gridsample_19_20(node: ir.Node, op):
    del op
    mode = node.attributes.get("mode")
    if mode is None:
        return None
    if mode.is_ref():
        raise VersionConverterError(
            f"Cannot convert GridSample node {node.name!r} with a referenced mode. "
            "Inline the local function first, or use the public ConvertVersionPass."
        )
    if mode.type != ir.AttributeType.STRING or mode.as_string() not in (
        "bilinear",
        "bicubic",
        "nearest",
    ):
        raise VersionConverterError(
            f"Cannot convert GridSample node {node.name!r}: mode must be "
            "'bilinear', 'bicubic', or 'nearest' in opset 19. Correct the mode attribute."
        )
    new_mode = {"bilinear": "linear", "bicubic": "cubic", "nearest": "nearest"}[
        mode.as_string()
    ]
    node.attributes["mode"] = ir.Attr("mode", ir.AttributeType.STRING, new_mode)
    return None


# Opset 20 -> 21


@register("GroupNormalization", node_version=20, up_conversion=True)
def groupnormalization_20_21(node: ir.Node, op):
    x = _require_input(node, 0)
    scale = _require_input(node, 1)
    bias = _require_input(node, 2)
    groups = _constant_attribute(node, "num_groups", "value_int", ir.AttributeType.INT)
    epsilon = _constant_attribute(node, "epsilon", "value_float", ir.AttributeType.FLOAT, 1e-5)
    # Decompose the legacy formula: opset 21 changes both parameter granularity
    # and accumulation precision. CastLike preserves the old dtype without
    # requiring static type or shape information.
    axes = op.Constant(value_ints=[0])
    groups_vector = op.Unsqueeze(op.Constant(value_int=groups), axes)
    grouped_shape = op.Concat(
        op.Constant(value_ints=[0]), groups_vector, op.Constant(value_ints=[-1]), axis=0
    )
    grouped = op.Reshape(x, grouped_shape)
    reduction_axes = op.Constant(value_ints=[2])
    mean = op.ReduceMean(grouped, reduction_axes, keepdims=1)
    mean_square = op.ReduceMean(op.Mul(grouped, grouped), reduction_axes, keepdims=1)
    variance = op.Sub(mean_square, op.Mul(mean, mean))
    epsilon_value = op.CastLike(op.Constant(value_float=epsilon), x)
    normalized = op.Div(op.Sub(grouped, mean), op.Sqrt(op.Add(variance, epsilon_value)))
    parameter_shape = op.Constant(value_ints=[1, -1, 1])
    scale_reshaped = op.Reshape(op.CastLike(scale, x), parameter_shape)
    bias_reshaped = op.Reshape(op.CastLike(bias, x), parameter_shape)
    result = op.Add(op.Mul(scale_reshaped, normalized), bias_reshaped)
    return op.Reshape(result, op.Shape(x))


class _VersionConverter:
    def __init__(self, target_version: int):
        self._target_version = target_version
        self._modified: bool = False
        # Default metadata merger: no merging should be needed; keep the first value.
        self._default_metadata_merger: metadata_merger.MetadataMerger = (
            metadata_merger.MetadataMerger(
                dict(),
            )
        )

    def process_node(
        self, node: ir.Node, from_version: int, up_conversion: bool = True
    ) -> Replacement | None:
        assert node.domain == ""
        adapter = registry.lookup_adapters(
            node.domain, node.op_type, from_version, up_conversion
        )
        if adapter is None:
            return None
        context = TapeBuilder()
        output = adapter(node, context)
        if output is not None:
            if isinstance(output, ir.Value):
                output = [output]
            return Replacement(output, context.nodes)
        return None

    def replace_node(self, node: ir.Node, replacement, root: ir.Graph | ir.Function) -> None:
        logger.debug("Replacing node: %s::%s %s", node.domain, node.op_type, node.name)

        ir_convenience.replace_nodes_and_values(
            root, node, [node], replacement.new_nodes, node.outputs, replacement.new_outputs
        )
        self._modified = True

    def visit_attribute(self, attr: ir.Attr, default_version: int | None) -> None:
        if attr.is_ref():
            return
        if attr.type == ir.AttributeType.GRAPH:
            self.visit_graph_or_function(attr.as_graph(), default_version)
        elif attr.type == ir.AttributeType.GRAPHS:
            for graph in attr.as_graphs():
                self.visit_graph_or_function(graph, default_version)

    def visit_node(
        self,
        node: ir.Node,
        root: ir.Graph | ir.Function,
        from_version: int,
        up_conversion: bool = True,
    ) -> Sequence[ir.Node]:
        self._modified = True
        if up_conversion:
            to_version = from_version + 1
        else:
            to_version = from_version - 1
        replacement = self.process_node(node, from_version, up_conversion)
        if replacement is None:
            node.version = to_version
            return [node]
        else:
            for new_node in replacement.new_nodes:
                new_node.version = to_version
            self._default_metadata_merger.copy_merged_metadata([node], replacement.new_nodes)
            self.replace_node(node, replacement, root)
            return replacement.new_nodes

    def visit_graph_or_function(
        self, graph_or_function: ir.Graph | ir.Function, default_version: int | None = None
    ) -> None:
        source_version = _get_onnx_opset_version(graph_or_function)
        if source_version is None:
            source_version = default_version
        uses_standard_domain = source_version is not None
        if source_version is not None:
            self._check_source_version(source_version)
        for node in tuple(graph_or_function):
            for attr in node.attributes.values():
                self.visit_attribute(attr, source_version)
            if node.domain not in ("", "ai.onnx"):
                continue
            uses_standard_domain = True
            node_version = node.version if node.version is not None else source_version
            if node_version is None:
                raise VersionConverterError(
                    f"Cannot convert node {node.name!r} ({node.op_type}): no source opset "
                    "version is available. Add a standard-domain opset import to its model "
                    "or local function."
                )
            self._check_node_version(node, node_version)
            if node.domain == "ai.onnx":
                node.domain = ""
                self._modified = True
            active_nodes: Sequence[ir.Node] = [node]
            for from_version in range(node_version, self._target_version):
                active_nodes = [
                    new_node
                    for current in active_nodes
                    for new_node in self.visit_node(current, graph_or_function, from_version)
                ]
        if uses_standard_domain:
            if graph_or_function.opset_imports.get("") != self._target_version or (
                "ai.onnx" in graph_or_function.opset_imports
            ):
                self._modified = True
            _set_onnx_opset_version(graph_or_function, self._target_version)

    def _check_node_version(self, node: ir.Node, source_version: int) -> None:
        if 0 < source_version < SUPPORTED_MIN_ONNX_OPSET:
            # Exporters may record the operator's schema revision, not the model's
            # imported opset. Accept it only if the same schema is active at opset 18.
            try:
                source_schema = onnx.defs.get_schema(node.op_type, source_version, "")
                supported_schema = onnx.defs.get_schema(
                    node.op_type, SUPPORTED_MIN_ONNX_OPSET, ""
                )
            except onnx.defs.SchemaError as e:
                raise VersionConverterError(
                    f"Cannot convert node {node.name!r} ({node.op_type}) from opset "
                    f"{source_version}: no standard-domain schema is available at its "
                    "source version or opset 18. Check the operator name and version."
                ) from e
            if source_schema.since_version == supported_schema.since_version:
                return
            raise VersionConverterError(
                f"Cannot convert node {node.name!r} ({node.op_type}) from source opset "
                f"{source_version} to {self._target_version}: its schema changed before "
                f"opset {SUPPORTED_MIN_ONNX_OPSET}, outside the native conversion range. "
                "Upgrade the node's original model or function using ONNX's converter "
                "before combining it with this model."
            )
        self._check_source_version(source_version)

    def _check_source_version(self, source_version: int) -> None:
        if not SUPPORTED_MIN_ONNX_OPSET <= source_version <= self._target_version:
            raise VersionConverterError(
                f"Cannot convert source opset {source_version} to {self._target_version}: "
                f"native conversion only supports upgrades from opset "
                f"{SUPPORTED_MIN_ONNX_OPSET} through {SUPPORTED_MAX_ONNX_OPSET}. "
                "Use fallback=True to request ONNX conversion outside this range."
            )

    def visit_model(self, model: ir.Model) -> None:
        self.visit_graph_or_function(model.graph)
        for function in model.functions.values():
            self.visit_graph_or_function(function)
        if self._modified:
            # TapeBuilder may create values with names that clash with existing graph
            # values when nodes are inserted via replace_nodes_and_values.
            # NameFixPass ensures all value names are unique before returning.
            ir_passes_common.NameFixPass()(model)


def convert_version(model: ir.Model, target_version: int) -> bool:
    """Convert the model in place and return whether it was modified."""
    if (target_version > SUPPORTED_MAX_ONNX_OPSET) or (
        target_version < SUPPORTED_MIN_ONNX_OPSET
    ):
        raise ValueError(
            f"Target opset version {target_version} is not supported. "
            f"Supported range: {SUPPORTED_MIN_ONNX_OPSET} to {SUPPORTED_MAX_ONNX_OPSET}."
        )
    version_converter = _VersionConverter(target_version=target_version)
    version_converter.visit_model(model)
    return version_converter._modified
