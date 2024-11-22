# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convert the model to the specified ONNX opset version."""

from __future__ import annotations

import dataclasses
import functools
import logging
from typing import Callable, Sequence, Union

import onnxscript.ir.convenience as ir_convenience
import onnxscript.rewriter.pattern as orp
from onnxscript import ir

logger = logging.getLogger(__name__)


CURRENT_MAX_ONNX_OPSET = 23


class VersionConverterError(RuntimeError):
    """Raised when an node's version cannot be upgraded/downgraded successfully."""


@dataclasses.dataclass
class Replacement:
    """A replacement for a node in the graph."""

    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]


# A version-adapter function takes a node, a RewriterContext and returns
# a Replacement for the node or None (if no replacement is needed).

ReturnValue = Union[Sequence[ir.Value], ir.Value, None]
AdapterFunction = Callable[[ir.Node, orp.RewriterContext], ReturnValue]


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


def _get_int_attribute(node: ir.Node, name: str, default: int | None = None) -> int | None:
    if name in node.attributes:
        attr = node.attributes[name]
        if not isinstance(attr, ir.Attr):
            return None
        attr_val = attr.value
        if isinstance(attr_val, int):
            return attr_val
        # This is an invalid model: attribute has invalid/unexpected type.
        # For now, we just return None. We could raise an error too.
        return None
    return default


def _get_str_attribute(node: ir.Node, name: str, default: str | None = None) -> str | None:
    if name in node.attributes:
        attr = node.attributes[name]
        if not isinstance(attr, ir.Attr):
            return None
        attr_val = attr.value
        if isinstance(attr_val, str):
            return attr_val
        # This is an invalid model: attribute has invalid/unexpected type.
        # For now, we just return None. We could raise an error too.
        return None
    return default


## Op-specific adapters

# Opset 19 -> 20


@register("DFT", node_version=19, up_conversion=True)
def dft_19_20(node: ir.Node, op):
    input = node.inputs[0]
    inverse = _get_int_attribute(node, "inverse", 0)
    onesided = _get_int_attribute(node, "onesided", 0)
    axis = _get_int_attribute(node, "axis", None)
    if axis is not None:
        axis_value = op.Constant(value_int=axis)
        return op.DFT(input, axis_value, inverse=inverse, onesided=onesided)
    return None


@register("GridSample", node_version=19, up_conversion=True)
def gridsample_19_20(node: ir.Node, op):
    x = node.inputs[0]
    grid = node.inputs[1]
    align_corners = _get_int_attribute(node, "align_corners", 0)
    mode = _get_str_attribute(node, "mode", "linear")
    padding_mode = _get_str_attribute(node, "padding_mode", "zeros")
    if mode == "bilinear":
        return op.GridSample(
            x, grid, align_corners=align_corners, mode="linear", padding_mode=padding_mode
        )
    elif mode == "bicubic":
        return op.GridSample(
            x, grid, align_corners=align_corners, mode="cubic", padding_mode=padding_mode
        )
    return None


# Opset 20 -> 21


@register("GroupNormalization", node_version=20, up_conversion=True)
def groupnormalization_20_21(node: ir.Node, op):
    x = _get_input(node, 0)
    scale = _get_input(node, 1)
    bias = _get_input(node, 2)
    if x is None or scale is None or bias is None:
        raise VersionConverterError(f"Missing input for {node}")

    x_shape = x.shape
    if x_shape is None:
        raise VersionConverterError(f"Missing required shape for {x}")
    num_channels = x_shape[1]
    if not isinstance(num_channels, int):
        return None

    scale_shape = scale.shape
    bias_shape = bias.shape
    if scale_shape is None or bias_shape is None:
        return None
    if not isinstance(scale_shape[0], int) or not isinstance(bias_shape[0], int):
        return None

    num_groups = _get_int_attribute(node, "num_groups", None)
    if num_groups is None:
        raise VersionConverterError("Missing required attribute: num_groups")
    if (
        num_groups != num_channels
        and num_groups == scale_shape[0]
        and num_groups == bias_shape[0]
    ):
        reshape_1_sizes = op.Constant(value_ints=[-1, 1])
        reshape_2_sizes = op.Constant(value_ints=[-1])
        c_div = int(num_channels / num_groups)
        expand_sizes = op.Constant(value_ints=[1, c_div])

        # Modify scale input
        scale_reshape_1 = op.Reshape(scale, reshape_1_sizes)
        scale_expand = op.Expand(scale_reshape_1, expand_sizes)
        scale_reshape_2 = op.Reshape(scale_expand, reshape_2_sizes)

        # Modify bias input
        bias_reshape_1 = op.Reshape(bias, reshape_1_sizes)
        bias_expand = op.Expand(bias_reshape_1, expand_sizes)
        bias_reshape_2 = op.Reshape(bias_expand, reshape_2_sizes)

        return op.GroupNormalization(x, scale_reshape_2, bias_reshape_2, num_groups=num_groups)
    return None


class _VersionConverter:
    opset_imports: dict[str, int]
    model_version: int

    def __init__(self, target_version: int):
        self.target_version = target_version

    def _upgrade_version(self, node: ir.Node, opset_version: int, up_conversion: bool) -> None:
        if up_conversion is True:
            node.version = opset_version + 1
        else:
            node.version = opset_version - 1

    def process_node(
        self, node: ir.Node, opset_version: int, up_conversion: bool = True
    ) -> Replacement | None:
        if node.domain not in {"", "ai.onnx"}:
            return None
        adapter = registry.lookup_adapters(
            node.domain, node.op_type, opset_version, up_conversion
        )
        if adapter is None:
            return None
        context = orp.RewriterContext()
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

    def visit_attribute(self, attr: ir.Attr | ir.RefAttr) -> None:
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.GRAPH:
                self.visit_graph(attr.value)  # type: ignore[arg-type]
            elif attr.type == ir.AttributeType.GRAPHS:
                for graph in attr.value:
                    self.visit_graph(graph)  # type: ignore[arg-type]

    def visit_node(
        self,
        node: ir.Node,
        root: ir.Graph | ir.Function,
        opset_version: int,
        up_conversion: bool = True,
    ) -> None:
        replacement = self.process_node(node, opset_version, up_conversion)
        if replacement is None:
            # No change. Process attributes.
            for attr in node.attributes.values():
                self.visit_attribute(attr)
            return None
        else:
            self.replace_node(node, replacement, root)
        return None

    def visit_graph(self, graph: ir.Graph) -> None:
        if self.target_version > CURRENT_MAX_ONNX_OPSET:
            logger.warning(
                "Conversion to target opset: %s not currently supported.",
                self.target_version,
            )
            return None
        for node in graph:
            up_conversion = True
            if node.version is None:
                node.version = self.model_version
            # Iterate each node from current node version -> target version
            # and updating node based on the correct adapter
            # Up-conversion [ver->ver+1] or down-conversion [ver->ver-1]
            # TODO(shubhambhokare1): Remove once down-conversion adapters are supoorted
            if self.target_version < node.version:
                up_conversion = False
                logger.warning(
                    "Target opset: %s less than %s, downstream version conversion not currently handled.",
                    self.target_version,
                    self.model_version,
                )
                return None
            for opset_version in range(node.version, self.target_version):
                try:
                    self.visit_node(node, graph, opset_version, up_conversion)
                    self._upgrade_version(node, opset_version, up_conversion)
                except VersionConverterError as e:
                    logger.warning(
                        "Skipping version conversion for node %s due to exception: %s",
                        node.op_type,
                        e,
                    )
        return None

    def visit_model(self, model: ir.Model) -> None:
        self.opset_imports = model.opset_imports
        model_version = self.opset_imports.get("")
        if model_version is None:
            model_version = model.opset_imports.get("ai.onnx")
            if model_version is None:
                return None
        self.model_version = model_version
        self.visit_graph(model.graph)
        return None


def convert_version(model: ir.Model, target_version: int) -> None:
    """Convert the model to the specified ONNX opset version."""
    version_converter = _VersionConverter(target_version=target_version)
    version_converter.visit_model(model)
