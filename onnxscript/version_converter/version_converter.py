# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convert the model to the specified ONNX opset version."""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Sequence, Union

import onnxscript.ir._convenience as _convenience
import onnxscript.rewriter.pattern as orp
from onnxscript import ir

logger = logging.getLogger(__name__)


CURRENT_MAX_ONNX_OPSET = 23


@dataclasses.dataclass
class Replacement:
    """A replacement for a node in the graph."""

    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]


# A version-adapter function takes a node, a RewriterContext and returns
# a Replacement for the node or None (if no replacement is needed).

ReturnValue = Union[Replacement, Sequence[ir.Value], ir.Value, None]
AdapterFunction = Callable[[ir.Node, orp.RewriterContext], ReturnValue]


@dataclasses.dataclass
class VersionAdapter:
    """A class that represents a version checker for a particular op.

    It is applicable for a specific version upgrade (orignal_version -> original_version + 1)
    or downgrade (orignal_version -> original_version - 1)of the op.
    """

    node_version: int
    up_conversion: bool
    function: AdapterFunction


class AdapterRegistry:
    """A class that maintains a registry of adapters for ops."""

    def __init__(self):
        self.op_adapters: dict[tuple[str, str, int, bool], VersionAdapter] = {}

    def lookup_adapters(
        self,
        domain: str,
        opname: str,
        original_version: int,
        up_conversion: bool = True,
    ) -> Union[VersionAdapter, None]:
        adapter = self.op_adapters.get((domain, opname, original_version, up_conversion), None)
        if adapter is not None:
            return adapter.function
        else:
            return None

    def register(
        self, opname: str, domain: str = "", node_version=None, up_conversion=True
    ) -> Callable[[AdapterFunction], AdapterFunction]:
        def decorator(function: AdapterFunction) -> AdapterFunction:
            self.op_adapters[(domain, opname, node_version, up_conversion)] = VersionAdapter(
                node_version, up_conversion, function
            )
            return function

        return decorator


registry: AdapterRegistry = AdapterRegistry()

register = registry.register


def _get_input(node: ir.Node, index: int) -> ir.Value | None:
    if index < len(node.inputs):
        return node.inputs[index]
    return None


def _get_output(node: ir.Node, index: int) -> ir.Value | None:
    if index < len(node.outputs):
        return node.outputs[index]
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
        return None

    x_shape = x.shape
    if x_shape is None:
        return None
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
        return None
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

    def __init__(self, target_version: int):
        self.target_version = target_version

    def process_node(self, node: ir.Node, opset_version, up_conversion: bool = True):
        if node.domain not in self.opset_imports:
            return None
        adapter = registry.lookup_adapters(
            node.domain, node.op_type, opset_version, up_conversion
        )
        if adapter is None:
            return
        context = orp.RewriterContext()
        output = adapter(node, context)
        if output is not None:
            if isinstance(output, Replacement):
                return output
            if isinstance(output, ir.Value):
                output = [output]
            return Replacement(output, context.nodes)

    def replace_node(self, node: ir.Node, replacement, root: ir.Graph | ir.Function):
        logger.debug("Replacing node: %s::%s %s", node.domain, node.op_type, node.name)

        _convenience.replace_nodes_and_values(
            root, node, [node], replacement.new_nodes, node.outputs, replacement.new_outputs
        )

    def visit_attribute(
        self, attr: ir.Attr | ir.RefAttr, opset_version: int, up_conversion: bool = True
    ) -> None:
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.GRAPH:
                self.visit_graph(attr.value, opset_version, up_conversion)  # type: ignore[arg-type]
            elif attr.type == ir.AttributeType.GRAPHS:
                for graph in attr.value:
                    self.visit_graph(graph, opset_version, up_conversion)  # type: ignore[arg-type]

    def visit_node(
        self,
        node: ir.Node,
        root: ir.Graph | ir.Function,
        opset_version: int,
        up_conversion: bool = True,
    ):
        replacement = self.process_node(node, opset_version, up_conversion)
        if replacement is None:
            # No change. Process attributes.
            for attr in node.attributes.values():
                self.visit_attribute(attr, opset_version, up_conversion)
            return None
        else:
            self.replace_node(node, replacement, root)

    def visit_graph(
        self, graph: ir.Graph, opset_version: int, up_conversion: bool = True
    ) -> None:
        for node in graph:
            self.visit_node(node, graph, opset_version, up_conversion)
            node.version = self.target_version

    def visit_model(self, model: ir.Model) -> None:
        self.opset_imports = model.opset_imports
        model_version = model.opset_imports.get("")
        if model_version is None:
            return None

        up_conversion = True
        # TODO (shubhambhokare1) : Remove once down-conversion adapters are supoorted
        if self.target_version < model_version:
            up_conversion = False
            logger.warning(
                "Target opset: %s less than %s, downstream version conversion not currently handled.",
                self.target_version,
                model_version,
            )
            return None
        # Iterate from current model version -> target version
        # Updating each node based on the correct adapter
        # Up-conversion [ver->ver+1] or down-conversion [ver->ver-1]
        for opset_version in range(model_version, self.target_version):
            if up_conversion is True and opset_version == CURRENT_MAX_ONNX_OPSET:
                logger.warning(
                    "Conversion from opset: %s to target opset: %s not currently supported.",
                    opset_version,
                    opset_version + 1,
                )
                return None

            self.visit_graph(model.graph, opset_version, up_conversion)


def version_convert(model: ir.Model, target_version: int) -> None:
    """Convert the model to the specified ONNX opset version."""
    version_converter = _VersionConverter(target_version=target_version)
    version_converter.visit_model(model)
