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


_ADAPTERS_18_19 = {
    "Equal",
    "AveragePool",
    "Cast",
    "CastLike",
    "Constant",
    "DequantizeLinear",
    "Identity",
    "If",
    "Loop",
    "Pad",
    "QuantizeLinear",
    "Reshape",
    "Scan",
    "Shape",
    "Size",
}


_ADAPTERS_19_20 = {
    "DFT",
    "ConstantOfShape",
    "IsInf",
    "IsNan",
    "ReduceMax",
    "ReduceMin",
    "GridSample",
}


_ADAPTERS_20_21 = {
    "Cast",
    "CastLike",
    "Constant",
    "ConstantOfShape",
    "DequantizeLinear",
    "Flatten",
    "GroupNormalization",
    "Identity",
    "If",
    "Loop",
    "Pad",
    "QLinearMatmul",
    "QuantizeLinear",
    "Reshape",
    "Scan",
    "Shape",
    "Size",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
}


_ADAPTERS_21_22 = {
    "EyeLike",
    "RandomUniform",
    "RandomNormal",
    "RandomUniformLike",
    "RandomNormalLike",
    "Multinomial",
    "Bernoulli",
    "ThresholdedRelu",
    "Selu",
    "Elu",
    "Mish",
    "HardSigmoid",
    "HardSwish",
    "Softsign",
    "Softplus",
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Sinh",
    "Cosh",
    "Asinh",
    "Acosh",
    "Atanh",
    "Round",
    "Det",
    "NegativeLogLikelihoodLoss",
    "AveragePool",
    "MaxPool",
    "MaxUnpool",
    "LpPool",
    "MaxRoiPool",
    "Conv",
    "ConvTranspose",
    "DeformConv",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "GlobalLpPool",
    "InstanceNormalization",
    "LpNormalization",
    "Dropout",
    "RoiAlign",
    "RNN",
    "GRU",
    "LSTM",
    "GridSample",
}


_ADAPTER_SETS = {
    (18, 19): _ADAPTERS_18_19,
    (19, 20): _ADAPTERS_19_20,
    (20, 21): _ADAPTERS_20_21,
    (21, 22): _ADAPTERS_21_22,
}


@dataclasses.dataclass
class Replacement:
    """A replacement for a node in the graph."""

    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]


# A partial-evaluator function takes a node, a RewriterContext, OptimizerState and returns
# a Replacement for the node or None (if no replacement is needed). It may also return just
# the ir.Value or ir.Values to replace the output values of the node, when the new nodes
# can be inferred from the RewriterContext used to build the new nodes.

ReturnValue = Union[Replacement, Sequence[ir.Value], ir.Value, None]
AdapterFunction = Callable[[ir.Node, orp.RewriterContext], ReturnValue]


@dataclasses.dataclass
class AdapterVersionChecker:
    """A class that represents a version checker for a particular op.

    It is applicable for a specific version upgrade (orignal_version -> target_version) of the op.
    """

    node_version: int | None
    upgrade_version: int | None
    function: AdapterFunction

    def valid_for(self, opname: str, original_version: int, target_version: int) -> bool:
        """Returns True if this evaluator is applicable for the given version upgrade."""
        adapter_set = tuple((original_version, target_version))
        if adapter_set not in _ADAPTER_SETS:
            return False
        if opname in _ADAPTER_SETS[adapter_set]:
            return True
        return False


class AdapterRegistry:
    """A class that maintains a registry of adapters for ops."""

    def __init__(self):
        self.op_adapters: dict[tuple[str, str], list[AdapterVersionChecker]] = {}

    def lookup_adapters(
        self, domain: str, opname: str, original_version: int, target_version: int
    ):
        adapter_list = self.op_adapters.get((domain, opname), [])
        return [
            adapter.function
            for adapter in adapter_list
            if adapter.valid_for(opname, original_version, target_version)
        ]

    def register(
        self, opname: str, domain: str = "", node_version=None, upgrade_version=None
    ) -> Callable[[AdapterFunction], AdapterFunction]:
        if (domain, opname) in self.op_adapters:
            adapter_list = self.op_adapters[(domain, opname)]
        else:
            adapter_list = []
            self.op_adapters[(domain, opname)] = adapter_list
        if node_version is None or upgrade_version is None:
            original_version = None
            target_version = None
        else:
            original_version = node_version
            target_version = upgrade_version

        def decorator(function: AdapterFunction) -> AdapterFunction:
            adapter_list.append(
                AdapterVersionChecker(original_version, target_version, function)
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


@register("DFT", node_version=19, upgrade_version=20)
def dft_19_20(node: ir.Node, op):
    input = node.inputs[0]
    inverse = _get_int_attribute(node, "inverse", 0)
    onesided = _get_int_attribute(node, "onesided", 0)
    axis = _get_int_attribute(node, "axis", None)
    if axis is not None:
        axis_value = op.Constant(value_int=axis)
        return op.DFT(input, axis_value, inverse=inverse, onesided=onesided)
    return None


@register("GridSample", node_version=19, upgrade_version=20)
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


@register("GroupNormalization", node_version=20, upgrade_version=21)
def groupnormalization_20_21(node: ir.Node, op):
    x = _get_input(node, 0)
    scale = _get_input(node, 1)
    bias = _get_input(node, 2)

    x_shape = x.shape
    num_channels = x_shape[1]
    if x_shape is None:
        return None
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

    def process_node(self, node: ir.Node, opset_version):
        if node.domain not in self.opset_imports:
            return None
        op_adapters = registry.lookup_adapters(
            node.domain, node.op_type, opset_version, opset_version + 1
        )
        for adapter in op_adapters:
            assert adapter
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

    def visit_attribute(self, attr: ir.Attr | ir.RefAttr, opset_version: int) -> None:
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.GRAPH:
                self.visit_graph(attr.value, opset_version)  # type: ignore[arg-type]
            elif attr.type == ir.AttributeType.GRAPHS:
                for graph in attr.value:
                    self.visit_graph(graph, opset_version)  # type: ignore[arg-type]

    def visit_node(self, node: ir.Node, root: ir.Graph | ir.Function, opset_version: int):
        replacement = self.process_node(node, opset_version)
        if replacement is None:
            # No change. Process attributes.
            for attr in node.attributes.values():
                self.visit_attribute(attr, opset_version)
            return None
        else:
            self.replace_node(node, replacement, root)

    def visit_graph(self, graph: ir.Graph, opset_version: int) -> None:
        for node in graph:
            self.visit_node(node, graph, opset_version)
            node.version = self.target_version

    def visit_function(self, function: ir.Function, opset_version: int) -> None:
        for node in function:
            self.visit_node(node, function, opset_version)

    def visit_model(self, model: ir.Model) -> None:
        self.opset_imports = model.opset_imports
        model_version = model.opset_imports.get("")

        if self.target_version < model_version:
            logger.warning(
                "Target opset: %s less than %s, downstream version conversion not currently supported.",
                self.target_version,
                model_version,
            )
            return
        # Iterate from current model version -> target version
        # Updating each node based on the correct adapter [ver->ver+1]
        for opset_version in range(model_version, self.target_version):
            if tuple((opset_version, opset_version + 1)) not in _ADAPTER_SETS:
                logger.warning(
                    "Conversion from opset: %s to target opset: %s not currently supported.",
                    opset_version,
                    opset_version + 1,
                )
                return

            self.visit_graph(model.graph, opset_version)
            for function in model.functions.values():
                self.visit_function(function, opset_version)


def version_convert(model: ir.Model, target_version: int) -> None:
    """Convert the model to the specified ONNX opset version."""
    version_converter = _VersionConverter(target_version=target_version)
    version_converter.visit_model(model)
