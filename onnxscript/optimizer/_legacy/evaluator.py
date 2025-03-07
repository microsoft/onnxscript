# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Callable, Protocol, Sequence, Union

import numpy as np
import onnx
import onnx.reference.ops

import onnxscript._legacy_ir as ir
from onnxscript.utils.utils import (
    get_node_attr_value,
)

logger = logging.getLogger(__name__)

# "Standard" evaluators are used to perform constant-folding.
# The API below works only for non-control-flow ops (ops without any graph-attributes).
# This currently used ONNX's reference implementation. But we could also
# use ORT's implementation if we want to.


class ReferenceEvaluator:
    def get_evaluator(self, domain: str, op: str, version: int) -> callable | None:
        try:
            op_impl_class = onnx.reference.ops.load_op(domain, op, version)
            return op_impl_class.eval  # noqa: TRY300
        except Exception:
            return None

    def evaluate(self, domain: str, op: str, version: int, *args, **kwargs) -> Any:
        logger.debug("Evaluating %s::%s", domain, op)
        evaluator = self.get_evaluator(domain, op, version)
        if evaluator is None:
            return None
        return evaluator(*args, **kwargs)


reference_evaluator = ReferenceEvaluator()

# The "partial evaluators" below are non-standard evaluators. They are used to perform
# partial evaluation and/or static program analysis (abstract interpretation).


class IRContext(Protocol):
    """A class that represents the context for partial evaluation.

    This is a placeholder, subject to simplification when a proper IR is defined.
    """

    def get_input(self, node: onnx.NodeProto, index: int) -> ir.Value | None: ...

    def get_output(self, node: onnx.NodeProto, index: int) -> ir.Value | None: ...

    def input_const_value(self, node: onnx.NodeProto, index: int) -> ir.ConcreteValue: ...

    def input_shape(
        self, node: onnx.NodeProto, index: int
    ) -> onnx.TensorShapeProto | None: ...

    def input_type(self, node: onnx.NodeProto, index: int) -> onnx.TypeProto | None: ...

    def input_element_type(self, node: onnx.NodeProto, index: int) -> int | None: ...

    def lookup_version(self, domain: str) -> int: ...

    def convert_attributes(self, attributes: Sequence[onnx.AttributeProto]) -> dict: ...

    def new_constant(self, name: str, value: Any) -> Sequence[onnx.NodeProto] | None: ...


# A partial-evaluator function takes an IRContext and a node, and returns a list of
# replacement nodes or None (if no replacement is needed). We return None instead
# of [input node] so the caller is aware that the node is not replaced. If the node
# is replaced, the caller will recursively visit the replacement nodes to process them.

PartialEvaluatorFunction = Union[
    Callable[[IRContext, onnx.NodeProto], Sequence[onnx.NodeProto]], None
]


@dataclasses.dataclass
class PartialEvaluator:
    """A class that represents a partial-evaluator for a particular op.

    It is applicable for a specific version range (min_version, max_version) of the op.
    The min_version and max_version can be None, indicating that there is no version
    constraint in that direction.
    """

    min_version: int | None
    max_version: int | None
    function: PartialEvaluatorFunction

    def valid_for(self, version: int) -> bool:
        """Returns True if this evaluator is applicable for the given version."""
        return (self.min_version is None or version >= self.min_version) and (
            self.max_version is None or version <= self.max_version
        )


class PartialEvaluatorRegistry:
    """A class that maintains a registry of evaluators for ops."""

    def __init__(self):
        self.op_evaluators: dict[tuple[str, str], list[PartialEvaluator]] = {}

    def lookup_evaluators(self, domain: str, opname: str, version: int):
        evaluator_list = self.op_evaluators.get((domain, opname), [])
        return [
            evaluator.function for evaluator in evaluator_list if evaluator.valid_for(version)
        ]

    def register(self, opname: str, domain: str = "", version=None):
        if (domain, opname) not in self.op_evaluators:
            evaluator_list = []
            self.op_evaluators[(domain, opname)] = evaluator_list
        else:
            evaluator_list = self.op_evaluators[(domain, opname)]
        if version is None:
            min_version = None
            max_version = None
        elif isinstance(version, int):
            min_version = version
            max_version = version
        elif isinstance(version, tuple):
            min_version, max_version = version

        def decorator(function: PartialEvaluatorFunction) -> PartialEvaluatorFunction:
            evaluator_list.append(PartialEvaluator(min_version, max_version, function))
            return function

        return decorator


registry: PartialEvaluatorRegistry = PartialEvaluatorRegistry()

register = registry.register


def get_bool_value(val) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray) and val.size == 1 and val.dtype == bool:
        return val.item(0)
    return None


def get_size_info(type: onnx.TypeProto) -> np.ndarray | None:
    if type.HasField("tensor_type") and type.tensor_type.HasField("shape"):
        if all(d.HasField("dim_value") for d in type.tensor_type.shape.dim):
            size = 1
            for d in type.tensor_type.shape.dim:
                size *= d.dim_value
            return np.array(size, dtype=np.int64)
    return None


def get_dim_info(type: onnx.TypeProto, dim: int) -> int | None:
    if type.HasField("tensor_type") and type.tensor_type.HasField("shape"):
        rank = len(type.tensor_type.shape.dim)
        dim = dim if dim >= 0 else dim + rank
        if dim < 0 or dim >= rank:
            return None
        if type.tensor_type.shape.dim[dim].HasField("dim_value"):
            return type.tensor_type.shape.dim[dim].dim_value
    return None


@register("Cast")
def cast(context: IRContext, node: onnx.NodeProto) -> Sequence[onnx.NodeProto] | None:
    if context.input_shape(node, 0) is not None:
        output_value = context.get_output(node, 0)
        output_value.type = onnx.TypeProto()
        output_value.type.CopyFrom(context.input_type(node, 0))
        output_value.type.tensor_type.elem_type = node.attribute[0].i
    return None


@register("CastLike")
def cast_like(context: IRContext, node: onnx.NodeProto):
    source_element_type = context.input_element_type(node, 0)
    target_element_type = context.input_element_type(node, 1)

    if target_element_type is None:
        return None
    if source_element_type == target_element_type:
        node.op_type = "Identity"
        del node.input[1]
        return [node]

    node.op_type = "Cast"
    del node.input[1]
    del node.attribute[:]
    node.attribute.append(onnx.helper.make_attribute("to", target_element_type))
    return [node]


@register("Shape")
def shape(context: IRContext, node: onnx.NodeProto):
    shape = context.input_shape(node, 0)
    if shape is None:
        return None
    start = get_node_attr_value(node, "start", 0)
    end = get_node_attr_value(node, "end", None)
    shape_slice = shape.dim[start:end]
    if all(d.HasField("dim_value") for d in shape_slice):
        return np.array([d.dim_value for d in shape_slice], dtype=np.int64)
    return None


@register("Size")
def size(context: IRContext, node: onnx.NodeProto):
    type = context.input_type(node, 0)
    size = get_size_info(type) if type is not None else None
    return size


@register("If")
def if_op(context: IRContext, node: onnx.NodeProto):
    cond = context.input_const_value(node, 0)
    if cond is ir.NotConstant:
        #  Visitor will recursively visit subgraphs to constant-fold them.
        return None
    cond = get_bool_value(cond)
    if cond is not None:
        # cond is a constant-value: inline the branch
        branch = "then_branch" if cond else "else_branch"
        graph = onnx.helper.get_node_attr_value(node, branch)

        formal_outs = list(graph.output)
        actual_outs = node.output
        renamings = {
            formal.name: actual
            for formal, actual in zip(formal_outs, actual_outs)
            if actual != ""
        }
        # TODO: Extend renaming to intermediate values.

        def rename(name):
            return renamings.get(name, name)

        for sub_node in graph.node:
            # TODO: handle renaming inside subgraphs in nodes
            sub_node.input[:] = [rename(name) for name in sub_node.input]
            sub_node.output[:] = [rename(name) for name in sub_node.output]
            # Avoid name collision.
            sub_node.name = f"{node.name}_{sub_node.name}"

        # TODO: we should handle initializers as well!
        return list(graph.node)
    return None


@register("Identity")
def identity(context: IRContext, node: onnx.NodeProto):
    input = context.get_input(node, 0)
    output = context.get_output(node, 0)
    if input is not None and output is not None:
        output.symbolic_value = input.name


@register("SequenceConstruct")
def sequence_construct(
    context: IRContext, node: onnx.NodeProto
) -> Sequence[onnx.NodeProto] | None:
    output = context.get_output(node, 0)
    if output is not None:
        output.symbolic_value = list(node.input)
    return None


@register("ConcatFromSequence")
def concat_from_sequence(
    context: IRContext, node: onnx.NodeProto
) -> Sequence[onnx.NodeProto] | None:
    input = context.get_input(node, 0)
    attrs = context.convert_attributes(node.attribute)
    new_axis = attrs.get("new_axis", 0)
    if input is not None and isinstance(input.symbolic_value, list):
        if new_axis == 0:
            node.op_type = "Concat"
            node.input[:] = input.symbolic_value
            logger.debug("ConcatFromSequence => Concat: %s", node.input)
            for i in range(len(node.attribute)):
                if node.attribute[i].name == "new_axis":
                    del node.attribute[i]
                    return [node]
            return [node]
        if new_axis == 1:
            # Unsqueeze the inputs with concat axis if new_axis is 1
            axis = attrs.get("axis", None)
            assert axis is not None
            output = context.get_output(node, 0)
            axis_node = context.new_constant(f"{output.name}_axis", np.array([axis]))[0]
            unsqueeze_nodes = []
            for node_input in input.symbolic_value:
                unsqueeze_node = onnx.helper.make_node(
                    "Unsqueeze",
                    [node_input, axis_node.output[0]],
                    [f"{node_input}_unsqueeze"],
                )
                unsqueeze_nodes.append(unsqueeze_node)
            unsqueeze_outputs = [n.output[0] for n in unsqueeze_nodes]
            unsqueeze_nodes = [axis_node, *unsqueeze_nodes]

            # Send unsqueezed outputs to Concat
            node.input[:] = unsqueeze_outputs
            node.op_type = "Concat"
            logger.debug(
                "ConcatFromSequence => UnSqueeze %s + Concat %s",
                unsqueeze_outputs,
                node.input,
            )
            for i in range(len(node.attribute)):
                if node.attribute[i].name == "new_axis":
                    del node.attribute[i]
                    break
            return [*unsqueeze_nodes, node]
    return None


@register("SplitToSequence")
def split_to_sequence(
    context: IRContext, node: onnx.NodeProto
) -> Sequence[onnx.NodeProto] | None:
    """Rewriting pattern.

    From

        splits = onnx::SplitToSequence(input, split, axis=axis)

    to

        split_0, split_1, ..., split_n = onnx::Split(input, split, axis=axis)
        splits = onnx::SequenceConstruct(split_0, split_1, ..., split_n)

    or

        split_0, split_1, ..., split_n = onnx::Split(input, axis=axis, num_outputs=n+1)
        splits = onnx::SequenceConstruct(split_0, split_1, ..., split_n)

    where number of output tensors in `splits` is statically known.
    onnx::SequenceConstruct will be further optimized away if possible, by its own designated evaluator.
    This allows downstream `SequenceAt` users to be replaced by `split_x` accordingly.
    """
    input = context.get_input(node, 0)
    split = context.get_input(node, 1)
    attrs = context.convert_attributes(node.attribute)
    output = context.get_output(node, 0)

    if input is None or split is None or output is None:
        return None

    axis = attrs.get("axis", 0)
    if input.type is None:
        return None
    split_dimension_size = get_dim_info(input.type, axis)
    if split_dimension_size is None:
        return None

    split_value = split.value
    if split_value is None or split_value is ir.NotConstant:
        return None
    assert isinstance(split_value, np.ndarray)

    if split_value.ndim == 0:
        # split into chunks all of size 'split' if possible.
        num_outputs = math.ceil(split_dimension_size / split_value.item())
        split_outputs = [f"{output.name}_split_{i}" for i in range(num_outputs)]
        split_node = onnx.helper.make_node(
            "Split",
            [input.name],
            split_outputs,
            axis=axis,
            num_outputs=num_outputs,
        )
    else:
        # split into 'size(split)' chunks
        num_outputs = split_value.size
        split_outputs = [f"{output.name}_split_{i}" for i in range(num_outputs)]
        split_node = onnx.helper.make_node(
            "Split",
            [input.name, split.name],
            split_outputs,
            axis=axis,
        )

    keepdims = attrs.get("keepdims", 1)
    squeeze_nodes = []
    if keepdims == 0:
        # squeeze the split dimension if keepdims is 0
        axis_node = context.new_constant(f"{output.name}_axis", np.array([axis]))[0]
        for i in range(num_outputs):
            squeeze_node = onnx.helper.make_node(
                "Squeeze",
                [split_outputs[i], axis_node.output[0]],
                [f"{split_outputs[i]}_squeeze"],
            )
            squeeze_nodes.append(squeeze_node)
        split_outputs = [n.output[0] for n in squeeze_nodes]
        squeeze_nodes = [axis_node, *squeeze_nodes]

    node.op_type = "SequenceConstruct"
    node.input[:] = split_outputs
    del node.attribute[:]
    logger.debug(
        "SplitToSequence => Split %s + SequenceConstruct %s",
        split_node.input,
        node.input,
    )
    return [split_node, *squeeze_nodes, node]


@register("SequenceAt")
def sequence_at(context: IRContext, node: onnx.NodeProto) -> Sequence[onnx.NodeProto] | None:
    input = context.get_input(node, 0)
    position = context.get_input(node, 1)
    output = context.get_output(node, 0)
    if input is not None and position is not None:
        input_vals = input.symbolic_value
        position_val = position.value
        if isinstance(input_vals, list) and position_val is not None:
            output.symbolic_value = input_vals[position_val]
            logger.debug("SequenceAt %s => %s", input, output.symbolic_value)
            new_node = onnx.helper.make_node(
                "Identity", [output.symbolic_value], [output.name]
            )
            return [new_node]
    return None
