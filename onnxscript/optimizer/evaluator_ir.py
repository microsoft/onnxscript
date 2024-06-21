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

import onnxscript.ir as ir
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


# class IRContext(Protocol):
#     """A class that represents the context for partial evaluation.

#     This is a placeholder, subject to simplification when a proper IR is defined.
#     """

#     def get_input(self, node: onnx.NodeProto, index: int) -> ir.Value | None: ...

#     def get_output(self, node: onnx.NodeProto, index: int) -> ir.Value | None: ...

#     def input_const_value(self, node: onnx.NodeProto, index: int) -> ir.ConcreteValue: ...

#     def input_shape(
#         self, node: onnx.NodeProto, index: int
#     ) -> onnx.TensorShapeProto | None: ...

#     def input_type(self, node: onnx.NodeProto, index: int) -> onnx.TypeProto | None: ...

#     def input_element_type(self, node: onnx.NodeProto, index: int) -> int | None: ...

#     def lookup_version(self, domain: str) -> int: ...

#     def convert_attributes(self, attributes: Sequence[onnx.AttributeProto]) -> dict: ...

#     def new_constant(self, name: str, value: Any) -> Sequence[onnx.NodeProto] | None: ...


# A partial-evaluator function takes an IRContext and a node, and returns a list of
# replacement nodes or None (if no replacement is needed). We return None instead
# of [input node] so the caller is aware that the node is not replaced. If the node
# is replaced, the caller will recursively visit the replacement nodes to process them.

PartialEvaluatorFunction = Callable[[ir.Node], Union[Sequence[ir.Node], None]]

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

def get_numpy_value(val: ir.Value) -> np.ndarray | None:
    const_value = val.const_value
    if hasattr(const_value, "numpy"):
        return const_value.numpy()
    return None

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


def get_dim_info(type: ir.Type, dim: int) -> int | None:
    if type.HasField("tensor_type") and type.tensor_type.HasField("shape"):
        rank = len(type.tensor_type.shape.dim)
        dim = dim if dim >= 0 else dim + rank
        if dim < 0 or dim >= rank:
            return None
        if type.tensor_type.shape.dim[dim].HasField("dim_value"):
            return type.tensor_type.shape.dim[dim].dim_value
    return None

def getInput(node:ir.Node, index: int) -> ir.Value | None:
    if index < len(node.inputs):
        return node.inputs[index]
    return None


@register("Cast")
def cast(node: ir.Node) -> Sequence[ir.Node] | None:
    if context.input_shape(node, 0) is not None:
        output_value = context.get_output(node, 0)
        output_value.type = onnx.TypeProto()
        output_value.type.CopyFrom(context.input_type(node, 0))
        output_value.type.tensor_type.elem_type = node.attribute[0].i
    return None


@register("CastLike")
def cast_like(op, node: ir.Node):
    input0 = node.inputs[0]
    input1 = node.inputs[1]
    source_element_type = input0.type.dtype.value
    target_element_type = input1.type.dtype.value

    if target_element_type is None:
        return None
    if source_element_type == target_element_type:
        return op.Identity(input0)
    return op.Cast(input0, to=target_element_type)


@register("Shape")
def shape(op, node: ir.Node):
    del op
    input = node.inputs[0]
    shape = input.shape
    if shape is None:
        return None
    start = node.attributes.get("start", 0)
    end = node.attributes.get("end", None)
    shape_slice = shape.dim[start:end]
    if all(d.HasField("dim_value") for d in shape_slice):
        return np.array([d.dim_value for d in shape_slice], dtype=np.int64)
    return None


@register("Size")
def size(op, node: ir.Node):
    del op
    shape = node.inputs[0].shape
    if shape is None:
        return None
    size = 1
    for d in shape:
        if not isinstance(d, int):
            return None
        size *= d
    return np.array(size, dtype=np.int64)

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
def identity(op, node: ir.Node):
    del op
    input = node.inputs[0]
    output = node.outputs[0]
    if input is not None and output is not None:
        output.symbolic_value = input
    return None


@register("SequenceConstruct")
def sequence_construct(op, node: ir.Node):
    del op
    output = node.outputs[0]
    if output is not None:
        output.symbolic_value = list(node.inputs)
    return None


@register("ConcatFromSequence")
def concat_from_sequence(op, node: ir.Node):
    input = node.inputs[0]
    inputs = input.symbolic_value
    if any(x is None for x in inputs):
        return None
    new_axis = node.attributes.get("new_axis", 0)
    axis = node.attributes["axis"]
    if input is not None and isinstance(input.symbolic_value, list):
        if new_axis == 0:
            logger.debug("ConcatFromSequence => Concat: %s", [x.name for x in inputs])
            return op.Concat(*inputs, axis=axis)
        if new_axis == 1:
            # Unsqueeze the inputs with concat axis if new_axis is 1
            axis_value = op.Constant(value_int=axis)
            unsqueezed_inputs = []
            for node_input in inputs:
                unsqueezed_input = op.Unsqueeze(node_input, axis_value, output=[f"{node_input.name}_unsqueeze"])
                unsqueezed_inputs.append(unsqueezed_input)
            # Send unsqueezed outputs to Concat
            logger.debug(
                "ConcatFromSequence => Concat %s",
                [x.name for x in unsqueezed_inputs]
            )
            return op.Concat(*unsqueezed_inputs, axis=axis)
    return None


@register("SplitToSequence")
def split_to_sequence(op, node: ir.Node):
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
    input = node.inputs[0]
    split = node.inputs[1]
    output = node.outputs[0]

    if input is None or split is None or output is None:
        return None

    axis = node.attributes.get("axis", 0)
    shape = input.shape
    if shape is None:
        return None
    rank = len(shape)
    if axis < 0:
        axis = axis + rank
    if axis < 0 or axis >= rank:
        return None
    split_dimension_size = shape[axis]
    if not isinstance(split_dimension_size, int):
        return None

    split_value = get_numpy_value(split)
    if split_value is None:
        return None
    assert isinstance(split_value, np.ndarray)

    if split_value.ndim == 0:
        # split into chunks all of size 'split' if possible.
        num_outputs = math.ceil(split_dimension_size / split_value.item())
        split_outputs = [f"{output.name}_split_{i}" for i in range(num_outputs)]
        split_values = op.Split(input, axis=axis, num_outputs=num_outputs, output=split_outputs)
    elif split_value.ndim == 1:
        # split into 'size(split)' chunks
        num_outputs = split_value.size
        split_outputs = [f"{output.name}_split_{i}" for i in range(num_outputs)]
        split_values = op.Split(input, split, axis=axis, output=split_outputs)
    else:
        return None

    keepdims = node.attributes.get("keepdims", 1)
    if keepdims == 0:
        # squeeze the split dimension if keepdims is 0
        axis_val = op.Constant(value_int=axis, outputs=[f"{output.name}_axis"])
        squeezed_values = []
        for i in range(num_outputs):
            squeezed = op.Squeeze(split_values[i], axis_val, output=[f"{split_outputs[i]}_squeeze"])
            squeezed_values.append(squeezed)
        split_values = squeezed_values

    logger.debug("SplitToSequence => Split + SequenceConstruct")

    return op.SequenceConstruct(*split_values)
    # return [split_node, *squeeze_nodes, node]


@register("SequenceAt")
def sequence_at(op, node: ir.Node):
    input = node.inputs[0]
    position = node.inputs[1]
    output = node.outputs[0]
    if input is not None and position is not None:
        input_vals = input.symbolic_value
        position_val = get_numpy_value(position)
        if isinstance(input_vals, list) and position_val is not None:
            if position_val.size != 1:
                return None
            position_val = position_val.item()
            try:
                result = input_vals[position_val]
            except IndexError:
                return None
            output.symbolic_value = result 
            logger.debug("SequenceAt %s => %s", input.name, result.name)
            return op.Identity(result)
    return None
