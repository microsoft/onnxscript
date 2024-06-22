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
import onnxscript.ir._convenience as _convenience
import onnxscript.optimizer.constant_folding as constant_folding
import onnxscript.rewriter.pattern
from onnxscript.utils.utils import (
    get_node_attr_value,
)

is_control_flow_op = constant_folding.is_control_flow_op
is_non_deterministic_op = constant_folding.is_non_deterministic_op
is_constant_op = constant_folding.is_constant_op
_DEFAULT_CONSTANT_FOLD_SIZE_LIMIT = constant_folding._DEFAULT_CONSTANT_FOLD_SIZE_LIMIT

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

def get_bool_value(val: ir.Value | None) -> bool | None:
    if val is None:
        return None
    val = get_numpy_value(val)
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray) and val.size == 1 and val.dtype == bool:
        return val.item(0)
    return None


def getInput(node:ir.Node, index: int) -> ir.Value | None:
    if index < len(node.inputs):
        return node.inputs[index]
    return None

def getOutput(node:ir.Node, index: int) -> ir.Value | None:
    if index < len(node.outputs):
        return node.outputs[index]
    return None

def updateType(value: ir.Value, type: ir.TypeProtocol) -> None:
    # TODO: merge types
    value.type = type

@register("Cast")
def cast(node: ir.Node) -> Sequence[ir.Node] | None:
    # This should not be necessary. Generic incremental shape-inference should handle this.
    input = getInput(node, 0)
    output = getOutput(node, 0)
    if input is not None and output is not None:
        updateType(output, input.type)
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
def if_op(op, node: ir.Node):
    cond = getInput(node, 0)
    cond = get_bool_value(cond)
    if cond is not None:
        # cond is a constant-value: inline the branch
        branch = "then_branch" if cond else "else_branch"
        graph = node.attributes.get(branch, None)
        if graph is None:
            return None
        formal_outs = graph.outputs
        actual_outs = node.outputs
        renamings = {
            formal.name: actual.name
            for formal, actual in zip(formal_outs, actual_outs)
            if actual is not None
        }
        # TODO: Extend renaming to intermediate values.

        def rename(name):
            return renamings.get(name, name)

        for sub_node in graph:
            # TODO: handle renaming inside subgraphs in nodes
            for v in sub_node.outputs:
                v.name = rename(v.name)
            # Avoid name collision.
            sub_node.name = f"{node.name}_{sub_node.name}"

        # TODO: we should handle initializers as well!
        return list(graph)
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


class ConstantFolder:
    opset_imports: dict[str, int]

    def new_constant(self, irvalue: ir.Value, value):
        # TODO(rama): Why do we need the conversion below?
        if isinstance(value, (int, float, np.ScalarType)):
            value = np.array(value)

        irvalue.const_value = value

        if not isinstance(value, np.ndarray):
            # ONNX does not have a way to represent non-tensor constants, eg. a sequence.
            # So, a constant-value of type sequence is not folded, but it can be used
            # to optimize subsequent operations when possible.
            logger.info(
                "Skip storing constant folded value %s due to unsupported type %s.",
                irvalue.name,
                type(value),
            )
            return None

        if value.nbytes > _DEFAULT_CONSTANT_FOLD_SIZE_LIMIT:
            logger.info(
                "Skip storing constant folded nvalue %s due to large size %s.",
                irvalue.name,
                value.nbytes,
            )
            return None

        tensor = onnx.numpy_helper.from_array(value, name)

        logger.debug(
            "New constant for value %s dtype: %s shape: %s",
            irvalue.name,
            value.dtype,
            value.shape,
        )

        # TODO(rama)
        # irvalue.type = onnx.helper.make_tensor_type_proto(
        #     onnx.helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
        # )
        attributes = _convenience.convert_attributes({"value": tensor})
        node = ir.Node("", "Constant", inputs=[], attributes=attributes, num_outputs=1)
        return [node]
    
    def process_node(self, node: ir.Node, root: ir.Graph | ir.Function):
        for i, value in enumerate(node.inputs):
            if value is not None and value.symbolic_value is not None:
                sym_value = value.symbolic_value
                if isinstance(sym_value, ir.Value):
                    node.replace_input_with(i, sym_value)
                    # TODO(rama): consider merging type/other info from both values

        # Do incremental shape inference

        if node.domain not in self.opset_imports:
            return None
        version  = self.opset_imports[node.domain]
        op_optimizers = registry.lookup_evaluators(node.domain, node.op_type, version)
        for optimizer in op_optimizers:
            assert optimizer
            context = onnxscript.rewriter.pattern.RewriterContext()
            output = optimizer(context, node)
            if output is not None:
                return output

        if is_control_flow_op(node) or is_non_deterministic_op(node):
            return None

        if any((x is not None and x.const_value is None) for x in node.inputs):
            return None

        input_values = [x.const_value.numpy() if x is not None else None for x in node.inputs]
        # Filter out bfloat16 cases?
        outputs = reference_evaluator.evaluate(node.domain, node.op_type, version, *input_values, **node.attributes)
        if outputs is None:
            return None
        if len(node.output) == 1 and not isinstance(outputs, (tuple, list)):
            replacement = self.new_constant(node.outputs[0], outputs)
            if is_constant_op(node):
                return None
            # self.add_count(op, outputs.size)
            return replacement
        else:
            logger.warning("Skipping constant folding for op %s with multiple outputs.", node.op_type)
        return None

    def replace_node(self, node: ir.Node, replacement, root: ir.Graph | ir.Function):
        # TODO: apply delta! what about opset_imports?

        for old_value, new_value in zip(old_values, new_values):
            # Propagate relevant info from old value to new value
            # TODO(Rama): Perhaps we should merge old and new types. As of now, new
            # values don't have type information. Note that this could be a problem
            # for semantics-altering rewrite-rules: we should allow users to override
            # this for such rules.
            new_value.type = old_value.type
            new_value.shape = old_value.shape
            new_value.const_value = old_value.const_value
            new_value.name = old_value.name

        # Reconnect the users of the deleted node to use the new outputs
        _convenience.replace_all_uses_with(old_values, new_values)
        # Update graph/function outputs if the node generates output
        replacement_mapping = dict(zip(old_values, new_values))
        for idx, graph_or_function_output in enumerate(root.outputs):
            if graph_or_function_output in replacement_mapping:
                root.outputs[idx] = replacement_mapping[graph_or_function_output]

        # insert new nodes after the index node
        root.insert_after(node, delta.new_nodes)
        root.remove(node, safe=True)

        # if isinstance(output, list):
        #     return output
        # else:
        #     # Currently handles single output only
        #     self.add_count(node.op_type, output.size)
        #     return self.new_constant(node.output[0], output)

    def visit_node(self, node: ir.Node):
        replacement = self.process_node(node)
        # logger.debug(
        #     "visit_node: %s::%s %s replacement %s",
        #     node.domain,
        #     node.op_type,
        #     node.name,
        #     "found" if replacement is not None else "missed",
        # )
        if replacement is None:
            # No change. Process attributes.
            for attr in node.attribute:
                self.visit_attribute(attr)
            return None
        else:
            self.replace_node(node, replacement)
