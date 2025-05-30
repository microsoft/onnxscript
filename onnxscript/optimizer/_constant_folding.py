# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This will eventually replace the existing constant_folding.py and evaluator.py files.

from __future__ import annotations

import dataclasses
import logging
import math
import typing
from typing import Any, Callable, Iterable, Sequence, Union

import numpy as np
import onnx
import onnx.reference.ops

import onnxscript.ir as ir
import onnxscript.utils.utils as utils
from onnxscript.ir import _tape

DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT = 1024

DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT = 1024 * 1024


def is_control_flow_op(node: ir.Node) -> bool:
    graph_types = {ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS}
    return any(attr.type in graph_types for attr in node.attributes.values())


non_deterministic_ops = frozenset(
    {
        "RandomUniform",
        "RandomNormal",
        "RandomUniformLike",
        "RandomNormalLike",
        "Multinomial",
    }
)


def is_non_deterministic_op(node: ir.Node) -> bool:
    return node.op_type in non_deterministic_ops and utils.is_onnx_domain(node.domain)


def is_onnx_op(node: ir.Node, op_type: str) -> bool:
    return node.op_type == op_type and utils.is_onnx_domain(node.domain)


def is_constant_op(node: ir.Node) -> bool:
    return node.op_type in {"Constant", "ConstantOfShape"} and utils.is_onnx_domain(
        node.domain
    )


logger = logging.getLogger(__name__)

# "Standard" evaluators are used to perform constant-folding.
# The API below works only for non-control-flow ops (ops without any graph-attributes).
# This currently used ONNX's reference implementation. But we could also
# use ORT's implementation if we want to.


def _process_constant_node(node: ir.Node) -> None:
    """Sets const_value of output value of a Constant op node."""
    if node.op_type != "Constant" or node.domain != "":
        return
    if len(node.attributes) != 1:
        return
    attr_name, attr_value = next(iter(node.attributes.items()))
    if len(node.outputs) != 1:
        return
    ir_value = node.outputs[0]

    if attr_value is None or not isinstance(attr_value, ir.Attr):
        return

    const_value: ir.TensorProtocol
    if attr_name in {"value_float", "value_floats"}:
        const_value = ir.Tensor(
            np.array(attr_value.value, dtype=np.float32), name=ir_value.name
        )
    elif attr_name in {"value_int", "value_ints"}:
        const_value = ir.Tensor(np.array(attr_value.value, dtype=np.int64), name=ir_value.name)
    elif attr_name in {"value_string", "value_strings"}:
        const_value = ir.StringTensor(
            np.array(attr_value.value, dtype=np.bytes_), name=ir_value.name
        )
    elif attr_name == "value":
        const_value = typing.cast(ir.TensorProtocol, attr_value.value)
    else:
        return

    ir_value.const_value = const_value
    ir_value.shape = const_value.shape  # type: ignore
    ir_value.dtype = const_value.dtype


def basic_constant_propagation(nodes: Iterable[ir.Node]) -> None:
    """Performs basic constant propagation for a sequence of nodes.

    Just marks the output values of Constant op nodes with their const_value.
    """
    for node in nodes:
        _process_constant_node(node)


class ReferenceEvaluator:
    def get_evaluator(self, domain: str, op: str, version: int) -> Callable | None:
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
        try:
            return evaluator(*args, **kwargs)
        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            return None


_reference_evaluator = ReferenceEvaluator()


@dataclasses.dataclass
class Replacement:
    """A replacement for a node in the graph."""

    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]


# The optimizer tracks an optional symbolic value for each value in the model.
# The symbolic value attached to a value X can be:
# - another IR value Y (indicating that X is equal to Y)
# - a list of IR values [Y1, Y2, ...] (indicating that X is a sequence of values Y1, Y2, ...)
# - a Shape object (indicating that X is a shape value)
# A Shape object as a symbolic value indicates that the corresponding value is
# 1-D (or 0-D) tensor of INT64 values. The values in this object may be constants
# or symbolic dimension values (like "batch_size", "sequence_length", etc.).
# Currently, we assume that symbolic dimensions are also guaranteed to be non-negative.
# TODO: Add support for negative symbolic dimensions.

SymbolicValue = Union[ir.Value, list[ir.Value], ir.Shape]


class OptimizerState:
    def __init__(self):
        self._sym_value_map: dict[ir.Value, SymbolicValue] = {}
        self._initializer_inputs: list[set[ir.Value]] = []

    @property
    def symbolic_value_map(self) -> dict[ir.Value, SymbolicValue]:
        return self._sym_value_map

    def get_sym_value(self, value: ir.Value | None) -> SymbolicValue | None:
        if value is None:
            return None
        return self._sym_value_map.get(value)

    def set_sym_value(self, value: ir.Value, sym_value: SymbolicValue) -> None:
        self._sym_value_map[value] = sym_value

    def push_initializer_inputs(self) -> None:
        self._initializer_inputs.append(set())

    def pop_initializer_inputs(self) -> None:
        self._initializer_inputs.pop()

    def add_initializer_input(self, value: ir.Value) -> None:
        assert self._initializer_inputs
        self._initializer_inputs[-1].add(value)

    def is_initializer_input(self, value: ir.Value) -> bool:
        return any(value in inputs for inputs in self._initializer_inputs)

    def get_shape_value(self, value: ir.Value | None) -> ir.Shape | None:
        const_value = _get_numpy_value(value, ir.DataType.INT64, size_limit=10)
        if const_value is not None:
            if const_value.ndim == 1:
                return ir.Shape(const_value.tolist())
            return None
        sym_value = self.get_sym_value(value)
        if isinstance(sym_value, ir.Shape):
            return sym_value
        # TODO use shape of value if available
        return None


# The "partial evaluators" below are non-standard evaluators. They are used to perform
# partial evaluation and/or static program analysis (abstract interpretation).

# A partial-evaluator function takes a node, a RewriterContext, OptimizerState and returns
# a Replacement for the node or None (if no replacement is needed). It may also return just
# the ir.Value or ir.Values to replace the output values of the node, when the new nodes
# can be inferred from the RewriterContext used to build the new nodes.

RewriterContext = _tape.Builder
ReturnValue = Union[Replacement, Sequence[ir.Value], ir.Value, None]
PartialEvaluatorFunction = Callable[[ir.Node, RewriterContext, OptimizerState], ReturnValue]


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

    def register(
        self, opname: str, domain: str = "", version=None
    ) -> Callable[[PartialEvaluatorFunction], PartialEvaluatorFunction]:
        if (domain, opname) in self.op_evaluators:
            evaluator_list = self.op_evaluators[(domain, opname)]
        else:
            evaluator_list = []
            self.op_evaluators[(domain, opname)] = evaluator_list
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


def _same_shape(shape1: ir.Shape, shape2: ir.Shape) -> bool:
    # Comparison of shapes as tuples works except if any dimension is None
    # (which represents an unknown dimension value). Thus, two shapes such
    # as (Batch, 1024) and (Batch, 1024) are considered equal, but (None, 1024)
    # and (None, 1024) are not considered equal.
    if any(isinstance(dim, ir.SymbolicDim) and dim.value is None for dim in shape1):
        return False
    return shape1.dims == shape2.dims


def _get_numpy_value(
    val: ir.Value | None, dtype: ir.DataType | None = None, size_limit: int | None = None
) -> np.ndarray | None:
    """Returns the numpy value of a constant value, if available.

    It returns None if the value is not a constant value, or if the value is not of
    the specified element dtype, or if the size of the value exceeds the specified
    size_limit.
    """
    if val is None:
        return None
    const_value = val.const_value
    if const_value is not None:
        if dtype is not None and const_value.dtype != dtype:
            return None
        if size_limit is not None and const_value.size > size_limit:
            return None
        try:
            # Reinterpret the array with `.view()` because some implementations of
            # ir.TensorProtocol (e.g. PyTorch<=2.7) do not use ml_dtypes for bfloat16 etc.
            array = const_value.numpy().view(const_value.dtype.numpy())
        except FileNotFoundError:
            # External data is not available.
            return None
        assert isinstance(array, np.ndarray)
        return array
    return None


def _get_bool_value(val: ir.Value | None) -> bool | None:
    if val is None:
        return None
    value = _get_numpy_value(val)
    if value is None:
        return None
    if value.size == 1 and value.dtype == bool:
        return value.item(0)
    return None


def _get_input(node: ir.Node, index: int) -> ir.Value | None:
    if index < len(node.inputs):
        return node.inputs[index]
    return None


def _get_output(node: ir.Node, index: int) -> ir.Value | None:
    if index < len(node.outputs):
        return node.outputs[index]
    return None


def _update_type(value: ir.Value, type: ir.TypeProtocol | None) -> None:
    if type is not None:
        # TODO: merge types
        value.type = type


def _get_input_element_type(node: ir.Node, index: int) -> int:
    input = _get_input(node, index)
    if input is not None and input.type is not None:
        return input.type.dtype.value
    return ir.DataType.UNDEFINED.value


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


@register("Abs")
def abs(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    """Replace an Abs node by Identity when applicable.

    Currently, addresses Abs applied to symbolic shapes.
    """
    input = _get_input(node, 0)
    input_sym_value = state.get_shape_value(input)
    if input_sym_value is None:
        return None
    if any(isinstance(d, int) and d < 0 for d in input_sym_value):
        return None
    # Abs applied to a symbolic shape of the form [1, 1, SequenceLength].
    # We assume that SequenceLength is a non-negative integer.
    # The Abs op is redundant in this case.
    return op.Identity(input)


@register("Gather")
def gather(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    """Replace a Gather node by a constant when applicable.

    Currently, handles the case of Gathering from a shape tensor.
    """
    input = _get_input(node, 0)
    indices = _get_input(node, 1)
    if input is None or indices is None:
        return None
    input_sym_value = state.get_shape_value(input)
    if input_sym_value is None:
        return None
    axis = _get_int_attribute(node, "axis", None)
    if axis != 0:
        return None
    indices_numpy_value = _get_numpy_value(indices)
    if indices_numpy_value is None:
        return None
    if indices_numpy_value.ndim != 1:
        return None
    gathered = [input_sym_value[i] for i in indices_numpy_value]
    output = _get_output(node, 0)
    if output is not None:
        state.set_sym_value(output, ir.Shape(gathered))
    if all(isinstance(d, int) for d in gathered):
        return op.Constant(value_ints=gathered)
    return None


@register("Reshape")
def reshape(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    """Replace a Reshape node by Identity when applicable."""
    input = _get_input(node, 0)
    shape = _get_input(node, 1)
    if input is None or shape is None:
        return None

    input_shape = input.shape
    shape_value = state.get_shape_value(shape)

    if shape_value is None or input_shape is None:
        return None

    # No need to check for special values like -1, 0, etc. here
    if _same_shape(input_shape, shape_value):
        return op.Identity(input)
    return None


@register("Cast")
def cast(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    input = _get_input(node, 0)
    output = _get_output(node, 0)

    if input is None or output is None:
        return None

    # TODO(rama): Parts of the following logic (implementing type/shape inference
    # for Cast op) should be unnecessary. Generic incremental shape-inference
    # should handle this. Only the optimization to eliminate redundant Cast ops
    # should be needed here.

    input_shape = input.shape
    if input_shape is not None:
        output.shape = input_shape.copy()

    input_dtype = _get_input_element_type(node, 0)
    output_dtype = _get_int_attribute(node, "to", None)
    if output_dtype is not None:
        if input_dtype == output_dtype:
            return op.Identity(input)
        output.type = ir.TensorType(ir.DataType(output_dtype))
    return None


@register("CastLike")
def cast_like(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    input0 = node.inputs[0]
    source_element_type = _get_input_element_type(node, 0)
    target_element_type = _get_input_element_type(node, 1)

    if target_element_type == ir.DataType.UNDEFINED:
        return None
    if source_element_type == target_element_type:
        return op.Identity(input0)
    return op.Cast(input0, to=target_element_type)


@register("Shape")
def shape(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    input = node.inputs[0]
    if input is None:
        return None
    shape = input.shape
    if shape is None:
        return None
    start = _get_int_attribute(node, "start", 0)
    end = _get_int_attribute(node, "end", None)
    shape_slice = shape[start:end]
    output = _get_output(node, 0)
    if output is not None:
        state.set_sym_value(output, ir.Shape(shape_slice))
    if all(isinstance(d, int) for d in shape_slice):
        return op.Constant(value_ints=list(shape_slice))
    return None


@register("Size")
def size(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    input = _get_input(node, 0)
    if input is None:
        return None
    shape = input.shape
    if shape is None:
        return None
    size = 1
    for d in shape:
        if not isinstance(d, int):
            return None
        size *= d
    return op.Constant(value_int=size)


@register("If")
def if_op(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    cond_input = _get_input(node, 0)
    cond = _get_bool_value(cond_input)
    if cond is not None:
        # cond is a constant-value: inline the branch
        branch = "then_branch" if cond else "else_branch"
        graph_attr = node.attributes.get(branch)
        if graph_attr is None:
            return None
        if graph_attr.type != ir.AttributeType.GRAPH:
            return None
        assert isinstance(graph_attr, ir.Attr)
        graph = graph_attr.as_graph()
        # Copy the graph outputs and clear the graph outputs so that the values are free to move
        formal_outs = list(graph.outputs)
        graph.outputs.clear()
        actual_outs = node.outputs
        renamings = {
            formal.name: actual.name
            for formal, actual in zip(formal_outs, actual_outs)
            if actual is not None
        }
        # TODO: Extend renaming to intermediate values.

        def rename(name):
            return renamings.get(name, name)

        graph_nodes = list(graph)
        graph.remove(graph_nodes)
        for sub_node in graph_nodes:
            # TODO: handle renaming inside subgraphs in nodes
            for v in sub_node.outputs:
                v.name = rename(v.name)
            # Avoid name collision.
            sub_node.name = f"{node.name}_{sub_node.name}"

        # TODO: we should handle initializers as well!
        return Replacement(formal_outs, graph_nodes)
    return None


@register("Identity")
def identity(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    del op
    input = node.inputs[0]
    output = node.outputs[0]
    if input is not None and output is not None:
        state.set_sym_value(output, input)
    return None


@register("SequenceConstruct")
def sequence_construct(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    del op
    output = node.outputs[0]
    if output is not None:
        state.set_sym_value(output, list(node.inputs))
    return None


@register("Concat")
def concat(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    """Replace a Concat node with a single input by Identity"""

    # Replace Concat(x) by Identity(x)
    inputs = node.inputs
    if len(inputs) == 1:
        return op.Identity(inputs[0])

    axis = _get_int_attribute(node, "axis", None)
    if axis is None:
        return None

    # Eliminate zero-length operands from Concat
    def has_zero_size(operand: ir.Value | None) -> bool:
        if operand is None:
            return False  # Invalid model
        if (shape := operand.shape) is None:
            return False
        try:
            # We have already checked that axis is an int value (!= None)
            dim_size = shape[axis]  # type: ignore[index]
        except IndexError:
            return False
        return dim_size == 0  # return False if symbolic or None or non-zero int value

    new_inputs = [x for x in inputs if not has_zero_size(x)]
    if len(new_inputs) != len(inputs):
        if new_inputs:
            # Remove zero-length operands from Concat
            logger.debug(
                "Concat: removing zero-length operand(s) %s => %s", inputs, new_inputs
            )
            return op.Concat(*new_inputs, axis=axis)
        elif inputs:
            # All operands are zero-length. Concat is a no-op, but we need to use one of the
            # inputs to get the other dimensions correct:
            logger.debug("Concat: removing all zero-length operands %s", inputs)
            return op.Identity(inputs[0])
        else:
            # No inputs: invalid model.
            return None

    # Track value of tensors that carry a shape value:

    # Check axis attribute is 0

    if axis != 0:
        return None
    shapes = [state.get_shape_value(input) for input in inputs]
    if any(shape is None for shape in shapes):
        return None
    concatenated = ir.Shape(dim for shape in shapes for dim in shape.dims)  # type: ignore[union-attr]
    output = node.outputs[0]
    if output is None:
        return None
    state.set_sym_value(output, concatenated)
    return None


@register("Dropout", version=(12, None))
def dropout(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    """Replace a Dropout by Identity when applicable."""

    def optimized_dropout():
        input = node.inputs[0]
        output = op.Identity(input)
        if len(node.outputs) == 1:
            return output
        else:
            true_tensor = ir.tensor([True])
            input_shape = op.Shape(input)
            mask = op.ConstantOfShape(input_shape, value=true_tensor)
            return output, mask

    inputs = node.inputs
    if (len(inputs) <= 2) or inputs[2] is None:
        # No training_mode specified:
        return optimized_dropout()
    if _get_bool_value(inputs[2]) is False:
        # training_mode is False: dropout is not applied.
        return optimized_dropout()
    ratio = _get_numpy_value(inputs[1])
    if ratio is None:
        return None
    if ratio.size != 1:  # Only scalar dropout ratio is supported.
        return None
    if ratio.item() == 0:
        # dropout ratio is 0: dropout is not applied.
        return optimized_dropout()
    return None


@register("Expand")
def expand(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    """Replace an Expand node by Identity when applicable."""
    if len(node.inputs) != 2:
        return None
    if (input := node.inputs[0]) is None:
        return None
    if (input_shape := input.shape) is None:
        # Input shape is not known.
        return None
    if (expanded_shape := _get_numpy_value(node.inputs[1])) is None:
        # Target shape is not known.
        expanded_sym_shape = state.get_shape_value(node.inputs[1])
        if expanded_sym_shape is None or not _same_shape(input_shape, expanded_sym_shape):
            return None
        return op.Identity(input)
    if expanded_shape.ndim != 1:
        # Target shape must be a 1D tensor. Erroneous model.
        return None
    if input_shape.dims == tuple(expanded_shape.tolist()):
        return op.Identity(input)
    return None


@register("ConcatFromSequence")
def concat_from_sequence(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    input = node.inputs[0]
    inputs = state.get_sym_value(input)
    if inputs is None or any(x is None for x in inputs):
        return None
    new_axis = _get_int_attribute(node, "new_axis", 0)
    axis = _get_int_attribute(node, "axis", None)
    if axis is None:
        return None
    if input is not None and isinstance(inputs, list):
        if new_axis == 0:
            logger.debug("ConcatFromSequence => Concat: %s", [x.name for x in inputs])
            return op.Concat(*inputs, axis=axis)
        if new_axis == 1:
            # Unsqueeze the inputs with concat axis if new_axis is 1
            axis_value = op.Constant(value_int=axis)
            unsqueezed_inputs = []
            for node_input in inputs:
                unsqueezed_input = op.Unsqueeze(
                    node_input, axis_value, _outputs=[f"{node_input.name}_unsqueeze"]
                )
                unsqueezed_inputs.append(unsqueezed_input)
            # Send unsqueezed outputs to Concat
            logger.debug(
                "ConcatFromSequence => Concat %s", [x.name for x in unsqueezed_inputs]
            )
            return op.Concat(*unsqueezed_inputs, axis=axis)
    return None


@register("SplitToSequence")
def split_to_sequence(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
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

    axis = _get_int_attribute(node, "axis", 0)
    if axis is None:
        return None
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

    split_value = _get_numpy_value(split)
    if split_value is None:
        return None
    assert isinstance(split_value, np.ndarray)

    if split_value.ndim == 0:
        # split into chunks all of size 'split' if possible.
        num_outputs = math.ceil(split_dimension_size / split_value.item())
        split_outputs = [f"{output.name}_split_{i}" for i in range(num_outputs)]
        split_values = op.Split(
            input, axis=axis, num_outputs=num_outputs, _outputs=split_outputs
        )
    elif split_value.ndim == 1:
        # split into 'size(split)' chunks
        num_outputs = split_value.size
        split_outputs = [f"{output.name}_split_{i}" for i in range(num_outputs)]
        split_values = op.Split(input, split, axis=axis, _outputs=split_outputs)
    else:
        return None

    # If Split returns a single value, we need to wrap it into a list.
    if isinstance(split_values, ir.Value):
        split_values = [split_values]

    keepdims = _get_int_attribute(node, "keepdims", 1)
    if keepdims is None:
        return None
    if keepdims == 0:
        # squeeze the split dimension if keepdims is 0
        axis_val = op.Constant(value_ints=[axis], _outputs=[f"{output.name}_axis"])
        squeezed_values = []
        for i in range(num_outputs):
            squeezed = op.Squeeze(
                split_values[i], axis_val, _outputs=[f"{split_outputs[i]}_squeeze"]
            )
            squeezed_values.append(squeezed)
        split_values = squeezed_values

    logger.debug("SplitToSequence => Split + SequenceConstruct")

    if isinstance(split_values, ir.Value):
        split_values = [split_values]
    return op.SequenceConstruct(*split_values)


@register("SequenceAt")
def sequence_at(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    input = node.inputs[0]
    position = node.inputs[1]
    output = node.outputs[0]
    if input is not None and position is not None:
        input_vals = state.get_sym_value(input)
        position_val = _get_numpy_value(position)
        if isinstance(input_vals, list) and position_val is not None:
            if position_val.size != 1:
                return None
            position_val = position_val.item()
            try:
                result = input_vals[position_val]  # type: ignore[index]
            except IndexError:
                return None
            state.set_sym_value(output, result)
            logger.debug("SequenceAt %s => %s", input.name, result.name)
            return op.Identity(result)
    return None


def _merge_shapes(shape1: ir.Shape | None, shape2: ir.Shape | None) -> ir.Shape | None:
    def merge_dims(dim1, dim2):
        if dim1 == dim2:
            return dim1
        if not isinstance(dim1, ir.SymbolicDim):
            return dim1  # Prefer int value over symbolic dim
        if not isinstance(dim2, ir.SymbolicDim):
            return dim2
        if dim1.value is None:
            return dim2
        return dim1

    if shape1 is None:
        return shape2
    if shape2 is None:
        return shape1
    if len(shape1) != len(shape2):
        raise ValueError("Shapes must have the same rank.")
    return ir.Shape([merge_dims(dim1, dim2) for dim1, dim2 in zip(shape1, shape2)])


class FoldConstantsPass(ir.passes.InPlacePass):
    def __init__(
        self,
        *,
        shape_inference: bool,
        input_size_limit: int,
        output_size_limit: int,
    ) -> None:
        self._shape_inference = shape_inference
        self._input_size_limit = input_size_limit
        self._output_size_limit = output_size_limit
        self.opset_imports: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.sizes: dict[str, int] = {}
        self.modified: bool = False
        self._state = OptimizerState()
        self._reset()

    def _reset(self) -> None:
        """Reset internal states for a new run."""
        self.counts = {}
        self.sizes = {}
        self.modified = False
        self._state = OptimizerState()

    def _do_inference(self, node: ir.Node) -> None:
        output_types = {}

        # TODO: handle optional inputs
        def get_constant_value(x: ir.Value) -> onnx.TensorProto | None:
            value = _get_numpy_value(x, size_limit=20)
            if value is not None:
                assert x.const_value is not None
                return ir.serde.serialize_tensor(x.const_value)
            return None

        def get_type(value: ir.Value) -> onnx.TypeProto | None:
            if value.type is not None:
                type_proto = ir.serde.serialize_type(value.type)
                if value.shape is not None:
                    ir.serde.serialize_shape_into(type_proto, value.shape)
                return type_proto
            return None

        input_types = {x.name: get_type(x) for x in node.inputs if x is not None}
        input_data = {x.name: get_constant_value(x) for x in node.inputs if x is not None}
        input_data = {k: v for k, v in input_data.items() if v is not None}
        if any(t is None for t in input_types.values()):
            logger.debug(
                "Skipping shape inference for node %s due to missing input type.",
                node.name,
            )
        else:
            # TODO: pass in constant values, ir_version
            try:
                schema = onnx.defs.get_schema(
                    node.op_type, self.opset_imports[node.domain], node.domain
                )
                output_types = onnx.shape_inference.infer_node_outputs(
                    schema,
                    ir.serde.serialize_node(node),
                    input_types,  # type: ignore[arg-type]
                    input_data,  # type: ignore[arg-type]
                )
                for output in node.outputs:
                    if output.name in output_types:
                        inferred_type = output_types[output.name]
                        # TODO: merge types, check for conflicts
                        inferred_shape = ir.serde.deserialize_type_proto_for_shape(
                            inferred_type
                        )
                        output.shape = _merge_shapes(output.shape, inferred_shape)
                        output.type = ir.serde.deserialize_type_proto_for_type(inferred_type)
            except Exception as e:
                logger.debug(
                    "Skipping shape inference for node %s due to exception: %s",
                    node.name,
                    e,
                )

    def new_constant(self, node: ir.Node, value) -> ir.Node | None:
        irvalue = node.outputs[0]
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

        tensor = ir.tensor(value)
        tensor.name = irvalue.name
        irvalue.const_value = tensor

        if value.nbytes > self._output_size_limit:
            # Handle examples like Transpose(weight) to be folded even if the size is large,
            # as long as weight has no other uses. This won't increase model size.
            removed_input_size = 0
            for input in node.inputs:
                if (input is not None) and (len(input.uses()) == 1):
                    array = _get_numpy_value(input)
                    if array is not None:
                        removed_input_size += array.nbytes
            increased_size = value.nbytes - removed_input_size
            if increased_size > 0:
                logger.info(
                    "Skip storing constant folded nvalue %s due to large size %s.",
                    irvalue.name,
                    value.nbytes,
                )
                return None

        logger.debug(
            "New constant for value %s dtype: %s shape: %s",
            irvalue.name,
            value.dtype,
            value.shape,
        )

        attributes = ir.convenience.convert_attributes({"value": tensor})
        node = ir.Node("", "Constant", inputs=[], attributes=attributes, num_outputs=1)
        return node

    def process_node(self, node: ir.Node) -> Replacement | None:
        for i, value in enumerate(node.inputs):
            sym_value = self._state.get_sym_value(value)
            if isinstance(sym_value, ir.Value):
                logger.debug(
                    "Node [%s]: Replacing input %s with %s",
                    node.name,
                    value.name,  # type: ignore[union-attr]
                    sym_value.name,
                )
                node.replace_input_with(i, sym_value)
                self.modified = True
                # TODO(rama): consider merging type/other info from both values

        # Do incremental shape inference
        if self._shape_inference and not is_control_flow_op(node):
            self._do_inference(node)

        if node.domain not in self.opset_imports:
            return None
        version = self.opset_imports[node.domain]
        op_optimizers = registry.lookup_evaluators(node.domain, node.op_type, version)
        for optimizer in op_optimizers:
            assert optimizer
            context = RewriterContext()
            output = optimizer(node, context, self._state)
            if output is not None:
                if isinstance(output, Replacement):
                    return output
                if isinstance(output, ir.Value):
                    output = [output]
                return Replacement(output, context.nodes)

        if is_control_flow_op(node) or is_non_deterministic_op(node):
            return None

        if is_onnx_op(node, "Constant"):
            _process_constant_node(node)
            return None

        input_values = [_get_numpy_value(x) for x in node.inputs]
        if any(x is None for x in input_values):
            return None

        if any(self._state.is_initializer_input(x) for x in node.inputs):  # type: ignore[arg-type]
            return None

        if any(input.nbytes > self._input_size_limit for input in input_values):  # type: ignore[union-attr]
            if logger.isEnabledFor(logging.DEBUG):
                input_sizes = [input.size for input in input_values]  # type: ignore[union-attr]
                logger.debug(
                    "Skipping constant folding for op %s due to large input size: %s",
                    node.op_type,
                    input_sizes,
                )
            return None

        # Filter out bfloat16 cases?
        def convert(av):
            if av.type == ir.AttributeType.TENSOR:
                return ir.serde.serialize_tensor(av.value)
            return av.value

        attr_values = {name: convert(attr) for name, attr in node.attributes.items()}
        outputs = _reference_evaluator.evaluate(
            node.domain, node.op_type, version, *input_values, **attr_values
        )

        if outputs is None:
            return None
        if len(node.outputs) == 1 and not isinstance(outputs, (tuple, list)):
            replacement = self.new_constant(node, outputs)
            if is_onnx_op(node, "ConstantOfShape") or replacement is None:
                return None
            return Replacement(replacement.outputs, [replacement])
        else:
            logger.warning(
                "Skipping constant folding for op %s with multiple outputs.", node.op_type
            )
        return None

    def replace_node(self, node: ir.Node, replacement, root: ir.Graph | ir.Function) -> None:
        logger.debug("Replacing node: %s::%s %s", node.domain, node.op_type, node.name)

        ir.convenience.replace_nodes_and_values(
            root, node, [node], replacement.new_nodes, node.outputs, replacement.new_outputs
        )

        self.modified = True

        # TODO: what about new opset_imports?
        # TODO: track statistics about replaced nodes and sizes of new constants

    def visit_attribute(self, attr: ir.Attr) -> None:
        if attr.is_ref():
            return
        if attr.type == ir.AttributeType.GRAPH:
            self.visit_graph(attr.as_graph())
        elif attr.type == ir.AttributeType.GRAPHS:
            for graph in attr.as_graphs():
                self.visit_graph(graph)

    def visit_node(self, node: ir.Node, root: ir.Graph | ir.Function) -> None:
        replacement = self.process_node(node)
        if replacement is None:
            # No change. Process attributes.
            for attr in node.attributes.values():
                self.visit_attribute(attr)
            return
        else:
            self.replace_node(node, replacement, root)

    def visit_graph(self, graph: ir.Graph) -> None:
        # Track inputs that have a const_value (which is really a default-value, and should not
        # be used for constant-folding).
        self._state.push_initializer_inputs()
        for input in graph.inputs:
            if input.const_value is not None:
                self._state.add_initializer_input(input)

        for node in graph:
            self.visit_node(node, graph)

        # Replace outputs if output nodes can be folded. This are typically outputs from
        # Identity nodes
        for i, output in enumerate(graph.outputs):
            if output is None:
                continue
            sym_value = self._state.get_sym_value(output)
            if not isinstance(sym_value, ir.Value):
                # An output must be a Value
                continue
            if not _sym_value_can_replace_graph_output(graph, sym_value, output):
                continue
            # Rename sym_value to match the output name
            sym_value.name = output.name
            graph.outputs[i] = sym_value
            self.modified = True

        self._state.pop_initializer_inputs()

    def visit_function(self, function: ir.Function) -> None:
        for node in function:
            self.visit_node(node, function)

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        self._reset()
        self.opset_imports = model.opset_imports
        self.visit_graph(model.graph)
        for function in model.functions.values():
            # TODO(rama): Should we specialize functions?
            self.visit_function(function)
        return FoldConstantsResult(model, self.modified, self._state.symbolic_value_map)


def _sym_value_can_replace_graph_output(
    graph: ir.Graph, sym_value: ir.Value, output: ir.Value
) -> bool:
    if (producer := sym_value.producer()) is None:
        # If the sym_value has no producer, it is some graph's input
        # ONNX does not allow a graph input to be a graph output
        return False
    if producer.graph is not graph:
        # The sym_value must be produced by a node in the graph to be an output of this graph
        return False
    if sym_value.is_graph_output():
        # If the sym_value is already an output of a graph, we cannot rename it
        # to this output name. Otherwise the graph output represented by sym_value
        # will lose its name.
        return False
    return True


@dataclasses.dataclass
class FoldConstantsResult(ir.passes.PassResult):
    symbolic_value_map: dict[ir.Value, SymbolicValue]

    # Add conversion to bool for backward compatibility. The previously returned value
    # for the fold_constants method was a boolean indicating whether the model was modified.
    def __bool__(self) -> bool:
        return self.modified


def fold_constants(
    model: ir.Model,
    *,
    onnx_shape_inference: bool = False,
    input_size_limit: int = DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT,
    output_size_limit: int = DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT,
) -> FoldConstantsResult:
    """
    Applies constant folding optimization to the model.

    Args:
        model: The ONNX model to optimize.
        onnx_shape_inference: Whether to enable ONNX shape inference during
            constant folding. Defaults to False.
        input_size_limit: The maximum size (in bytes) of input tensors
            that can be considered for constant folding. Defaults to
            `DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT`.
        output_size_limit: The maximum size (in bytes) of output tensors
            that can be stored after constant folding. Defaults to
            `DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT`.

    Returns:
        An instance of `FoldConstantsResult`.

    """
    folder_pass = FoldConstantsPass(
        shape_inference=onnx_shape_inference,
        input_size_limit=input_size_limit,
        output_size_limit=output_size_limit,
    )
    return folder_pass(model)  # type: ignore[return-value]
