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
import onnxscript.ir._convenience as _convenience
import onnxscript.rewriter.pattern as orp
import onnxscript.utils.utils as utils

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
    if node.op_type != "Constant" or node.domain not in {"", "ai.onnx"}:
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
        return evaluator(*args, **kwargs)


_reference_evaluator = ReferenceEvaluator()


@dataclasses.dataclass
class Replacement:
    """A replacement for a node in the graph."""

    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]


class OptimizerState:
    def __init__(self):
        self._sym_value_map: dict[ir.Value, Any] = {}

    def get_sym_value(self, value: ir.Value | None) -> Any:
        if value is None:
            return None
        return self._sym_value_map.get(value)

    def set_sym_value(self, value: ir.Value, sym_value: Any) -> None:
        self._sym_value_map[value] = sym_value


# The "partial evaluators" below are non-standard evaluators. They are used to perform
# partial evaluation and/or static program analysis (abstract interpretation).

# A partial-evaluator function takes a node, a RewriterContext, OptimizerState and returns
# a Replacement for the node or None (if no replacement is needed). It may also return just
# the ir.Value or ir.Values to replace the output values of the node, when the new nodes
# can be inferred from the RewriterContext used to build the new nodes.

ReturnValue = Union[Replacement, Sequence[ir.Value], ir.Value, None]
PartialEvaluatorFunction = Callable[
    [ir.Node, orp.RewriterContext, OptimizerState], ReturnValue
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


def _get_numpy_value(val: ir.Value | None) -> np.ndarray | None:
    if val is None:
        return None
    const_value = val.const_value
    if const_value is not None:
        try:
            return const_value.numpy()
        except FileNotFoundError:
            # External data is not available.
            return None
    return None


def _get_bool_value(val: ir.Value | None) -> bool | None:
    if val is None:
        return None
    value = _get_numpy_value(val)
    if value is None:
        return None
    # TODO: cleanup following checks, which seem redundant. But need to also ensure
    # the invariant when setting the value (and also use clearly defined representation
    # types in evaluators, such a reference-evaluator).
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray) and value.size == 1 and value.dtype == bool:
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


# TODO(rama): The following should not be necessary. Generic incremental shape-inference
# should handle this. This essentially implements type/shape-inference for Cast op.
@register("Cast")
def cast(node: ir.Node, op, state: OptimizerState) -> ReturnValue:
    input = _get_input(node, 0)
    output = _get_output(node, 0)
    if input is not None and output is not None:
        input_shape = input.shape
        if input_shape is not None:
            output.shape = input_shape.copy()
    if output is not None:
        output_dtype = _get_int_attribute(node, "to", None)
        if output_dtype is not None:
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
        graph: ir.Graph = graph_attr.value
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


class ConstantFolder:
    opset_imports: dict[str, int]

    def __init__(
        self,
        *,
        external_data_folder: str,
        shape_inference: bool,
        input_size_limit: int,
        output_size_limit: int,
    ) -> None:
        self._external_data_folder = external_data_folder
        self._shape_inference = shape_inference
        self._input_size_limit = input_size_limit
        self._output_size_limit = output_size_limit
        self._init()

    def _init(self) -> None:
        self.counts: dict[str, int] = {}
        self.sizes: dict[str, int] = {}
        self.modified = False
        self._state = OptimizerState()

    def _do_inference(self, node: ir.Node) -> None:
        output_types = {}

        # TODO: handle optional inputs
        def get_constant_value(x: ir.Value) -> onnx.TensorProto | None:
            value = _get_numpy_value(x)
            if isinstance(value, np.ndarray) and value.size < 20:
                return onnx.numpy_helper.from_array(value, x.name)
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
                        output.shape = ir.serde.deserialize_type_proto_for_shape(inferred_type)
                        output.type = ir.serde.deserialize_type_proto_for_type(inferred_type)
            except Exception as e:
                logger.debug(
                    "Skipping shape inference for node %s due to exception: %s",
                    node.name,
                    e,
                )

    def new_constant(self, irvalue: ir.Value, value):
        # TODO(rama): Why do we need the conversion below?
        if isinstance(value, (int, float, np.ScalarType)):
            value = np.array(value)

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

        irvalue.const_value = _convenience.tensor(value)

        if value.nbytes > self._output_size_limit:
            logger.info(
                "Skip storing constant folded nvalue %s due to large size %s.",
                irvalue.name,
                value.nbytes,
            )
            return None

        tensor = onnx.numpy_helper.from_array(value, irvalue.name)

        logger.debug(
            "New constant for value %s dtype: %s shape: %s",
            irvalue.name,
            value.dtype,
            value.shape,
        )

        attributes = _convenience.convert_attributes({"value": tensor})
        node = ir.Node("", "Constant", inputs=[], attributes=attributes, num_outputs=1)
        return node

    def process_node(self, node: ir.Node):
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
            context = orp.RewriterContext()
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

        if any(input.size > self._input_size_limit for input in input_values):  # type: ignore[union-attr]
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
            replacement = self.new_constant(node.outputs[0], outputs)
            if is_onnx_op(node, "ConstantOfShape") or replacement is None:
                return None
            return Replacement(replacement.outputs, [replacement])
        else:
            logger.warning(
                "Skipping constant folding for op %s with multiple outputs.", node.op_type
            )
        return None

    def replace_node(self, node: ir.Node, replacement, root: ir.Graph | ir.Function):
        logger.debug("Replacing node: %s::%s %s", node.domain, node.op_type, node.name)

        _convenience.replace_nodes_and_values(
            root, node, [node], replacement.new_nodes, node.outputs, replacement.new_outputs
        )

        # TODO: what about new opset_imports?
        # TODO: track statistics about replaced nodes and sizes of new constants

    def visit_attribute(self, attr: ir.Attr | ir.RefAttr) -> None:
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.GRAPH:
                self.visit_graph(attr.value)  # type: ignore[arg-type]
            elif attr.type == ir.AttributeType.GRAPHS:
                for graph in attr.value:
                    self.visit_graph(graph)  # type: ignore[arg-type]

    def visit_node(self, node: ir.Node, root: ir.Graph | ir.Function):
        replacement = self.process_node(node)
        if replacement is None:
            # No change. Process attributes.
            for attr in node.attributes.values():
                self.visit_attribute(attr)
            return None
        else:
            self.replace_node(node, replacement, root)

    def visit_graph(self, graph: ir.Graph) -> None:
        for node in graph:
            self.visit_node(node, graph)

    def visit_function(self, function: ir.Function) -> None:
        for node in function:
            self.visit_node(node, function)

    def visit_model(self, model: ir.Model) -> None:
        self._init()
        self.opset_imports = model.opset_imports
        self.visit_graph(model.graph)
        for function in model.functions.values():
            # TODO(rama): Should we specialize functions?
            self.visit_function(function)


def fold_constants(
    model: ir.Model,
    external_data_folder: str = "",
    *,
    onnx_shape_inference: bool = False,
    input_size_limit: int = DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT,
    output_size_limit: int = DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT,
) -> bool:
    """
    Applies constant folding optimization to the model.
    Returns true iff the model was modified.
    """
    folder = ConstantFolder(
        external_data_folder=external_data_folder,
        shape_inference=onnx_shape_inference,
        input_size_limit=input_size_limit,
        output_size_limit=output_size_limit,
    )
    folder.visit_model(model)
    for op in folder.counts:
        logger.info(
            "Constant-folded '%s' %s times, with %s size.",
            op,
            folder.counts[op],
            folder.sizes[op],
        )
    return folder.modified
