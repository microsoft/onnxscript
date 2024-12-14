# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import abc
import contextlib
import dataclasses
import inspect
import itertools
import math
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    MutableSequence,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import onnxscript.optimizer
from onnxscript import ir
from onnxscript.ir import _convenience, _tape

T = TypeVar("T")


class Pattern(Protocol[T]):  # type: ignore[misc]
    """This is essentially a Predicate[T], that is, a Callable[[T], bool] bound to the name "matches"."""

    def matches(self, item: T) -> bool: ...


class StringPattern(abc.ABC, Pattern[str]):
    """Abstract base class for string patterns."""

    @abc.abstractmethod
    def matches(self, item: str) -> bool:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass


class StringConstantPattern(StringPattern):
    """Matches strings with given value."""

    def __init__(self, value: str):
        self._value = value

    def matches(self, item: str) -> bool:
        return item == self._value

    def __str__(self) -> str:
        return self._value

    def value(self) -> str:
        return self._value


class PrefixPattern(StringPattern):
    """Matches strings with a given prefix."""

    def __init__(self, value: str) -> None:
        self._value = value

    def matches(self, value: str) -> bool:
        return value.startswith(self._value)

    def __str__(self) -> str:
        return f"{self._value}*"


class AttrPattern(Pattern[Union[ir.Attr, ir.RefAttr]]):
    """Base class for an attribute pattern. Matches any attribute value by default."""

    def __init__(self, name: str | None):
        self._name = name

    @property
    def name(self) -> str | None:
        return self._name

    def matches(self, attr: ir.Attr | ir.RefAttr) -> bool:
        return True

    def __str__(self) -> str:
        return self._name if self._name is not None else "anonymous:" + str(id(self))


# TODO: Support tensors. Align with usage elsewhere.
SupportedAttrTypes = Union[
    int,
    float,
    str,
    Sequence[int],
    Sequence[float],
    Sequence[str],
]


class AttrConstantPattern(AttrPattern):
    """Matches attributes with given value.

    Uses standard equality for matching. For list-valued attributes, the order of elements matters.
    If order is immaterial, we need to define a separate pattern for that.
    """

    def __init__(self, value: SupportedAttrTypes):
        super().__init__(None)
        self._value = value

    def matches(self, attr: ir.Attr | ir.RefAttr) -> bool:
        return isinstance(attr, ir.Attr) and attr.value == self._value

    def __str__(self) -> str:
        return str(self._value)


def _to_attr_pattern(value: AttrPattern | ValuePattern | SupportedAttrTypes) -> AttrPattern:
    """Represents promotion of values allowed as keyword-arguments in a pattern-builder call to an AttrPattern."""
    if isinstance(value, AttrPattern):
        return value
    if type(value) is ValuePattern:
        # This is a hack. Currently, when we create pattern-variables, we create them as ValuePattern,
        # and change them to AttrPattern if/when used in an attribute context. We could use type
        # annotations to distinguish between ValuePattern and AttrPattern, but forces users to
        # use these type annotations.
        # TODO: check for misuse at rule-creation time. (Currently will be caught by matcher at match-time.)
        return AttrPattern(value.name)
    if isinstance(value, (int, float, str)):
        return AttrConstantPattern(value)
    if isinstance(value, Sequence):
        if all(isinstance(i, (int, float)) for i in value):
            return AttrConstantPattern(value)
        if all(isinstance(i, str) for i in value):
            return AttrConstantPattern(value)
        raise ValueError("Only lists of int/float/str can be used as an AttrPattern")
    raise TypeError(f"Cannot convert {type(value)} to AttrPattern")


class OpsetPatternBuilder:
    """Represents an opset pattern and a pattern builder.

    (i) It is used to create a NodePattern (via OpPatternBuilder).
    Example usage:
    ::

        z = op.Matmul(x, y)

    Here, `op` is an instance of OpsetPatternBuilder and `op.Matmul` is an instance
    of OpPatternBuilder, and  `op.Matmul(x, y)` is an instance of NodePattern.

    (ii) It contains a domain pattern matched against the actual opset domain used in the
    input model.
    """

    def __init__(self, domain: StringPattern | str, record: bool = False) -> None:
        if isinstance(domain, str):
            domain = StringConstantPattern(domain)
        self._domain_pattern = domain
        if record:
            self._nodes: list[NodePattern] | None = []
        else:
            self._nodes = None

    def domain_pattern(self) -> StringPattern:
        return self._domain_pattern

    def __getattr__(self, op_name: str) -> OpPatternBuilder:
        return OpPatternBuilder(self, op_name)

    def submodule(self, name: str) -> OpPatternBuilder:
        """This method is used to match against submodule ops with prefix."""
        return OpPatternBuilder(self, PrefixPattern(name))

    def __str__(self) -> str:
        return str(self._domain_pattern)

    def add_node(self, node: NodePattern) -> None:
        if self._nodes is not None:
            self._nodes.append(node)

    def nodes(self) -> Sequence[NodePattern]:
        if self._nodes is None:
            raise ValueError("Nodes were not recorded.")
        return self._nodes


onnxop = OpsetPatternBuilder("")

torch_module_op = OpsetPatternBuilder(PrefixPattern("pkg.torch"))


class OpPatternBuilder:
    """A utility class to build a NodePattern.

    It is used primarily to create a NodePattern.
    Example usage:
    ::

        z = op.Matmul(x, y)

    Here, `op` is an instance of OpsetPatternBuilder and `op.Matmul` is an instance
    of OpPatternBuilder, and  `op.Matmul(x, y)` is an instance of NodePattern.

    """

    def __init__(
        self,
        pattern_builder: OpsetPatternBuilder,
        op_name: str | Pattern[str],
    ) -> None:
        self.pattern_builder = pattern_builder
        self.op_name = op_name

    def __call__(
        self,
        *args,
        _domain: str | None = None,
        _version: int | None = None,
        _outputs: int | list[str | None] = 1,
        _allow_other_attributes: bool | None = None,
        _allow_other_inputs: bool | None = None,
        **kwargs,
    ):
        if _version is not None:
            raise ValueError(
                "The pattern builder does not support '_version' keyword argument. "
                "Version restrictions should be handled by rewrite rules."
            )
        if _domain is None:
            opset_pattern = self.pattern_builder.domain_pattern()
        elif isinstance(_domain, str):
            opset_pattern = StringConstantPattern(_domain)
        else:
            # TODO(rama): allow OpsetPatternBuilder as _domain.
            raise TypeError("_domain must be a string.")

        if isinstance(_outputs, int):
            _outputs = [None for _ in range(_outputs)]
        elif not isinstance(_outputs, Sequence) or not all(
            isinstance(x, (str, type(None))) for x in _outputs
        ):
            raise ValueError("_outputs must be an int or a list[str|None].")
        inputs = [_to_value_pattern(x) for x in args]
        attributes = {name: _to_attr_pattern(value) for (name, value) in kwargs.items()}
        node_pattern = NodePattern(
            opset_pattern,
            self.op_name,
            inputs,
            attributes,
            _outputs,
            allow_other_attributes=_allow_other_attributes,
            allow_other_inputs=_allow_other_inputs,
        )
        self.pattern_builder.add_node(node_pattern)
        output_values = node_pattern.outputs
        # Unpack outputs if there is only one output, the common case.
        if len(output_values) == 1:
            return output_values[0]
        else:
            return output_values


def _to_value_pattern(
    x: ValuePattern | int | float | None,
) -> ValuePattern | None:
    """Promotes an input-value used to construct a NodePattern to a ValuePattern.

    Example usage:
    ::
        x = op.MatMul(a, b)
        z = op.Add(x, 0)

    In this example, `a, `b`, and `x` are ValuePatterns used to construct a NodePattern.
    `0` is a constant (int) value, and is automatically promoted to a ValuePattern.

    Note that this is a shorthand for creating a Constant pattern. The user can more
    explicitly write this as:
    ::
        z = op.Add(x, op.Constant(0))
    """
    if x is None or isinstance(x, ValuePattern):
        return x
    if isinstance(x, (int, float)):
        return Constant(x)
    if isinstance(x, Sequence):
        if all(isinstance(i, (int, float)) for i in x):
            return Constant(x)
        raise ValueError("Only lists of int/float can be used as a ValuePattern")

    raise TypeError(f"Cannot convert {type(x)} to ValuePattern")


class MatchResult:
    """Represents the result of a match operation.

    A match can either succeed or fail.
    If it succeeds, it returns a list of nodes that matched the pattern
    and a set of bindings for the variables in the pattern.

    Example:
    ::
        def pattern(x, shape1, shape2):
            t1 = op.Reshape(x, shape1)
            t2 = op.Reshape(t1, shape2)
            return t2
    The above pattern matches a sequence of two Reshape ops.
    The matched_nodes will contain the two Reshape ops, and the bindings will
    contain the values that are bound to the variables `x`, `shape1`, and `shape2`.
    """

    def __init__(self) -> None:
        self._success: bool = True
        # For a successful match, _matched_nodes is a list of values that matched the pattern.
        # These include the internal nodes of the pattern that were matched, but not
        # the leaves (sub-trees) that match against the variables in the pattern.
        # These represent the values that will be replaced by the replacement pattern.
        self._matched_nodes: MutableSequence[ir.Node] = []
        # For a successful match, bindings is a dictionary of mapping pattern-variable-names
        # to values.
        self.bindings: dict[str, Any] = {}
        self.outputs: list[ir.Value] = []
        # For a failed match, _reason is a string that describes the reason for the failure.
        self._reason: str = ""

    def __bool__(self):
        return self._success

    def fail(self, reason: str = "") -> MatchResult:
        self._success = False
        self._reason = reason
        return self

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def nodes(self) -> MutableSequence[ir.Node]:
        return self._matched_nodes

    def bind(self, var: str, value: Any) -> bool:
        """Binds a pattern variable name to a value from the matched IR.

        Returns True if the binding is successful, False otherwise (when the binding is inconsistent).
        """
        if var in self.bindings:
            # TODO(rama): Use appropriate equality-check here.
            if self.bindings[var] == value:
                return True
            self._success = False
            return False
        self.bindings[var] = value
        return True

    def extend(self, other: MatchResult | bool):
        if not self._success:
            return
        if not other:
            self._success = False
            return
        if isinstance(other, bool):
            return
        for var, val in other.bindings.items():
            if var in self.bindings:
                # TODO: handle attribute var bindings
                if self.bindings[var] != val:
                    self._success = False
                    return
            else:
                self.bindings[var] = val
        assert self._matched_nodes is not None, "_matched_nodes should not be None."
        self._matched_nodes.extend(other._matched_nodes)  # type: ignore[attr-defined]


_pattern_builder: OpsetPatternBuilder = onnxop


@contextlib.contextmanager
def pattern_builder(builder: OpsetPatternBuilder):
    global _pattern_builder
    prev_builder = _pattern_builder
    _pattern_builder = builder
    yield
    _pattern_builder = prev_builder


class ValuePattern:
    """Base class for all patterns that match against IR values.

    This is used primarily to provide operator overloadings for arithmetic
    operations, so that we can write patterns like `x + 1` and `1 + x`.
    """

    def __init__(self, name: str | None) -> None:
        self._name = name
        # Note: uses will be computed only when the full graph-pattern is constructed.
        self._uses: list[tuple[NodePattern, int]] = []

    def clone(self, node_map: dict[NodePattern, NodePattern]) -> ValuePattern:
        del node_map
        return ValuePattern(self._name)

    @property
    def name(self) -> str | None:
        return self._name

    def producer(self) -> None | NodePattern:
        return None

    def uses(self) -> Sequence[tuple[NodePattern, int]]:
        return self._uses

    def append_use(self, node: NodePattern, index: int):
        self._uses.append((node, index))

    def __repr__(self) -> str:
        return f"ValuePattern({self._name!r})"

    def __add__(self, other):
        return _pattern_builder.Add(self, other)

    def __radd__(self, other):
        return _pattern_builder.Add(other, self)

    def __sub__(self, other):
        return _pattern_builder.Sub(self, other)

    def __rsub__(self, other):
        return _pattern_builder.Sub(other, self)

    def __mul__(self, other):
        return _pattern_builder.Mul(self, other)

    def __rmul__(self, other):
        return _pattern_builder.Mul(other, self)

    def __truediv__(self, other):
        return _pattern_builder.Div(self, other)

    def __rtruediv__(self, other):
        return _pattern_builder.Div(other, self)

    def __pow__(self, other):
        return _pattern_builder.Pow(self, other)

    def __str__(self) -> str:
        return self._name if self._name is not None else "anonymous:" + str(id(self))


class NodePattern:
    """Represents a pattern that matches against a Node.

    This differs from a NodeOutputPattern in that it matches against a node (which
    may produce 1 or more outputs), whereas a NodeOutputPattern matches against
    a specific output of a node.

    Args:
        domain: pattern to match against the domain of the node.
        op: pattern or string constant to match against the op_type of the node.
        inputs: sequence of ValuePatterns (or constants) to match against the inputs of the node.
        attributes: dictionary of attribute patterns to match against the attributes of the node.
        outputs: specifies pattern-variable-name for outputs (or None)
        allow_other_attributes: specifies whether other attributes (not mentioned in `attributes`)
          are allowed in the node.
    """

    def __init__(
        self,
        domain: StringPattern,
        op: str | Pattern[str],
        inputs: Sequence[int | float | ValuePattern | None],
        attributes: dict[str, AttrPattern],
        outputs: Sequence[str | None],
        *,
        allow_other_attributes: bool | None,
        allow_other_inputs: bool | None,
    ):
        if allow_other_attributes is None:
            # Default behavior: allow other unmatched attributes in the node.
            allow_other_attributes = True
        if allow_other_inputs is None:
            # TODO(rama): Should we default to True? For now, we preserve the current behavior.
            allow_other_inputs = False
        self.domain = domain
        self.op = StringConstantPattern(op) if isinstance(op, str) else op
        self.inputs = [_to_value_pattern(x) for x in inputs]
        self.attributes = attributes
        self.allow_other_attributes = allow_other_attributes
        self.allow_other_inputs = allow_other_inputs
        # In the common case, domain and op are constants, which can be used to optimize matching.
        if isinstance(op, str) and isinstance(domain, StringConstantPattern):
            # TODO(rama): support overloaded operators.
            overload = ""
            self._op_identifier: tuple[str, str, str] | None = (
                domain.value(),
                op,
                overload,
            )
        else:
            self._op_identifier = None
        self.outputs = [NodeOutputPattern(self, i, name) for i, name in enumerate(outputs)]

        # Update uses for inputs.
        for index, value in enumerate(self.inputs):
            if value is not None:
                value.append_use(self, index)

    def __str__(self) -> str:
        inputs = ", ".join(str(v) for v in self.inputs)
        outputs = ", ".join(str(v) for v in self.outputs)
        attributes = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
        op = str(self.op)
        domain = str(self.domain)
        qualified_op = f"{domain}.{op}" if domain else op
        inputs_and_attributes = f"{inputs}, {attributes}" if attributes else inputs
        return f"{outputs} = {qualified_op} ({inputs_and_attributes})"

    def op_identifier(self) -> Tuple[str, str, str] | None:
        return self._op_identifier

    @property
    def op_type(self) -> str:
        return str(self.op)

    def matches(self, node: ir.Node, match: MatchResult) -> MatchResult:
        """Matches the pattern represented by self against a node.

        This is purely a local node-level match, and does not consider the subgraph rooted at the node.
        We check the domain, op_type, and attributes of the node, but not the inputs.
        """
        # TODO(rama): Ensure we handle "" and "onnx.ai" correctly.
        if not self.domain.matches(node.domain):
            return match.fail(f"Domain mismatch: expected {self.domain}, got {node.domain}.")
        if not self.op.matches(node.op_type):
            return match.fail(f"OpType mismatch: expected {self.op}, got {node.op_type}.")

        for name, attr_pattern in self.attributes.items():
            attr_value = node.attributes.get(name)
            if attr_value is None:
                return match.fail(f"Attribute {name} not found in node.")
            if not attr_pattern.matches(attr_value):
                return match.fail(
                    f"Attribute {name} mismatch: expected {attr_pattern}, got {attr_value}."
                )
            if attr_pattern.name is not None:
                if not match.bind(attr_pattern.name, attr_value):
                    return match

        if not self.allow_other_attributes:
            for name in node.attributes:
                # TODO: Support matching default nodes for attributes.
                if name not in self.attributes:
                    return match.fail(f"Attribute {name} not expected in node.")

        return match

    def clone(self, node_map: dict[NodePattern, NodePattern], swap: bool) -> NodePattern:
        inputs = [(v.clone(node_map) if v is not None else None) for v in self.inputs]
        if swap:
            assert (
                len(inputs) == 2
            ), "Internal error: commutative swap applies only to binary ops."
            inputs = [inputs[1], inputs[0]]
        outputs = [value.name for value in self.outputs]
        copied = NodePattern(
            self.domain,
            self.op,
            inputs,
            self.attributes,
            outputs,
            allow_other_attributes=self.allow_other_attributes,
            allow_other_inputs=self.allow_other_inputs,
        )
        node_map[self] = copied
        return copied


class NodeOutputPattern(ValuePattern):
    """Represents a pattern that matches against a specific output of a Node.

    This is the primary pattern used to match against computed values, that
    is values computed using a specific op.
    """

    def __init__(
        self, producer: NodePattern, output_index: int, name: str | None = None
    ) -> None:
        super().__init__(name)
        self._producer = producer
        self._output_index = output_index

    def clone(self, node_map: dict[NodePattern, NodePattern]) -> NodeOutputPattern:
        return node_map[self._producer].outputs[self._output_index]
        # return NodeOutputPattern(node_map[self._producer], self._output_index, self._name)

    @property
    def output_index(self) -> int:
        return self._output_index

    def producer(self) -> NodePattern:
        return self._producer


Var = ValuePattern


def _is_pattern_variable(x: Any) -> bool:
    # The derived classes of ValuePattern represent constant patterns and node-output patterns.
    return type(x) is ValuePattern


class Constant(ValuePattern):
    """Represents a pattern that matches against a scalar constant value."""

    def __init__(
        self,
        value: int | float | Sequence[int] | Sequence[float],
        rel_tol: float = 1e-5,
        abs_tol: float = 1e-8,
    ) -> None:
        super().__init__(None)
        self._value = list(value) if isinstance(value, Sequence) else value
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    def clone(self, node_map: dict[NodePattern, NodePattern]) -> Constant:
        del node_map
        return Constant(self._value, self._rel_tol, self._abs_tol)

    @property
    def value(self) -> int | float | list[int] | list[float]:
        return self._value

    def matches(self, value: ir.Value, match: MatchResult) -> MatchResult:
        constant_value = value.const_value
        if constant_value is None:
            return match.fail(f"Value is not a constant, expecting {self.value}.")

        constant_value_numpy = constant_value.numpy()
        if isinstance(self._value, list):
            if constant_value_numpy.shape != (len(self._value),):
                return match.fail(f"Value has mismatching shape, expecting ({self.value},).")
            if not all(
                math.isclose(
                    constant_value_numpy.item(i),
                    self._value[i],
                    rel_tol=self._rel_tol,
                    abs_tol=self._abs_tol,
                )
                for i in range(len(self._value))
            ):
                return match.fail(
                    f"Value mismatch: expected {self._value}, got {constant_value_numpy}."
                )
            return match

        # Scalar constant case:
        # TODO (rama): allow users to specify shape requirement, if desired.
        if constant_value_numpy.size != 1:
            return match.fail(f"Value is not a scalar, expecting {self.value}.")

        if not math.isclose(
            constant_value_numpy.item(),
            self._value,
            rel_tol=self._rel_tol,
            abs_tol=self._abs_tol,
        ):
            match.fail(
                f"Value mismatch: expected {self._value}, got {constant_value_numpy.item()}."
            )

        # Note: If the value is produced by a Constant node, we could include
        # the Constant node in the return_value list. However, we don't do that.
        # Instead, we will rely on DCE to remove the constant node if it is not
        # used elsewhere.
        return match

    def __str__(self) -> str:
        return str(self._value)


def _nodes_in_pattern(outputs: Sequence[ValuePattern]) -> list[NodePattern]:
    """Returns all nodes used in a pattern, given the outputs of the pattern."""
    node_patterns: list[NodePattern] = []

    def visit(value_patterns: Sequence[ValuePattern | None]) -> None:
        for value_pattern in value_patterns:
            if isinstance(value_pattern, NodeOutputPattern):
                node_pattern = value_pattern.producer()
                if node_pattern not in node_patterns:
                    node_patterns.append(node_pattern)
                    visit(node_pattern.inputs)

    visit(outputs)
    node_patterns.reverse()
    return node_patterns


def _add_backward_slice(node: NodePattern, backward_slice: set[NodePattern]) -> None:
    """Adds all nodes in the backward slice of given node to the set `backward_slice`.

    The backward slice of a node is the set of all nodes that are reachable from the node
    in a backward traversal from the given node.
    """
    if node in backward_slice:
        return
    backward_slice.add(node)
    for value_pattern in node.inputs:
        if isinstance(value_pattern, NodeOutputPattern):
            _add_backward_slice(value_pattern.producer(), backward_slice)


class GraphPattern:
    """Represents a pattern that can be matched against a subgraph."""

    def __init__(
        self,
        inputs: Sequence[ValuePattern],
        outputs: Sequence[ValuePattern],
        nodes: Sequence[NodePattern],
    ) -> None:
        self._inputs = inputs
        self._outputs = outputs
        if len(outputs) == 0:
            raise ValueError("GraphPattern must have at least one output")
        self._nodes = nodes  # _nodes_in_pattern(outputs)

        # Determine the output nodes of the pattern. These are a minimal set of nodes
        # whose backward-slices cover the entire pattern.
        output_nodes: set[NodePattern] = set()
        covered: set[NodePattern] = set()
        for value_pattern in outputs:
            if not isinstance(value_pattern, ValuePattern):
                raise TypeError(
                    f"Invalid type {type(value_pattern)} for graph pattern output."
                )
            if isinstance(value_pattern, Constant):
                raise NotImplementedError(
                    "Constant values are not allowed as graph pattern outputs."
                )
            if isinstance(value_pattern, NodeOutputPattern):
                candidate = value_pattern.producer()
                if candidate not in covered:
                    output_nodes.add(candidate)
                    _add_backward_slice(candidate, covered)

        self.output_nodes: list[NodePattern] = list(output_nodes)

    @property
    def output_node(self) -> NodePattern:
        if len(self.output_nodes) != 1:
            raise ValueError("GraphPattern does not have unique output node.")
        return self.output_nodes[0]

    def node(self, index: int) -> NodePattern:
        return self._nodes[index]

    def num_nodes(self) -> int:
        return len(self._nodes)

    def __len__(self) -> int:
        return self.num_nodes()

    @property
    def inputs(self) -> Sequence[ValuePattern]:
        return self._inputs

    @property
    def outputs(self) -> Sequence[ValuePattern]:
        return self._outputs

    def __iter__(self) -> Iterator[NodePattern]:
        return iter(self._nodes)

    def __reversed__(self) -> Iterator[NodePattern]:
        return reversed(self._nodes)

    @property
    def has_single_output_node(self) -> bool:
        return len(self.output_nodes) == 1

    @property
    def num_outputs(self) -> int:
        return len(self._outputs)

    def commute(self) -> Sequence[GraphPattern]:
        def commute_node(node: NodePattern) -> Iterable[bool]:
            if node.op_identifier() == ("", "Add", "") or node.op_identifier() == (
                "",
                "Mul",
                "",
            ):
                # Try with and without swapping inputs.
                return [False, True]
            # No swapping of inputs
            return [False]

        iteration_space = [commute_node(node) for node in self._nodes]

        def copy_graph(swap_list: Iterable[bool]) -> GraphPattern:
            if not any(swap_list):
                # No need to swap inputs of any node
                return self
            # Create a copy of the graph, with swapped inputs for the nodes that need it.
            node_map: dict[NodePattern, NodePattern] = {}
            new_inputs = [v.clone(node_map) for v in self._inputs]
            new_nodes = [
                node.clone(node_map, swap) for node, swap in zip(self._nodes, swap_list)
            ]
            new_outputs = [v.clone(node_map) for v in self._outputs]
            return GraphPattern(new_inputs, new_outputs, new_nodes)

        return [copy_graph(swap_list) for swap_list in itertools.product(*iteration_space)]

    def __str__(self) -> str:
        inputs = ", ".join(str(v) for v in self._inputs)
        outputs = ", ".join(str(v) for v in self._outputs)
        nodes = "\n   ".join(str(n) for n in self._nodes)
        return f"pattern ({inputs}) {{\n   {nodes}\n   return {outputs}\n}}"


def _to_graph_pattern(pattern_constructor: Callable) -> GraphPattern:
    """Convert a pattern-construction function to a GraphPattern.

    A pattern-construction function will return values as below:
    ::
        def pattern(op, x: Var, shape1: Var, shape2: Var):
            ...
            return outputs

    We create a pattern graph by creating pattern-variables for each parameter of the function,
    and calling the function. The returned values are normalized to a list of ValuePatterns,
    which represent the outputs of the pattern graph.

    Args:
        pattern_constructor: Callable

    Returns:
        GraphPattern: A representation of the pattern that can be matched against a subgraph.
    """
    _pattern_vars = inspect.signature(pattern_constructor).parameters
    pattern_inputs = [Var(v) for v in _pattern_vars][1:]  # Skip the first parameter
    builder = OpsetPatternBuilder("", record=True)
    with pattern_builder(builder):
        pattern_outputs = pattern_constructor(builder, *pattern_inputs)
    # TODO(rama): classify inputs as value/attribute vars
    # Returned value could be a single ValuePattern or a list of ValuePatterns.
    # Normalize representation to a list of ValuePatterns.
    if isinstance(pattern_outputs, ValuePattern):
        pattern_outputs = [pattern_outputs]
    return GraphPattern(pattern_inputs, pattern_outputs, builder.nodes())


def _valid_to_replace(
    matched_nodes: Sequence[ir.Node], output_values: Sequence[ir.Value]
) -> bool:
    """Check that values computed by the matched_nodes, except for output_values, are used only by the matched_nodes."""
    # * Must check that all values matched by pattern are used only by pattern,
    # except for the value that is replaced.
    # * Must ensure that replacement subgraph does not use any of the deleted
    # (intermediate) values. (Not necessary for now. Guaranteed.)
    for n in matched_nodes:
        for v in n.outputs:
            if v in output_values:
                continue
            if v.is_graph_output():
                # value is an output-value of the graph/function.
                return False
            for consumer, _ in v.uses():
                if consumer not in matched_nodes:
                    return False
    return True


RewriterContext = _tape.Builder


@dataclasses.dataclass
class ReplacementSubgraph:
    """A subgraph that will replace the matched pattern."""

    match: MatchResult
    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]
    used_opsets: _tape.UsedOpsets


def always_true(*args, **kwargs) -> bool:
    """A condition function that always returns True.

    This is used when no condition function is provided for a rewrite rule.
    """
    return True


class ReplacementPatternFunction:
    """The replacement pattern that will replace the targeted pattern.

    Attributes:
        function (Callable): The replacement function that will be used to replace the matched pattern.
    """

    def __init__(self, function) -> None:
        self._function = function

    def get_replacement(self, match: MatchResult) -> ReplacementSubgraph | None:
        context = RewriterContext()
        new_outputs = self._function(context, **match.bindings)
        if new_outputs is None:
            return None  # Failed to create replacement subgraph
        if not isinstance(new_outputs, Sequence):
            new_outputs = [new_outputs]
        return ReplacementSubgraph(match, new_outputs, context.nodes, context.used_opsets)


def _update_opset_imports(
    graph_or_function: ir.Graph | ir.Function, delta: ReplacementSubgraph
):
    imports = graph_or_function.opset_imports
    for domain, version in delta.used_opsets:
        if domain not in imports:
            # use 1 as default version if not explicitly specified
            imports[domain] = version if version is not None else 1
        elif version is not None and version != imports[domain]:
            raise ValueError(
                f"Multiple versions of opset {domain} used. "
                f"Expected version {imports[domain]}, but got {version}."
            )


class PatternMatcher(abc.ABC):
    def __init__(self, pattern: GraphPattern) -> None:
        self.pattern = pattern

    @abc.abstractmethod
    def match(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        verbose: int = 0,
    ) -> MatchResult:
        """Match the pattern against the subgraph ending at the given node."""

    def __str__(self) -> str:
        return str(self.pattern)


class SimplePatternMatcher(PatternMatcher):
    def __init__(self, pattern: GraphPattern) -> None:
        super().__init__(pattern)

    def fail(self, reason: str) -> bool:
        if self._verbose:
            if self._matched:  # Print only if at least one node successfully matched.
                count = len(self._matched)
                print(f"Match failed after {count} nodes: {reason}")
        self._match.fail(reason)
        return False

    def _match_constant(self, pattern_constant: Constant, value: ir.Value) -> bool:
        """Match a Constant pattern against a value.

        If the constant value is produced by a Constant node, we do not include
        the constant node as part of the matched graph. Thus, it will not be deleted,
        if subgraph replacement happens. But subsequent DCE will remove the constant
        node if it is not used elsewhere.
        """
        constant_value = value.const_value
        if constant_value is None:
            return self.fail(
                f"Value {value.name} is not a constant, expecting {pattern_constant.value}.",
            )

        try:
            constant_value_numpy = constant_value.numpy()
        except FileNotFoundError:
            return self.fail(f"Constant value of {value.name} not available.")

        pattern_constant_value = pattern_constant._value

        if isinstance(pattern_constant_value, list):
            expected_shape = (len(pattern_constant_value),)
            if constant_value_numpy.shape != expected_shape:
                return self.fail(f"Value has mismatching shape, expecting {expected_shape}.")
            if not all(
                math.isclose(
                    constant_value_numpy.item(i),
                    pattern_constant_value[i],
                    rel_tol=pattern_constant._rel_tol,
                    abs_tol=pattern_constant._abs_tol,
                )
                for i in range(len(pattern_constant_value))
            ):
                return self.fail(
                    f"Value mismatch: expected {pattern_constant_value}, got {constant_value_numpy}."
                )
            return True

        # TODO (rama): allow users to specify shape requirement, if desired.
        if constant_value_numpy.size != 1:
            return self.fail(
                f"Value {value.name} is not a scalar, expecting {pattern_constant_value}.",
            )

        if not math.isclose(
            constant_value_numpy.item(),
            pattern_constant_value,
            rel_tol=pattern_constant._rel_tol,
            abs_tol=pattern_constant._abs_tol,
        ):
            return self.fail(
                f"Constant value mismatch: expected {pattern_constant_value}, got {constant_value_numpy.item()}.",
            )

        return True

    def _match_node(self, pattern_node: NodePattern, node: ir.Node) -> bool:
        """Matches a pattern subgraph against subgraph rooted at node."""

        # Graph-matching: we do not allow the same pattern node to be matched against
        # different graph nodes.
        if pattern_node in self._matched:
            if self._matched[pattern_node] is not node:
                return self.fail("Same pattern node is matched against different graph nodes.")
            return True
        match = self._match
        if not pattern_node.matches(node, match):
            return self.fail(match.reason)

        if self._verbose:
            print(f"Matched: {node.op_type}")

        self._matched[pattern_node] = node

        # TODO: Revisit this to handle optional trailing inputs better.
        if pattern_node.allow_other_inputs:
            if len(node.inputs) < len(pattern_node.inputs):
                return self.fail(
                    f"Number of inputs ({len(node.inputs)}) is less than expected ({len(pattern_node.inputs)})"
                )
        else:
            if len(node.inputs) != len(pattern_node.inputs):
                return self.fail(
                    f"Input nums mismatch. {len(node.inputs)} vs {len(pattern_node.inputs)}"
                )

        for arg_value, arg_pattern in zip(node.inputs, pattern_node.inputs):
            # arg_pattern could be a Var, if it's the original arg.
            if arg_pattern is None:
                if arg_value is None:
                    continue
                else:
                    return self.fail("(Optional) input is expected to be None but is not.")
            if not self._match_value(arg_pattern, arg_value):
                return False

        for i, output_value_pattern in enumerate(pattern_node.outputs):
            if not self._bind_value(output_value_pattern, node.outputs[i]):
                return False

        match.nodes.append(node)
        return True

    def _bind_value(self, pattern_value: ValuePattern, value: ir.Value | None) -> bool:
        """Bind a ValuePattern var to ir Value."""
        if pattern_value.name is not None:
            match = self._match
            if pattern_value.name in match.bindings:
                # TODO(rama): Use appropriate equality-check here: future extension possibility.
                if match.bindings[pattern_value.name] == value:
                    return True
                return self.fail(f"Variable {pattern_value.name} is bound to multiple values.")
            match.bindings[pattern_value.name] = value
        return True

    def _match_value(self, pattern_value: ValuePattern, value: ir.Value | None) -> bool:
        """Match an IR value against a ValuePattern instance."""
        if not self._bind_value(pattern_value, value):
            return False

        if isinstance(pattern_value, NodeOutputPattern):
            if value is None:
                return self.fail("Mismatch: Computed node pattern does not match None.")
            return self._match_node_output(pattern_value, value)
        if isinstance(pattern_value, Constant):
            if value is None:
                return self.fail("Mismatch: Constant pattern does not match None.")
            return self._match_constant(pattern_value, value)
        return True

    def _match_node_output(self, pattern_value: NodeOutputPattern, value: ir.Value) -> bool:
        """Match an IR value against a NodeOutputPattern instance."""
        node = value.producer()
        if node is None:
            return self.fail(
                "Mismatch: Computed node pattern does not match uncomputed IR value."
            )
        if value.index() != pattern_value.output_index:
            return self.fail(
                f"Node output index mismatch: expected {pattern_value._output_index}, got {value.index()}."
            )
        return self._match_node(pattern_value.producer(), node)

    def _init_match(self, verbose: int) -> None:
        """Initialize the match state. Invoked before starting a new match."""
        self._verbose = verbose
        self._matched: dict[NodePattern, ir.Node] = {}
        self._match: MatchResult = MatchResult()

    def _get_output_values(self) -> list[ir.Value] | None:
        """Get values bound to the output variables of the pattern."""
        output_values: list[ir.Value] = []
        unbound_values: list[str] = []
        for j, value_pattern in enumerate(self.pattern.outputs):
            if value_pattern.name is not None:
                if value_pattern.name in self._match.bindings:
                    output_values.append(self._match.bindings[value_pattern.name])
                else:
                    unbound_values.append(value_pattern.name)
            elif isinstance(value_pattern, NodeOutputPattern):
                i = value_pattern.output_index
                node = value_pattern.producer()
                if node in self._matched:
                    output_values.append(self._matched[node].outputs[i])
                else:
                    unbound_values.append(f"output_{j}")
            elif isinstance(value_pattern, Constant):
                raise NotImplementedError("Constant values as return-values not supported.")
        if unbound_values:
            self._match.fail(f"Error: Output values not found: {unbound_values}")
            return None
        return output_values

    def _match_single_output_node(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
    ) -> MatchResult:
        del model
        del graph_or_function

        pattern = self.pattern
        match = self._match

        if not pattern.has_single_output_node:
            return match.fail(
                "Internal Error: SimplePatternMatcher should not be used for patterns with multiple output nodes."
            )

        if not self._match_node(pattern.output_node, node):
            return match

        output_values = self._get_output_values()
        if output_values is None:
            return match
        if not _valid_to_replace(match.nodes, output_values):
            return match.fail("Matched nodes have other uses preventing replacement.")

        match.outputs.extend(output_values)
        return match

    def _multi_match(self, candidate: Iterable[ir.Node]) -> MatchResult:
        """Find a match for a pattern with multiple output nodes.

        For a pattern with K output nodes, the input candidate should specify K nodes
        in the graph that will be matched against the pattern output nodes.

        Args:
            candidate: An iterable of nodes that will be matched against the pattern output nodes.
        """
        match = self._match
        for pattern_node, node in zip(self.pattern.output_nodes, candidate):
            if not self._match_node(pattern_node, node):
                return match
        output_values = self._get_output_values()
        if output_values is None:
            return match

        if not _valid_to_replace(match.nodes, output_values):
            return match.fail("Matched nodes have other uses preventing replacement.")

        match.outputs.extend(output_values)
        return match

    def match(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        verbose: int = 0,
    ) -> MatchResult:
        """Match the pattern against the subgraph ending at the given node.

        For patterns with multiple output nodes, the given node is matched
        against the first output node in the pattern. For the remaining
        output nodes in the pattern, we use a brute-force algorithm that
        enumerates all possible combinations of nodes from the graph (with
        a filter based on op-type).

        TODO: Consider omitting parameters model and graph_or_function. With
        the new IR, the graph can be obtained from the node, and the model is
        not used. But this is a shared abstract method of the Matcher interface,
        so other matcher implementation also needs to be updated. More importantly,
        matching in the presence of subgraphs (control-flow) can introduce some
        complications which require careful consideration.
        """

        if self.pattern.has_single_output_node:
            self._init_match(verbose)
            return self._match_single_output_node(model, graph_or_function, node)
        else:
            # Note: This is a potentially expensive algorithm for matching patterns with
            # multiple output nodes. For patterns with N output nodes, we try all possible
            # combinations of N nodes from the graph, and check if they match the pattern.
            # The first node is fixed to the node argument in this method call. We do
            # some simple filtering by restricting the candidates for each remaining
            # output nodes to graph nodes with the same op_type as the corresponding pattern
            # node. For now, this is intended to be a simple, but robust, implementation
            # that can be used for debugging and testing. The GenericPatternMatcher is a
            # more sophisticated implementation, but incomplete.
            pattern_output_nodes = self.pattern.output_nodes
            op_to_nodes: dict[tuple[str, str, str], list[ir.Node]] = {}
            for n in graph_or_function:
                op_to_nodes.setdefault(n.op_identifier(), []).append(n)
            all_nodes = iter(graph_or_function)

            def get_nodes(pattern_node):
                id = pattern_node.op_identifier()
                if id is None:
                    return all_nodes
                return op_to_nodes.get(id, [])

            candidates = [iter([node])] + [get_nodes(pn) for pn in pattern_output_nodes[1:]]
            match = None
            for combination in itertools.product(*candidates):
                self._init_match(verbose)
                match = self._multi_match(combination)
                if match:
                    return match
            if match is None:
                return MatchResult().fail("No match found.")
            return match


class RewriteRule:
    def __init__(
        self,
        target_pattern: GraphPattern | Callable,
        replacement_pattern: ReplacementPatternFunction | Callable,
        condition_function: Callable | None = None,
        matcher: PatternMatcher | Callable[[GraphPattern], PatternMatcher] | None = None,
        verbose: int = 0,
        name: str | None = None,
    ) -> None:
        """Create a rewrite rule.

        Args:
            target_pattern: The GraphPattern that will be matched against the IR.
                If a callable is provided, it will be converted to a GraphPattern.
            replacement_pattern: The ReplacementPatternFunction that will be used to
                replace the matched pattern. If a callable is provided, it will be
                converted to a ReplacementPatternFunction.
            condition_function: The condition function that will be used to check if
                the pattern match found should be rewritten.
            matcher: The pattern matcher that will be used to match the pattern.
                If not provided, a default matcher will be used.
            verbose: The verbosity level of the rule.
            name: An optional name for the pattern that will show up in verbose logging.
        """

        if not isinstance(target_pattern, GraphPattern):
            target_pattern = _to_graph_pattern(target_pattern)
        self._target_pattern = target_pattern

        if not isinstance(replacement_pattern, ReplacementPatternFunction):
            replacement_pattern = ReplacementPatternFunction(replacement_pattern)
        self._replacement_pattern = replacement_pattern
        self._condition_function = condition_function or always_true
        if isinstance(matcher, PatternMatcher):
            self._matcher = matcher
        elif matcher is None:
            if target_pattern.has_single_output_node:
                self._matcher = SimplePatternMatcher(self._target_pattern)
            else:
                import onnxscript.rewriter.generic_pattern as generic_pattern

                self._matcher = generic_pattern.GenericPatternMatcher(self._target_pattern)
        else:
            self._matcher = matcher(self._target_pattern)
        self._verbose = verbose
        self.name = name

    def __str__(self) -> str:
        if self.name:
            return f"{self.__class__.__name__}(..., name={self.name!r})"
        return (
            f"{self.__class__.__name__}({self._target_pattern}, {self._replacement_pattern})"
        )

    def try_rewrite(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        verbose: int | None = None,
    ) -> ReplacementSubgraph | None:
        """If the node matches the pattern, then replace the node with the replacement pattern."""
        if verbose and verbose > 2:
            print(f"[try_rewrite] {self}")
        verbose = verbose if verbose is not None else self._verbose
        match = self._matcher.match(model, graph_or_function, node, verbose=verbose)
        if match:
            context = None  # TODO(rama)
            for var in self._target_pattern.inputs:
                if var.name is not None:
                    if var.name not in match.bindings:
                        match.bindings[var.name] = None
            if not self._condition_function(context, **match.bindings):
                return None
            replacement_subgraph = self._replacement_pattern.get_replacement(match)
            if replacement_subgraph is None:
                return None
            if len(replacement_subgraph.new_outputs) != self._target_pattern.num_outputs:
                raise ValueError(
                    f"Number of outputs from replacement function does not match the number of outputs from the target pattern. "
                    f"Expected {self._target_pattern.num_outputs}, but got {len(replacement_subgraph.new_outputs)}."
                )
            # TODO(rama): Remove the opset imports from deleted nodes?
            _update_opset_imports(graph_or_function, replacement_subgraph)
            _update_opset_imports(model.graph, replacement_subgraph)
            return replacement_subgraph
        return None

    def apply_to_model(
        self, model: ir.Model, *, commute: bool = False, verbose: int | None = None
    ):
        # A convenience method to apply the rule to a model. We use a RewriteRuleSet to
        # handle commutative rules.
        return RewriteRuleSet([self], commute=commute).apply_to_model(model, verbose=verbose)

    def commute(self) -> Sequence[RewriteRule]:
        def replace_pattern(new_pattern):
            """Return a shallow copy of self with node_pattern replaced by new_pattern."""
            # TODO(rama): Maybe we should use a better alternative to construct new matcher.
            matcher_class = type(self._matcher)
            return RewriteRule(
                new_pattern,
                self._replacement_pattern,
                self._condition_function,
                matcher_class(new_pattern),
                self._verbose,
            )

        return [replace_pattern(p) for p in self._target_pattern.commute()]


class RewriteRuleAsClass:
    """Defines a class grouping method pattern, rewrite, check.
    This class is then given to function :func:`make_rewrite_rule_from_class`
    to define a new rule.
    """

    @classmethod
    def pattern(cls, op, *_) -> Any:
        raise NotImplementedError("Method 'pattern' must be overwritten.")

    @classmethod
    def rewrite(cls, op, *_) -> Any:
        raise NotImplementedError("Method 'rewrite' must be overwritten.")

    @classmethod
    def check(cls, context, *_, **__) -> bool:
        return True


def make_rewrite_rule_from_class(
    rule_class: type | RewriteRuleAsClass, generic: bool = False
) -> RewriteRule:
    """Creates a RewriteRule from a class defining the function
    pattern, rewrite, check with class method. It makes it is easier
    to read when a module contains multiple patterns.

    Example::

        class TransposeIdentity(RewriteRuleAsClass):
            @classmethod
            def pattern(cls, op, x, perm):
                return op.Transpose(x, perm=perm)

            @classmethod
            def check(cls, context, x: ir.Value, perm: ir.Attr | ir.RefAttr) -> bool:
                if isinstance(perm, ir.RefAttr):
                    return False
                if perm.type == ir.AttributeType.INTS:
                    if perm.value == list(range(len(perm.value))):
                        return True
                return False

            @classmethod
            def rewrite(cls, op, x: ir.Value, perm: ir.Attr | None = None):
                return op.Identity(x)

        transpose_identity_rule = make_rewrite_rule_from_class(TransposeIdentity)
    """
    assert hasattr(rule_class, "pattern"), f"Method 'pattern' is missing from {rule_class!r}."
    assert hasattr(rule_class, "rewrite"), f"Method 'rewrite' is missing from {rule_class!r}."
    assert hasattr(rule_class, "check"), f"Method 'check' is missing from {rule_class!r}."
    if generic:
        import onnxscript.rewriter.generic_pattern as orpp

        return RewriteRule(
            rule_class.pattern,
            rule_class.rewrite,
            rule_class.check,
            orpp.GenericPatternMatcher,
            name=rule_class.__name__,  # type: ignore[union-attr]
        )
    return RewriteRule(
        rule_class.pattern,
        rule_class.rewrite,
        rule_class.check,
        name=rule_class.__name__,  # type: ignore[union-attr]
    )


# Variation of RewriteRuleAsClass that is based on instance methods instead of class methods.
# Useful to implement a family of rules to support pattern variations.
# TODO: cleanup the naming conventions for these inter-related classes.
class RewriteRuleClassBase:
    @classmethod
    def rule(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        return RewriteRule(
            instance.pattern, instance.rewrite, instance.check, name=instance.name
        )

    @property
    def name(self):
        """Default implementation of name property."""
        return self.__class__.__name__

    def pattern(self, op, *args, **kwargs):
        raise NotImplementedError("Method 'pattern' must be implemented by derived class.")

    def check(self, op, *args, **kwargs):
        raise NotImplementedError("Method 'check' must be implemented by derived class.")

    def rewrite(self, op, *args, **kwargs):
        raise NotImplementedError("Method 'rewrite' must be implemented by derived class.")


class RewriteRuleSet:
    def __init__(self, rules: Sequence[RewriteRule], *, commute: bool = False) -> None:
        if commute:
            rules = list(itertools.chain.from_iterable([rule.commute() for rule in rules]))
        self.rules = rules

    def _apply_to_graph_or_function(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        verbose: int | None,
    ) -> int:
        count = 0

        # NOTE: Rules should be prioritized in the order they are added to the RewriteRuleSet.
        # And the graph is applied in order.
        for rule in self.rules:
            for node in graph_or_function:
                delta = rule.try_rewrite(model, graph_or_function, node, verbose=verbose)
                if delta is None:
                    continue
                assert isinstance(delta, ReplacementSubgraph)
                # TODO: This does not yet handle the problem of determining the correct insertion point
                # for inserted nodes in the case of patterns with multiple output-nodes. The following
                # is sufficient for patterns with a single output-node "node", which can serve as the
                # insertion-point.
                onnxscript.optimizer.basic_constant_propagation(delta.new_nodes)
                _convenience.replace_nodes_and_values(
                    graph_or_function,
                    node,
                    delta.match.nodes,
                    delta.new_nodes,
                    delta.match.outputs,
                    delta.new_outputs,
                )
                count += 1

        return count

    def apply_to_model(self, model: ir.Model, verbose: int | None = None) -> int:
        assert isinstance(model, ir.Model)
        onnxscript.optimizer.basic_constant_propagation(model.graph)
        count = self._apply_to_graph_or_function(model, model.graph, verbose=verbose)
        for function in model.functions.values():
            onnxscript.optimizer.basic_constant_propagation(function)
            count += self._apply_to_graph_or_function(model, function, verbose=verbose)
        return count

    def __iter__(self):
        yield from self.rules
