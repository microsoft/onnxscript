# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The Pattern IR: used to describe (source) patterns of rewrite rules."""

from __future__ import annotations

import abc
import contextlib
import inspect
import itertools
from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

import onnxscript.rewriter._basics as _basics
from onnxscript import ir

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


class AttrPattern(Pattern[ir.Attr]):
    """Base class for an attribute pattern. Matches any attribute value by default."""

    def __init__(self, name: str | None):
        self._name = name

    @property
    def name(self) -> str | None:
        return self._name

    def matches(self, attr: ir.Attr) -> bool:
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

    def matches(self, attr: ir.Attr) -> bool:
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

    def producer(self) -> NodePattern | None:
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
            self._op_identifier: ir.OperatorIdentifier | None = (
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

    def op_identifier(self) -> ir.OperatorIdentifier | None:
        return self._op_identifier

    @property
    def op_type(self) -> str:
        return str(self.op)

    def matches(self, node: ir.Node, match: _basics.MatchResult) -> _basics.MatchResult:
        """Matches the pattern represented by self against a node.

        This is purely a local node-level match, and does not consider the subgraph rooted at the node.
        We check the domain, op_type, and attributes of the node, but not the inputs.
        """
        # TODO(rama): Ensure we handle "" and "onnx.ai" correctly.
        if not self.op.matches(node.op_type):
            return match.fail(
                f"OpType mismatch: expected {self.op}, got {node.op_type}.", node
            )
        if not self.domain.matches(node.domain):
            return match.fail(
                f"Domain mismatch: expected {self.domain}, got {node.domain}.", node
            )

        for name, attr_pattern in self.attributes.items():
            attr_value = node.attributes.get(name)
            if attr_value is None:
                return match.fail(f"Attribute {name} not found in node.", node)
            if not attr_pattern.matches(attr_value):
                return match.fail(
                    f"Attribute {name} mismatch: expected {attr_pattern}, got {attr_value}.",
                    node,
                )
            if attr_pattern.name is not None:
                if not match.bind(attr_pattern.name, attr_value):
                    return match

        if not self.allow_other_attributes:
            for name in node.attributes:
                # TODO: Support matching default nodes for attributes.
                if name not in self.attributes:
                    return match.fail(f"Attribute {name} not expected in node.", node)

        return match

    def clone(self, node_map: dict[NodePattern, NodePattern], swap: bool) -> NodePattern:
        inputs = [(v.clone(node_map) if v is not None else None) for v in self.inputs]
        if swap:
            assert len(inputs) == 2, (
                "Internal error: commutative swap applies only to binary ops."
            )
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


class AnyValue(ValuePattern):
    """Represents a pattern that matches against any value."""

    def __init__(self) -> None:
        super().__init__(None)

    def clone(self, node_map: dict[NodePattern, NodePattern]) -> AnyValue:
        # A single instance of AnyValue suffices.
        return self


ANY_VALUE = AnyValue()


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

    def __str__(self) -> str:
        return str(self._value)


class OpIdDispatchOr(ValuePattern):
    """Represents a (restricted) form of value pattern disjunction that enables deterministic matching."""

    def __init__(
        self,
        op_to_pattern: Mapping[ir.OperatorIdentifier, tuple[Any, ValuePattern]],
        name: str | None = None,
        tag_var: str | None = None,
    ) -> None:
        """
        Initialize an OpIdDispatchOr pattern.

        Args:
            op_to_pattern: A dictionary mapping operator identifiers to tuples of tag values and patterns.
                The keys are operator identifiers, and the values are tuples containing a tag value
                and a pattern to match against.
            name: An optional variable name for the pattern. Defaults to None. If present,
                this name will be bound to the value matched by the pattern.
            tag_var: An optional variable name for the tag. Defaults to None. If present,
                it will be bound to a value indicating which alternative was matched.
        """
        super().__init__(name)
        self._op_to_pattern = op_to_pattern
        self._tag_var = tag_var

    @property
    def tag_var(self) -> str | None:
        """Returns the tag variable associated with the OrValue pattern."""
        return self._tag_var

    def clone(self, node_map: dict[NodePattern, NodePattern]) -> OpIdDispatchOr:
        return OpIdDispatchOr(
            {k: (v[0], v[1].clone(node_map)) for k, v in self._op_to_pattern.items()},
            self.name,
            self._tag_var,
        )

    def get_pattern(self, value: ir.Value) -> tuple[Any, ValuePattern] | None:
        """Returns the pattern that should be tried for the given value."""
        producer = value.producer()
        if producer is not None:
            id = producer.op_identifier()
            if id is not None and id in self._op_to_pattern:
                return self._op_to_pattern[id]
        return None


class BacktrackingOr(ValuePattern):
    """Represents an unrestricted form of OR pattern implemented using backtracking."""

    def __init__(
        self,
        values: Sequence[ValuePattern],
        name: str | None = None,
        tag_var: str | None = None,
        tag_values: Sequence[Any] | None = None,
    ) -> None:
        """
        Initialize a BacktrackingOr pattern.

        Args:
            values: A sequence of value patterns to match against.
            name: An optional variable name for the pattern. Defaults to None. If present,
                this name will be bound to the value matched by the pattern.
            tag_var: An optional variable name for the tag. Defaults to None. If present,
                it will be bound to a value (from tag_values) indicating which alternative was matched.
            tag_values: An optional sequence of values to bind to the tag_var. Defaults to None.
                If present, the length of tag_values must match the number of alternatives in values.
                In a successful match, tag-var will be bound to the i-th value in tag_values if the i-th
                alternative pattern matched. If omitted, the default value of (0, 1, 2, ...) will be used.
        """
        super().__init__(name)
        if tag_values is not None:
            if tag_var is None:
                raise ValueError("tag_var must be specified if tag_values is provided.")
            if len(tag_values) != len(values):
                raise ValueError(
                    "tag_values must have the same length as the number of alternatives."
                )
        else:
            tag_values = tuple(range(len(values)))
        self._tag_var = tag_var
        self._tag_values = tag_values
        self._values = values

    @property
    def tag_var(self) -> str | None:
        """Returns the tag variable associated with the OrValue pattern."""
        return self._tag_var

    def clone(self, node_map: dict[NodePattern, NodePattern]) -> BacktrackingOr:
        return BacktrackingOr(
            [v.clone(node_map) for v in self._values],
            self.name,
            self._tag_var,
            self._tag_values,
        )


def OrValue(
    values: Sequence[ValuePattern],
    name: str | None = None,
    tag_var: str | None = None,
    tag_values: Sequence[Any] | None = None,
) -> ValuePattern:
    """
    Creates an OR pattern.

    Args:
        values: A sequence of value patterns to match against.
        name: An optional variable name for the pattern. Defaults to None. If present,
            this name will be bound to the value matched by the pattern.
        tag_var: An optional variable name for the tag. Defaults to None. If present,
            it will be bound to a value (from tag_values) indicating which alternative was matched.
        tag_values: An optional sequence of values to bind to the tag_var. Defaults to None.
            If present, the length of tag_values must match the number of alternatives in values.
            In a successful match, tag-var will be bound to the i-th value in tag_values if the i-th
            alternative pattern matched. If omitted, the default value of (0, 1, 2, ...) will be used.
    """
    if tag_values is not None:
        if tag_var is None:
            raise ValueError("tag_var must be specified if tag_values is provided.")
        if len(tag_values) != len(values):
            raise ValueError(
                "tag_values must have the same length as the number of alternatives."
            )
    else:
        tag_values = tuple(range(len(values)))

    def make_op_id_or_pattern() -> OpIdDispatchOr | None:
        mapping: dict[ir.OperatorIdentifier, tuple[Any, NodeOutputPattern]] = {}
        for i, alternative in enumerate(values):
            if not isinstance(alternative, NodeOutputPattern):
                return None
            producer = alternative.producer()
            id = producer.op_identifier()
            if id is None or id in mapping:
                return None
            mapping[id] = (tag_values[i], alternative)
        return OpIdDispatchOr(mapping, name, tag_var)

    optimized_pattern = make_op_id_or_pattern()
    return optimized_pattern or BacktrackingOr(
        values, name, tag_var, tag_values if tag_var else None
    )


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


def _add_backward_slice(
    node: NodePattern,
    backward_slice: set[NodePattern],
    backward_slice_values: set[ValuePattern],
) -> None:
    """Adds all nodes in the backward slice of given node to the set `backward_slice`.

    The backward slice of a node is the set of all nodes that are reachable from the node
    in a backward traversal from the given node.
    """
    if node in backward_slice:
        return
    backward_slice.add(node)
    for value_pattern in node.inputs:
        if isinstance(value_pattern, NodeOutputPattern):
            _add_backward_slice(
                value_pattern.producer(), backward_slice, backward_slice_values
            )
        elif isinstance(value_pattern, (OpIdDispatchOr, BacktrackingOr)):
            backward_slice_values.add(value_pattern)


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
        choice_values_returned: set[ValuePattern] = set()
        covered_choice_values: set[ValuePattern] = set()
        for value_pattern in outputs:
            if not isinstance(value_pattern, ValuePattern):
                raise TypeError(
                    f"Invalid type {type(value_pattern)} for graph pattern output."
                )
            if isinstance(value_pattern, NodeOutputPattern):
                candidate = value_pattern.producer()
                if candidate not in covered:
                    output_nodes.add(candidate)
                    _add_backward_slice(candidate, covered, covered_choice_values)
            elif isinstance(value_pattern, (OpIdDispatchOr, BacktrackingOr)):
                choice_values_returned.add(value_pattern)

        # check if all choice_values_returned are contained in covered_choice_values:
        # We don't yet support the use of a choice-value as a "root" of the search.
        # This is a limitation of the current implementation, and will be fixed in the future.
        if not (choice_values_returned <= covered_choice_values):
            raise NotImplementedError("Returning uncovered choice-values is not supported.")

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
