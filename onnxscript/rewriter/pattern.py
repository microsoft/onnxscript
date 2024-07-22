# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import abc
import dataclasses
import inspect
import itertools
import math
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    MutableSequence,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from onnxscript import ir
from onnxscript.ir import _convenience
from onnxscript.rewriter import _ir_utils, _tape

T = TypeVar("T")


class Pattern(Protocol[T]):  # type: ignore[misc]
    """This is essentially a Predicate[T], that is, a Callable[[T], bool] bound to the name "matches"."""

    def matches(self, item: T) -> bool: ...


class StringConstantPattern(Pattern[str]):
    """Matches strings with given value."""

    def __init__(self, value: str):
        self._value = value

    def matches(self, item: str) -> bool:
        return item == self._value

    def __str__(self) -> str:
        return self._value


class PrefixPattern(Pattern[str]):
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
    if type(value) == ValuePattern:
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


class OpsetPatternBuilder(Pattern[str]):
    """Represents an opset pattern.

    (i) It is used to create a NodePattern (via OpPatternBuilder).
    Example usage:
    ::

        z = op.Matmul(x, y)

    Here, `op` is an instance of OpsetPatternBuilder and `op.Matmul` is an instance
    of OpPatternBuilder, and  `op.Matmul(x, y)` is an instance of NodePattern.

    (ii) An opset pattern is also matched against the actual opset domain used in the
    input model.
    """

    def __init__(self, domain: Pattern[str] | str) -> None:
        if isinstance(domain, str):
            self._domain_name: str | None = domain
            self._domain_pattern: Pattern[str] = StringConstantPattern(domain)
        else:
            self._domain_name = None
            self._domain_pattern = domain

    @property
    def domain_name(self) -> str | None:
        return self._domain_name

    def matches(self, domain):
        return self._domain_pattern.matches(domain)

    def __getattr__(self, op_name: str) -> OpPatternBuilder:
        return OpPatternBuilder(self, op_name)

    def submodule(self, name: str) -> OpPatternBuilder:
        """This method is used to match against submodule ops with prefix."""
        return OpPatternBuilder(self, PrefixPattern(name))

    def __str__(self) -> str:
        return str(self._domain_pattern)


onnxop = OpsetPatternBuilder("")

msft_op = OpsetPatternBuilder("com.microsoft")

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
        opset_pattern: OpsetPatternBuilder,
        op_name: str | Pattern[str],
    ) -> None:
        self.opset_pattern = opset_pattern
        self.op_name = op_name

    def __call__(
        self,
        *args,
        domain: str | None = None,
        version: int | None = None,
        outputs: int | list[str | None] = 1,
        _allow_other_attributes: bool | None = None,
        **kwargs,
    ):
        if version is not None:
            raise ValueError(
                "The pattern builder does not support 'version' keyword argument. "
                "Version restrictions should be handled by rewrite rules."
            )
        if domain is None:
            opset_pattern = self.opset_pattern
        elif isinstance(domain, str):
            opset_pattern = OpsetPatternBuilder(domain)
        else:
            # TODO(rama): allow OpsetPatternBuilder as domain.
            raise TypeError("domain must be a string.")

        if isinstance(outputs, int):
            outputs = [None for _ in range(outputs)]
        elif not isinstance(outputs, Sequence) or not all(
            isinstance(x, (str, type(None))) for x in outputs
        ):
            raise ValueError("outputs must be an int or a list[str|None].")
        inputs = [_to_value_pattern(x) for x in args]
        attributes = {name: _to_attr_pattern(value) for (name, value) in kwargs.items()}
        node_pattern = NodePattern(
            opset_pattern, self.op_name, inputs, attributes, outputs, _allow_other_attributes
        )
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
    # TODO(rama): support lists of int/float
    # if isinstance(x, list):
    #     if all(isinstance(i, (int, float)) for i in x):
    #         return Constant(x)
    #     raise ValueError("Only lists of int/float can be used as a ValuePattern")
    # TODO(titaiwang): Could this be wrapped Constant?
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


class ValuePattern:
    """Base class for all patterns that match against IR values.

    This is used primarily to provide operator overloadings for arithmetic
    operations, so that we can write patterns like `x + 1` and `1 + x`.
    """

    def __init__(self, name: str | None) -> None:
        self._name = name
        # Note: uses will be computed only when the full graph-pattern is constructed.
        self._uses: list[tuple[NodePattern, int]] = []

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

    def commute(self) -> Sequence[ValuePattern]:
        """Return a list of commuted patterns.

        This is used to handle commutative operations like addition and multiplication.
        A single pattern is converted into a list of equivalent patterns by swapping
        the parameters of commutative operations.
        """
        return [self]

    def __add__(self, other):
        return onnxop.Add(self, other)

    def __radd__(self, other):
        return onnxop.Add(other, self)

    def __sub__(self, other):
        return onnxop.Sub(self, other)

    def __rsub__(self, other):
        return onnxop.Sub(other, self)

    def __mul__(self, other):
        return onnxop.Mul(self, other)

    def __rmul__(self, other):
        return onnxop.Mul(other, self)

    def __truediv__(self, other):
        return onnxop.Div(self, other)

    def __rtruediv__(self, other):
        return onnxop.Div(other, self)

    def __pow__(self, other):
        return onnxop.Pow(self, other)

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
        domain: OpsetPatternBuilder,
        op: str | Pattern[str],
        inputs: Sequence[int | float | ValuePattern | None],
        attributes: dict[str, AttrPattern],
        outputs: Sequence[str | None],
        allow_other_attributes: bool | None,
    ):
        if allow_other_attributes is None:
            # Default behavior: allow other unmatched attributes in the node.
            allow_other_attributes = True
        self.domain = domain
        self.op = StringConstantPattern(op) if isinstance(op, str) else op
        self.inputs = [_to_value_pattern(x) for x in inputs]
        self.attributes = attributes
        self.allow_other_attributes = allow_other_attributes
        # In the common case, domain and op are constants, which can be used to optimize matching.
        if isinstance(op, str) and domain.domain_name is not None:
            # TODO(rama): support overloaded operators.
            overload = ""
            self._op_identifier: tuple[str, str, str] | None = (
                domain.domain_name,
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

    def commute(self) -> Sequence[NodePattern]:
        list_of_lists = [
            [None] if pattern is None else pattern.commute() for pattern in self.inputs
        ]  # type: ignore[attr-defined]

        def enumerate_inputs(inputs, index):
            if index >= len(inputs):
                yield []
            else:
                for pattern in inputs[index]:
                    for rest in enumerate_inputs(inputs, index + 1):
                        yield [pattern, *rest]

        inputs = list(enumerate_inputs(list_of_lists, 0))
        if self.domain.matches("") and (self.op.matches("Add") or self.op.matches("Mul")):
            # TODO: handle cases where number of inputs is not 2.
            swapped = [[x[1], x[0]] for x in inputs]
            inputs.extend(swapped)
        outputs = [value.name for value in self.outputs]
        return [
            NodePattern(
                self.domain,
                self.op,
                input,
                self.attributes,
                outputs,
                self.allow_other_attributes,
            )
            for input in inputs
        ]


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

    @property
    def output_index(self) -> int:
        return self._output_index

    def commute(self) -> Sequence[ValuePattern]:
        # TODO
        return [
            NodeOutputPattern(pattern, self._output_index, self.name)
            for pattern in self._producer.commute()
        ]

    def producer(self) -> NodePattern:
        return self._producer


Var = ValuePattern


class Constant(ValuePattern):
    """Represents a pattern that matches against a scalar constant value."""

    def __init__(
        self, value: int | float, rel_tol: float = 1e-5, abs_tol: float = 1e-8
    ) -> None:
        super().__init__(None)
        self._value = value
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    @property
    def value(self) -> int | float:
        return self._value

    def matches(self, value: ir.Value, match: MatchResult) -> MatchResult:
        value = _ir_utils.propagate_const_value(value)
        constant_value = value.const_value
        if constant_value is None:
            return match.fail(f"Value is not a constant, expecting {self.value}.")

        constant_value_numpy = constant_value.numpy()
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

    def commute(self) -> list[ValuePattern]:
        return [self]

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


class GraphPattern:
    """Represents a pattern that can be matched against a subgraph."""

    def __init__(
        self, inputs: Sequence[ValuePattern], outputs: Sequence[ValuePattern]
    ) -> None:
        self._inputs = inputs
        self._outputs = outputs
        if len(outputs) == 0:
            raise ValueError("GraphPattern must have at least one output")
        self._nodes = _nodes_in_pattern(outputs)

        # Check if all outputs are produced by the same node.
        output_node = None
        for i, value_pattern in enumerate(outputs):
            if not isinstance(value_pattern, ValuePattern):
                raise TypeError(
                    f"Invalid type {type(value_pattern)} for graph pattern output."
                )
            if not isinstance(value_pattern, NodeOutputPattern) or (
                value_pattern.output_index != i
            ):
                output_node = None
            elif i == 0:
                output_node = value_pattern.producer()
            elif value_pattern.producer() is not output_node:
                output_node = None
        self._output_node = output_node

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
        return self._output_node is not None

    @property
    def num_outputs(self) -> int:
        return len(self._outputs)

    def commute(self) -> Sequence[GraphPattern]:
        if self._output_node is None:
            raise NotImplementedError(
                "Cannot commute a graph pattern with multiple output nodes."
            )
        nodes = self._output_node.commute()
        return [
            GraphPattern(
                self._inputs, [NodeOutputPattern(n, i) for i in range(self.num_outputs)]
            )
            for n in nodes
        ]

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
    pattern_outputs = pattern_constructor(onnxop, *pattern_inputs)
    # TODO(rama): classify inputs as value/attribute vars
    # Returned value could be a single ValuePattern or a list of ValuePatterns.
    # Normalize representation to a list of ValuePatterns.
    if isinstance(pattern_outputs, ValuePattern):
        pattern_outputs = [pattern_outputs]
    return GraphPattern(pattern_inputs, pattern_outputs)


def _valid_to_replace(matched_nodes: Sequence[ir.Node]) -> bool:
    """Check that values computed by the matched_nodes, except for the last one, are used only by the matched_nodes."""
    # * Must check that all values matched by pattern are used only by pattern,
    # except for the value that is replaced.
    # * Must ensure that replacement subgraph does not use any of the deleted
    # (intermediate) values. (Not necessary for now. Guaranteed.)
    deleted_nodes = matched_nodes[:-1]
    for n in deleted_nodes:
        for v in n.outputs:
            if v.is_graph_output():
                # value is an output-value of the graph/function.
                return False
            for consumer, _ in v.uses():
                if consumer not in matched_nodes:
                    return False
    return True


# A type representing the domains/versions used in creating a replacement subgraph
UsedOpsets = List[Tuple[str, Optional[int]]]


class RewriterContext:
    """Context parameter used to build the replacement pattern."""

    # TODO(justinchuby): Merge with the rest of pattern building methods
    def __init__(self):
        self._tape = _tape.Tape()
        self._used_opsets: UsedOpsets = []

    def __getattr__(self, op_type: str) -> Any:
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def _make_node(self, op_type: str, inputs: Sequence[ir.Value], kwargs: dict[str, Any]):
        # TODO(rama): some of the following logic should move into the tape.
        domain = kwargs.pop("domain", "")
        version = kwargs.pop("version", None)
        outputs = kwargs.pop("outputs", 1)
        if isinstance(outputs, Sequence):
            num_outputs = len(outputs)
        else:
            assert isinstance(outputs, int)
            num_outputs = outputs

        self._used_opsets.append((domain, version))
        if num_outputs == 1:
            value = self._tape.op(op_type, inputs=inputs, attributes=kwargs, domain=domain)
            if isinstance(outputs, Sequence):
                value.name = outputs[0]
            return value
        values = self._tape.op_multi_output(
            op_type, inputs=inputs, attributes=kwargs, domain=domain, num_outputs=num_outputs
        )
        if isinstance(outputs, Sequence):
            for value, name in zip(values, outputs):
                value.name = name
        return values

    @property
    def nodes(self) -> Sequence[ir.Node]:
        # TODO(rama): The current tape-based implementation will not track nodes added
        # via overloaded operators, eg., `x + y`. One possible way to fix this is to
        # have values/nodes know which tape they belong to (instead of a graph/function).
        # However, it is unclear we need this feature for rewriting: we could also
        # identify the nodes to be inserted from the replacement values (by tracing back).
        return self._tape.nodes

    @property
    def used_opsets(self) -> UsedOpsets:
        return self._used_opsets


@dataclasses.dataclass
class ReplacementSubgraph:
    """A subgraph that will replace the matched pattern."""

    match: MatchResult
    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]
    used_opsets: UsedOpsets


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
        pass

    def __str__(self) -> str:
        return str(self.pattern)


class SimplePatternMatcher(PatternMatcher):
    def __init__(self, pattern: GraphPattern) -> None:
        assert (
            pattern.has_single_output_node
        ), "SimplePatternMatcher only supports patterns with a single output node."
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
        value = _ir_utils.propagate_const_value(value)
        constant_value = value.const_value
        if constant_value is None:
            return self.fail(
                f"Value {value.name} is not a constant, expecting {pattern_constant.value}.",
            )

        constant_value_numpy = constant_value.numpy()
        # TODO (rama): allow users to specify shape requirement, if desired.
        if constant_value_numpy.size != 1:
            return self.fail(
                f"Value {value.name} is not a scalar, expecting {pattern_constant.value}.",
            )

        if not math.isclose(
            constant_value_numpy.item(),
            pattern_constant._value,
            rel_tol=pattern_constant._rel_tol,
            abs_tol=pattern_constant._abs_tol,
        ):
            return self.fail(
                f"Constant value mismatch: expected {pattern_constant._value}, got {constant_value_numpy.item()}.",
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

        for arg_value, previous_node_output_pattern in zip(node.inputs, pattern_node.inputs):
            # previous_node_output_pattern could be a Var, if it's the original arg.
            if arg_value is None and previous_node_output_pattern is None:
                continue
            if arg_value is None or previous_node_output_pattern is None:
                msg = (
                    "Input not expected to be None"
                    if arg_value is None
                    else "Input expected to be None"
                )
                return self.fail(msg)
            if not self._match_value(previous_node_output_pattern, arg_value):
                return False

        for i, output_value_pattern in enumerate(pattern_node.outputs):
            if not self._bind_value(output_value_pattern, node.outputs[i]):
                return False

        match.nodes.append(node)
        return True

    def _bind_value(self, pattern_value: ValuePattern, value: ir.Value) -> bool:
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

    def _match_value(self, pattern_value: ValuePattern, value: ir.Value) -> bool:
        """Match an IR value against a ValuePattern instance."""
        if not self._bind_value(pattern_value, value):
            return False

        if isinstance(pattern_value, NodeOutputPattern):
            return self._match_node_output(pattern_value, value)
        if isinstance(pattern_value, Constant):
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

    def match(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        verbose: int = 0,
    ) -> MatchResult:
        del model
        del graph_or_function
        self._verbose = verbose
        self._matched: dict[NodePattern, ir.Node] = {}
        self._match: MatchResult = MatchResult()

        pattern = self.pattern
        match = self._match
        if len(node.outputs) != pattern.num_outputs:
            return match.fail(
                f"Number of node outputs mismatch: expected {pattern.num_outputs}, got {len(node.outputs)}."
            )
        if pattern._output_node is None:
            return match.fail(
                "Internal Error: SimplePatternMatcher should not be used for patterns with multiple output nodes."
            )

        if self._match_node(pattern._output_node, node):
            if not _valid_to_replace(match.nodes):
                return match.fail("Matched nodes have other uses preventing replacement.")

        match.outputs.extend(node.outputs)
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


def _apply_delta(
    graph_or_function: ir.Graph | ir.Function,
    node: ir.Node,
    delta: ReplacementSubgraph,
):
    """Applies delta.

    This code is valid is the considered pattern has only one output.
    In case of multi output replacements, there is not need to rename
    the outputs.

    In case of multi-output design, the nodes may not be necessary inserted
    all at the same position. To be convinced, you can take a pattern
    producing two outputs, but the second one needs the first one and
    another input appeared after the first outputs. What could be
    the right place to inserted all of the node.

    The current implementation insert all the nodes at the same position
    but checks there is not inconsistency. In that case, it fails.
    We could reorder (long) or do more clever changes.
    The reordering would probably happen not very often.
    """

    assert isinstance(delta, ReplacementSubgraph)
    # Replace matched nodes with new nodes, matched values with new values
    old_values = delta.match.outputs
    new_values = delta.new_outputs

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
    for idx, graph_or_function_output in enumerate(graph_or_function.outputs):
        if graph_or_function_output in replacement_mapping:
            graph_or_function.outputs[idx] = replacement_mapping[graph_or_function_output]

    # insert new nodes after the index node
    graph_or_function.insert_after(node, delta.new_nodes)
    graph_or_function.remove(delta.match.nodes, safe=True)


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
                _apply_delta(graph_or_function, node, delta)
                count += 1

        return count

    def apply_to_model(self, model: ir.Model, verbose: int | None = None) -> int:
        assert isinstance(model, ir.Model)
        count = self._apply_to_graph_or_function(model, model.graph, verbose=verbose)
        for function in model.functions.values():
            count += self._apply_to_graph_or_function(model, function, verbose=verbose)
        return count

    def __iter__(self):
        yield from self.rules
