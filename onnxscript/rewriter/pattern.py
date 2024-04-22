from __future__ import annotations

import inspect
import itertools
import math
from typing import Any, Callable, Sequence

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.printer

from onnxscript import ir
from onnxscript.ir import _convenience
from onnxscript.rewriter import _ir_utils

# Overview of the pattern module: The classes below are used to define both
# patterns (that we search for) and replacements for rewrite rules.
# The matches() method of a pattern is used to check if an IR component
# matches the pattern.
# The to_ir() method of a pattern is used to create a new IR component
# TODO: Ensure that all matches() methods have same type signature (where
# appropriate) and that all to_ir() methods have same type signature (where
# appropriate).


class PythonPattern:
    def __init__(self, value: int | str | Sequence, name: str | None = None) -> None:
        self._value = value
        self._name = name

    @property
    def value(self) -> int | str | Sequence:
        return self._value

    @property
    def name(self) -> str | None:
        return self._name

    def matches(self, value: int | str | Sequence) -> bool:
        return value == self.value

    def to_ir(self, model, bindings=None) -> int | str | Sequence:
        return self.value


class StringConstantPattern:
    def __init__(self, value: str, name: str) -> None:
        self._value = value
        self._name = name

    @property
    def value(self) -> str:
        return self._value

    @property
    def name(self) -> str:
        return self._name

    def matches(self, attr: ir.AttrString) -> bool:
        return attr.value == self.value

    def to_ir(self, model, bindings=None) -> ir.AttrString:
        return ir.AttrString(value=self.value, name=self.name)


class IntConstantPattern:
    def __init__(self, value: int, name: str) -> None:
        self._value = value
        self._name = name

    @property
    def value(self) -> int:
        return self._value

    @property
    def name(self) -> str:
        return self._name

    def matches(self, attr: ir.AttrInt64) -> bool:
        return attr.value == self.value

    def to_ir(self, model, bindings=None) -> ir.AttrInt64:
        return ir.AttrInt64(value=self.value, name=self.name)


class ListConstantPattern:
    def __init__(self, value: Sequence[int | str | float], name: str) -> None:
        self._value = value
        self._name = name

    @property
    def value(self) -> Sequence[int | str | float]:
        return self._value

    @property
    def name(self) -> str:
        return self._name

    def matches(self, attr: ir.AttrFloat32s | ir.AttrInt64s | ir.AttrStrings) -> bool:
        # TODO: Need more data points to determine if this is the right way to compare lists.
        return attr.value == self.value

    def to_ir(self, model, bindings=None) -> ir.AttrFloat32s | ir.AttrInt64s | ir.AttrStrings:
        the_first_non_none_item = next((item for item in self.value if item is not None), None)
        if isinstance(the_first_non_none_item, int):
            return ir.AttrInt64s(value=self.value, name=self.name)  # type: ignore
        if isinstance(the_first_non_none_item, str):
            return ir.AttrStrings(value=self.value, name=self.name)  # type: ignore
        if isinstance(the_first_non_none_item, float):
            return ir.AttrFloat32s(value=self.value, name=self.name)  # type: ignore
        raise TypeError(
            f"Cannot convert list of {type(the_first_non_none_item)} to ConstantPattern"
        )


class PrefixPattern:
    """This pattern is used to simplify submodule opset pattern matching."""

    def __init__(self, value: str) -> None:
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    def matches(self, value: str) -> bool:
        return value.startswith(self.value)

    def to_ir(self, model, bindings=None) -> str:
        raise NotImplementedError("PrefixPattern should not be converted to IR")


class FloatConstantPattern:
    def __init__(
        self, value: float, name: str, rel_tol: float = 1e-5, abs_tol: float = 1e-8
    ) -> None:
        self._value = value
        self._name = name
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        return self._name

    def matches(self, attr: ir.AttrFloat32):
        return math.isclose(
            attr.value, self.value, rel_tol=self._rel_tol, abs_tol=self._abs_tol
        )

    def to_ir(self, model, bindings=None) -> ir.AttrFloat32:
        return ir.AttrFloat32(self.name, self.value)


class TensorConstantPattern:
    def __init__(
        self, value: ir.TensorProtocol, name, rel_tol: float = 1e-3, abs_tol: float = 1e-3
    ) -> None:
        self._value = value
        self._name = name
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        return self._name

    def matches(self, attr: ir.AttrTensor):
        return (
            attr.value.dtype == self._value.dtype
            and attr.value.shape == self._value.shape
            and np.allclose(
                attr.value,
                self._value,
                rtol=self._rel_tol,
                atol=self._abs_tol,
            )
        )

    def to_ir(self, model, bindings=None) -> ir.AttrTensor:
        return ir.AttrTensor(self.name, self.value)


def _make_constant_pattern(
    value: float | int | Sequence | ir.TensorProtocol, name: str
) -> (
    IntConstantPattern
    | FloatConstantPattern
    | TensorConstantPattern
    | StringConstantPattern
    | ListConstantPattern
):
    """Convert an attrbute value to a ConstantPattern."""
    if isinstance(value, float):
        return FloatConstantPattern(value, name)
    if isinstance(value, int):
        return IntConstantPattern(value, name)
    if isinstance(value, str):
        return StringConstantPattern(value, name)
    if isinstance(value, Sequence):
        return ListConstantPattern(value, name)
    if isinstance(value, ir.TensorProtocol):
        return TensorConstantPattern(value, name)
    raise TypeError(f"Cannot convert {type(value)} to ConstantPattern")


class AnyPattern:
    def matches(self, value) -> bool:
        return True


class AttrPattern:
    def __init__(
        self, value: Var | int | float | Sequence | ir.TensorProtocol, name: str
    ) -> None:
        if isinstance(value, Var):
            self.value_pattern = value
        elif isinstance(value, (int, float, Sequence, ir.TensorProtocol)):
            self.value_pattern = _make_constant_pattern(value, name)  # type: ignore[assignment]
        else:
            raise TypeError(f"Cannot convert {type(value)} to AttrPattern")

    def matches(
        self,
        attr_val: int | float | Sequence | Var | ir.TensorProtocol | ir.Value,
        model: ir.Model,
    ) -> MatchResult:
        if isinstance(self.value_pattern, Var):
            return self.value_pattern.matches(attr_val, model)  # type: ignore[arg-type]
        return self.value_pattern.matches(attr_val)

    def to_ir(self, model: ir.Model, rewrite_cache: RewriteCache, bindings=None) -> ir.Value:
        if isinstance(self.value_pattern, Var):
            val, nodes = self.value_pattern.to_ir(
                model, bindings, 1, rewrite_cache
            )  # TODO: handle multiple outputs
            return val
        # constant pattern
        return self.value_pattern.to_ir(model, bindings)


class OpsetPattern:
    """Represents an opset pattern.

    It is used primarily to create a NodePattern (via OpPattern).
    Example usage:
    ::

        z = op.Matmul(x, y)

    Here, `op` is an instance of OpsetPattern and `op.Matmul` is an instance
    of OpPattern, and  `op.Matmul(x, y)` is an instance of NodePattern.

    An opset pattern is also matched against the actual opset used in the
    input model. Typically, we match against an ONNX opset (ignoring the
    version), but we can match against a specific version of the opset too.
    However, it is preferable that version-dependences are handled at the
    level of a rewrite rule, rather than at the level of a pattern.

    """

    def __init__(
        self,
        domain_pattern: PythonPattern | PrefixPattern,
        version_pattern: PythonPattern | AnyPattern,
    ) -> None:
        self.domain_pattern = domain_pattern
        self.version_pattern = version_pattern

    @classmethod
    def singleton(cls, domain: str, version: int) -> OpsetPattern:
        return cls(PythonPattern(domain), PythonPattern(version))

    @classmethod
    def domain(cls, domain: str) -> OpsetPattern:
        return cls(PythonPattern(domain), AnyPattern())

    @classmethod
    def domain_prefix(cls, domain: str) -> OpsetPattern:
        return cls(PrefixPattern(domain), AnyPattern())

    def matches(self, opset):
        domain, version = opset
        return self.domain_pattern.matches(domain) and self.version_pattern.matches(version)

    def to_ir(self, model, bindings=None) -> str:
        domain = self.domain_pattern.to_ir(model, bindings)
        assert isinstance(domain, str), f"Expected str, got {type(domain)}"
        # TODO: Should we ban other custom domains?
        if domain not in model.opset_imports:
            assert isinstance(
                self.version_pattern, PythonPattern
            ), f"custom domain {domain} needs to have a specific version."
            model.opset_imports[self.domain_pattern.value] = self.version_pattern.value
        return domain

    def __getattr__(self, name: str) -> Any:
        return OpPattern(self, PythonPattern(name))

    def submodule(self, name: str) -> Any:
        """This method is used to match against submodule ops with prefix."""
        return OpPattern(self, PrefixPattern(name))


opset17 = OpsetPattern.singleton("", 17)

onnxop = OpsetPattern.domain("")

msft_op = OpsetPattern.singleton("com.microsoft", 1)

torch_module_op = OpsetPattern.domain_prefix("pkg.torch")


class OpPattern:
    """A utility class to build a NodePattern.

    It is used primarily to create a NodePattern.
    Example usage:
    ::

        z = op.Matmul(x, y)

    Here, `op` is an instance of OpsetPattern and `op.Matmul` is an instance
    of OpPattern, and  `op.Matmul(x, y)` is an instance of NodePattern.

    """

    def __init__(
        self,
        opset_pattern: OpsetPattern,
        op_name_pattern: PythonPattern | PrefixPattern,
    ) -> None:
        self.opset_pattern = opset_pattern
        self.op_name_pattern = op_name_pattern

    def __call__(self, *args, **kwargs):
        if "_num_outputs" in kwargs:
            num_outputs = kwargs["_num_outputs"]
            del kwargs["_num_outputs"]
        else:
            num_outputs = 1
        attributes = {
            name: AttrPattern(value=value, name=name) for (name, value) in kwargs.items()
        }
        node_pattern = NodePattern(self.opset_pattern, self.op_name_pattern, args, attributes)
        if num_outputs == 1:
            return NodeOutputPattern(node_pattern, 0)
        else:
            return [NodeOutputPattern(node_pattern, i) for i in range(num_outputs)]


def _to_value_pattern(
    x: ValuePattern | int | float,
) -> NodeOutputPattern | Constant | Var | ValuePattern:
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
    if isinstance(x, ValuePattern):
        return x
    if isinstance(x, (int, float, Sequence)):
        return Constant(x)
    # TODO(titaiwang): Could this be wrapped Constant?
    raise TypeError(f"Cannot convert {type(x)} to ValuePattern")


class MatchResult:
    """Represents the result of a match operation.

    A match can either succeed or fail.
    If it succeeds, it returns a list of IR values that matched the pattern
    and a set of bindings for the variables in the pattern.

    Example:
    ::
        def pattern(x, shape1, shape2):
            t1 = op.Reshape(x, shape1)
            t2 = op.Reshape(t1, shape2)
            return t2
    The above pattern matches a sequence of two Reshape ops.
    The matched_values will contain the values representing the (output of)
    the two Reshape ops, and the bindings will contain the values that
    are bound to the variables `x`, `shape1`, and `shape2`.
    """

    def __init__(
        self, matched_values=None, bindings: dict[str, ir.Value | Any] | None = None
    ) -> None:
        assert matched_values is None or isinstance(matched_values, list)
        self.success: bool = matched_values is not None
        # For a successful match, matched_values is a list of values that matched the pattern.
        # These include the internal nodes of the pattern that were matched, but not
        # the leaves (sub-trees) that match against the variables in the pattern.
        # These represent the values that will be replaced by the replacement pattern.
        self.matched_values: Sequence[Any] | None = matched_values
        # For a successful match, bindings is a dictionary of mapping pattern-variable-names
        # to values.
        self.bindings: dict[str, Any] = bindings if bindings is not None else {}

    def __bool__(self):
        return self.success

    @classmethod
    def FAIL(cls):
        return cls(None)

    @property
    def values(self) -> Sequence[Any] | None:
        return self.matched_values

    def fail(self):
        self.success = False
        self.matched_values = None
        self.bindings = {}

    def extend(self, other: MatchResult | bool, model):
        del model  # Unused
        if not self.success:
            return
        if not other:
            self.fail()
            return
        if isinstance(other, bool):
            return
        for var, val in other.bindings.items():
            if var in self.bindings:
                # TODO: handle attribute var bindings
                if self.bindings[var] != val:
                    self.fail()
                    return
            else:
                self.bindings[var] = val
        assert self.matched_values is not None, "matched_values should not be None."
        self.matched_values.extend(other.matched_values)  # type: ignore[attr-defined]


class ValuePattern:
    """Base class for all patterns that match against IR values.

    This is used primarily to provide operator overloadings for arithmetic
    operations, so that we can write patterns like `x + 1` and `1 + x`.
    """

    def __init__(self) -> None:
        pass

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


class NodePattern:
    """Represents a pattern that matches against a Node.

    This differs from a NodeOutputPattern in that it matches against a node (which
    may produce 1 or more outputs), whereas a NodeOutputPattern matches against
    a specific output of a node.
    """

    def __init__(
        self,
        domain: OpsetPattern,
        op: PythonPattern | PrefixPattern,
        inputs: Sequence[int | float | ValuePattern],
        attributes: dict[str, AttrPattern],
    ):
        self.domain = domain
        self.op = op
        self.inputs = [_to_value_pattern(x) for x in inputs]
        self.attributes = attributes
        self.bound_value = None

    def matches(self, value: ir.Value, model: ir.Model):
        if self.bound_value is not None:
            # DAG-matching, not Tree-matching.
            if self.bound_value.is_same_as(value):
                return MatchResult([])
            else:
                return MatchResult.FAIL()
        node = value.producer()
        if node is None:
            # Eg., value could be an input parameter, which will not match a value
            # computed by the op in this pattern.
            return MatchResult.FAIL()
        return self.matches_node(node, model)

    def matches_node(self, node: ir.Node, model: ir.Model) -> MatchResult:
        """Examine if the IR node matches the self pattern."""
        node_version = model.graph.opset_imports.get(node.domain, 0)
        if not self.domain.matches((node.domain, node_version)):
            return MatchResult.FAIL()
        if not self.op.matches(node.op_type):
            return MatchResult.FAIL()
        match = MatchResult([])
        # TODO: We should add filtered logging starting from here to emit why
        # matching failed. This should cut a lot of noises compared to logging everything,
        # because at least the starting node op_type is already matched.
        for arg_value, previous_node_output_pattern in zip(node.inputs, self.inputs):
            # previous_node_output_pattern could be a Var, if it's the original arg.
            sub_match = previous_node_output_pattern.matches(arg_value, model)  # type: ignore[attr-defined]
            match.extend(sub_match, model)
            if not match:  # If sub-match failed,
                return match
        # Sub-graphs not handled yet.
        for name, attr_pattern in self.attributes.items():
            attr_value = node.attributes.get(name)
            if attr_value is None:
                return MatchResult.FAIL()
            sub_match = attr_pattern.matches(attr_value, model)  # type: ignore[arg-type]
            if not sub_match:
                return MatchResult.FAIL()
            match.extend(sub_match, model)
        for name in node.attributes:
            # TODO: Support matching default values for attributes.
            if name not in self.attributes:
                return MatchResult.FAIL()
        assert match.values is not None, "Matched values should not be None."
        match.values.append(node)  #  type: ignore[attr-defined]
        return match

    def to_ir(
        self,
        model: ir.Model,
        bindings: dict[str, ir.Value | Any],
        num_outputs: int,
        rewrite_cache: RewriteCache,
    ) -> tuple[Sequence[ir.Value], Sequence[ir.Node]]:
        domain = self.domain.to_ir(model)
        op = self.op.to_ir(model)
        assert isinstance(op, str), f"Expected str, got {type(op)}"
        inputs = []
        nodes: list[ir.Node] = []
        for val_pattern in self.inputs:
            if (
                value_and_node := rewrite_cache.get_node_output_pattern(val_pattern)  # type: ignore[arg-type]
            ) is not None:
                val, n = value_and_node
            else:
                val, n = val_pattern.to_ir(model, bindings, 1, rewrite_cache)  # type: ignore[attr-defined]
                rewrite_cache.set_node_output_pattern_with_ir(val_pattern, val, n)  # type: ignore[arg-type]
                nodes.extend(n)  # type: ignore[arg-type]
            # If one of the inputs was a the output of a previous node,
            # unpack the new output ir value that is created for that node
            if isinstance(val, tuple):
                # TODO: Move implementation of output_index to NodeOutputPatter.to_ir
                inputs.append(val[val_pattern.output_index])
            else:
                inputs.append(val)
        attributes = (
            attr_pattern.to_ir(model, rewrite_cache, bindings)
            for attr_pattern in self.attributes.values()
        )
        new_node = ir.Node(
            domain=domain,
            op_type=op,
            inputs=inputs,
            attributes=attributes,  # type: ignore[arg-type]
            num_outputs=num_outputs,
        )
        nodes.append(new_node)
        return new_node.outputs, nodes

    def commute(self) -> Sequence[NodePattern]:
        list_of_lists = [pattern.commute() for pattern in self.inputs]  # type: ignore[attr-defined]

        def enumerate_inputs(inputs, index):
            if index >= len(inputs):
                yield []
            else:
                for pattern in inputs[index]:
                    for rest in enumerate_inputs(inputs, index + 1):
                        yield [pattern, *rest]

        inputs = list(enumerate_inputs(list_of_lists, 0))
        if self.domain.matches(("", None)) and (
            self.op.matches("Add") or self.op.matches("Mul")
        ):
            # TODO: handle cases where number of inputs is not 2.
            swapped = [[x[1], x[0]] for x in inputs]
            inputs.extend(swapped)
        return [NodePattern(self.domain, self.op, input, self.attributes) for input in inputs]


class NodeOutputPattern(ValuePattern):
    """Represents a pattern that matches against a specific output of a Node.

    This is the primary pattern used to match against computed values, that
    is values computed using a specific op.
    """

    def __init__(self, node_pattern: NodePattern, output_index: int) -> None:
        self.node_pattern = node_pattern
        self.output_index = output_index

    def matches(self, value: ir.Value, model: ir.Model):
        """Match the StaticValueInfo from IR with the `matches_node()` in node pattern."""
        node = value.producer()
        if node is None:
            return MatchResult.FAIL()
        if value.index() != self.output_index:
            return MatchResult.FAIL()
        return self.node_pattern.matches_node(node, model)

    def to_ir(
        self,
        model: ir.Model,
        bindings: dict[str, ir.Value | Any],
        num_outputs: int,
        rewrite_cache: RewriteCache,
    ) -> tuple[Sequence[ir.Value], Sequence[ir.Node]]:
        assert self.output_index == 0, "TODO: handle multiple outputs"
        return self.node_pattern.to_ir(model, bindings, num_outputs, rewrite_cache)


class Var(ValuePattern):
    """Represents a pattern variable."""

    def __init__(self, name: str) -> None:
        self.pattern_var_name = name
        self.bound_value = None

    def __repr__(self) -> str:
        return f"Var({self.pattern_var_name!r})"

    def matches(self, value: ir.Value, model: ir.Model):
        return MatchResult([], {self.pattern_var_name: value})

    def to_ir(
        self,
        model: ir.Model,
        bindings: dict[str, ir.Value | Any],
        num_outputs: int,
        rewrite_cache: RewriteCache,
    ) -> tuple[ir.Value, Sequence]:
        del model  # Unused
        del num_outputs  # Unused
        del rewrite_cache  # Unused
        return bindings[self.pattern_var_name], []

    def commute(self) -> Sequence[ValuePattern]:
        return [self]


class Constant(ValuePattern):
    """Represents a pattern that matches against a scalar constant value."""

    def __init__(
        self, value: int | float, rel_tol: float = 1e-5, abs_tol: float = 1e-8
    ) -> None:
        self.value = value
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def match_scalar(self, scalar_value, return_value: Sequence[ir.Node]):
        if math.isclose(scalar_value, self.value, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return MatchResult(return_value)
        return MatchResult.FAIL()

    def matches(self, value: ir.Value, model: ir.Model):
        value = _ir_utils.propagate_const_value(value)
        constant_value = _ir_utils.get_numpy_from_ir_value(value)
        if constant_value is None:
            return MatchResult.FAIL()

        # TODO (rama): allow users to specify shape requirement, if desired.
        if constant_value.size != 1:
            return MatchResult.FAIL()

        return_value: list[ir.Node] = []
        # Note: If the value is produced by a Constant node, we could include
        # the Constant node in the return_value list. However, we don't do that.
        # Instead, we will rely on DCE to remove the constant node if it is not
        # used elsewhere.

        return self.match_scalar(constant_value.item(), return_value)

    def commute(self) -> list[ValuePattern]:
        return [self]


def _handle_pattern_return_value(
    node_output_pattern: NodeOutputPattern | list[NodeOutputPattern],
) -> tuple[NodePattern, int]:
    """This checks and cleans up the return value of a pattern-construction function.

    A pattern-construction function will return values as below:
    ::
        def pattern(x, shape1, shape2):
            ...
            return op.SomeOp(...)
    However, `SomeOp` may represent an ONNX op that produces multiple outputs.
    This function validates that the return values represent the outputs of
    a single NodePattern. It returns the node_pattern and the number of outputs.

    This follows an important restriction of the pattern-matcher algorithm: it
    only matches against subgraphs that end in a single terminal node. If we
    permit two terminal nodes, then we would have to match against all possible
    pairs of nodes in the graph, which produces an extra quadratic factor in the
    complexity of the pattern-matching algorithm. In general, the complexity becomes
    exponential in the number of terminal nodes.

    Args:
        node_output_pattern: NodeOutputPattern | Sequence[NodeOutputPattern]

    Returns:
        tuple[NodePattern, int]: The last node_pattern, num_outputs
    """
    if isinstance(node_output_pattern, NodeOutputPattern):
        node_pattern = node_output_pattern.node_pattern
        num_outputs = 1
    elif isinstance(node_output_pattern, Sequence):
        node_pattern = node_output_pattern[0].node_pattern
        num_outputs = len(node_output_pattern)
        for i, p in enumerate(node_output_pattern):
            assert isinstance(p, NodeOutputPattern)
            assert p.node_pattern is node_pattern
            assert p.output_index == i
    else:
        raise TypeError(f"Invalid type {type(node_output_pattern)} for pattern")
    return node_pattern, num_outputs


# Currently, the replacement graph function is the same as the pattern function.
# This may change in the future.
_handle_replacement_return_value = _handle_pattern_return_value


def _valid_to_replace(matched_nodes: Sequence[Any]) -> bool:
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
            for consumer, _ in v.consumers():
                if consumer not in matched_nodes:
                    return False
    return True


class TargetPatternFunction:
    """The targeted pattern that will be replaced by the replacement pattern.

    Attributes:
        function (Callable): The pattern function that will be matched against the IR.
    """

    def __init__(self, function: Callable) -> None:
        self._function = function

    @property
    def function(self) -> Callable:
        return self._function

    def get_pattern(self, *variables: Sequence[Var]) -> tuple[NodePattern, int]:
        node_output_pattern = self._function(*variables)
        return _handle_pattern_return_value(node_output_pattern)


class ReplacementPatternFunction:
    """The replacement pattern that will replace the targeted pattern.

    Attributes:
        function (Callable): The replacement function that will be used to replace the matched pattern.
        delay_run (bool): If True, the replacement function will not be run until the matched pattern is found.
            This is useful when we want to extract certain metavalue from the matched pattern and use it in the
            replacement pattern.
    """

    def __init__(self, function, *, delay_run: bool = False):
        self._function = function
        self._delay_run = delay_run

    @property
    def function(self) -> Callable:
        return self._function

    @property
    def delay_run(self) -> bool:
        return self._delay_run

    # TODO: How do we merge it with to_ir function?
    def get_pattern(
        self,
        *vars: Sequence[Var],
        match_bindings: dict[str, ir.Value | Any] | None = None,
    ) -> tuple[NodePattern | None, int | None]:
        if self._delay_run:
            if match_bindings is None:
                return None, None
            node_output_pattern = self._function(*vars, match_bindings)
        else:
            node_output_pattern = self._function(*vars)
        return _handle_pattern_return_value(node_output_pattern)


class RewriteCache:
    def __init__(self):
        self._node_output_pattern_to_ir: dict[NodeOutputPattern, tuple[ir.Value, ir.Node]] = (
            dict()
        )

    def get_node_output_pattern(
        self, node_output_pattern: NodeOutputPattern
    ) -> tuple[ir.Value, ir.Node] | None:
        return self._node_output_pattern_to_ir.get(node_output_pattern, None)

    def set_node_output_pattern_with_ir(
        self, node_output_pattern: NodeOutputPattern, value: ir.Value, node: ir.Node
    ) -> None:
        self._node_output_pattern_to_ir[node_output_pattern] = (value, node)


class RewriteRule:
    def __init__(
        self,
        target_pattern: TargetPatternFunction | Callable | None = None,
        replacement_pattern: ReplacementPatternFunction | Callable | None = None,
        condition_function: Callable | None = None,
    ) -> None:
        """Create a rewrite rule.

        Args:
            target_pattern: The pattern function that will be
                matched against the IR.
            replacement_pattern: The replacement function that
                will be used to replace the matched pattern.
            condition_function: The condition function that
                will be used to check if the pattern matches the IR with ir.Values
                constraints in consideration.

        """
        if target_pattern is None:
            # NOTE: commute() generated rules will have target_pattern as None
            # ReplacementPatternFunction is still needed in try_rewrite
            assert replacement_pattern is None
            assert condition_function is None
            self._replacement_pattern = ReplacementPatternFunction(replacement_pattern)
            return
        elif replacement_pattern is None:
            raise ValueError(
                "replacement_pattern must be provided if target_pattern is provided"
            )
        # TODO: Do we want to tolerate Callable inputs?
        if callable(target_pattern):
            target_pattern = TargetPatternFunction(target_pattern)
        if callable(replacement_pattern):
            replacement_pattern = ReplacementPatternFunction(replacement_pattern)

        self._target_pattern = target_pattern
        self._replacement_pattern = replacement_pattern
        self._condition_function = condition_function

        _pattern_vars = inspect.signature(self._target_pattern.function).parameters
        _replacement_vars = inspect.signature(self._replacement_pattern.function).parameters
        # TODO: accept _replacement_vars being subset of _pattern_vars?
        assert len(_pattern_vars) == len(_replacement_vars)

        self._vars = [Var(v) for v in _pattern_vars]
        # Get the last node pattern and number of outputs from the pattern function
        self._target_node_pattern, self._target_num_outputs = self._target_pattern.get_pattern(
            *self._vars  # type: ignore[arg-type]
        )
        # NOTE: Return Nones if the replacement pattern is delayed running
        self._replace_node_pattern, _replacement_num_outputs = replacement_pattern.get_pattern(
            *self._vars  # type: ignore[arg-type]
        )
        if _replacement_num_outputs is not None:
            assert self._target_num_outputs == _replacement_num_outputs

    def matches(self, node: ir.Node, model: ir.Model) -> MatchResult:
        """Check if the node from IR matches the pattern."""
        if len(node.outputs) != self._target_num_outputs:
            return MatchResult.FAIL()
        match = self._target_node_pattern.matches_node(node, model)
        if (
            self._condition_function is not None
            and match
            and not self._condition_function(match.bindings)
        ):
            return MatchResult.FAIL()
        return match

    def try_rewrite(
        self, model: ir.Model, node: ir.Node
    ) -> tuple[Sequence[Any], Sequence[ir.Node]] | None:
        """If the node matches the pattern, then replace the node with the replacement pattern."""
        match = self.matches(node, model)
        if match:
            assert match.values is not None, "Matched values should not be None."
            if _valid_to_replace(match.values):
                # NOTE: delayed running as the replacement pattern needs bindings
                if self._replacement_pattern.delay_run:
                    # bindings will be consumed by the replacement function
                    self._replace_node_pattern, _replacement_num_outputs = (
                        self._replacement_pattern.get_pattern(
                            *self._vars[:-1],  # type: ignore[arg-type]
                            match_bindings=match.bindings,
                        )
                    )
                    assert self._target_num_outputs == _replacement_num_outputs
                rewrite_cache = RewriteCache()
                assert self._replace_node_pattern is not None, "Replacement pattern is None."
                _, _to_insert = self._replace_node_pattern.to_ir(
                    model, match.bindings, self._target_num_outputs, rewrite_cache
                )

                return (match.values, _to_insert)  # type: ignore[return-value]
        return None

    def apply_to_model(self, model: ir.Model, *, commute: bool = False):
        # TODO(titaiwang): Why do we need RewriteRuleSet?
        return RewriteRuleSet([self], commute=commute).apply_to_model(model)

    def count_matches(self, model: ir.Model, *, commute: bool = False):
        return RewriteRuleSet([self], commute=commute).count_matches(model)

    def commute(self) -> Sequence[RewriteRule]:
        def replace_pattern(new_pattern):
            """Return a shallow copy of self with node_pattern replaced by new_pattern."""
            rule = RewriteRule()
            rule._condition_function = self._condition_function
            rule._target_node_pattern = new_pattern
            rule._target_num_outputs = self._target_num_outputs
            rule._replace_node_pattern = self._replace_node_pattern
            return rule

        return [replace_pattern(p) for p in self._target_node_pattern.commute()]


def _apply_deltas(
    graph_or_function: ir.Graph | ir.Function,
    # TODO(jutinchuby): Use a more descriptive data structure to store deltas
    deltas: Sequence[tuple[int, tuple[Sequence[ir.Node], Sequence[ir.Node]]]],
):
    """Applies deltas.

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
    existing_ids = {id(n): (i, n) for i, n in enumerate(graph_or_function)}
    to_delete: set[ir.Node] = set()
    to_insert: list[tuple[ir.Node, list[ir.Node]]] = []

    for i, delta in reversed(deltas):
        if len(delta) == 3:
            # multi-outut strategy
            n_matches, deleted_nodes, inserted_nodes = delta
            for d in deleted_nodes:
                assert id(d) in existing_ids
                to_delete.add(d)

            # the position to insert must be chosen.
            # we'll try position i
            assert i not in to_insert  # conflicts should avoid that case
            to_insert.append((graph_or_function[i], inserted_nodes))
        else:
            deleted_nodes, inserted_nodes = delta
            # Replace deleted nodes with inserted nodes.
            # However, we merge the last deleted node and last inserted node
            # to avoid replacing the values produced by the last deleted node
            # in all places where they are used. So, we reuse the output
            # values from the last deleted node and replace the node itself
            # TODO: simplify this
            last_deleted = deleted_nodes[-1]
            last_inserted = inserted_nodes[-1]
            # Reconnect the users of the deleted node to use the new outputs
            _convenience.replace_all_uses_with(last_deleted.outputs, last_inserted.outputs)
            # Update graph/function outputs if the node generates output
            replacement_mapping = dict(zip(last_deleted.outputs, last_inserted.outputs))
            for idx, graph_or_function_output in enumerate(graph_or_function.outputs):
                if graph_or_function_output in replacement_mapping:
                    graph_or_function.outputs[idx] = replacement_mapping[
                        graph_or_function_output
                    ]

            # insert new nodes after the index node
            graph_or_function.insert_after(last_deleted, inserted_nodes)
            graph_or_function.remove(deleted_nodes, safe=True)

    for replaced_node, inserted_nodes in to_insert:
        graph_or_function.insert_after(replaced_node, inserted_nodes)
        # TODO: improve this
        # This is updating the graph/function outputs to use the new outputs
        for inserted_node in inserted_nodes:
            for new_output in inserted_node.outputs:
                if (index := new_output.meta.get(_ir_utils.GRAPH_OUTPUT_META_KEY)) is not None:  # type: ignore[assignment]
                    graph_or_function.outputs[index] = new_output

    graph_or_function.remove(to_delete, safe=True)


class RewriteRuleSet:
    def __init__(self, rules: Sequence[RewriteRule], *, commute: bool = False) -> None:
        if commute:
            rules = list(itertools.chain.from_iterable([rule.commute() for rule in rules]))
        self.rules = rules

    def _apply_to_graph_or_function(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
    ) -> int:
        count = 0
        marked = set()
        # NOTE: Rules should be prioritized in the order they are added to the RewriteRuleSet.
        # And the graph is applied in order.
        for rule in self.rules:
            deltas = []
            for i, node in enumerate(graph_or_function):
                delta = rule.try_rewrite(model, node)

                if delta is None:
                    continue

                matched_nodes, _ = delta[-2:]

                conflict = False
                for n in matched_nodes:
                    if id(n) in marked:
                        # The same node cannot be matched twice with different patterns.
                        conflict = True
                        break

                if conflict:
                    # Some nodes are already marked as rewritten.
                    continue

                marked |= set(map(id, matched_nodes))

                deltas.append((i, delta))
                count += 1

            _apply_deltas(graph_or_function, deltas)
        return count

    def apply_to_model(self, model: ir.Model) -> int:
        assert isinstance(model, ir.Model)
        count = self._apply_to_graph_or_function(model, model.graph)
        for function in model.functions.values():
            count += self._apply_to_graph_or_function(model, function)
        return count

    def _count_matches_in_graph_or_function(
        self, model: ir.Model, graph_or_function: ir.Graph | ir.Function
    ) -> int:
        count = 0
        for node in graph_or_function:
            for rule in self.rules:
                if rule.matches(node, model):
                    count += 1
                    break
        return count

    def count_matches(self, model: onnx.ModelProto | ir.Model):
        if isinstance(model, onnx.ModelProto):
            model = ir.serde.deserialize_model(model)
        else:
            assert isinstance(model, ir.Model)
        count = self._count_matches_in_graph_or_function(model, model.graph)
        for function in model.functions.values():
            count += self._count_matches_in_graph_or_function(model, function)
        return count
