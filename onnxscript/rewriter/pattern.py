from __future__ import annotations

import inspect
import itertools
import math
from typing import Any, Callable, Sequence

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.printer

import onnxscript._legacy_ir as ir
from onnxscript._legacy_ir import irbuilder

# Overview of the pattern module: The classes below are used to define both
# patterns (that we search for) and replacements for rewrite rules.
# The matches() method of a pattern is used to check if an IR component
# matches the pattern.
# The to_ir() method of a pattern is used to create a new IR component
# TODO: Ensure that all matches() methods have same type signature (where
# appropriate) and that all to_ir() methods have same type signature (where
# appropriate).


class ConstantPattern:
    def __init__(self, value: int | str | list) -> None:
        self._value = value

    @property
    def value(self) -> int | str | list:
        return self._value

    def matches(self, value: int | str | list) -> bool:
        return value == self.value

    def to_ir(self, model, bindings=None) -> int | str | list:
        return self.value


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
    def __init__(self, value: float, rel_tol: float = 1e-5, abs_tol: float = 1e-8) -> None:
        self._value = value
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    @property
    def value(self):
        return self._value

    def matches(self, value: float):
        return math.isclose(value, self.value, rel_tol=self._rel_tol, abs_tol=self._abs_tol)

    def to_ir(self, model, bindings=None) -> float:
        return self.value


class TensorConstantPattern:
    def __init__(
        self, value: np.ndarray, rel_tol: float = 1e-3, abs_tol: float = 1e-3
    ) -> None:
        self._value = value
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    @property
    def value(self):
        return self._value

    def matches(self, value: np.ndarray):
        return (
            value.dtype == self._value.dtype
            and value.shape == self._value.shape
            and np.allclose(
                value,
                self._value,
                rtol=self._rel_tol,
                atol=self._abs_tol,
            )
        )

    def to_ir(self, model, bindings=None) -> onnx.TensorProto:
        return onnx.helper.make_tensor(
            "",
            onnx.helper.np_dtype_to_tensor_dtype(self.value.dtype),
            self.value.shape,
            self.value,
        )


def _make_constant_pattern(
    value: float | int | list | np.ndarray,
) -> ConstantPattern | FloatConstantPattern | TensorConstantPattern:
    """Convert an attrbute value to a ConstantPattern."""
    if isinstance(value, float):
        return FloatConstantPattern(value)
    if isinstance(value, (int, list)):
        return ConstantPattern(value)
    if isinstance(value, np.ndarray):
        return TensorConstantPattern(value)
    raise TypeError(f"Cannot convert {type(value)} to ConstantPattern")


class AnyPattern:
    def matches(self, value) -> bool:
        return True


class AttrPattern:
    def __init__(self, value: Var | int | float | list | np.ndarray) -> None:
        if isinstance(value, Var):
            self.value_pattern = value
        elif isinstance(value, (int, float, list, np.ndarray)):
            self.value_pattern = _make_constant_pattern(value)
        else:
            raise TypeError(f"Cannot convert {type(value)} to AttrPattern")

    def matches(self, attr_val: int | float | list, model: ir.Model) -> MatchResult:
        if isinstance(self.value_pattern, Var):
            return self.value_pattern.matches(attr_val, model)
        return self.value_pattern.matches(attr_val)

    def to_ir(self, model: ir.Model, rewrite_cache: RewriteCache, bindings=None) -> ir.Val:
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
        domain_pattern: ConstantPattern | PrefixPattern,
        version_pattern: ConstantPattern | AnyPattern,
    ) -> None:
        self.domain_pattern = domain_pattern
        self.version_pattern = version_pattern

    @classmethod
    def singleton(cls, domain: str, version: int) -> OpsetPattern:
        return cls(ConstantPattern(domain), ConstantPattern(version))

    @classmethod
    def domain(cls, domain: str) -> OpsetPattern:
        return cls(ConstantPattern(domain), AnyPattern())

    @classmethod
    def domain_prefix(cls, domain: str) -> OpsetPattern:
        return cls(PrefixPattern(domain), AnyPattern())

    def matches(self, opset):
        domain, version = opset
        return self.domain_pattern.matches(domain) and self.version_pattern.matches(version)

    def to_ir(self, model, bindings=None) -> str:
        domain = self.domain_pattern.to_ir(model, bindings)
        # TODO: Should we ban other custom domains?
        if domain not in model.version_map:
            model.version_map[self.domain_pattern.value] = self.version_pattern.value
        return domain

    def __getattr__(self, name: str) -> Any:
        return OpPattern(self, ConstantPattern(name))

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
        op_name_pattern: ConstantPattern | PrefixPattern,
    ) -> None:
        self.opset_pattern = opset_pattern
        self.op_name_pattern = op_name_pattern

    def __call__(self, *args, **kwargs):
        if "_num_outputs" in kwargs:
            num_outputs = kwargs["_num_outputs"]
            del kwargs["_num_outputs"]
        else:
            num_outputs = 1
        attributes = {name: AttrPattern(value) for (name, value) in kwargs.items()}
        node_pattern = NodePattern(self.opset_pattern, self.op_name_pattern, args, attributes)
        if num_outputs == 1:
            return NodeOutputPattern(node_pattern, 0)
        else:
            return [NodeOutputPattern(node_pattern, i) for i in range(num_outputs)]


def _to_value_pattern(x: ValuePattern | int | float) -> ValuePattern:
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
    if isinstance(x, (int, float, list)):
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
                if not self.bindings[var].is_same_as(val):
                    self.fail()
                    return
            else:
                self.bindings[var] = val
        self.matched_values.extend(other.matched_values)


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


# NOTE(bowbao): Based on reading code, this is (nearly) the only place where `model` is used
# for (nearly) all the functions that passes `model` around. It seems the goal is to be able
# create unique value names.
def _make_node(
    model: ir.Model,
    domain: str,
    op: str,
    input,
    attributes,
    num_outputs: int,
) -> tuple[list[ir.Value], ir.Node]:
    inputnames = [x.name for x in input]
    outputs = [model.make_new_name() for i in range(num_outputs)]
    node = onnx.helper.make_node(op, inputnames, outputs, domain=domain, **attributes)
    newnode = ir.Node(node)
    newnode.set_version_if_custom_op(model.version_map)
    newvalues = [ir.Value(name=v, node=newnode, output_index=i) for i, v in enumerate(outputs)]
    newnode.inputs = input
    newnode.outputs = newvalues
    newnode.attributes = attributes  # TODO
    return newvalues, newnode


class NodePattern:
    """Represents a pattern that matches against a Node.

    This differs from a NodeOutputPattern in that it matches against a node (which
    may produce 1 or more outputs), whereas a NodeOutputPattern matches against
    a specific output of a node.
    """

    def __init__(
        self,
        domain: OpsetPattern,
        op: ConstantPattern,
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
        node = value.def_node()
        if node is None:
            # Eg., value could be an input parameter, which will not match a value
            # computed by the op in this pattern.
            return MatchResult.FAIL()
        return self.matches_node(node, model)

    def matches_node(self, node: ir.Node, model: ir.Model) -> MatchResult:
        """Examine if the IR node matches the self pattern."""
        if not self.domain.matches((node.domain, node.version)):
            return MatchResult.FAIL()
        if not self.op.matches(node.op_type):
            return MatchResult.FAIL()
        match = MatchResult([])
        # TODO: We should add filtered logging starting from here to emit why
        # matching failed. This should cut a lot of noises compared to logging everything,
        # because at least the starting node op_type is already matched.
        for arg_value, previous_node_output_pattern in zip(node.inputs, self.inputs):
            # previous_node_output_pattern could be a Var, if it's the original arg.
            sub_match = previous_node_output_pattern.matches(arg_value, model)
            match.extend(sub_match, model)
            if not match:  # If sub-match failed,
                return match
        # Sub-graphs not handled yet.
        for name, attr_pattern in self.attributes.items():
            attr_value = node.get_attribute(name)
            if attr_value is None:
                return MatchResult.FAIL()
            sub_match = attr_pattern.matches(attr_value, model)
            if not sub_match:
                return MatchResult.FAIL()
            match.extend(sub_match, model)
        for name in node.attributes:
            # TODO: Support matching default values for attributes.
            if name not in self.attributes:
                return MatchResult.FAIL()
        match.values.append(node)
        return match

    def to_ir(
        self,
        model: ir.Model,
        bindings: dict[str, ir.Value | Any],
        num_outputs: int,
        rewrite_cache: RewriteCache,
    ) -> tuple[list[ir.Value], list[ir.Node]]:
        domain = self.domain.to_ir(model)
        op = self.op.to_ir(model)
        inputs = []
        nodes = []
        for val_pattern in self.inputs:
            if (
                value_and_node := rewrite_cache.get_node_output_pattern(val_pattern)
            ) is not None:
                val, n = value_and_node
            else:
                val, n = val_pattern.to_ir(model, bindings, 1, rewrite_cache)
                rewrite_cache.set_node_output_pattern_with_ir(val_pattern, val, n)
                nodes.extend(n)
            # If one of the inputs was a the output of a previous node,
            # unpack the new output ir value that is created for that node
            if isinstance(val, list):
                # TODO: Move implementation of output_index to NodeOutputPatter.to_ir
                inputs.append(val[val_pattern.output_index])
            else:
                inputs.append(val)
        attributes = {
            name: attr_pattern.to_ir(model, rewrite_cache, bindings)
            for (name, attr_pattern) in self.attributes.items()
        }
        newvals, newnode = _make_node(model, domain, op, inputs, attributes, num_outputs)
        nodes.append(newnode)
        return newvals, nodes

    def commute(self) -> list[ValuePattern]:
        list_of_lists = [pattern.commute() for pattern in self.inputs]

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
        node = value.def_node()
        if node is None:
            return MatchResult.FAIL()
        if value.def_index() != self.output_index:
            return MatchResult.FAIL()
        return self.node_pattern.matches_node(node, model)

    def to_ir(
        self,
        model: ir.Model,
        bindings: dict[str, ir.Value | Any],
        num_outputs: int,
        rewrite_cache: RewriteCache,
    ) -> tuple[list[ir.Value], list[ir.Node]]:
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
    ) -> tuple[ir.Value, list[None]]:
        del model  # Unused
        del num_outputs  # Unused
        del rewrite_cache  # Unused
        return bindings[self.pattern_var_name], []

    def commute(self) -> list[ValuePattern]:
        return [self]


class Constant(ValuePattern):
    """Represents a pattern that matches against a scalar constant value."""

    def __init__(
        self, value: int | float, rel_tol: float = 1e-5, abs_tol: float = 1e-8
    ) -> None:
        self.value = value
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def match_scalar(self, scalar_value, return_value: list[ir.Node]):
        if math.isclose(scalar_value, self.value, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return MatchResult(return_value)
        else:
            return MatchResult.FAIL()

    def matches(self, value: ir.Value, model: ir.Model):
        del model  # Unused
        constant_value = value.value_as_np_array
        if isinstance(constant_value, np.ndarray):
            # TODO (rama): allow users to specify shape requirement, if desired.
            if constant_value.size != 1:
                return MatchResult.FAIL()

            return_value = []
            # Note: If the value is produced by a Constant node, we could include
            # the Constant node in the return_value list. However, we don't do that.
            # Instead, we will rely on DCE to remove the constant node if it is not
            # used elsewhere.

            return self.match_scalar(constant_value.item(), return_value)
        return MatchResult.FAIL()

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
        node_output_pattern: NodeOutputPattern | list[NodeOutputPattern]

    Returns:
        tuple[NodePattern, int]: The last node_pattern, num_outputs
    """
    if isinstance(node_output_pattern, NodeOutputPattern):
        node_pattern = node_output_pattern.node_pattern
        num_outputs = 1
    elif isinstance(node_output_pattern, (list, tuple)):
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


def _valid_to_replace(matched_nodes: Sequence[ir.Node]) -> bool:
    """Check that values computed by the matched_nodes, except for the last one, are used only by the matched_nodes."""
    # * Must check that all values matched by pattern are used only by pattern,
    # except for the value that is replaced.
    # * Must ensure that replacement subgraph does not use any of the deleted
    # (intermediate) values. (Not necessary for now. Guaranteed.)
    deleted_nodes = matched_nodes[:-1]
    for n in deleted_nodes:
        for v in n.outputs:
            if v.is_output:
                # value is an output-value of the graph/function.
                return False
            for use in v.uses:
                if use not in matched_nodes:
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
    ) -> bool:
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
            *self._vars
        )
        # NOTE: Return Nones if the replacement pattern is delayed running
        self._replace_node_pattern, _replacement_num_outputs = replacement_pattern.get_pattern(
            *self._vars
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
    ) -> tuple[list[ir.Node], list[ir.Node]] | None:
        """If the node matches the pattern, then replace the node with the replacement pattern."""
        match = self.matches(node, model)
        if match:
            if _valid_to_replace(match.values):
                # NOTE: delayed running as the replacement pattern needs bindings
                if self._replacement_pattern.delay_run:
                    # bindings will be consumed by the replacement function
                    self._replace_node_pattern, _replacement_num_outputs = (
                        self._replacement_pattern.get_pattern(
                            *self._vars[:-1], match_bindings=match.bindings
                        )
                    )
                    assert self._target_num_outputs == _replacement_num_outputs
                rewrite_cache = RewriteCache()
                _, _to_insert = self._replace_node_pattern.to_ir(
                    model, match.bindings, self._target_num_outputs, rewrite_cache
                )

                return (match.values, _to_insert)
        return None

    def apply_to_model(self, model: ir.Model, *, commute: bool = False):
        # TODO(titaiwang): Why do we need RewriteRuleSet?
        return RewriteRuleSet([self], commute=commute).apply_to_model(model)

    def count_matches(self, model: ir.Model, *, commute: bool = False):
        return RewriteRuleSet([self], commute=commute).count_matches(model)

    def commute(self) -> list[RewriteRule]:
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
    deltas: list[tuple[int, tuple[list[ir.Node], list[ir.Node]]]],
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
    nodes = graph_or_function.nodes
    existing_ids = {id(n): (i, n) for i, n in enumerate(nodes)}
    to_delete = set()
    to_insert = {}
    path_2 = False

    for i, delta in reversed(deltas):
        if len(delta) == 3:
            # multi-outut strategy
            n_matches, deleted_nodes, inserted_nodes = delta
            for d in deleted_nodes:
                assert id(d) in existing_ids
                to_delete.add(id(d))

            # the position to insert must be chosen.
            # we'll try position i
            assert i not in to_insert  # conflicts should avoid that case
            to_insert[i] = inserted_nodes

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

            assert len(last_deleted.outputs) == len(last_inserted.outputs)
            del last_inserted.outputs[:]
            for v in last_deleted.outputs:
                v.node = last_inserted
                last_inserted.outputs.append(v)

            del nodes[i]

            for new_node in reversed(inserted_nodes):
                nodes.insert(i, new_node)
                # bind the outputs to the graph
                for output_name, value in zip(new_node.output_names, new_node.outputs):
                    graph_or_function.values[output_name] = value
            path_2 = True

    assert not to_delete or not path_2, (
        "Two different rules were applied. It will solved later. "
        "Right now, the functions assumes all the changes come from one "
        "rule."
    )

    if path_2:
        for _, delta in deltas:
            deleted_nodes, inserted_nodes = delta
            inserted_input_output = []
            for nd in inserted_nodes:
                inserted_input_output += nd.inputs + nd.outputs
            for old_node in deleted_nodes[0:-1]:
                # Delete intermediary outputs from graph that are not used as
                # outputs of the graph
                for output in old_node.outputs:
                    if not output.is_output and output not in inserted_input_output:
                        graph_or_function.values.pop(output.name)
                nodes.remove(old_node)

    for i in to_delete:
        position = existing_ids[i][0]
        nodes[position] = None

    for position, insert in sorted(to_insert.items(), reverse=True):
        for v in reversed(insert):
            nodes.insert(position, v)

    position_to_delete = []
    for i, n in enumerate(nodes):
        if n is None:
            position_to_delete.append(i)

    for p in reversed(position_to_delete):
        del nodes[p]


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
        bridge = None
        # NOTE: Rules should be prioritized in the order they are added to the RewriteRuleSet.
        # And the graph is applied in order.
        for rule in self.rules:
            deltas = []
            for i, node in enumerate(graph_or_function.nodes):
                if hasattr(rule, "pattern"):
                    from onnxscript.rewriter.generic_pattern import (
                        GenericRewriteRule,
                        ModelWithGraphStructure,
                    )

                    assert isinstance(
                        rule, GenericRewriteRule
                    ), f"Unexpected type {type(rule)}"
                    # The successors and the predecessors do not change
                    # until the deltas are applied. We cache the structure
                    # to avoid building them again.
                    if bridge is None:
                        bridge = ModelWithGraphStructure(model)
                    delta = rule.try_rewrite(bridge, node)
                else:
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
        for function in model.functions:
            count += self._apply_to_graph_or_function(model, function)
        return count

    def _count_matches_in_graph_or_function(
        self, model: ir.Model, graph_or_funciton: ir.Graph | ir.Function
    ) -> int:
        count = 0
        for node in graph_or_funciton.nodes:
            for rule in self.rules:
                if rule.matches(node, model):
                    count += 1
                    break
        return count

    def count_matches(self, model: onnx.ModelProto | ir.Model):
        if isinstance(model, onnx.ModelProto):
            model = irbuilder.build_ir(model)
        else:
            assert isinstance(model, ir.Model)
        count = self._count_matches_in_graph_or_function(model, model.graph)
        for function in model.functions:
            count += self._count_matches_in_graph_or_function(model, function)
        return count
