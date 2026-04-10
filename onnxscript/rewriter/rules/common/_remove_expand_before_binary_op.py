# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fusion rule to remove an Expand node before a binary operator.

This implements the optimization:

    BinaryOp(Expand(x, shape), y) -> BinaryOp(x, y)
    BinaryOp(x, Expand(y, shape)) -> BinaryOp(x, y)

This is valid when the binary operator's broadcasting semantics would produce
the same output shape as first expanding the input and then applying the op.
"""

from __future__ import annotations

from onnxscript import ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._ir_utils import get_numpy_value
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet

# Binary operators in ONNX standard opset that support numpy-style broadcasting.
_BROADCAST_BINARY_OPS: tuple[str, ...] = (
    "Add",
    "And",
    "BitShift",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Mod",
    "Mul",
    "Or",
    "Pow",
    "PRelu",
    "Sub",
    "Xor",
)


def _compute_broadcast_dim(d1, d2):
    """Return the numpy broadcast of two dimension values.

    Each dimension value may be an ``int`` or an ``onnx_ir.SymbolicDim``.
    Returns ``None`` when the result cannot be determined statically (e.g. two
    distinct symbolic values neither of which is known to be 1).
    """
    if d1 == 1:
        return d2
    if d2 == 1:
        return d1
    if d1 == d2:
        return d1
    return None


def _compute_broadcast_shape(shape1: ir.Shape, shape2: ir.Shape) -> list | None:
    """Compute numpy-style broadcast shape symbolically.

    Returns the broadcast shape as a list of dimension values (``int`` or
    ``SymbolicDim``), or ``None`` when the result cannot be determined (e.g.
    unknown ranks or incompatible static dims).
    """
    rank1 = shape1.rank()
    rank2 = shape2.rank()
    if rank1 is None or rank2 is None:
        return None
    rank = max(rank1, rank2)
    result = []
    for i in range(rank):
        idx1 = rank1 - rank + i
        d1 = shape1[idx1] if idx1 >= 0 else 1
        idx2 = rank2 - rank + i
        d2 = shape2[idx2] if idx2 >= 0 else 1
        d = _compute_broadcast_dim(d1, d2)
        if d is None:
            return None
        result.append(d)
    return result


def _check_dims_sufficient(
    expand_shape: ir.Shape,
    x_shape: ir.Shape,
    y_shape: ir.Shape,
) -> MatchResult:
    """Check that x and y together cover every dimension of the expand target.

    For each dimension ``i`` of *expand_shape* (right-aligned) the expand is
    considered redundant when at least one of the following holds:

    - ``expand_shape[i] == 1`` - expand cannot shrink a dim, so ``x_d`` must
      also be 1 and both with and without expand produce ``y_d``.
    - ``x_d == expand_shape[i]`` - the expand is a no-op at this dim.
    - ``y_d == expand_shape[i]`` - ``y`` already supplies this expansion.

    Comparisons work for both ``int`` and ``SymbolicDim`` values.
    """
    check_result = MatchResult()
    e_rank = expand_shape.rank()
    x_rank = x_shape.rank()
    y_rank = y_shape.rank()
    if e_rank is None:
        return check_result.fail("Expand output rank is unknown.")

    for rev_i in range(e_rank):
        i = e_rank - 1 - rev_i
        e_d = expand_shape[i]

        if isinstance(e_d, int) and e_d == 1:
            continue  # expand cannot shrink; x_d is also 1, no-op

        x_idx = x_rank - 1 - rev_i
        x_d = x_shape[x_idx] if x_idx >= 0 else 1
        if x_d == e_d:
            continue  # expand is a no-op at this dimension

        y_idx = y_rank - 1 - rev_i
        y_d = y_shape[y_idx] if y_idx >= 0 else 1
        if y_d == e_d:
            continue  # y already supplies this dimension

        return check_result.fail(
            f"Cannot verify that removing Expand is safe at dimension {i}: "
            f"x_d={x_d!r}, expand_d={e_d!r}, y_d={y_d!r}."
        )

    return check_result


def _check_expand_removable(
    expand_input: ir.Value,
    shape: ir.Value,
    other_input: ir.Value,
    expand_output: ir.Value | None = None,
    binary_op_output: ir.Value | None = None,
) -> MatchResult:
    """Check if an Expand node can be safely removed before a binary op.

    The Expand ``expanded_x = Expand(x, expand_shape)`` before a binary op
    ``out = BinaryOp(expanded_x, y)`` is redundant when the binary op's own
    broadcasting produces the same output as if the expand had been applied.

    Three strategies are tried in order:

    1. **Constant expand shape** - When ``shape`` is a compile-time constant,
       the dimension values are extracted from it and the check is performed
       directly.

    2. **Expand output shape annotation** - When ``shape`` is dynamic but the
       Expand node's output value already carries a shape annotation (e.g.
       after ONNX shape inference has been applied to the model), those
       dimension values are used for the check.

    3. **Binary op output shape** - When neither of the above is available,
       the rule verifies that ``broadcast(x.shape, y.shape)`` symbolically
       equals the binary op's output shape.  If they agree, the binary op's
       own broadcasting already accounts for all the expansion and the
       Expand is redundant.

    Args:
        expand_input: The value fed into the Expand node (``x``).
        shape: The target shape operand of the Expand node.
        other_input: The other operand of the binary op (``y``).
        expand_output: The output value of the Expand node.  Required for
            strategy 2.
        binary_op_output: The output value of the binary op.  Required for
            strategy 3.

    Returns:
        A :class:`MatchResult` that is successful when the Expand can be
        removed.
    """
    check_result = MatchResult()

    x_shape = expand_input.shape
    y_shape = other_input.shape
    if x_shape is None or y_shape is None:
        return check_result.fail("Input shapes are not known.")

    x_rank = x_shape.rank()
    y_rank = y_shape.rank()

    # --- Strategy 1: expand target shape is a compile-time constant ---
    expand_shape_val = get_numpy_value(shape)
    if expand_shape_val is not None:
        expand_shape = tuple(int(v) for v in expand_shape_val.tolist())
        expand_rank = len(expand_shape)

        for rev_i in range(expand_rank):
            i = expand_rank - 1 - rev_i
            e_d = expand_shape[i]  # always a known integer from numpy

            if e_d == 1:
                continue  # expand cannot shrink; x_d is also 1, no-op

            x_idx = x_rank - 1 - rev_i
            x_d = x_shape[x_idx] if x_idx >= 0 else 1

            if isinstance(x_d, int) and x_d == e_d:
                continue  # expand is a no-op at this dimension

            y_idx = y_rank - 1 - rev_i
            y_d = y_shape[y_idx] if y_idx >= 0 else 1

            if isinstance(y_d, int) and y_d == e_d:
                continue  # y already supplies this dimension

            return check_result.fail(
                f"Cannot verify that removing Expand is safe at dimension {i}: "
                f"x_d={x_d!r}, expand_d={e_d}, y_d={y_d!r}."
            )

        return check_result

    # --- Strategy 2: Expand output shape is known (e.g. from shape inference) ---
    if expand_output is not None and expand_output.shape is not None:
        return _check_dims_sufficient(expand_output.shape, x_shape, y_shape)

    # --- Strategy 3: use the binary op's output shape ---
    # broadcast(x.shape, y.shape) must equal the binary op's output shape.
    # If it does, the binary op's own broadcasting already produces the same
    # result as first expanding x and then broadcasting.
    if binary_op_output is not None and binary_op_output.shape is not None:
        op_output_shape = binary_op_output.shape
        if op_output_shape.rank() is not None:
            computed = _compute_broadcast_shape(x_shape, y_shape)
            if computed is not None and len(computed) == op_output_shape.rank():
                if all(c == a for c, a in zip(computed, op_output_shape)):
                    return check_result
        return check_result.fail(
            "broadcast(x.shape, y.shape) does not match the binary op output shape."
        )

    return check_result.fail(
        "Expand target shape is not a constant and no shape annotations are available."
    )


class _ExpandFirstInput(RewriteRuleClassBase):
    """Removes ``BinaryOp(Expand(x, shape), y)`` -> ``BinaryOp(x, y)``."""

    def __init__(self, op_type: str) -> None:
        super().__init__(f"ExpandFirst_{op_type}", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x: ir.Value, shape: ir.Value, y: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(op.Expand(x, shape), y)

    def check(self, context, x: ir.Value, shape: ir.Value, y: ir.Value) -> MatchResult:
        expand_output = context.root.inputs[0] if context.root.inputs else None
        binary_op_output = context.root.outputs[0] if context.root.outputs else None
        return _check_expand_removable(
            x, shape, y, expand_output=expand_output, binary_op_output=binary_op_output
        )

    def rewrite(self, op, x: ir.Value, shape: ir.Value, y: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(x, y)


class _ExpandSecondInput(RewriteRuleClassBase):
    """Removes ``BinaryOp(x, Expand(y, shape))`` -> ``BinaryOp(x, y)``."""

    def __init__(self, op_type: str) -> None:
        super().__init__(f"ExpandSecond_{op_type}", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x: ir.Value, y: ir.Value, shape: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(x, op.Expand(y, shape))

    def check(self, context, x: ir.Value, y: ir.Value, shape: ir.Value) -> MatchResult:
        expand_output = context.root.inputs[1] if context.root.inputs else None
        binary_op_output = context.root.outputs[0] if context.root.outputs else None
        return _check_expand_removable(
            y, shape, x, expand_output=expand_output, binary_op_output=binary_op_output
        )

    def rewrite(self, op, x: ir.Value, y: ir.Value, shape: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(x, y)


def _make_expand_before_binary_op_rules() -> list:
    """Create rewrite rules for removing Expand before each supported binary op."""
    rules = []
    for op_type in _BROADCAST_BINARY_OPS:
        rules.append(_ExpandFirstInput.rule(op_type))
        rules.append(_ExpandSecondInput.rule(op_type))
    return rules


expand_before_binary_op_rules = RewriteRuleSet(_make_expand_before_binary_op_rules())
