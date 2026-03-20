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


def _get_shape_tensor_length(shape_value: ir.Value) -> int | None:
    """Try to determine the number of elements in a 1-D shape tensor.

    Returns the length as an int, or ``None`` if it cannot be determined.
    """
    const = get_numpy_value(shape_value)
    if const is not None:
        return len(const)

    # Use the tensor's own shape annotation (should be 1-D).
    tensor_shape = shape_value.shape
    if tensor_shape is not None and tensor_shape.rank() == 1:
        dim = tensor_shape[0]
        if isinstance(dim, int):
            return dim

    # Trace through Concat and Shape nodes.
    producer = shape_value.producer()
    if producer is None:
        return None

    if producer.op_type == "Concat":
        total = 0
        for inp in producer.inputs:
            if inp is None:
                return None
            seg_len = _get_shape_tensor_length(inp)
            if seg_len is None:
                return None
            total += seg_len
        return total

    if producer.op_type == "Shape":
        x_input = producer.inputs[0] if producer.inputs else None
        if x_input is None:
            return None
        start_attr = producer.attributes.get("start")
        end_attr = producer.attributes.get("end")
        start = start_attr.value if start_attr is not None else 0
        if end_attr is not None:
            return end_attr.value - start
        # end defaults to rank of x
        if x_input.shape is not None:
            x_rank = x_input.shape.rank()
            if x_rank is not None:
                return x_rank - start
        return None

    return None


def _get_dim_from_shape_value(shape_value: ir.Value, index: int):
    """Try to extract the ``index``-th element from a 1-D shape tensor.

    This traces the computation graph through ``Concat`` and ``Shape`` nodes
    to resolve individual elements without requiring the whole tensor to be a
    compile-time constant.

    Returns an ``int``, a ``SymbolicDim``, or ``None`` if the element cannot
    be determined.
    """
    const = get_numpy_value(shape_value)
    if const is not None:
        if 0 <= index < len(const):
            return int(const[index])
        return None

    producer = shape_value.producer()
    if producer is None:
        return None  # graph input or initializer, can't trace

    if producer.op_type == "Concat":
        offset = 0
        for inp in producer.inputs:
            if inp is None:
                return None
            seg_len = _get_shape_tensor_length(inp)
            if seg_len is None:
                return None
            if offset <= index < offset + seg_len:
                return _get_dim_from_shape_value(inp, index - offset)
            offset += seg_len
        return None

    if producer.op_type == "Shape":
        x_input = producer.inputs[0] if producer.inputs else None
        if x_input is None:
            return None
        x_shape = x_input.shape
        if x_shape is None:
            return None
        start_attr = producer.attributes.get("start")
        start = start_attr.value if start_attr is not None else 0
        actual_idx = start + index
        x_rank = x_shape.rank()
        if x_rank is not None and 0 <= actual_idx < x_rank:
            return x_shape[actual_idx]  # int or SymbolicDim
        return None

    return None


def _check_expand_removable(
    expand_input: ir.Value,
    shape: ir.Value,
    other_input: ir.Value,
) -> MatchResult:
    """Check if an Expand node can be safely removed before a binary op.

    The Expand node ``expanded_x = Expand(x, expand_shape)`` before a binary op
    ``out = BinaryOp(expanded_x, y)`` can be removed when the binary op's
    own broadcasting produces the same output shape as the explicit expand.

    Two strategies are tried in order:

    1. **Constant expand shape**: When the expand target shape is a compile-time
       constant, each dimension is checked individually (right-aligned).  At
       dimension ``i`` the expand is safe to remove if any of the following hold:

       - ``expand_shape[i] == 1`` - expand can never shrink a dim, so x_d is
         also 1 and both paths produce ``y_d``.
       - ``x_d == expand_shape[i]`` - expand is a no-op here.
       - ``y_d == expand_shape[i]`` - y already covers the expansion.

    2. **Dynamic expand shape**: When the target shape is not a compile-time
       constant, the rule traces through ``Shape`` and ``Concat`` nodes to
       extract individual dimension values from the shape tensor.  The same
       dimension-by-dimension safety check is then applied.  This handles
       patterns such as ``Expand(x, Concat(Shape(x, 0, 1), Shape(x, 1, 2)))``
       where the expand is provably a no-op.

    Args:
        expand_input: The value fed into the Expand node (``x``).
        shape: The target shape operand of the Expand node.
        other_input: The other operand of the binary op (``y``).

    Returns:
        A MatchResult that is successful when the Expand can be removed.
    """
    check_result = MatchResult()

    expand_input_shape = expand_input.shape
    other_shape = other_input.shape
    if expand_input_shape is None or other_shape is None:
        return check_result.fail("Input shapes are not known.")

    x_rank = expand_input_shape.rank()
    y_rank = other_shape.rank()

    # --- Path 1: expand target shape is a compile-time constant ---
    expand_shape_val = get_numpy_value(shape)
    if expand_shape_val is not None:
        expand_shape = tuple(int(v) for v in expand_shape_val.tolist())
        expand_rank = len(expand_shape)

        for rev_i in range(expand_rank):
            i = expand_rank - 1 - rev_i
            e_d = expand_shape[i]  # always a known integer

            # expand cannot shrink a dim, so x_d must also be 1 here;
            # both with and without expand the output is y_d.
            if e_d == 1:
                continue

            x_idx = x_rank - 1 - rev_i
            x_d = expand_input_shape[x_idx] if x_idx >= 0 else 1

            if isinstance(x_d, int) and x_d == e_d:
                continue  # expand is a no-op at this dimension

            y_idx = y_rank - 1 - rev_i
            y_d = other_shape[y_idx] if y_idx >= 0 else 1

            if isinstance(y_d, int) and y_d == e_d:
                continue  # y already supplies this dimension

            return check_result.fail(
                f"Cannot verify that removing Expand is safe at dimension {i}: "
                f"x_d={x_d!r}, expand_d={e_d}, y_d={y_d!r}."
            )

        return check_result

    # --- Path 2: expand target shape is dynamic ---
    # Trace through Shape/Concat nodes to extract individual elements of the
    # shape tensor, then apply the same dimension-by-dimension check.
    expand_rank = _get_shape_tensor_length(shape)
    if expand_rank is None:
        return check_result.fail(
            "Expand target shape is dynamic and its length cannot be determined."
        )

    for i in range(expand_rank):
        e_d = _get_dim_from_shape_value(shape, i)
        if e_d is None:
            return check_result.fail(
                f"Cannot determine expand shape at dimension {i}."
            )

        if isinstance(e_d, int) and e_d == 1:
            continue  # expand is a no-op at this dimension

        x_idx = x_rank - expand_rank + i
        x_d = expand_input_shape[x_idx] if x_idx >= 0 else 1

        # e_d == x_d works for both int and SymbolicDim (same symbolic name).
        if x_d == e_d:
            continue  # expand is a no-op at this dimension

        y_idx = y_rank - expand_rank + i
        y_d = other_shape[y_idx] if y_idx >= 0 else 1

        if y_d == e_d:
            continue  # y already supplies this dimension

        return check_result.fail(
            f"Cannot verify that removing Expand is safe at dimension {i}: "
            f"x_d={x_d!r}, expand_d={e_d!r}, y_d={y_d!r}."
        )

    return check_result


class _ExpandFirstInput(RewriteRuleClassBase):
    """Removes ``BinaryOp(Expand(x, shape), y)`` -> ``BinaryOp(x, y)``."""

    def __init__(self, op_type: str) -> None:
        super().__init__(f"ExpandFirst_{op_type}", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x: ir.Value, shape: ir.Value, y: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(op.Expand(x, shape), y)

    def check(self, context, x: ir.Value, shape: ir.Value, y: ir.Value) -> MatchResult:
        del context  # Unused
        return _check_expand_removable(x, shape, y)

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
        del context  # Unused
        return _check_expand_removable(y, shape, x)

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
