# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Collapses ``Reshape -> Transpose -> Reshape`` into a single ``Transpose``.

A Reshape never reorders elements: it only re-labels the same row-major
sequence of values under a new shape. A Transpose is the op that reorders
elements. So the two Reshapes around a Transpose can only ever be collapsed
away when they are pure bookkeeping around the Transpose:

- R1 splits each *original* input axis into a contiguous run of finer axes
  (never merging two different original axes together).
- The Transpose moves each such run around as a whole, contiguous,
  internally-ordered block (never scrambling axes within a run, never
  splitting a run apart).
- R2 merges each relocated block back into a single axis, exactly as the
  block structure predicts.

When all three hold, the composite is provably identical, element for
element, to a single ``Transpose`` of the *original* input axes, and this
rule derives that Transpose's ``perm`` from the block structure above. When
any of the three cannot be established from statically known shapes, the
rule fails closed and leaves the graph unchanged: it never falls back to
comparing shapes alone, since a Reshape/Transpose/Reshape chain can produce
an output shape that looks identical to (or looks like a valid permutation
of) the input shape while actually reordering data within an axis.

This is deliberately the smallest provably-correct slice of the problem
(see onnxscript issue #1775). This first implementation rejects any chain
containing a size-0 or size-1 dimension anywhere in the input, intermediate,
or output shapes -- not only where such a dimension is actually split or
merged. This is deliberately conservative, not something the underlying
theorem requires: a dimension of size 0 or 1 makes axis-boundary provenance
ambiguous in the greedy grouping algorithm below (multiplying a running
product by 0 or 1 never moves it, so the algorithm cannot always tell which
side of a boundary such an axis belongs on), even when that specific axis
is never reordered at all. Supporting size-0/1 dimensions safely is a
self-contained follow-up, not attempted here.

Also out of scope for now, safe to relax later behind its own proof and
tests:

- Symbolic or otherwise not-fully-static shapes anywhere in the chain.

Note on ``allowzero``: this rule deliberately does *not* check it. It only
governs how a literal ``0`` in the raw shape *input tensor* is resolved,
and this rule never reads that tensor -- it trusts the already-resolved
``Value.shape``, which by the time it's inspected here is required to have
every dimension >= 2 (see above). There is no ``0`` left for ``allowzero``
to have had an interpretive choice about, so it cannot affect this rule's
correctness regardless of its value on either Reshape node.
"""

from __future__ import annotations

from typing import Sequence

from onnxscript import ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _static_dims(shape: ir.Shape | None) -> list[int] | None:
    """Returns `shape` as a list of concrete ints, or None if any dim is unknown/symbolic."""
    if shape is None:
        return None
    dims = list(shape)
    if not all(isinstance(d, int) for d in dims):
        return None
    return dims  # type: ignore[return-value]


def _no_trivial_dims(*dim_lists: Sequence[int]) -> bool:
    """True if no dimension in any of `dim_lists` is 0 or 1.

    A run of consecutive axis sizes is located by growing a running product
    until it hits a target size exactly (see `_split_into_original_axis_groups`
    and `_derive_block_permutation` below). Multiplying by 0 or 1 never
    changes that running product, so whenever such an axis sits at a
    candidate boundary, the boundary is genuinely ambiguous rather than
    computable. This rule sidesteps the ambiguity entirely by declining to
    fire whenever a 0 or 1 dimension appears anywhere in the chain.
    """
    return all(d >= 2 for dims in dim_lists for d in dims)


def _split_into_original_axis_groups(
    original_dims: Sequence[int], split_dims: Sequence[int]
) -> list[list[int]] | None:
    """Partitions R1's output axis positions into one contiguous run per original axis.

    Walks `original_dims` and `split_dims` left to right, growing a run of
    consecutive `split_dims` positions until its product matches the next
    original axis's size exactly. Returns the list of runs (each a list of
    positions into `split_dims`), or None if some original axis boundary
    cannot be matched -- i.e. R1 merges across it, mixing two different
    original axes into one intermediate axis, which no Transpose of the
    original axes could ever undo.
    """
    groups: list[list[int]] = []
    pos = 0
    end = len(split_dims)
    for size in original_dims:
        start = pos
        running = 1
        while pos < end and running < size:
            running *= split_dims[pos]
            pos += 1
        if running != size:
            return None
        groups.append(list(range(start, pos)))
    if pos != end:
        return None
    return groups


def _derive_block_permutation(
    groups: Sequence[Sequence[int]], perm: Sequence[int]
) -> list[int] | None:
    """Checks that `perm` moves each of `groups` as a whole, ordered block.

    `groups[g]` lists the R1-output positions belonging to original axis
    `g`, in original order. This walks the Transpose's output (i.e. `perm`
    applied to those positions) and requires it to decompose into exactly
    `len(groups)` contiguous runs, each containing exactly one group's
    positions in their original relative order.

    Returns the resulting permutation of the *original* axes (block `i` of
    the Transpose's output came from original axis `result[i]`), or None if
    a group is split across non-adjacent output positions, interleaved with
    another group, or internally reordered.
    """
    group_of_position: dict[int, int] = {}
    local_order_of_position: dict[int, int] = {}
    for group_index, positions in enumerate(groups):
        for local_index, position in enumerate(positions):
            group_of_position[position] = group_index
            local_order_of_position[position] = local_index

    block_order: list[int] = []
    seen_groups: set[int] = set()
    i = 0
    end = len(perm)
    while i < end:
        group_index = group_of_position[perm[i]]
        if group_index in seen_groups:
            return None  # This group's axes are not contiguous in the output.
        seen_groups.add(group_index)
        run_start = i
        while i < end and group_of_position[perm[i]] == group_index:
            i += 1
        run = perm[run_start:i]
        if [local_order_of_position[p] for p in run] != list(range(len(run))):
            return None  # This group's internal axis order was scrambled.
        block_order.append(group_index)
    return block_order


class ReshapeTransposeReshape(RewriteRuleClassBase):
    """Replaces ``Reshape -> Transpose -> Reshape`` with a single ``Transpose``
    when the chain is provably equivalent to permuting the original input's
    axes. See the module docstring for the exact conditions and scope.
    """

    def pattern(self, op, x, shape1, perm, shape2):
        return op.Reshape(
            op.Transpose(
                op.Reshape(x, shape1, _outputs=["r1_out"]),
                perm=perm,
            ),
            shape2,
        )

    def check(
        self,
        context,
        x: ir.Value,
        shape1: ir.Value,
        perm: ir.Attr,
        shape2: ir.Value,
        r1_out: ir.Value,
    ) -> MatchResult:
        # The raw shape *inputs* are never inspected: this rule trusts the
        # already shape-inferred output shapes instead (same strategy as
        # MaterializeReshapeShape), so it does not need to re-derive Reshape's
        # 0/-1/allowzero resolution itself. allowzero is not checked at all --
        # see the module docstring for why it cannot matter here.
        del shape1, shape2
        check_result = MatchResult()

        if perm.is_ref() or perm.type != ir.AttributeType.INTS:
            return check_result.fail("Transpose permutation is not a concrete attribute.")
        perm_ints = list(perm.as_ints())

        original_dims = _static_dims(x.shape)
        split_dims = _static_dims(r1_out.shape)
        final_dims = _static_dims(context.output_values[0].shape)
        if original_dims is None or split_dims is None or final_dims is None:
            return check_result.fail(
                "Input, intermediate, or output shape is not fully statically known."
            )

        if len(final_dims) != len(original_dims):
            return check_result.fail("Output rank differs from input rank.")

        if not _no_trivial_dims(original_dims, split_dims, final_dims):
            return check_result.fail(
                "An axis of size 0 or 1 appears in this chain; out of scope for "
                "this conservative rule (see module docstring)."
            )

        # Defensive: perm must be a valid permutation of R1's output axes
        # before it's used to index into position-keyed dicts below. A
        # malformed perm (wrong length, out-of-range or repeated index)
        # should fail closed here, not raise a KeyError further down.
        if sorted(perm_ints) != list(range(len(split_dims))):
            return check_result.fail(
                "Transpose perm is not a valid permutation of R1's output axes."
            )

        groups = _split_into_original_axis_groups(original_dims, split_dims)
        if groups is None:
            return check_result.fail(
                "R1 merges across an original axis boundary; not a pure split."
            )

        pi = _derive_block_permutation(groups, perm_ints)
        if pi is None:
            return check_result.fail(
                "Transpose does not move R1's axis groups as whole, internally-ordered blocks."
            )

        # Explicit invariant, not relied upon implicitly: `pi` must name each
        # original axis exactly once. `_no_trivial_dims` already makes the
        # only known way to violate this (an empty group from a size-0/1
        # original axis, see `_split_into_original_axis_groups`) unreachable
        # today, but this rewrite's safety should not depend on that guard
        # being the only thing keeping `pi` well-formed.
        n = len(original_dims)
        if len(pi) != n or len(groups) != n or sorted(pi) != list(range(n)):
            return check_result.fail(
                "Derived permutation failed an internal well-formedness check "
                "(expected a permutation of range(len(original_dims)))."
            )

        expected_final_dims = [original_dims[axis] for axis in pi]
        if expected_final_dims != final_dims:
            return check_result.fail(
                "R2's output shape does not match what the derived permutation "
                "predicts; R2 does not merge exactly the Transpose's blocks."
            )

        self._pi = pi
        return check_result

    def rewrite(
        self,
        op,
        x: ir.Value,
        shape1: ir.Value,
        perm: ir.Attr,
        shape2: ir.Value,
        r1_out: ir.Value,
    ):
        del shape1, perm, shape2, r1_out
        pi = self._pi
        if pi == list(range(len(pi))):
            return op.Identity(x)
        return op.Transpose(x, perm=pi)


reshape_transpose_reshape_rule = ReshapeTransposeReshape.rule()

rules = RewriteRuleSet([reshape_transpose_reshape_rule])
