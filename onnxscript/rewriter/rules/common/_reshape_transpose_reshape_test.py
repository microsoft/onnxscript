# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.checker
import onnx.helper as oh
import onnxruntime as ort

from onnxscript import ir, optimizer
from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _reshape_transpose_reshape as rtr


def _dims_to_str(dims: list[int]) -> str:
    return ", ".join(str(d) for d in dims)


def _build_chain(
    input_dims: list[int],
    r1_dims: list[int],
    perm: list[int],
    output_dims: list[int],
    *,
    r1_shape_known: bool = True,
    output_shape_known: bool = True,
) -> ir.Model:
    """Builds ``Reshape -> Transpose -> Reshape`` with non-constant shape
    inputs (Shape() of a dummy tensor), then manually stamps the
    shape-inference results this rule relies on -- mirroring
    ``_materialize_reshape_shape_test.py``'s convention of simulating shape
    inference having already run.
    """
    n1 = len(r1_dims)
    n2 = len(output_dims)
    output_sig = (
        f"float[{_dims_to_str(output_dims)}] output" if output_shape_known else "float output"
    )
    text = f"""
        <ir_version: 7, opset_import: [ "" : 17]>
        agraph (float[{_dims_to_str(input_dims)}] data, int64[{n1}] shape1, int64[{n2}] shape2)
            => ({output_sig})
        {{
            r1 = Reshape(data, shape1)
            t = Transpose<perm=[{_dims_to_str(perm)}]>(r1)
            output = Reshape(t, shape2)
        }}
    """
    model = ir.from_onnx_text(text)
    for node in model.graph:
        if node.outputs[0].name == "r1" and r1_shape_known:
            node.outputs[0].shape = ir.Shape(r1_dims)
        if node.outputs[0].name == "output" and output_shape_known:
            node.outputs[0].shape = ir.Shape(output_dims)
    return model


def _op_types(model: ir.Model) -> list[str]:
    return [n.op_type for n in model.graph]


class ReshapeTransposeReshapeFiresTest(unittest.TestCase):
    """Positive cases: the rewrite must fire and collapse to a single Transpose
    (or Identity, when the derived permutation is trivial), and the result
    must be numerically identical to the original three-op chain.
    """

    def test_simple_grouped_axis(self):
        """[2,15,7] -(split M=3*5)-> [2,3,5,7] -T[3,0,1,2]-> merge -> [7,2,15].

        Axis1 (15) splits into (3,5); the Transpose moves D to the front and
        keeps (3,5) together in order; R2 merges them straight back. Derived
        permutation: [2,0,1].
        """
        model = _build_chain([2, 15, 7], [2, 3, 5, 7], [3, 0, 1, 2], [7, 2, 15])
        count = rtr.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(_op_types(model), ["Transpose"])
        transpose = next(n for n in model.graph if n.op_type == "Transpose")
        self.assertEqual(list(transpose.attributes["perm"].as_ints()), [2, 0, 1])

    def test_multiple_grouped_axes(self):
        """Two original axes split simultaneously: [6,4,10] -> [2,3,4,2,5],
        T moves the (E,F) and (B,C) blocks around the untouched D axis.
        Derived permutation: [2,1,0].
        """
        model = _build_chain([6, 4, 10], [2, 3, 4, 2, 5], [3, 4, 2, 0, 1], [10, 4, 6])
        count = rtr.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(_op_types(model), ["Transpose"])
        transpose = next(n for n in model.graph if n.op_type == "Transpose")
        self.assertEqual(list(transpose.attributes["perm"].as_ints()), [2, 1, 0])

    def test_non_trivial_permutation_no_split(self):
        """R1/R2 are identity reshapes (no group has more than one axis);
        the rule must still recognize the plain Transpose hiding inside.
        """
        model = _build_chain([2, 3, 4], [2, 3, 4], [1, 0, 2], [3, 2, 4])
        count = rtr.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(_op_types(model), ["Transpose"])
        transpose = next(n for n in model.graph if n.op_type == "Transpose")
        self.assertEqual(list(transpose.attributes["perm"].as_ints()), [1, 0, 2])

    def test_identity_chain_becomes_identity(self):
        """When the derived permutation is the identity, emit Identity(x)
        instead of a no-op Transpose, matching TransposeIdentity's convention.
        """
        model = _build_chain([2, 15, 7], [2, 3, 5, 7], [0, 1, 2, 3], [2, 15, 7])
        count = rtr.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(_op_types(model), ["Identity"])

    def test_equal_size_axes_swapped_as_whole_blocks(self):
        """[6,6] with both axes split identically (2,3); repeated top-level
        dimension values must not confuse the grouping walk. Derived
        permutation: [1,0].
        """
        model = _build_chain([6, 6], [2, 3, 2, 3], [2, 3, 0, 1], [6, 6])
        count = rtr.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(_op_types(model), ["Transpose"])
        transpose = next(n for n in model.graph if n.op_type == "Transpose")
        self.assertEqual(list(transpose.attributes["perm"].as_ints()), [1, 0])

    def test_numerical_equivalence_simple_grouped_axis(self):
        """Verify actual numerical equivalence (not just structure) for the
        canonical grouped-axis case, via ONNX Runtime.
        """
        original = _build_chain([2, 15, 7], [2, 3, 5, 7], [3, 0, 1, 2], [7, 2, 15])
        rewritten = _build_chain([2, 15, 7], [2, 3, 5, 7], [3, 0, 1, 2], [7, 2, 15])
        rtr.rules.apply_to_model(rewritten)
        data = np.arange(2 * 15 * 7, dtype=np.float32).reshape(2, 15, 7)
        shape1 = np.array([2, 3, 5, 7], dtype=np.int64)
        shape2 = np.array([7, 2, 15], dtype=np.int64)
        testing.assert_numerically_equal(original, rewritten, (data, shape1, shape2))

    def test_numerical_equivalence_multiple_grouped_axes(self):
        """Verify numerical equivalence for the multi-axis-split case."""
        original = _build_chain([6, 4, 10], [2, 3, 4, 2, 5], [3, 4, 2, 0, 1], [10, 4, 6])
        rewritten = _build_chain([6, 4, 10], [2, 3, 4, 2, 5], [3, 4, 2, 0, 1], [10, 4, 6])
        rtr.rules.apply_to_model(rewritten)
        data = np.arange(6 * 4 * 10, dtype=np.float32).reshape(6, 4, 10)
        shape1 = np.array([2, 3, 4, 2, 5], dtype=np.int64)
        shape2 = np.array([10, 4, 6], dtype=np.int64)
        testing.assert_numerically_equal(original, rewritten, (data, shape1, shape2))


class ReshapeTransposeReshapeDoesNotFireTest(unittest.TestCase):
    """Negative cases: the rewrite must NOT fire, and the graph must be left
    completely unchanged (still the original three ops).
    """

    def _assert_does_not_fire(self, model: ir.Model) -> None:
        before = _op_types(model)
        count = rtr.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(_op_types(model), before)

    def test_r1_crosses_original_axis_boundary(self):
        """[2,6] reshaped to [3,4]: mixes original axis0 and axis1's indices
        into new axes that don't align with either original axis boundary.
        No permutation of the original axes can reproduce this (see the
        design's adversarial-validation writeup for issue #1775).
        """
        model = _build_chain([2, 6], [3, 4], [0, 1], [3, 4])
        self._assert_does_not_fire(model)

    def test_transpose_interleaves_pieces_of_one_original_axis(self):
        """R1 splits axis1 (15=3*5) into (B,C); perm moves A between B and C,
        so B and C are no longer contiguous in the Transpose's output --
        R2 could not even be written as a single Reshape merging them back.
        """
        model = _build_chain([2, 15, 7], [2, 3, 5, 7], [2, 0, 1, 3], [2, 15, 7])
        self._assert_does_not_fire(model)

    def test_transpose_reorders_pieces_internally(self):
        """The classic trap: final shape [2,15,7] is bit-identical to the
        input shape, so a shape-only check would call this a no-op. But the
        Transpose swaps B and C (perm=[0,2,1,3]) before R2 merges them, so
        the merged 15-axis holds C-major data where the input held B-major
        data. Must be rejected purely from the internal-order check, since
        the shapes alone give no signal that anything is wrong.
        """
        model = _build_chain([2, 15, 7], [2, 3, 5, 7], [0, 2, 1, 3], [2, 15, 7])
        self._assert_does_not_fire(model)

    def test_r2_partially_merges_incompatible_groups(self):
        """T's output blocks are (D)(A)(B,C) with sizes (7)(2)(3,5). R2
        instead merges (D,A) together (14) while leaving (B,C) split (3,5)
        -- a real, rank-3, product-matching shape, but not the merge the
        block structure requires.
        """
        model = _build_chain([2, 15, 7], [2, 3, 5, 7], [3, 0, 1, 2], [14, 3, 5])
        self._assert_does_not_fire(model)

    def test_adversarial_same_shape_as_a_valid_case(self):
        """The most dangerous case: final shape (7,2,15) is IDENTICAL to the
        shape produced by the genuinely valid `test_simple_grouped_axis`
        case above. But here perm=[3,0,2,1] reverses B and C's internal
        order while relocating the block, so the merged 15-axis holds
        C-major, not B-major, data. A heuristic that accepts any chain whose
        output shape matches "a known-good pattern" would wrongly fire here;
        this rule correctly rejects it because it tracks block-internal
        order (condition 2), not just the final shape (condition 3).
        This is the case discovered during adversarial validation of the
        #1775 design and is included here specifically to guard against a
        future regression that weakens the internal-order check.
        """
        model = _build_chain([2, 15, 7], [2, 3, 5, 7], [3, 0, 2, 1], [7, 2, 15])
        self._assert_does_not_fire(model)

    def test_unknown_intermediate_shape(self):
        """R1's output shape was never resolved by shape inference (e.g. a
        Shape()-derived, non-constant reshape target with no downstream
        annotation) -- the rule must fail closed rather than guess.
        """
        model = _build_chain(
            [2, 15, 7], [2, 3, 5, 7], [3, 0, 1, 2], [7, 2, 15], r1_shape_known=False
        )
        self._assert_does_not_fire(model)

    def test_symbolic_dim_outside_validated_scope(self):
        """A symbolic dimension participating in the split is out of scope
        for this first, fully-static-only implementation.
        """
        model = _build_chain([2, 15, 7], [2, 3, 5, 7], [3, 0, 1, 2], [7, 2, 15])
        for node in model.graph:
            if node.outputs[0].name == "r1":
                node.outputs[0].shape = ir.Shape([2, "B", 5, 7])
        self._assert_does_not_fire(model)

    def test_unresolved_output_shape(self):
        """The final Reshape's shape input is non-constant and its output
        shape was never resolved -- another realistic "shape information is
        unavailable" scenario, distinct from the intermediate-shape case
        above.
        """
        model = _build_chain(
            [2, 15, 7], [2, 3, 5, 7], [3, 0, 1, 2], [7, 2, 15], output_shape_known=False
        )
        self._assert_does_not_fire(model)

    def test_size_one_axis_ambiguity(self):
        """A provably-valid Transpose exists for this chain (pi=[2,1,0], per
        the brute-force oracle used during design validation), but a size-1
        axis participates in the split, which this conservative first
        implementation explicitly excludes rather than risk an ambiguous
        group-boundary decision. A documented false negative, not a
        correctness bug.
        """
        model = _build_chain([1, 15, 7], [1, 3, 5, 7], [3, 1, 2, 0], [7, 15, 1])
        self._assert_does_not_fire(model)

    def test_product_compatible_but_provenance_incompatible(self):
        """[2,3,4] -> R1 shape [4,6]: same total element count (24 == 24),
        but 4 and 6 do not align with any prefix of the original axes
        (2, 6, 24) at all -- distinct from the rank-2 boundary-crossing case
        above, showing the same guard holds for a higher-rank original
        shape where "the totals match" is the only thing that lines up.
        """
        model = _build_chain([2, 3, 4], [4, 6], [0, 1], [2, 3, 4])
        self._assert_does_not_fire(model)

    def test_untouched_size_one_axis_still_rejected(self):
        """Confirms the blanket size-0/1 rejection is deliberately broad,
        not limited to dimensions that participate in a split or merge.

        Here R1/R2 are identity reshapes (no axis is actually split), and
        the leading size-1 axis is never reordered relative to anything --
        this is, in substance, nothing more than Transpose(x, [0,2,1])
        wrapped in redundant reshapes, and a provably valid permutation
        exists for it. This rule still rejects it, purely because a `1`
        appears somewhere in the shapes, exactly as documented in the
        module docstring: the restriction is "anywhere", not "only where
        actually split/merged". This is a deliberate, accepted false
        negative -- not a bug -- and this test locks in that behavior so a
        future change doesn't silently narrow the guard without a
        corresponding update to the module docstring and the invariant in
        `check()`.
        """
        model = _build_chain([1, 6, 7], [1, 6, 7], [0, 2, 1], [1, 7, 6])
        self._assert_does_not_fire(model)

    def test_malformed_perm_fails_closed_instead_of_raising(self):
        """A Transpose `perm` attribute that is not a valid permutation of
        R1's output axes (here: wrong length) must be rejected by `check()`
        before it is ever used to index into position-keyed structures --
        it must not raise (e.g. KeyError/IndexError) partway through.
        """
        model = _build_chain([2, 15, 7], [2, 3, 5, 7], [3, 0, 1, 2], [7, 2, 15])
        for node in model.graph:
            if node.op_type == "Transpose":
                node.attributes["perm"] = ir.AttrInt64s("perm", [3, 0, 1])  # too short
        count = rtr.rules.apply_to_model(model)  # must not raise
        self.assertEqual(count, 0)


class ReshapeTransposeReshapeInvariantTest(unittest.TestCase):
    """Direct tests of the internal grouping/permutation helpers, covering
    the well-formedness invariant added in `check()` after human review.

    `_no_trivial_dims` makes the scenario below unreachable through the
    public `check()`/pattern-matching surface today (any chain containing a
    size-1 axis is rejected before the helpers below are ever called), so
    it cannot be exercised end-to-end without either an illegal ONNX graph
    or bypassing that guard. These tests call the helpers directly instead,
    to (a) document precisely why the explicit `len(pi) == n` invariant in
    `check()` exists, and (b) keep protecting it as a real invariant -- not
    an incidental one -- if `_no_trivial_dims` is ever relaxed in a future
    change.
    """

    def test_untouched_size_one_axis_yields_an_empty_group(self):
        """With size-1 dims allowed, the greedy grouping walk assigns a
        size-1 original axis ZERO R1-output positions (its running product
        already equals its target before consuming anything), silently
        absorbing the literal size-1 R1 axis into the NEXT group instead.
        """
        original_dims = [2, 1, 7]
        split_dims = [2, 1, 7]  # identity reshape; axis1=1 is never "split"
        groups = rtr._split_into_original_axis_groups(original_dims, split_dims)
        self.assertEqual(groups, [[0], [], [1, 2]])  # group for axis1 is empty

    def test_empty_group_yields_a_permutation_shorter_than_original_rank(self):
        """Continuing from the empty group above: `_derive_block_permutation`
        never visits a group with no positions, so its result is missing an
        entry for that original axis entirely -- shorter than `len(groups)`.
        This is exactly the malformed `pi` the explicit invariant in
        `check()` (`len(pi) == len(groups) == len(original_dims)`) exists
        to catch, rather than relying on it being caught only incidentally
        by a later shape-length mismatch.
        """
        groups = [[0], [], [1, 2]]
        pi = rtr._derive_block_permutation(groups, [0, 1, 2])
        self.assertIsNotNone(pi)
        self.assertNotEqual(len(pi), len(groups))  # 2 != 3: the invariant must reject this
        self.assertEqual(pi, [0, 2])


class ReshapeTransposeReshapeIntegrationTest(unittest.TestCase):
    """End-to-end regression through the REAL default optimizer pipeline
    (not this rule in isolation), covering the integration issue found
    during human review: `MaterializeReshapeShape` runs immediately before
    this rule in `_DEFAULT_REWRITE_RULES` and can replace a Reshape's shape
    input with a concrete Constant, setting `allowzero=1` on the new node.
    This rule must still collapse the resulting chain -- it must not depend
    on `allowzero` being 0.
    """

    def test_materialize_reshape_shape_then_collapse(self):
        """R1's shape input (`shape1`) is a genuine, non-constant graph
        input -- nothing folds it away. R1's output shape is only known
        because the model carries a real ONNX `value_info` entry for it
        (exactly what a prior shape-inference pass or exporter would
        attach), not anything manually poked via IR APIs. Run through the
        public `optimizer.optimize()`, exactly as a real caller would.
        """
        data = oh.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 15, 7])
        shape1 = oh.make_tensor_value_info("shape1", onnx.TensorProto.INT64, [4])
        output = oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [7, 2, 15])
        r1_value_info = oh.make_tensor_value_info("r1", onnx.TensorProto.FLOAT, [2, 3, 5, 7])

        r1 = oh.make_node("Reshape", ["data", "shape1"], ["r1"])
        t = oh.make_node("Transpose", ["r1"], ["t"], perm=[3, 0, 1, 2])
        shape2_const = oh.make_node(
            "Constant",
            [],
            ["shape2"],
            value=oh.make_tensor("shape2_val", onnx.TensorProto.INT64, [3], [7, 2, 15]),
        )
        r2 = oh.make_node("Reshape", ["t", "shape2"], ["output"])

        graph = oh.make_graph(
            [r1, t, shape2_const, r2],
            "g",
            [data, shape1],
            [output],
            value_info=[r1_value_info],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        self.assertEqual(
            [n.op_type for n in model.graph.node],
            ["Reshape", "Transpose", "Constant", "Reshape"],
        )

        optimized = optimizer.optimize(model, num_iterations=2)

        self.assertEqual([n.op_type for n in optimized.graph.node], ["Transpose"])
        onnx.checker.check_model(optimized)

        data_np = np.arange(2 * 15 * 7, dtype=np.float32).reshape(2, 15, 7)
        shape1_np = np.array([2, 3, 5, 7], dtype=np.int64)
        sess_original = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        out_original = sess_original.run(None, {"data": data_np, "shape1": shape1_np})[0]
        sess_optimized = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        out_optimized = sess_optimized.run(None, {"data": data_np, "shape1": shape1_np})[0]
        np.testing.assert_allclose(out_original, out_optimized, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
