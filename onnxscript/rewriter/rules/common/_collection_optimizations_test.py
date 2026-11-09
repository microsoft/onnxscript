# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _collection_optimizations


def _rewrite(text: str) -> tuple[ir.Model, ir.Model, int]:
    original = ir.from_onnx_text(text)
    optimized = ir.from_onnx_text(text)
    count = _collection_optimizations.rules.apply_to_model(optimized)
    return original, optimized, count


def _assert_multi_output_equal(
    original: ir.Model, optimized: ir.Model, feeds: dict[str, np.ndarray]
) -> None:
    expected = ReferenceEvaluator(ir.serde.serialize_model(original)).run(None, feeds)
    actual = ReferenceEvaluator(ir.serde.serialize_model(optimized)).run(None, feeds)
    assert len(expected) == len(actual)
    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(expected_value, actual_value, equal_nan=True)


def test_concat_empty_removes_zero_axis_input():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, 0] empty, float[2, 3] x, float[2, 2] z)
            => (float[2, 5] y) {
          y = Concat <axis = 1> (empty, x, z)
        }
        """
    )
    assert count >= 1
    assert [node.op_type for node in optimized.graph] == ["Concat"]
    assert len(optimized.graph[0].inputs) == 2
    testing.assert_numerically_equal(
        original,
        optimized,
        (
            np.empty((2, 0), dtype=np.float32),
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(4, dtype=np.float32).reshape(2, 2),
        ),
    )


def test_concat_empty_rejects_unknown_dimension():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, N] a, float[2, 3] b) => (float[2, M] y) {
          y = Concat <axis = -1> (a, b)
        }
        """
    )
    assert count == 0
    assert optimized.graph[0].op_type == "Concat"


def test_concat_empty_rejects_removing_every_input():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, 0] a, float[2, 0] b) => (float[2, 0] y) {
          y = Concat <axis = 1> (a, b)
        }
        """
    )
    assert count == 0
    assert optimized.graph[0].op_type == "Concat"


def test_gather_gather_composes_int32_indices_and_preserves_shared_inner():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[8, 2] x) => (float[2, 2] y, float[4, 2] shared) {
          inner_idx = Constant <value = int32[4] {6, 2, 5, 1}> ()
          shared = Gather <axis = 0> (x, inner_idx)
          outer_idx = Constant <value = int64[2] {-1, 0}> ()
          y = Gather <axis = 0> (shared, outer_idx)
        }
        """
    )
    assert count >= 1
    gathers = [node for node in optimized.graph if node.op_type == "Gather"]
    assert len(gathers) == 1
    assert any(node.op_type == "Slice" for node in optimized.graph)
    _assert_multi_output_equal(
        original,
        optimized,
        {"x": np.arange(16, dtype=np.float32).reshape(8, 2)},
    )


def test_gather_gather_matches_default_axis_attributes():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[5] x) => (float[2] y) {
          inner_indices = Constant <value = int64[3] {4, 1, 3}> ()
          inner = Gather (x, inner_indices)
          outer_indices = Constant <value = int64[2] {2, 0}> ()
          y = Gather (inner, outer_indices)
        }
        """
    )
    assert count >= 1
    testing.assert_numerically_equal(original, optimized, (np.arange(5, dtype=np.float32),))


def test_gather_gather_rejects_out_of_bounds_unselected_inner_index():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4] x) => (float[1] y) {
          inner_indices = Constant <value = int64[2] {0, 9}> ()
          inner = Gather (x, inner_indices)
          outer_indices = Constant <value = int64[1] {0}> ()
          y = Gather (inner, outer_indices)
        }
        """
    )
    assert count == 0
    assert len([node for node in optimized.graph if node.op_type == "Gather"]) == 2
    inner = next(
        node
        for node in optimized.graph
        if node.op_type == "Gather" and node.inputs[0].name == "x"
    )
    np.testing.assert_array_equal(inner.inputs[1].const_value.numpy(), np.array([0, 9]))


def test_gather_gather_rejects_symbolic_input_dimension():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[N] x) => (float[1] y) {
          inner_indices = Constant <value = int64[2] {0, 1}> ()
          inner = Gather (x, inner_indices)
          outer_indices = Constant <value = int64[1] {0}> ()
          y = Gather (inner, outer_indices)
        }
        """
    )
    assert count == 0
    assert len([node for node in optimized.graph if node.op_type == "Gather"]) == 2


def test_gather_to_slice_vector_progression():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, 8] x) => (float[2, 3] y) {
          indices = Constant <value = int32[3] {1, 3, 5}> ()
          y = Gather <axis = -1> (x, indices)
        }
        """
    )
    assert count >= 1
    assert optimized.graph[-1].op_type == "Slice"
    testing.assert_numerically_equal(
        original, optimized, (np.arange(16, dtype=np.float32).reshape(2, 8),)
    )


def test_gather_to_slice_rejects_out_of_bounds_positive_index():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4] x) => (float[2] y) {
          indices = Constant <value = int64[2] {3, 5}> ()
          y = Gather (x, indices)
        }
        """
    )
    assert count == 0
    assert optimized.graph[-1].op_type == "Gather"


def test_gather_to_slice_vector_supports_opset_10():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 10]>
        graph (float[5] x) => (float[2] y) {
          indices = Constant <value = int64[2] {1, 3}> ()
          y = Gather (x, indices)
        }
        """
    )
    assert count >= 1
    assert optimized.graph[-1].op_type == "Slice"
    testing.assert_numerically_equal(original, optimized, (np.arange(5, dtype=np.float32),))


def test_gather_to_slice_scalar_rejects_opset_12():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 12]>
        graph (float[5] x) => (float y) {
          index = Constant <value = int64 {1}> ()
          y = Gather (x, index)
        }
        """
    )
    assert count == 0
    assert optimized.graph[-1].op_type == "Gather"


def test_gather_concat_routes_vector_indices_to_one_input():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2] a, float[3] b, float[2] c)
            => (float[2] y, float[7] shared) {
          shared = Concat <axis = 0> (a, b, c)
          indices = Constant <value = int32[2] {2, 4}> ()
          y = Gather <axis = 0> (shared, indices)
        }
        """
    )
    assert count >= 1
    assert any(node.op_type == "Concat" for node in optimized.graph)
    _assert_multi_output_equal(
        original,
        optimized,
        {
            "a": np.array([10, 11], dtype=np.float32),
            "b": np.array([20, 21, 22], dtype=np.float32),
            "c": np.array([30, 31], dtype=np.float32),
        },
    )


def test_gather_concat_rejects_indices_spanning_inputs():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2] a, float[3] b) => (float[2] y) {
          joined = Concat <axis = 0> (a, b)
          indices = Constant <value = int64[2] {1, 2}> ()
          y = Gather <axis = 0> (joined, indices)
        }
        """
    )
    assert count == 0
    assert any(node.op_type == "Concat" for node in optimized.graph)


def test_gather_concat_supports_variadic_concat():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[1] a, float[1] b, float[3] c, float[1] d, float[1] e)
            => (float[2] y) {
          joined = Concat <axis = 0> (a, b, c, d, e)
          indices = Constant <value = int64[2] {2, 4}> ()
          y = Gather (joined, indices)
        }
        """
    )
    assert count >= 1
    assert all(node.op_type != "Concat" for node in optimized.graph)
    testing.assert_numerically_equal(
        original,
        optimized,
        (
            np.array([10], dtype=np.float32),
            np.array([20], dtype=np.float32),
            np.array([30, 31, 32], dtype=np.float32),
            np.array([40], dtype=np.float32),
            np.array([50], dtype=np.float32),
        ),
    )


def test_concat_slice_routes_exact_operand_and_preserves_shared_concat():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2] a, float[3] b, float[1] c)
            => (float[3] y, float[6] shared) {
          shared = Concat <axis = 0> (a, b, c)
          starts = Constant <value = int64[1] {2}> ()
          ends = Constant <value = int64[1] {5}> ()
          axes = Constant <value = int64[1] {0}> ()
          steps = Constant <value = int64[1] {1}> ()
          y = Slice (shared, starts, ends, axes, steps)
        }
        """
    )
    assert count == 1
    assert any(node.op_type == "Concat" for node in optimized.graph)
    assert all(node.op_type != "Slice" for node in optimized.graph)
    _assert_multi_output_equal(
        original,
        optimized,
        {
            "a": np.array([1, 2], dtype=np.float32),
            "b": np.array([3, 4, 5], dtype=np.float32),
            "c": np.array([6], dtype=np.float32),
        },
    )


def test_concat_twice_relu_moves_expensive_work_before_concat():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, 3] x) => (float[2, 6] y) {
          joined = Concat <axis = -1> (x, x)
          y = Relu (joined)
        }
        """
    )
    assert count == 1
    assert [node.op_type for node in optimized.graph] == ["Relu", "Concat"]
    testing.assert_numerically_equal(
        original,
        optimized,
        (np.arange(-3, 3, dtype=np.float32).reshape(2, 3),),
    )


def test_concat_twice_reciprocal_preserves_signed_zero_nan_and_inf():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4] x) => (float[8] y) {
          joined = Concat <axis = 0> (x, x)
          y = Reciprocal (joined)
        }
        """
    )
    assert count == 1
    feeds = {"x": np.array([-0.0, 0.0, np.inf, np.nan], dtype=np.float32)}
    _assert_multi_output_equal(original, optimized, feeds)
    expected = ReferenceEvaluator(ir.serde.serialize_model(original)).run(None, feeds)[0]
    actual = ReferenceEvaluator(ir.serde.serialize_model(optimized)).run(None, feeds)[0]
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_concat_twice_cast_preserves_saturate_attribute():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 19]>
        graph (float[2] x) => (float16[4] y) {
          joined = Concat <axis = 0> (x, x)
          y = Cast <to = 10, saturate = 0> (joined)
        }
        """
    )
    assert count == 1
    cast = next(node for node in optimized.graph if node.op_type == "Cast")
    assert cast.attributes["to"].as_int() == 10
    assert cast.attributes["saturate"].as_int() == 0


def test_concat_twice_cast_preserves_round_mode_attribute():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 29]>
        graph (float[2] x) => (y) {
          joined = Concat <axis = 0> (x, x)
          y = Cast <to = 24, saturate = 1, round_mode = "up"> (joined)
        }
        """
    )
    assert count == 1
    cast = next(node for node in optimized.graph if node.op_type == "Cast")
    assert cast.attributes["to"].as_int() == 24
    assert cast.attributes["saturate"].as_int() == 1
    assert cast.attributes["round_mode"].as_string() == "up"


def test_gather_upstream_add_handles_right_aligned_broadcast():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4, 3] x, float[3] bias) => (float[2, 3] y) {
          added = Add (x, bias)
          indices = Constant <value = int64[2] {3, 1}> ()
          y = Gather <axis = 0> (added, indices)
        }
        """
    )
    assert count >= 1
    op_types = [node.op_type for node in optimized.graph]
    assert op_types.index("Gather") < op_types.index("Add")
    testing.assert_numerically_equal(
        original,
        optimized,
        (
            np.arange(12, dtype=np.float32).reshape(4, 3),
            np.array([10, 20, 30], dtype=np.float32),
        ),
    )


def test_gather_upstream_rejects_shared_producer():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4] x) => (float[2] y, float[4] shared) {
          shared = Exp (x)
          indices = Constant <value = int64[2] {0, 2}> ()
          y = Gather <axis = 0> (shared, indices)
        }
        """
    )
    # The Gather may still become a Slice, but Exp must remain before it.
    assert count >= 1
    assert optimized.graph[0].op_type == "Exp"


def test_gather_upstream_cast_supports_scalar_and_preserves_saturate():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 19]>
        graph (float[3, 2] x) => (float16[2] y) {
          cast = Cast <to = 10, saturate = 0> (x)
          index = Constant <value = int64 {1}> ()
          y = Gather <axis = 0> (cast, index)
        }
        """
    )
    assert count >= 1
    op_types = [node.op_type for node in optimized.graph]
    assert op_types.index("Slice") < op_types.index("Cast")
    cast = next(node for node in optimized.graph if node.op_type == "Cast")
    assert cast.attributes["saturate"].as_int() == 0


def test_gather_upstream_cast_rejects_string_input_that_may_fail_conversion():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (string[2] x) => (float[1] y) {
          cast = Cast <to = 1> (x)
          index = Constant <value = int64[1] {0}> ()
          y = Gather <axis = 0> (cast, index)
        }
        """
    )
    assert count == 0
    assert [node.op_type for node in optimized.graph if node.op_type != "Constant"] == [
        "Cast",
        "Gather",
    ]


def test_gather_upstream_unary_supports_scalar_indices():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[3, 2] x) => (float[2] y) {
          transformed = Exp (x)
          index = Constant <value = int64 {1}> ()
          y = Gather <axis = 0> (transformed, index)
        }
        """
    )
    assert count >= 1
    op_types = [node.op_type for node in optimized.graph]
    assert op_types.index("Slice") < op_types.index("Exp")
    testing.assert_numerically_equal(
        original,
        optimized,
        (np.arange(6, dtype=np.float32).reshape(3, 2),),
    )


def test_gather_upstream_unary_supports_default_axis():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[3, 2] x) => (float[2, 2] y) {
          transformed = Tanh (x)
          indices = Constant <value = int64[2] {2, 0}> ()
          y = Gather (transformed, indices)
        }
        """
    )
    assert count == 1
    assert [node.op_type for node in optimized.graph][-2:] == ["Gather", "Tanh"]
    testing.assert_numerically_equal(
        original,
        optimized,
        (np.arange(-3, 3, dtype=np.float32).reshape(3, 2),),
    )


def test_gathers_split_scalar_partition():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, 3] x) => (float[3] y0, float[3] y1) {
          i1 = Constant <value = int64 {-1}> ()
          y0 = Gather <axis = 0> (x, i1)
          i0 = Constant <value = int32 {0}> ()
          y1 = Gather <axis = 0> (x, i0)
        }
        """
    )
    assert count >= 1
    assert len([node for node in optimized.graph if node.op_type == "Split"]) == 1
    assert len([node for node in optimized.graph if node.op_type == "Squeeze"]) == 2
    _assert_multi_output_equal(
        original,
        optimized,
        {"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
    )


def test_gathers_split_rejects_opset_12_input_form_split():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 12]>
        graph (float[2] x) => (float[1] y0, float[1] y1) {
          i0 = Constant <value = int64[1] {0}> ()
          y0 = Gather <axis = 0> (x, i0)
          i1 = Constant <value = int64[1] {1}> ()
          y1 = Gather <axis = 0> (x, i1)
        }
        """
    )
    assert count == 2
    assert all(node.op_type != "Split" for node in optimized.graph)


def test_gathers_split_three_way_permuted_partition():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[3, 2] x) => (float[1, 2] y0, float[1, 2] y1, float[1, 2] y2) {
          i2 = Constant <value = int64[1] {-1}> ()
          y0 = Gather (x, i2)
          i0 = Constant <value = int32[1] {0}> ()
          y1 = Gather (x, i0)
          i1 = Constant <value = int64[1] {1}> ()
          y2 = Gather (x, i1)
        }
        """
    )
    assert count >= 1
    assert len([node for node in optimized.graph if node.op_type == "Split"]) == 1
    _assert_multi_output_equal(
        original, optimized, {"x": np.arange(6, dtype=np.float32).reshape(3, 2)}
    )


def test_gathers_split_four_way_partition():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4] x) => (float y0, float y1, float y2, float y3) {
          i0 = Constant <value = int64 {0}> ()
          y0 = Gather <axis = 0> (x, i0)
          i1 = Constant <value = int64 {1}> ()
          y1 = Gather <axis = 0> (x, i1)
          i2 = Constant <value = int64 {2}> ()
          y2 = Gather <axis = 0> (x, i2)
          i3 = Constant <value = int64 {-1}> ()
          y3 = Gather <axis = 0> (x, i3)
        }
        """
    )
    assert count == 1
    assert len([node for node in optimized.graph if node.op_type == "Squeeze"]) == 4
    _assert_multi_output_equal(original, optimized, {"x": np.arange(4, dtype=np.float32)})


def test_gathers_split_three_rejects_duplicate_index():
    _, optimized, _ = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[3] x) => (float[1] y0, float[1] y1, float[1] y2) {
          i0 = Constant <value = int64[1] {0}> ()
          y0 = Gather <axis = 0> (x, i0)
          i1 = Constant <value = int64[1] {1}> ()
          y1 = Gather <axis = 0> (x, i1)
          duplicate = Constant <value = int64[1] {1}> ()
          y2 = Gather <axis = 0> (x, duplicate)
        }
        """
    )
    assert all(node.op_type != "Split" for node in optimized.graph)


def test_slices_split_complete_partition():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[5] x) => (float[3] y0, float[2] y1) {
          a = Constant <value = int64[1] {0}> ()
          one = Constant <value = int64[1] {1}> ()
          two = Constant <value = int64[1] {2}> ()
          five = Constant <value = int64[1] {5}> ()
          y0 = Slice (x, two, five, a, one)
          y1 = Slice (x, a, two, a, one)
        }
        """
    )
    assert count == 1
    assert len([node for node in optimized.graph if node.op_type == "Split"]) == 1
    assert all(node.op_type != "Slice" for node in optimized.graph)
    _assert_multi_output_equal(
        original,
        optimized,
        {"x": np.arange(5, dtype=np.float32)},
    )


def test_slices_split_rejects_opset_12_input_form_split():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 12]>
        graph (float[4] x) => (float[2] y0, float[2] y1) {
          axis = Constant <value = int64[1] {0}> ()
          step = Constant <value = int64[1] {1}> ()
          zero = Constant <value = int64[1] {0}> ()
          two = Constant <value = int64[1] {2}> ()
          four = Constant <value = int64[1] {4}> ()
          y0 = Slice (x, zero, two, axis, step)
          y1 = Slice (x, two, four, axis, step)
        }
        """
    )
    assert count == 0
    assert len([node for node in optimized.graph if node.op_type == "Slice"]) == 2


def test_slices_split_three_way_permuted_partition():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[6] x) => (float[2] y0, float[2] y1, float[2] y2) {
          axis = Constant <value = int64[1] {0}> ()
          step = Constant <value = int64[1] {1}> ()
          zero = Constant <value = int64[1] {0}> ()
          two = Constant <value = int64[1] {2}> ()
          four = Constant <value = int64[1] {4}> ()
          six = Constant <value = int64[1] {6}> ()
          y0 = Slice (x, four, six, axis, step)
          y1 = Slice (x, zero, two, axis, step)
          y2 = Slice (x, two, four, axis, step)
        }
        """
    )
    assert count == 1
    assert len([node for node in optimized.graph if node.op_type == "Split"]) == 1
    _assert_multi_output_equal(original, optimized, {"x": np.arange(6, dtype=np.float32)})


def test_slices_split_four_way_partition():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[8] x) => (float[2] y0, float[2] y1, float[2] y2, float[2] y3) {
          axis = Constant <value = int64[1] {0}> ()
          step = Constant <value = int64[1] {1}> ()
          zero = Constant <value = int64[1] {0}> ()
          two = Constant <value = int64[1] {2}> ()
          four = Constant <value = int64[1] {4}> ()
          six = Constant <value = int64[1] {6}> ()
          eight = Constant <value = int64[1] {8}> ()
          y0 = Slice (x, zero, two, axis, step)
          y1 = Slice (x, two, four, axis, step)
          y2 = Slice (x, four, six, axis, step)
          y3 = Slice (x, six, eight, axis, step)
        }
        """
    )
    assert count == 1
    assert len([node for node in optimized.graph if node.op_type == "Split"]) == 1
    _assert_multi_output_equal(original, optimized, {"x": np.arange(8, dtype=np.float32)})


def test_slices_split_three_rejects_overlap():
    _, optimized, _ = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[6] x) => (float[3] y0, float[3] y1, float[1] y2) {
          axis = Constant <value = int64[1] {0}> ()
          step = Constant <value = int64[1] {1}> ()
          zero = Constant <value = int64[1] {0}> ()
          three = Constant <value = int64[1] {3}> ()
          two = Constant <value = int64[1] {2}> ()
          five = Constant <value = int64[1] {5}> ()
          six = Constant <value = int64[1] {6}> ()
          y0 = Slice (x, zero, three, axis, step)
          y1 = Slice (x, two, five, axis, step)
          y2 = Slice (x, five, six, axis, step)
        }
        """
    )
    assert all(node.op_type != "Split" for node in optimized.graph)


def test_gather_slice_to_split_complete_partition():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4, 2] x) => (float[2] y0, float[3, 2] y1) {
          index = Constant <value = int64 {0}> ()
          y0 = Gather <axis = 0> (x, index)
          start = Constant <value = int64[1] {1}> ()
          end = Constant <value = int64[1] {4}> ()
          axis = Constant <value = int64[1] {0}> ()
          step = Constant <value = int64[1] {1}> ()
          y1 = Slice (x, start, end, axis, step)
        }
        """
    )
    assert count == 1
    assert len([node for node in optimized.graph if node.op_type == "Split"]) == 1
    assert len([node for node in optimized.graph if node.op_type == "Squeeze"]) == 1
    _assert_multi_output_equal(
        original,
        optimized,
        {"x": np.arange(8, dtype=np.float32).reshape(4, 2)},
    )


def test_gather_slices_to_split_three_way_partition():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[5] x) => (float y0, float[2] y1, float[2] y2) {
          index = Constant <value = int64 {0}> ()
          y0 = Gather <axis = 0> (x, index)
          axis = Constant <value = int64[1] {0}> ()
          step = Constant <value = int64[1] {1}> ()
          one = Constant <value = int64[1] {1}> ()
          three = Constant <value = int64[1] {3}> ()
          five = Constant <value = int64[1] {5}> ()
          y1 = Slice (x, one, three, axis, step)
          y2 = Slice (x, three, five, axis, step)
        }
        """
    )
    assert count >= 1
    assert len([node for node in optimized.graph if node.op_type == "Split"]) == 1
    assert all(node.op_type not in {"Gather", "Slice"} for node in optimized.graph)
    _assert_multi_output_equal(
        original,
        optimized,
        {"x": np.arange(5, dtype=np.float32)},
    )


def test_slice_concat_to_space_to_depth_for_single_channel():
    original, optimized, count = _rewrite(
        f"""
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[1, 1, 4, 6] x) => (float[1, 4, 2, 3] y) {{
          axes = Constant <value = int64[2] {{2, 3}}> ()
          ends = Constant <value = int64[2] {{{_collection_optimizations._INT64_MAX}, {_collection_optimizations._INT64_MAX}}}> ()
          steps = Constant <value = int64[2] {{2, 2}}> ()
          p00 = Constant <value = int64[2] {{0, 0}}> ()
          p01 = Constant <value = int64[2] {{0, 1}}> ()
          p10 = Constant <value = int64[2] {{1, 0}}> ()
          p11 = Constant <value = int64[2] {{1, 1}}> ()
          s0 = Slice (x, p00, ends, axes, steps)
          s1 = Slice (x, p01, ends, axes, steps)
          s2 = Slice (x, p10, ends, axes, steps)
          s3 = Slice (x, p11, ends, axes, steps)
          y = Concat <axis = 1> (s0, s1, s2, s3)
        }}
        """
    )
    assert count == 1
    assert [node.op_type for node in optimized.graph] == ["SpaceToDepth"]
    testing.assert_numerically_equal(
        original,
        optimized,
        (np.arange(24, dtype=np.float32).reshape(1, 1, 4, 6),),
    )


def test_slice_concat_to_space_to_depth_rejects_multiple_channels():
    _, optimized, count = _rewrite(
        f"""
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[1, 2, 4, 4] x) => (float[1, 8, 2, 2] y) {{
          axes = Constant <value = int64[2] {{2, 3}}> ()
          ends = Constant <value = int64[2] {{{_collection_optimizations._INT64_MAX}, {_collection_optimizations._INT64_MAX}}}> ()
          steps = Constant <value = int64[2] {{2, 2}}> ()
          p00 = Constant <value = int64[2] {{0, 0}}> ()
          p01 = Constant <value = int64[2] {{0, 1}}> ()
          p10 = Constant <value = int64[2] {{1, 0}}> ()
          p11 = Constant <value = int64[2] {{1, 1}}> ()
          s0 = Slice (x, p00, ends, axes, steps)
          s1 = Slice (x, p01, ends, axes, steps)
          s2 = Slice (x, p10, ends, axes, steps)
          s3 = Slice (x, p11, ends, axes, steps)
          y = Concat <axis = 1> (s0, s1, s2, s3)
        }}
        """
    )
    assert count == 0
    assert optimized.graph[-1].op_type == "Concat"


def test_gather_to_slice_scalar_squeezes():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, 4, 3] x) => (float[2, 3] y) {
          index = Constant <value = int64 {-1}> ()
          y = Gather <axis = 1> (x, index)
        }
        """
    )
    assert count == 1
    op_types = [node.op_type for node in optimized.graph]
    assert "Slice" in op_types
    assert op_types[-1] == "Squeeze"
    testing.assert_numerically_equal(
        original, optimized, (np.arange(24, dtype=np.float32).reshape(2, 4, 3),)
    )


def test_gather_to_slice_matches_default_axis_attribute():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[6, 2] x) => (float[3, 2] y) {
          indices = Constant <value = int64[3] {0, 2, 4}> ()
          y = Gather (x, indices)
        }
        """
    )
    assert count == 1
    assert optimized.graph[-1].op_type == "Slice"
    testing.assert_numerically_equal(
        original,
        optimized,
        (np.arange(12, dtype=np.float32).reshape(6, 2),),
    )


def test_gather_to_slice_rejects_non_progression():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[8] x) => (float[3] y) {
          indices = Constant <value = int64[3] {0, 2, 5}> ()
          y = Gather <axis = 0> (x, indices)
        }
        """
    )
    assert count == 0
    assert optimized.graph[-1].op_type == "Gather"


def test_slice_slice_fuses_disjoint_negative_axis():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[5, 6] x) => (float[3, 3] y) {
          s0 = Constant <value = int64[1] {1}> ()
          e0 = Constant <value = int64[1] {4}> ()
          a0 = Constant <value = int64[1] {0}> ()
          st0 = Constant <value = int64[1] {1}> ()
          t = Slice (x, s0, e0, a0, st0)
          s1 = Constant <value = int64[1] {0}> ()
          e1 = Constant <value = int64[1] {6}> ()
          a1 = Constant <value = int64[1] {-1}> ()
          st1 = Constant <value = int64[1] {2}> ()
          y = Slice (t, s1, e1, a1, st1)
        }
        """
    )
    assert count == 1
    assert len([node for node in optimized.graph if node.op_type == "Slice"]) == 1
    testing.assert_numerically_equal(
        original, optimized, (np.arange(30, dtype=np.float32).reshape(5, 6),)
    )


def test_slice_slice_rejects_same_axis():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[8] x) => (float[2] y) {
          s0 = Constant <value = int64[1] {1}> ()
          e0 = Constant <value = int64[1] {7}> ()
          a0 = Constant <value = int64[1] {0}> ()
          st0 = Constant <value = int64[1] {1}> ()
          t = Slice (x, s0, e0, a0, st0)
          s1 = Constant <value = int64[1] {1}> ()
          e1 = Constant <value = int64[1] {3}> ()
          y = Slice (t, s1, e1, a0, st0)
        }
        """
    )
    assert count == 0
    assert len([node for node in optimized.graph if node.op_type == "Slice"]) == 2


def test_split_concat_eliminates_exact_rejoin():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, 6] x) => (float[2, 6] y) {
          sizes = Constant <value = int64[3] {1, 2, 3}> ()
          a, b, c = Split <axis = -1> (x, sizes)
          y = Concat <axis = 1> (a, b, c)
        }
        """
    )
    assert count == 1
    assert [node.op_type for node in optimized.graph] == ["Identity"]
    testing.assert_numerically_equal(
        original, optimized, (np.arange(12, dtype=np.float32).reshape(2, 6),)
    )


def test_split_concat_matches_default_split_axis():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4, 2] x) => (float[4, 2] y) {
          sizes = Constant <value = int64[2] {1, 3}> ()
          a, b = Split (x, sizes)
          y = Concat <axis = 0> (a, b)
        }
        """
    )
    assert count == 1
    assert optimized.graph[0].op_type == "Identity"
    testing.assert_numerically_equal(
        original,
        optimized,
        (np.arange(8, dtype=np.float32).reshape(4, 2),),
    )


def test_split_concat_rejects_shared_output():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[4] x) => (float[4] y, float[2] shared) {
          sizes = Constant <value = int64[2] {2, 2}> ()
          a, shared = Split <axis = 0> (x, sizes)
          y = Concat <axis = 0> (a, shared)
        }
        """
    )
    assert count == 0
    assert any(node.op_type == "Split" for node in optimized.graph)


def test_sequence_construct_at_eliminates_complete_permuted_reads():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2] x, float[2] z) => (float[2] y0, float[2] y1) {
          sequence = SequenceConstruct (x, z)
          i1 = Constant <value = int64 {1}> ()
          y0 = SequenceAt (sequence, i1)
          i0 = Constant <value = int64 {0}> ()
          y1 = SequenceAt (sequence, i0)
        }
        """
    )
    assert count == 1
    assert all(
        node.op_type not in {"SequenceConstruct", "SequenceAt"} for node in optimized.graph
    )
    testing.assert_numerically_equal(
        original,
        optimized,
        (np.array([1, 2], dtype=np.float32), np.array([3, 4], dtype=np.float32)),
    )


def test_sequence_construct_at_rejects_partial_reads():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2] x, float[2] z) => (float[2] y) {
          sequence = SequenceConstruct (x, z)
          i0 = Constant <value = int64 {0}> ()
          y = SequenceAt (sequence, i0)
        }
        """
    )
    assert count == 0
    assert any(node.op_type == "SequenceConstruct" for node in optimized.graph)


def test_sequence_construct_at_eliminates_three_way_permutation():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[1] a, float[1] b, float[1] c)
            => (float[1] y0, float[1] y1, float[1] y2) {
          sequence = SequenceConstruct (a, b, c)
          i2 = Constant <value = int64 {-1}> ()
          y0 = SequenceAt (sequence, i2)
          i0 = Constant <value = int64 {0}> ()
          y1 = SequenceAt (sequence, i0)
          i1 = Constant <value = int64 {1}> ()
          y2 = SequenceAt (sequence, i1)
        }
        """
    )
    assert count == 1
    assert all(
        node.op_type not in {"SequenceConstruct", "SequenceAt"} for node in optimized.graph
    )
    _assert_multi_output_equal(
        original,
        optimized,
        {
            "a": np.array([1], dtype=np.float32),
            "b": np.array([2], dtype=np.float32),
            "c": np.array([3], dtype=np.float32),
        },
    )


def test_sequence_construct_at_eliminates_four_way_reads():
    original, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[1] a, float[1] b, float[1] c, float[1] d)
            => (float[1] y0, float[1] y1, float[1] y2, float[1] y3) {
          sequence = SequenceConstruct (a, b, c, d)
          i0 = Constant <value = int64 {0}> ()
          y0 = SequenceAt (sequence, i0)
          i1 = Constant <value = int64 {1}> ()
          y1 = SequenceAt (sequence, i1)
          i2 = Constant <value = int64 {2}> ()
          y2 = SequenceAt (sequence, i2)
          i3 = Constant <value = int64 {3}> ()
          y3 = SequenceAt (sequence, i3)
        }
        """
    )
    assert count == 1
    assert all(
        node.op_type not in {"SequenceConstruct", "SequenceAt"} for node in optimized.graph
    )
    _assert_multi_output_equal(
        original,
        optimized,
        {
            "a": np.array([1], dtype=np.float32),
            "b": np.array([2], dtype=np.float32),
            "c": np.array([3], dtype=np.float32),
            "d": np.array([4], dtype=np.float32),
        },
    )


def test_sequence_construct_at_three_rejects_duplicate_reads():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[1] a, float[1] b, float[1] c)
            => (float[1] y0, float[1] y1, float[1] y2) {
          sequence = SequenceConstruct (a, b, c)
          i0 = Constant <value = int64 {0}> ()
          y0 = SequenceAt (sequence, i0)
          i1 = Constant <value = int64 {1}> ()
          y1 = SequenceAt (sequence, i1)
          duplicate = Constant <value = int64 {1}> ()
          y2 = SequenceAt (sequence, duplicate)
        }
        """
    )
    assert count == 0
    assert any(node.op_type == "SequenceConstruct" for node in optimized.graph)


def test_rules_reach_fixed_point_without_reapplying():
    _, optimized, count = _rewrite(
        """
        <ir_version: 10, opset_import: ["" : 18]>
        graph (float[2, 3] x) => (float[2, 6] y) {
          joined = Concat <axis = 1> (x, x)
          y = Tanh (joined)
        }
        """
    )
    assert count == 1
    assert _collection_optimizations.rules.apply_to_model(optimized) == 0
