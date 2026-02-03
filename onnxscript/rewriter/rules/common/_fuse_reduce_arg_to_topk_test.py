# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.passes.common import onnx_checker, shape_inference
from parameterized import parameterized

from onnxscript.rewriter import MatchingTracer, MatchStatus, testing
from onnxscript.rewriter.rules.common._fuse_reduce_arg_to_topk import (
    reduce_max_argmax_to_topk_rule,
    reduce_min_argmin_to_topk_rule,
    rules,
)


class FuseReduceArgToTopKTestBase(unittest.TestCase):
    @property
    def rng(self):
        return np.random.default_rng(20260127)

    def clone_model(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    def run_test(
        self,
        base_model: ir.Model,
        expected_op_types: list[str],
    ):
        onnx_checker.CheckerPass(True)(base_model)
        base_model = shape_inference.infer_shapes(base_model)
        updated_model = self.clone_model(base_model)
        count = rules.apply_to_model(updated_model)

        # Check that the rule was applied
        self.assertGreater(count, 0)

        # Check expected op_types
        self.assertEqual([node.op_type for node in updated_model.graph], expected_op_types)

        # Check inference
        inputs = (
            self.rng.uniform(
                low=-10.0,
                high=10.0,
                size=(2, *updated_model.graph.inputs[0].shape[1:]),
            ).astype(np.float32),
        )

        testing.assert_numerically_equal(
            base_model,
            updated_model,
            inputs,
        )

        # Validate serialized model
        output_model_proto = ir.serde.serialize_model(updated_model)
        onnx.checker.check_model(output_model_proto, full_check=True)

    def run_failed_condition_test(
        self,
        base_model: ir.Model,
        rule,
        expected_message: str,
    ):
        onnx_checker.CheckerPass(True)(base_model)

        updated_model = self.clone_model(base_model)
        tracer = MatchingTracer()
        count = rule.apply_to_model(updated_model, tracer=tracer)

        # Check that the model is unchanged
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[rule][0]
        self.assertEqual(tracer_match.status.value, MatchStatus.CONDITION_FAILED)
        self.assertRegex(tracer_match.match_result.reason, expected_message)


class TestFuseReduceMaxArgMaxToTopK(FuseReduceArgToTopKTestBase):
    @parameterized.expand(
        [
            ("keepdims_1_axis_1", 1, 1),
            ("keepdims_1_axis_2", 1, 2),
            ("keepdims_1_axis_neg1", 1, -1),
            ("keepdims_0_axis_1", 0, 1),
            ("keepdims_0_axis_2", 0, 2),
            ("keepdims_0_axis_neg1", 0, -1),
        ]
    )
    def test_successful_fuse_reduce_argmax_to_topk(self, _, keepdims, axis):
        """Test fusion of ReduceMax + ArgMax into TopK with various keepdims and axis values."""
        # When keepdims=0, the output rank is reduced by 1
        if keepdims == 0:
            output_shape_str = "[N, ?, ?]"
        else:
            output_shape_str = "[N, ?, ?, ?]"

        # Test with opset 13 (axes as attribute)
        base_model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float{output_shape_str} max_val, int64{output_shape_str} max_idx)
            {{
                max_val = ReduceMax<axes=[{axis}], keepdims={keepdims}>(X)
                max_idx = ArgMax<axis={axis}, keepdims={keepdims}>(X)
            }}
        """)

        # Expected: Constant for K, TopK, possibly (Constant + Squeeze) x2 for keepdims=0
        if keepdims == 0:
            expected_op_types = ["Constant", "TopK", "Constant", "Squeeze", "Squeeze"]
        else:
            expected_op_types = ["Constant", "TopK"]

        self.run_test(base_model, expected_op_types)

    @parameterized.expand(
        [
            ("keepdims_1_axis_1", 1, 1),
            ("keepdims_0_axis_2", 0, 2),
        ]
    )
    def test_successful_fuse_reduce_argmax_to_topk_opset18(self, _, keepdims, axis):
        """Test fusion with opset 18+ (axes as input)."""
        if keepdims == 0:
            output_shape_str = "[N, ?, ?]"
        else:
            output_shape_str = "[N, ?, ?, ?]"

        # In opset 18+, axes must be passed as the second input to ReduceMax
        base_model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 18] >
            test_model (float[N, 32, 14, 17] X) => (float{output_shape_str} max_val, int64{output_shape_str} max_idx)
            <int64[1] axes = {{{axis}}}>
            {{
                max_val = ReduceMax<keepdims={keepdims}>(X, axes)
                max_idx = ArgMax<axis={axis}, keepdims={keepdims}>(X)
            }}
        """)

        # Expected: Constant for K, TopK, possibly (Constant + Squeeze) x2 for keepdims=0
        if keepdims == 0:
            expected_op_types = ["Constant", "TopK", "Constant", "Squeeze", "Squeeze"]
        else:
            expected_op_types = ["Constant", "TopK"]

        self.run_test(base_model, expected_op_types)

    def test_fuse_reduce_argmax_explicit_axis_0(self):
        """Test fusion with explicit axis=0."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 14, 17] X) => (float[1, 14, 17] max_val, int64[1, 14, 17] max_idx)
            {
                max_val = ReduceMax<axes=[0], keepdims=1>(X)
                max_idx = ArgMax<axis=0, keepdims=1>(X)
            }
        """)

        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_successful_fuse_reduce_argmax_mixed_negative_positive_axes(self):
        """Test fusion when ReduceMax uses negative axis and ArgMax uses positive axis.

        Input shape is [N, 32, 14, 17], rank is 4. Axis -1 is equivalent to axis 3.
        The rule should normalize both axes before comparison.
        """
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, 32, 14, 1] max_val, int64[N, 32, 14, 1] max_idx)
            {
                max_val = ReduceMax<axes=[-1], keepdims=1>(X)
                max_idx = ArgMax<axis=3, keepdims=1>(X)
            }
        """)
        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_fail_keepdims_mismatch(self):
        """Test that fusion fails when keepdims values don't match."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] max_val, int64[N, ?, ?] max_idx)
            {
                max_val = ReduceMax<axes=[1], keepdims=1>(X)
                max_idx = ArgMax<axis=1, keepdims=0>(X)
            }
        """)

        self.run_failed_condition_test(
            base_model, reduce_max_argmax_to_topk_rule, "keepdims mismatch"
        )

    def test_fail_axis_mismatch(self):
        """Test that fusion fails when axes don't match."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] max_val, int64[N, ?, ?, ?] max_idx)
            {
                max_val = ReduceMax<axes=[1], keepdims=1>(X)
                max_idx = ArgMax<axis=2, keepdims=1>(X)
            }
        """)

        self.run_failed_condition_test(
            base_model, reduce_max_argmax_to_topk_rule, "Axis mismatch"
        )

    def test_fail_multiple_axes_reduce_max(self):
        """Test that fusion fails when ReduceMax operates on multiple axes."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] max_val, int64[N, ?, ?, ?] max_idx)
            {
                max_val = ReduceMax<axes=[1, 2], keepdims=1>(X)
                max_idx = ArgMax<axis=1, keepdims=1>(X)
            }
        """)

        self.run_failed_condition_test(
            base_model,
            reduce_max_argmax_to_topk_rule,
            "ReduceMax must operate on a single axis",
        )

    def test_fail_select_last_index_argmax(self):
        """Test that fusion fails when ArgMax has select_last_index=1."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] max_val, int64[N, ?, ?, ?] max_idx)
            {
                max_val = ReduceMax<axes=[1], keepdims=1>(X)
                max_idx = ArgMax<axis=1, keepdims=1, select_last_index=1>(X)
            }
        """)

        self.run_failed_condition_test(
            base_model,
            reduce_max_argmax_to_topk_rule,
            "ArgMax has select_last_index=1, which is not supported by TopK.",
        )

    def test_successful_fuse_with_default_keepdims(self):
        """Test fusion with default keepdims (should be 1)."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] max_val, int64[N, ?, ?, ?] max_idx)
            {
                max_val = ReduceMax<axes=[1]>(X)
                max_idx = ArgMax<axis=1>(X)
            }
        """)

        # Both should use default keepdims=1, so fusion should succeed
        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_successful_fuse_with_default_axis(self):
        """Test fusion with default axis (should be 0)."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 14, 17] X) => (float[1, 14, 17] max_val, int64[1, 14, 17] max_idx)
            {
                max_val = ReduceMax<axes=[0], keepdims=1>(X)
                max_idx = ArgMax<keepdims=1>(X)
            }
        """)

        # ArgMax should use default axis=0, so fusion should succeed
        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_successful_fuse_with_all_defaults(self):
        """Test fusion with all default values (keepdims=1, axis=0)."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 14, 17] X) => (float[1, 14, 17] max_val, int64[1, 14, 17] max_idx)
            {
                max_val = ReduceMax<axes=[0]>(X)
                max_idx = ArgMax(X)
            }
        """)

        # Both should use defaults: keepdims=1, axis=0
        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_no_fusion_different_inputs(self):
        """Test that fusion doesn't happen when nodes have different inputs."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X, float[N, 32, 14, 17] Y) => (float[N, ?, ?, ?] max_val, int64[N, ?, ?, ?] max_idx)
            {
                max_val = ReduceMax<axes=[1], keepdims=1>(X)
                max_idx = ArgMax<axis=1, keepdims=1>(Y)
            }
        """)

        # Pattern won't match at all because inputs are different
        updated_model = self.clone_model(base_model)
        count = rules.apply_to_model(updated_model)
        self.assertEqual(count, 0)

        # Model should be unchanged
        self.assertEqual(
            [node.op_type for node in base_model.graph],
            [node.op_type for node in updated_model.graph],
        )


class TestFuseReduceMinArgMinToTopK(FuseReduceArgToTopKTestBase):
    """Test cases for ReduceMin + ArgMin → TopK(largest=0) fusion."""

    @parameterized.expand(
        [
            ("keepdims_1_axis_1", 1, 1),
            ("keepdims_1_axis_2", 1, 2),
            ("keepdims_1_axis_neg1", 1, -1),
            ("keepdims_0_axis_1", 0, 1),
            ("keepdims_0_axis_2", 0, 2),
            ("keepdims_0_axis_neg1", 0, -1),
        ]
    )
    def test_successful_fuse_reduce_argmin_to_topk(self, _, keepdims, axis):
        """Test fusion of ReduceMin + ArgMin into TopK with various keepdims and axis values."""
        if keepdims == 0:
            output_shape_str = "[N, ?, ?]"
        else:
            output_shape_str = "[N, ?, ?, ?]"

        base_model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float{output_shape_str} min_val, int64{output_shape_str} min_idx)
            {{
                min_val = ReduceMin<axes=[{axis}], keepdims={keepdims}>(X)
                min_idx = ArgMin<axis={axis}, keepdims={keepdims}>(X)
            }}
        """)

        # Expected: Constant for K, TopK, possibly (Constant + Squeeze) x2 for keepdims=0
        if keepdims == 0:
            expected_op_types = ["Constant", "TopK", "Constant", "Squeeze", "Squeeze"]
        else:
            expected_op_types = ["Constant", "TopK"]

        self.run_test(base_model, expected_op_types)

    @parameterized.expand(
        [
            ("keepdims_1_axis_1", 1, 1),
            ("keepdims_0_axis_2", 0, 2),
        ]
    )
    def test_successful_fuse_reduce_argmin_to_topk_opset18(self, _, keepdims, axis):
        """Test fusion with opset 18+ (axes as input) for Min operations."""
        if keepdims == 0:
            output_shape_str = "[N, ?, ?]"
        else:
            output_shape_str = "[N, ?, ?, ?]"

        base_model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 18] >
            test_model (float[N, 32, 14, 17] X) => (float{output_shape_str} min_val, int64{output_shape_str} min_idx)
            <int64[1] axes = {{{axis}}}>
            {{
                min_val = ReduceMin<keepdims={keepdims}>(X, axes)
                min_idx = ArgMin<axis={axis}, keepdims={keepdims}>(X)
            }}
        """)

        if keepdims == 0:
            expected_op_types = ["Constant", "TopK", "Constant", "Squeeze", "Squeeze"]
        else:
            expected_op_types = ["Constant", "TopK"]

        self.run_test(base_model, expected_op_types)

    def test_fuse_reduce_argmin_explicit_axis_0(self):
        """Test fusion with explicit axis=0."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 14, 17] X) => (float[1, 14, 17] min_val, int64[1, 14, 17] min_idx)
            {
                min_val = ReduceMin<axes=[0], keepdims=1>(X)
                min_idx = ArgMin<axis=0, keepdims=1>(X)
            }
        """)

        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_successful_fuse_reduce_argmin_mixed_axes(self):
        """Test fusion with mixed negative/positive axes for Min operations.

        Axis -2 is equivalent to axis 2 for rank-4 tensors.
        """
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, 32, 1, 17] min_val, int64[N, 32, 1, 17] min_idx)
            {
                min_val = ReduceMin<axes=[-2], keepdims=1>(X)
                min_idx = ArgMin<axis=2, keepdims=1>(X)
            }
        """)
        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_fail_axis_mismatch(self):
        """Test that fusion fails when axes don't match for Min operations."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] min_val, int64[N, ?, ?, ?] min_idx)
            {
                min_val = ReduceMin<axes=[1], keepdims=1>(X)
                min_idx = ArgMin<axis=2, keepdims=1>(X)
            }
        """)
        self.run_failed_condition_test(
            base_model, reduce_min_argmin_to_topk_rule, "Axis mismatch"
        )

    def test_fail_keepdims_mismatch(self):
        """Test that fusion fails when keepdims values don't match for Min operations."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] min_val, int64[N, ?, ?] min_idx)
            {
                min_val = ReduceMin<axes=[1], keepdims=1>(X)
                min_idx = ArgMin<axis=1, keepdims=0>(X)
            }
        """)
        self.run_failed_condition_test(
            base_model, reduce_min_argmin_to_topk_rule, "keepdims mismatch"
        )

    def test_fail_multiple_axes_reduce_min(self):
        """Test that fusion fails when ReduceMin operates on multiple axes."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] min_val, int64[N, ?, ?, ?] min_idx)
            {
                min_val = ReduceMin<axes=[1, 2], keepdims=1>(X)
                min_idx = ArgMin<axis=1, keepdims=1>(X)
            }
        """)

        self.run_failed_condition_test(
            base_model,
            reduce_min_argmin_to_topk_rule,
            "ReduceMin must operate on a single axis",
        )

    def test_fail_select_last_index_argmin(self):
        """Test that fusion fails when ArgMin has select_last_index=1."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] min_val, int64[N, ?, ?, ?] min_idx)
            {
                min_val = ReduceMin<axes=[1], keepdims=1>(X)
                min_idx = ArgMin<axis=1, keepdims=1, select_last_index=1>(X)
            }
        """)
        self.run_failed_condition_test(
            base_model,
            reduce_min_argmin_to_topk_rule,
            "ArgMin has select_last_index=1, which is not supported by TopK.",
        )

    def test_successful_fuse_with_default_keepdims(self):
        """Test fusion with default keepdims (should be 1)."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] min_val, int64[N, ?, ?, ?] min_idx)
            {
                min_val = ReduceMin<axes=[1]>(X)
                min_idx = ArgMin<axis=1>(X)
            }
        """)

        # Both should use default keepdims=1, so fusion should succeed
        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_successful_fuse_with_default_axis(self):
        """Test fusion with default axis (should be 0)."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 14, 17] X) => (float[1, 14, 17] min_val, int64[1, 14, 17] min_idx)
            {
                min_val = ReduceMin<axes=[0], keepdims=1>(X)
                min_idx = ArgMin<keepdims=1>(X)
            }
        """)

        # ArgMin should use default axis=0, so fusion should succeed
        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_successful_fuse_with_all_defaults(self):
        """Test fusion with all default values (keepdims=1, axis=0)."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 14, 17] X) => (float[1, 14, 17] min_val, int64[1, 14, 17] min_idx)
            {
                min_val = ReduceMin<axes=[0]>(X)
                min_idx = ArgMin(X)
            }
        """)

        # Both should use defaults: keepdims=1, axis=0
        expected_op_types = ["Constant", "TopK"]
        self.run_test(base_model, expected_op_types)

    def test_no_fusion_different_inputs(self):
        """Test that fusion doesn't happen when nodes have different inputs."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 13] >
            test_model (float[N, 32, 14, 17] X, float[N, 32, 14, 17] Y) => (float[N, ?, ?, ?] min_val, int64[N, ?, ?, ?] min_idx)
            {
                min_val = ReduceMin<axes=[1], keepdims=1>(X)
                min_idx = ArgMin<axis=1, keepdims=1>(Y)
            }
        """)

        # Pattern won't match at all because inputs are different
        updated_model = self.clone_model(base_model)
        count = rules.apply_to_model(updated_model)
        self.assertEqual(count, 0)

        # Model should be unchanged
        self.assertEqual(
            [node.op_type for node in base_model.graph],
            [node.op_type for node in updated_model.graph],
        )


if __name__ == "__main__":
    unittest.main()