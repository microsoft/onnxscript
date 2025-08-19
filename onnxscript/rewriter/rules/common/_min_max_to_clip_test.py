# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.passes.common import onnx_checker, shape_inference
from parameterized import parameterized

from onnxscript.rewriter import MatchingTracer, MatchStatus, RewriteRule, testing
from onnxscript.rewriter.rules.common._min_max_to_clip import (
    fuse_successive_max_min_rule,
    fuse_successive_max_rule,
    fuse_successive_min_max_rule,
    fuse_successive_min_rule,
    rules,
)


class _TestMinMaxToClipBase(unittest.TestCase):
    @property
    def rng(self):
        return np.random.default_rng(20250817)

    def clone_model(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    def run_test(
        self,
        base_model: ir.Model,
        expected_op_types: list[str],
        dtype: str = "float",
    ):
        onnx_checker.CheckerPass(True)(base_model)
        base_model = shape_inference.infer_shapes(base_model)
        updated_model = self.clone_model(base_model)
        _ = rules.apply_to_model(updated_model)

        # Check expected op_types
        self.assertEqual([node.op_type for node in updated_model.graph], expected_op_types)

        # Check inference
        inputs = (
            self.rng.integers(
                low=-10,
                high=10,
                size=(2, *updated_model.graph.inputs[0].shape[1:]),
                dtype=np.int32,
            ),
        )
        if dtype == "float":
            inputs = (inputs[0].astype(np.float32),)

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
        rewrite_rule: RewriteRule,
        expected_message: str,
    ):
        onnx_checker.CheckerPass(True)(base_model)

        updated_model = self.clone_model(base_model)
        tracer = MatchingTracer()
        count = rewrite_rule.apply_to_model(updated_model, tracer=tracer)

        # Check that the model is unchanged
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[rewrite_rule][0]
        self.assertEqual(tracer_match.status.value, MatchStatus.CONDITION_FAILED)
        self.assertRegex(tracer_match.match_result.reason, expected_message)


class TestFuseSuccesiveMinOrMax(_TestMinMaxToClipBase):
    @parameterized.expand(
        [
            ("int32_min", "int32", "Min"),
            ("int32_max", "int32", "Max"),
            ("float32_min", "float", "Min"),
            ("float32_max", "float", "Max"),
        ]
    )
    def test_successful_fuse_succesive_min_or_max(self, _, dtype, op_type):
        base_model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model ({dtype}[N, 32, 14, 17] X) => ({dtype} [N, ?, ?, ?] Y)
            <{dtype}[1] cst1 = {{3}}, {dtype}[1] cst2 = {{6}}>
            {{
                x1 = {op_type}(X, cst1)
                Y = {op_type}(x1, cst2)
            }}
        """)
        self.run_test(base_model, expected_op_types=[op_type], dtype=dtype)

    @parameterized.expand(
        [
            ("int32_min_multi", "int32", "Min"),
            ("int32_max_multi", "int32", "Max"),
            ("float32_min_multi", "float", "Min"),
            ("float32_max_multi", "float", "Max"),
        ]
    )
    def test_successful_fuse_succesive_min_or_max_multiple_inputs(self, _, dtype, op_type):
        base_model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model ({dtype}[N, 3, 3] X) => ({dtype}[N, 3, 3] Y)
            <{dtype}[3] cst1 = {{2, 5, 8}},
             {dtype}[1] cst2 = {{4}},
             {dtype}[3] cst3 = {{3, 1, -6}},
             {dtype}[1] cst4 = {{10}},
             {dtype}[3] cst5 = {{-2, 7, 9}},
             {dtype}[1] cst6 = {{0}},
             {dtype}[3] cst7 = {{11, -3, 4}}>
            {{
                x1 = {op_type}(X, cst1, cst2, cst3, cst4)
                Y  = {op_type}(x1, cst5, cst6, cst7)
            }}
        """)
        self.run_test(base_model, expected_op_types=[op_type], dtype=dtype)

    @parameterized.expand(
        [
            ("int32_min", "Min"),
            ("int32_max", "Max"),
            ("float32_min", "Min"),
            ("float32_max", "Max"),
        ]
    )
    def test_successful_fuse_succesive_min_or_max_constants(self, _, op_type):
        base_model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] cst1 = {{3}}>
            {{
                x1 = {op_type}(X, cst1)
                cst2 = Constant<value_float=6.0>()
                Y = {op_type}(x1, cst2)
            }}
        """)
        self.run_test(base_model, expected_op_types=["Constant", op_type])

    @parameterized.expand(
        [
            ("min_nonconst", "Min", fuse_successive_min_rule),
            ("max_nonconst", "Max", fuse_successive_max_rule),
        ]
    )
    def test_failure_fuse_successive_min_or_max_non_constant(self, _, op_type, rewrite_rule):
        model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float[N, ?, ?, ?] Y)
            <float[1] cst2 = {{6}}>
            {{
                cst1 = ReduceMean<keepdims=0>(X)
                x1 = {op_type}(X, cst1)
                Y = {op_type}(x1, cst2)
            }}
        """)
        self.run_failed_condition_test(model, rewrite_rule, "is not a constant.")

    @parameterized.expand(
        [
            ("min_graph_input", "Min", fuse_successive_min_rule),
            ("max_graph_input", "Max", fuse_successive_max_rule),
        ]
    )
    def test_failure_fuse_successive_min_or_max_graph_inputs(self, _, op_type, rewrite_rule):
        base_model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X, float[1] cst1, float[1] cst2) => (float[N, ?, ?, ?] Y)
            {{
                x1 = {op_type}(X, cst1)
                Y = {op_type}(x1, cst2)
            }}
        """)
        self.run_failed_condition_test(base_model, rewrite_rule, "is a graph input")


class TestMinMaxToClip(_TestMinMaxToClipBase):
    def test_successful_min_max_to_clip(self):
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] min = {10.0}, float[1] max = {6.0}>
            {
                x1 = Min(X, min)
                Y = Max(x1, max)
            }
        """)
        self.run_test(base_model, expected_op_types=["Clip"])

    def test_successful_min_max_to_clip_constants(self):
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] min = {10.0}>
            {
                x1 = Min(X, min)
                max = Constant<value_float=6.0>()
                Y = Max(x1, max)
            }
        """)
        self.run_test(base_model, expected_op_types=["Constant", "Clip"])

    def test_failure_min_max_to_clip_invalid_bounds(self):
        """Min node should have the max value and Max node should have the min value."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] min = {2.0}, float[1] max = {6.0}>
            {
                x1 = Min(X, min)
                Y = Max(x1, max)
            }
        """)
        self.run_failed_condition_test(
            base_model, fuse_successive_min_max_rule, "Invalid bounds:"
        )

    def test_failure_fuse_min_max_to_clip_non_constant(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] max = {6.0}>
            {
                min = ReduceMean<keepdims=0>(X)
                x1 = Min(X, min)
                Y = Max(x1, max)
            }
        """)
        self.run_failed_condition_test(
            model, fuse_successive_min_max_rule, "is not a constant."
        )

    def test_failure_min_max_to_clip_graph_inputs(self):
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X, float[1] min, float[1] max) => (float [N, ?, ?, ?] Y)
            {
                x1 = Min(X, min)
                Y = Max(x1, max)
            }
        """)
        self.run_failed_condition_test(
            base_model, fuse_successive_min_max_rule, "is a graph input"
        )

    def test_failure_min_max_to_clip_need_scalars(self):
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 4, 4] X) => (float [N, ?, ?] Y)
            <float[4] min = {1.0, 2.0, -3.0, 0.0}, float[1] max = {6.0}>
            {
                x1 = Min(X, min)
                Y = Max(x1, max)
            }
        """)
        self.run_failed_condition_test(
            base_model, fuse_successive_min_max_rule, "is not a scalar"
        )


class TestMaxMinToClip(_TestMinMaxToClipBase):
    def test_successful_max_min_to_clip(self):
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] min = {10.0}, float[1] max = {6.0}>
            {
                x1 = Max(X, max)
                Y = Min(x1, min)
            }
        """)
        self.run_test(base_model, expected_op_types=["Clip"])

    def test_successful_max_min_to_clip_constants(self):
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] max = {6.0}>
            {
                x1 = Max(X, max)
                min = Constant<value_float=10.0>()
                Y = Min(x1, min)
            }
        """)
        self.run_test(base_model, expected_op_types=["Constant", "Clip"])

    def test_failure_max_min_to_clip_invalid_bounds(self):
        """Min node should have the max value and Max node should have the min value."""
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] min = {2.0}, float[1] max = {6.0}>
            {
                x1 = Max(X, max)
                Y = Min(x1, min)
            }
        """)
        self.run_failed_condition_test(
            base_model, fuse_successive_max_min_rule, "Invalid bounds:"
        )

    def test_failure_fuse_max_min_to_clip_non_constant(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X) => (float [N, ?, ?, ?] Y)
            <float[1] max = {6.0}>
            {
                min = ReduceMean<keepdims=0>(X)
                x1 = Max(X, max)
                Y = Min(x1, min)
            }
        """)
        self.run_failed_condition_test(
            model, fuse_successive_max_min_rule, "is not a constant."
        )

    def test_failure_max_min_to_clip_graph_inputs(self):
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14, 17] X, float[1] max, float[1] min) => (float [N, ?, ?, ?] Y)
            {
                x1 = Max(X, max)
                Y = Min(x1, min)
            }
        """)
        self.run_failed_condition_test(
            base_model, fuse_successive_max_min_rule, "is a graph input"
        )

    def test_failure_max_min_to_clip_need_scalars(self):
        base_model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 4, 4] X) => (float [N, ?, ?] Y)
            <float[4] min = {1.0, 2.0, -3.0, 0.0}, float[1] max = {6.0}>
            {
                x1 = Max(X, min)
                Y = Min(x1, max)
            }
        """)
        self.run_failed_condition_test(
            base_model, fuse_successive_max_min_rule, "is not a scalar"
        )


class TestIntegrationMinMaxToClip(_TestMinMaxToClipBase):
    def test_successful_full_chain_fusion(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float [N, ?, ?] Y)
            <float[1] min1 = {10.0}, float[1] min2 = {12.0}, float[1] min3 = {20.0}, float[1] min4 = {5.0},
            float[1] max1 = {6.0}, float[1] max2 = {8.0}, float[1] max3 = {5.0}>
            {
                x1 = Min(X, min1)
                x2 = Min(x1, min2)
                x3 = Max(x2, max1)
                x4 = Max(x3, max2)
                x5 = Min(x4, min3)
                x6 = Max(x5, max3)
                Y = Min(x6, min4)
            }
        """)
        self.run_test(model, expected_op_types=["Clip", "Clip", "Clip"])


if __name__ == "__main__":
    unittest.main()
