# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import parameterized
from onnx_ir.passes.common import onnx_checker, shape_inference

from onnxscript.rewriter import fuse_relus_clips, testing
from onnxscript.rewriter import pattern as orp
from onnxscript.rewriter.fuse_relus_clips import (
    fuse_successive_clip_relu_rule,
    fuse_successive_clip_rule,
    fuse_successive_relu_clip_rule,
)


class _FuseReluClipTestBase(unittest.TestCase):
    @property
    def rng(self):
        return np.random.default_rng(20250621)

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
        _ = fuse_relus_clips.fuse_relus_clips_rules().apply_to_model(updated_model)

        # Check expected op_types
        self.assertEqual([node.op_type for node in updated_model.graph], expected_op_types)

        # Check inference
        inputs = (self.rng.integers(low=-10, high=10, size=(2, 32, 14), dtype=np.int32),)
        if dtype == "float":
            inputs = (inputs[0].astype(np.float32),)

        # onnxruntime has an optimization that fuses Clip(Relu) and
        # it doesn't support int data, that's why we disable ort optimization
        # see https://github.com/microsoft/onnxruntime/blob/c98a0e014b641e289ed25f42b792bca1893ccb03/onnxruntime/core/optimizer/relu_clip_fusion.cc#L60
        testing.assert_numerically_equal(
            base_model,
            updated_model,
            inputs,
            ort_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )

        # Validate serialized model
        output_model_proto = ir.serde.serialize_model(updated_model)
        onnx.checker.check_model(output_model_proto, full_check=True)

    def run_failed_condition_test(
        self,
        base_model: ir.Model,
        rewrite_rule: orp.RewriteRule,
        expected_message: str,
    ):
        onnx_checker.CheckerPass(True)(base_model)

        updated_model = self.clone_model(base_model)
        tracer = orp.MatchingTracer()
        count = rewrite_rule.apply_to_model(updated_model, tracer=tracer)

        # Check that the model is unchanged
        self.assertEqual(count, 0)

        # Check that the error message is the expected one
        tracer_match = tracer.best_matches_map[rewrite_rule][0]
        self.assertEqual(tracer_match.status.value, orp.MatchStatus.CONDITION_FAILED)
        self.assertRegex(tracer_match.match_result.reason, expected_message)


class FuseSuccessiveReluTest(_FuseReluClipTestBase):
    def test_successful_fuse_successive_relus(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float [N, ?, ?] Y)
            {
                x1 = Relu(X)
                x2 = Relu(x1)
                Y = Relu(x2)
            }
        """)
        self.run_test(model, expected_op_types=["Relu"])


class FuseSuccessiveReluClipTest(_FuseReluClipTestBase):
    @parameterized.parameterized.expand(
        [
            (
                "relu_then_clip",
                """
                    x1 = Relu(X)
                    Y = Clip(x1, min, max)
                """,
                "float",
            ),
            (
                "clip_then_relu",
                """
                    x1 = Clip(X, min, max)
                    Y = Relu(x1)
                """,
                "float",
            ),
            (
                "int_relu_then_clip",
                """
                    x1 = Relu(X)
                    Y = Clip(x1, min, max)
                """,
                "int32",
            ),
            (
                "int_clip_then_relu",
                """
                    x1 = Clip(X, min, max)
                    Y = Relu(x1)
                """,
                "int32",
            ),
        ]
    )
    def test_successful_fuse_successive_relu_clip(self, _, nodes, dtype):
        model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model ({dtype}[N, 32, 14] X) => ({dtype} [N, ?, ?] Y)
            <{dtype} min = {{1}}, {dtype} max = {{6}}>
            {{
                {nodes}
            }}
        """)
        self.run_test(model, expected_op_types=["Clip"], dtype=dtype)

    @parameterized.parameterized.expand(
        [
            (
                "relu_then_clip",
                """
                    x1 = Relu(X)
                    min = Constant<value_float=-2.0>()
                    Y = Clip(x1, min)
                """,
            ),
            (
                "clip_then_relu",
                """
                    min = Constant<value_float=-2.0>()
                    x1 = Clip(X, min)
                    Y = Relu(x1)
                """,
            ),
        ]
    )
    def test_successful_fuse_successive_relu_clip_constant_nodes(self, _, nodes):
        model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float[N, ?, ?] Y)
            {{
                {nodes}
            }}
        """)
        self.run_test(model, expected_op_types=["Constant", "Clip"])

    @parameterized.parameterized.expand(
        [
            (
                "relu_then_clip",
                """
                    x1 = Relu(X)
                    Y = Clip(x1,,max)
                """,
            ),
            (
                "clip_then_relu",
                """
                    x1 = Clip(X,,max)
                    Y = Relu(x1)
                """,
            ),
        ]
    )
    def test_successful_fuse_successive_relu_clip_no_min(self, _, nodes):
        model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float [N, ?, ?] Y)
            <float max = {{6.0}}>
            {{
                {nodes}
            }}
        """)
        self.run_test(model, expected_op_types=["Clip"])

    @parameterized.parameterized.expand(
        [
            (
                "relu_then_clip",
                """
                    x1 = Relu(X)
                    Y = Clip(x1, min)
                """,
                fuse_successive_clip_relu_rule,
            ),
            (
                "clip_then_relu",
                """
                    x1 = Clip(X, min)
                    Y = Relu(x1)
                """,
                fuse_successive_relu_clip_rule,
            ),
        ]
    )
    def test_fail_fuse_successive_relu_clip_non_initializers(self, _, nodes, rewrite_rule):
        model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float [N, ?, ?] Y)
            {{
                min = ReduceMean<keepdims=0>(X)
                {nodes}
            }}
        """)
        self.run_failed_condition_test(model, rewrite_rule, "is not a constant.")

    @parameterized.parameterized.expand(
        [
            (
                "relu_then_clip",
                """
                    x1 = Relu(X)
                    Y = Clip(x1, min)
                """,
                fuse_successive_clip_relu_rule,
            ),
            (
                "clip_then_relu",
                """
                    x1 = Clip(X, min)
                    Y = Relu(x1)
                """,
                fuse_successive_relu_clip_rule,
            ),
        ]
    )
    def test_fail_fuse_successive_relu_clip_graph_inputs(self, _, nodes, rewrite_rule):
        model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X, float min) => (float [N, ?, ?] Y)
            {{
                {nodes}
            }}
        """)
        self.run_failed_condition_test(model, rewrite_rule, "is a graph input.")


class FuseSuccessiveClipTest(_FuseReluClipTestBase):
    @parameterized.parameterized.expand(
        [
            ("float", "float"),
            ("int32", "int32"),
        ]
    )
    def test_successful_fuse_successive_clips(self, _, dtype):
        model = ir.from_onnx_text(f"""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model ({dtype}[N, 32, 14] X) => ({dtype} [N, ?, ?] Y)
            <{dtype} max1 = {{4}}, {dtype} min2 = {{0}},
             {dtype} max2 = {{11}}, {dtype} min3 = {{1}},
            {dtype} max3 = {{7}}, {dtype} max4 = {{13}}>
            {{
                x1 = Clip(X)
                x2 = Clip(x1,,max1)
                x3 = Clip(x2, min2, max2)
                x4 = Clip(x3, min3, max3)
                x5  = Clip(x4,,max4)
                Y = Clip(x5)
            }}
        """)
        self.run_test(model, expected_op_types=["Clip"], dtype=dtype)

    def test_successful_fuse_successive_clips_node_constants(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float [N, ?, ?] Y)
            {
                min1 = Constant<value_float=-2.0>()
                max1 = Constant<value_float=6.0>()
                min2 = Constant<value_float=-3.0>()
                max2 = Constant<value_float=3.0>()
                x1 = Clip(X, min1, max1)
                Y  = Clip(x1, min2, max2)
            }
        """)
        self.run_test(
            model, expected_op_types=["Constant", "Constant", "Constant", "Constant", "Clip"]
        )

    def test_successful_fuse_successive_clips_no_min(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float [N, ?, ?] Y)
            <float max1 = {4.0}, float max2 = {7.0}>
            {
                x1 = Clip(X,, max1)
                Y  = Clip(x1,, max2)
            }
        """)
        self.run_test(model, expected_op_types=["Clip"])

    def test_fail_fuse_successive_clips_non_initializers(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float [N, ?, ?] Y)
            <float max = {6.0}>
            {
                min1 = ReduceMean<keepdims=0>(X)
                min2 = ReduceMax<keepdims=0>(X)
                x1 = Clip(X, min1)
                Y = Clip(x1, min2)
            }
        """)
        self.run_failed_condition_test(model, fuse_successive_clip_rule, "is not a constant.")

    def test_fail_fuse_successive_clips_graph_inputs(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X, float min1, float min2) => (float [N, ?, ?] Y)
            <float max = {6.0}>
            {
                x1 = Clip(X, min1)
                Y = Clip(x1, min2)
            }
        """)
        self.run_failed_condition_test(model, fuse_successive_clip_rule, "is a graph input.")


class FuseReluClipIntegrationTest(_FuseReluClipTestBase):
    def test_successful_full_chain_fusion(self):
        model = ir.from_onnx_text("""
            < ir_version: 10, opset_import: ["" : 20] >
            test_model (float[N, 32, 14] X) => (float [N, ?, ?] Y)
            {
                x1 = Relu(X)
                x2 = Relu(x1)
                x3 = Relu(x2)
                x4 = Relu(x3)
                x5 = Clip(x4)
                x6 = Relu(x5)
                Y = Clip(x6)
            }
        """)
        self.run_test(model, expected_op_types=["Clip"])


if __name__ == "__main__":
    unittest.main()
