# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnx

import onnxscript.testing
from onnxscript import ir, optimizer


def _create_model(model_text: str) -> ir.Model:
    """Create a model from the given text."""
    model = onnx.parser.parse_model(model_text)
    return ir.serde.deserialize_model(model)


class FunctionFoldingTest(unittest.TestCase):
    def test_identity(self):
        model = _create_model(
            """
            <ir_version: 7, opset_import: ["" : 17, "local" : 1]>
            agraph (float[N] x1, bool cond1) => (float[N] z1) {
                z1 = local.fun1(x1, cond1)
            }
            <opset_import: ["" : 17], domain: "local">
            fun1 (x, cond) => (z) {
                t = Identity(x)
                t2 = Identity(t)
                t3 = If (cond) <
                    then_branch = then_graph() => (t4) {
                        t5 = Identity(t2)
                        t4 = Identity(t5)
                    },
                    else_branch = else__graph() => (t6) {
                        t7 = Identity(t)
                        t6 = Identity(t7)
                    }
                >
                t4 = Add(t3, t3)
                z = Identity(t4)
            }"""
        )
        optimized = optimizer.optimize(
            model, onnx_shape_inference=False, num_iterations=1, inline=True
        )
        self.assertEqual(len(optimized.functions), 0)
        self.assertEqual(len(optimized.graph), 2)

    def test_sequence_concat(self):
        model = _create_model(
            """
            <ir_version: 7, opset_import: ["" : 17, "local" : 1]>
            agraph (float[N] x1) => (float[M] z1) {
                z1 = local.fun1(x1)
            }
            <opset_import: ["" : 17], domain: "local">
            fun1 (x) => (z) {
                t0 = Add (x, x)
                t2 = Add (x, x)
                t3 = SequenceConstruct (x, t0, t2, x)
                z = ConcatFromSequence <axis=0> (t3)
            }"""
        )
        optimized = optimizer.optimize(
            model, onnx_shape_inference=False, num_iterations=1, inline=False
        )
        function = optimized.functions[("local", "fun1", "")]
        self.assertEqual(len(function), 3)
        self.assertEqual(function[2].op_type, "Concat")

    def test_sequence_at(self):
        model = _create_model(
            """
            <ir_version: 7, opset_import: ["" : 17]>
            agraph (float[N] x) => (float[M] z) {
                t0 = Add (x, x)
                t1 = Mul (x, x)
                s = SequenceConstruct (x, t0, t1)
                one = Constant <value = int64 {1}> ()
                z = SequenceAt (s, one)
            }"""
        )
        optimized = optimizer.optimize(
            model, onnx_shape_inference=False, num_iterations=1, inline=False
        )
        expected = _create_model(
            """
            <ir_version: 7, opset_import: ["" : 17]>
            agraph (float[N] x) => (float[M] z) {
                z = Add (x, x)
            }"""
        )
        # TODO(justinchuby): Implement assert_isomorphic_graph for IR objects
        onnxscript.testing.assert_isomorphic_graph(
            ir.to_proto(optimized.graph), ir.to_proto(expected.graph)
        )

    def test_single_user_function_is_modified_inplace_after_folding(self):
        model = _create_model(
            """
            <ir_version: 7, opset_import: ["" : 17, "local" : 1]>
            agraph (float[N] x1) => (float[M] z1) {
                z1 = local.fun1(x1)
            }
            <opset_import: ["" : 17], domain: "local">
            fun1 (x) => (z) {
                t0 = Add (x, x)
                t2 = Add (x, x)
                t3 = SequenceConstruct (x, t0, t2, x)
                z = ConcatFromSequence (t3)
            }"""
        )
        optimized = optimizer.optimize(
            model, onnx_shape_inference=False, num_iterations=1, inline=False
        )
        self.assertEqual(next(iter(optimized.functions.values())).name, "fun1")

    def test_fold_nested_if_function_succeeds(self):
        model = _create_model(
            """
            <
            ir_version: 9,
            opset_import: ["this" : 1, "" : 18]
            >
            func (float[1,512] x, float[1,512] y) => ( out) {
            out = this.foldable_func (x, y)
            }
            <
            domain: "this",
            opset_import: ["" : 18]
            >
            foldable_func (x, y) => (z_6)
            {
            cond = Constant <value: tensor = bool cond {1}> ()
            z_6 = If (cond) <then_branch: graph = thenGraph_4 () => ( z_2) {
                cond_0 = Not (cond)
                z_2 = If (cond_0) <then_branch: graph = thenGraph_5 () => ( z) {
                    z = Add (x, x)
                }, else_branch: graph = elseGraph_5 () => ( z_1) {
                    z_1 = Identity (x)
                }>
            }, else_branch: graph = elseGraph_4 () => ( z_5) {
                z_5 = If (cond) <then_branch: graph = thenGraph_10 () => ( z_3) {
                    z_3 = Add (y, y)
                }, else_branch: graph = elseGraph_10 () => ( z_4) {
                    z_4 = Add (x, y)
                }>
            }>
            }"""
        )
        optimized = optimizer.optimize(model, onnx_shape_inference=False, inline=True)

        self.assertEqual(len(optimized.functions), 0)
        self.assertEqual(len(optimized.graph), 2)
        self.assertNotIn("If", {n.op_type for n in optimized.graph})


if __name__ == "__main__":
    unittest.main()
