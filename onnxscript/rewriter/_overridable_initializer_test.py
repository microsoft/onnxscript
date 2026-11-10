# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import parameterized

from onnxscript import optimizer, rewriter
from onnxscript.rewriter import _ir_utils
from onnxscript.rewriter.ort_fusions import (
    cos_sin_cache,
    fused_matmul_rule_sets,
    instance_to_group_normalization,
)
from onnxscript.rewriter.rules.common import (
    _basic_rules,
    _broadcast_to_matmul,
    _collapse_slices,
    _fuse_pad_into_conv,
    _materialize_reshape_shape,
    _min_max_to_clip,
    _redundant_scatter_nd,
    _remove_optional_bias,
)


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


class OverridableInitializerTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("Add", 0.0),
            ("Sub", 0.0),
            ("Mul", 1.0),
            ("Div", 1.0),
        ]
    )
    def test_public_optimizer_preserves_overrides(self, op_type, default):
        for representation in ("overridable", "initializer", "Constant"):
            for reverse in (False, True) if op_type in ("Add", "Mul") else (False,):
                with self.subTest(representation=representation, reverse=reverse):
                    tensor = onnx.numpy_helper.from_array(np.array(default, np.float32), "c")
                    inputs = [onnx.helper.make_tensor_value_info("x", 1, [2])]
                    nodes = []
                    if representation == "overridable":
                        inputs.append(onnx.helper.make_tensor_value_info("c", 1, []))
                    if representation == "Constant":
                        nodes.append(
                            onnx.helper.make_node("Constant", [], ["c"], value=tensor)
                        )
                    nodes.append(
                        onnx.helper.make_node(
                            op_type, ["c", "x"] if reverse else ["x", "c"], ["y"]
                        )
                    )
                    model = onnx.helper.make_model(
                        onnx.helper.make_graph(
                            nodes,
                            "arithmetic",
                            inputs,
                            [onnx.helper.make_tensor_value_info("y", 1, [2])],
                            [] if representation == "Constant" else [tensor],
                        ),
                        ir_version=9,
                        opset_imports=[onnx.helper.make_opsetid("", 18)],
                    )
                    original = model.SerializeToString()
                    optimized = optimizer.optimize(model)
                    self.assertEqual(model.SerializeToString(), original)
                    onnx.checker.check_model(optimized)
                    baseline, actual = _session(model), _session(optimized)
                    feeds = {"x": np.array([2.0, 4.0], np.float32)}
                    np.testing.assert_array_equal(
                        actual.run(None, feeds)[0], baseline.run(None, feeds)[0]
                    )
                    if representation == "overridable":
                        self.assertEqual([v.name for v in optimized.graph.input], ["x", "c"])
                        self.assertEqual(optimized.graph.initializer[0], tensor)
                        self.assertIn(op_type, [n.op_type for n in optimized.graph.node])
                        feeds["c"] = np.array(3.0, np.float32)
                        np.testing.assert_array_equal(
                            actual.run(None, feeds)[0], baseline.run(None, feeds)[0]
                        )
                        self.assertFalse(
                            np.array_equal(
                                actual.run(None, feeds)[0],
                                actual.run(None, {"x": feeds["x"]})[0],
                            )
                        )
                    else:
                        self.assertNotIn(op_type, [n.op_type for n in optimized.graph.node])

    @parameterized.parameterized.expand(
        [
            (
                "expand",
                "float[1] x",
                "int64[1] c = {1}",
                "y = Expand(x, c)",
                _basic_rules.ExpandIdentity.rule(),
                {"x": np.array([2.0], np.float32)},
                [3],
            ),
            (
                "reshape",
                "float[2,3] x",
                "int64[2] c = {2,3}, int64[2] s = {3,2}",
                "t = Reshape(x, s)\n y = Reshape(t, c)",
                _basic_rules.ReshapeReshape.rule(),
                {"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
                [3, 2],
            ),
            (
                "slice",
                "float[4] x",
                "int64[1] c = {0}, int64[1] e = {4}, int64[1] a = {0}, int64[1] s = {1}",
                "y = Slice(x, c, e, a, s)",
                _collapse_slices.collapse_slice_rule,
                {"x": np.arange(4, dtype=np.float32)},
                [1],
            ),
            (
                "min",
                "float[4] x",
                "float c = {1}, float b = {6}",
                "t = Min(x, c)\n y = Min(t, b)",
                _min_max_to_clip.min_min_rule,
                {"x": np.array([-3.0, 0.0, 4.0, 8.0], np.float32)},
                5.0,
            ),
            (
                "max",
                "float[4] x",
                "float c = {1}, float b = {6}",
                "t = Max(x, c)\n y = Max(t, b)",
                _min_max_to_clip.max_max_rule,
                {"x": np.array([-3.0, 0.0, 4.0, 8.0], np.float32)},
                10.0,
            ),
            (
                "clip",
                "float[4] x",
                "float c = {1}, float b = {6}",
                "t = Max(x, c)\n y = Min(t, b)",
                _min_max_to_clip.max_min_rule,
                {"x": np.array([-3.0, 0.0, 4.0, 8.0], np.float32)},
                5.0,
            ),
            (
                "bias",
                "float[2,2] x",
                "float[2] c = {0,0}, float[2,2] w = {1,0,0,1}",
                "y = Gemm(x, w, c)",
                _remove_optional_bias.remove_optional_bias_from_gemm_rule,
                {"x": np.arange(4, dtype=np.float32).reshape(2, 2)},
                [2.0, 3.0],
            ),
            (
                "scatter",
                "float[2] x, float[2] updates",
                "int64[2,1] c = {0,1}",
                "y = ScatterND(x, c, updates)",
                _redundant_scatter_nd.no_op_static_scatter_nd_rule,
                {
                    "x": np.array([1.0, 2.0], np.float32),
                    "updates": np.array([4.0, 5.0], np.float32),
                },
                [[1], [0]],
            ),
            (
                "pad",
                "float[1,1,2] x",
                "float c = {0}, int64[6] pads = {0,0,1,0,0,1}, float[1,1,1] w = {1}",
                "t = Pad(x, pads, c)\n y = Conv(t, w)",
                _fuse_pad_into_conv.FuseConvPad.rule(),
                {"x": np.array([[[1.0, 2.0]]], np.float32)},
                3.0,
            ),
            (
                "matmul_shape",
                "float[1,2,2] x, float[1,2,2] z",
                "int64[3] c = {1,2,2}, int64[3] s = {1,2,2}",
                "a = Reshape(x, s)\n b = Reshape(z, s)\n t = MatMul(a, b)\n y = Reshape(t, c)",
                _broadcast_to_matmul.two_reshapes_matmul_reshape_rule,
                {
                    "x": np.arange(4, dtype=np.float32).reshape(1, 2, 2),
                    "z": np.arange(4, dtype=np.float32).reshape(1, 2, 2),
                },
                [2, 1, 2],
            ),
            (
                "ort_divisor",
                "float[2,2] x, float[2,2] z",
                "float c = {2}",
                "t = MatMul(x, z)\n y = Div(t, c)",
                fused_matmul_rule_sets.FusedMatMulDiv1.rule(),
                {
                    "x": np.arange(4, dtype=np.float32).reshape(2, 2),
                    "z": np.arange(4, dtype=np.float32).reshape(2, 2),
                },
                4.0,
            ),
        ]
    )
    def test_rule_preserves_default_and_override(
        self, _, inputs, initializers, body, rule, feeds, override
    ):
        model = ir.from_onnx_text(
            f"""
            <ir_version: 9, opset_import: ["" : 18, "com.microsoft": 1]>
            graph ({inputs}) => (float[] y) <{initializers}> {{ {body} }}
            """
        )
        # Leave result dimensions unconstrained so runtime overrides may change them.
        model.graph.outputs[0].shape = None
        if model.graph[-1].op_type == "ScatterND":
            model.graph.outputs[0].shape = ir.Shape([2])
        baseline = _session(ir.to_proto(model))
        default = model.graph.initializers["c"]
        default_tensor = default.const_value
        assert default_tensor is not None
        for overridable in (True, False):
            with self.subTest(overridable=overridable):
                candidate = model.clone()
                if overridable:
                    candidate.graph.inputs.append(candidate.graph.initializers["c"])
                count = rewriter.RewriteRuleSet([rule]).apply_to_model(candidate)
                self.assertEqual(count, 0 if overridable else 1)
                if overridable:
                    candidate = rewriter.rewrite(candidate)
                actual = _session(ir.to_proto(candidate))
                np.testing.assert_allclose(
                    actual.run(None, feeds)[0], baseline.run(None, feeds)[0]
                )
                if overridable:
                    self.assertEqual([v.name for v in candidate.graph.inputs], [*feeds, "c"])
                    candidate_tensor = candidate.graph.initializers["c"].const_value
                    assert candidate_tensor is not None
                    np.testing.assert_array_equal(
                        candidate_tensor.numpy(), default_tensor.numpy()
                    )
                    source = model.clone()
                    source.graph.inputs.append(source.graph.initializers["c"])
                    overridden_feeds = {
                        **feeds,
                        "c": np.array(override, dtype=default_tensor.dtype.numpy()),
                    }
                    expected = _session(ir.to_proto(source)).run(None, overridden_feeds)[0]
                    np.testing.assert_allclose(actual.run(None, overridden_feeds)[0], expected)
                    self.assertFalse(np.array_equal(expected, baseline.run(None, feeds)[0]))

    def test_materialize_preserves_overridable_shape(self):
        model = ir.from_onnx_text("""
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[2,3] x, int64[2] shape) => (float[2,3] y)
            <int64[2] shape = {2,3}>
            { y = Reshape(x, shape) }
        """)
        self.assertEqual(_materialize_reshape_shape.rules.apply_to_model(model), 0)

    @parameterized.parameterized.expand(
        [(name,) for name in ("begin0", "end0", "axes0", "begin1", "end1", "axes1")]
    )
    def test_slices_split_rejects_each_overridable_parameter(self, name):
        model = ir.from_onnx_text("""
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[4] x) => (float[?] y0, float[?] y1)
            <int64[1] begin0 = {0}, int64[1] end0 = {2}, int64[1] axes0 = {0},
             int64[1] begin1 = {2}, int64[1] end1 = {4}, int64[1] axes1 = {0}>
            {
                y0 = Slice(x, begin0, end0, axes0)
                y1 = Slice(x, begin1, end1, axes1)
            }
        """)
        for overridable in (True, False):
            with self.subTest(overridable=overridable):
                candidate = model.clone()
                if overridable:
                    candidate.graph.inputs.append(candidate.graph.initializers[name])
                result = _basic_rules.SlicesSplit().check(
                    None, candidate.graph.inputs[0], **candidate.graph.initializers
                )
                self.assertEqual(bool(result), not overridable)

    def test_constant_helpers_preserve_defaults(self):
        model = ir.from_onnx_text("""
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float c) => (float y) <float c = {1}>
            { y = Identity(c) }
        """)
        value = model.graph.inputs[0]
        default = value.const_value
        self.assertIsNone(_ir_utils.get_const_value(value))
        self.assertIsNone(_ir_utils.get_numpy_value(value))
        self.assertIsNone(_ir_utils.get_singleton_value(value))
        self.assertFalse(_ir_utils.is_singleton_value(value, 1))
        self.assertIs(value.const_value, default)
        model.graph.inputs.remove(value)
        self.assertIs(_ir_utils.get_const_value(value), default)

    def test_cos_sin_cache_rejects_overridable_frequencies(self):
        model = ir.from_onnx_text("""
            <ir_version: 9, opset_import: ["" : 18]>
            graph (int64[1,2] positions) => (int64[1,2] y)
            <float[1,2,1] inv_freq = {1,2}, float[1,2,2] freqs = {1,2,3,4},
             int64[1] axes = {1}>
            { y = Identity(positions) }
        """)
        for const_freqs, name in ((False, "inv_freq"), (True, "freqs")):
            with self.subTest(const_freqs=const_freqs):
                rule = cos_sin_cache.CosSinCacheFusion("test", const_freqs=const_freqs)
                args = (
                    None,
                    model.graph.initializers["inv_freq"],
                    model.graph.inputs[0],
                    model.graph.initializers["freqs"],
                    model.graph.initializers["axes"],
                )
                self.assertTrue(rule.check(*args))
                value = model.graph.initializers[name]
                model.graph.inputs.append(value)
                self.assertFalse(rule.check(*args))
                model.graph.inputs.remove(value)

    @parameterized.parameterized.expand(
        [
            (name,)
            for name in (
                "adjusted_input_shape",
                "original_input_shape",
                "weight_for_norm",
                "bias_for_norm",
            )
        ]
    )
    def test_instance_norm_rejects_overridable_parameters(self, name):
        model = ir.from_onnx_text("""
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[1,2,2,2] x) => (float[1,2,2,2] y)
            <int64[3] adjusted_input_shape = {0,2,-1},
             int64[4] original_input_shape = {1,2,2,2},
             float[2] weight_for_norm = {1,1}, float[2] bias_for_norm = {0,0},
             float[2,1,1] weight_full = {1,1}, float[2,1,1] bias_full = {0,0}>
            { y = Identity(x) }
        """)
        check = instance_to_group_normalization.check_if_simulated_instance_norm_is_used
        self.assertTrue(check(None, model.graph.inputs[0], **model.graph.initializers))
        model.graph.inputs.append(model.graph.initializers[name])
        self.assertFalse(check(None, model.graph.inputs[0], **model.graph.initializers))

    def test_nested_graph_capture_is_not_a_constant(self):
        model = ir.from_onnx_text("""
            <ir_version: 9, opset_import: ["" : 18]>
            graph (float[2] x, float c, bool condition) => (float[2] y)
            <float c = {0}>
            {
                y = If<
                    then_branch: graph = then_graph () => (float[2] t) { t = Add(x, c) },
                    else_branch: graph = else_graph () => (float[2] e) { e = Sub(x, c) }
                >(condition)
            }
        """)
        rewriter.RewriteRuleSet(rewriter._DEFAULT_REWRITE_RULES).apply_to_model(model)
        node = model.graph[0]
        self.assertEqual(node.attributes["then_branch"].as_graph()[0].op_type, "Add")
        self.assertEqual(node.attributes["else_branch"].as_graph()[0].op_type, "Sub")
        actual = _session(ir.to_proto(model))
        for condition in (True, False):
            for c in (None, np.array(3.0, np.float32)):
                feeds = {
                    "x": np.array([2.0, 4.0], np.float32),
                    "condition": np.array(condition),
                }
                if c is not None:
                    feeds["c"] = c
                expected = feeds["x"] + (0 if c is None else c * (1 if condition else -1))
                np.testing.assert_array_equal(actual.run(None, feeds)[0], expected)


if __name__ == "__main__":
    unittest.main()
