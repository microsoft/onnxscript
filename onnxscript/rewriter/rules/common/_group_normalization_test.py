# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx.parser
import onnx.reference
import onnx_ir as ir
import onnxruntime as ort
import parameterized

from onnxscript import optimizer, rewriter
from onnxscript.rewriter.ort_fusions import optimize_for_ort
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose, ort_run
from onnxscript.rewriter.rules.common import instance_norm_to_onnx_group_norm_rule


def _make_model(opset=21, groups=2, channels=4, dtype="float", affine_shape=None):
    if affine_shape is None:
        affine_shape = (channels, 1, 1)
    model_proto = onnx.parser.parse_model(
        f"""
            <ir_version: 9, opset_import: [ "" : {opset}]>
            agraph ({dtype}[2, {channels}, 4, 4] image)
                => ({dtype}[2, {channels}, 4, 4] output)
            {{
                adjusted_input_shape = Constant<value: tensor = int64[3] {{0, {groups}, -1}}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {{2, {channels}, 4, 4}}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }}
            """
    )
    np_dtype = {"float": np.float32, "float16": np.float16, "double": np.float64}[dtype]
    rng = np.random.default_rng(0)
    initializers = {
        "weight_for_norm": np.ones(groups, dtype=np_dtype),
        "bias_for_norm": np.zeros(groups, dtype=np_dtype),
        "weight_full": rng.random(affine_shape).astype(np_dtype),
        "bias_full": rng.random(affine_shape).astype(np_dtype),
    }
    for name, value in initializers.items():
        model_proto.graph.initializer.append(
            ir.serde.serialize_tensor(ir.tensor(value, name=name))
        )
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    model = ir.from_proto(model_proto)
    inputs = {"image": rng.standard_normal((2, channels, 4, 4)).astype(np_dtype)}
    return model, inputs


class OptimizeGroupNormalizationTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (optimize, opset, groups, channels, dtype, affine_channels)
            for optimize in (optimizer.optimize, rewriter.rewrite, optimize_for_ort)
            for opset in (17, 20, 21, 23)
            for groups, channels in ((32, 320), (32, 32), (8, 40), (2, 3))
            for dtype in ("float", "float16")
            for affine_channels in (channels, 1)
        ]
    )
    def test_optimization_uses_supported_group_normalization(
        self, optimize, opset, groups, channels, dtype, affine_channels
    ):
        model, inputs = _make_model(opset, groups, channels, dtype, (affine_channels, 1, 1))
        expected = ort_run("original_group_norm", model, inputs)

        optimize(model)

        self.assertEqual(model.opset_imports[""], opset)
        self.assertNotIn(
            ("com.microsoft", "GroupNorm"),
            {(node.domain, node.op_type) for node in model.graph},
        )
        group_norms = [node for node in model.graph if node.op_type == "GroupNormalization"]
        if opset >= 21 and affine_channels == channels and channels % groups == 0:
            self.assertEqual(len(group_norms), 1)
            group_norm = group_norms[0]
            self.assertEqual(group_norm.domain, "")
            self.assertEqual(group_norm.attributes["num_groups"].as_int(), groups)
            self.assertAlmostEqual(group_norm.attributes["epsilon"].as_float(), 1e-6)
            inferred_model = ir.from_proto(
                onnx.shape_inference.infer_shapes(ir.to_proto(model), strict_mode=True)
            )
            inferred_group_norm = next(
                node for node in inferred_model.graph if node.op_type == "GroupNormalization"
            )
            self.assertEqual(inferred_group_norm.inputs[1].shape, ir.Shape([channels]))
            self.assertEqual(inferred_group_norm.inputs[2].shape, ir.Shape([channels]))
            self.assertEqual(inferred_group_norm.inputs[1].dtype, model.graph.inputs[0].dtype)
            self.assertEqual(inferred_group_norm.inputs[2].dtype, model.graph.inputs[0].dtype)
        else:
            self.assertEqual(len(group_norms), 0)
            self.assertIn("InstanceNormalization", {node.op_type for node in model.graph})
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
        assert_allclose(ort_run("optimized_group_norm", model, inputs), expected)

    @parameterized.parameterized.expand(
        [(optimize,) for optimize in (optimizer.optimize, rewriter.rewrite, optimize_for_ort)]
    )
    def test_double_precision_is_preserved(self, optimize):
        model, inputs = _make_model(dtype="double")
        for name, value in (("weight_full", 1.0), ("bias_full", 0.0)):
            model.graph.initializers[name].const_value = ir.tensor(
                np.full((4, 1, 1), value, dtype=np.float64)
            )
        inputs["image"] = np.resize(
            np.array([1e8 - 1, 1e8 + 1], dtype=np.float64), (2, 4, 4, 4)
        )
        expected = onnx.reference.ReferenceEvaluator(ir.to_proto(model)).run(None, inputs)
        np.testing.assert_allclose(expected[0], inputs["image"] - 1e8, atol=1e-6)

        optimize(model)

        self.assertNotIn("GroupNormalization", {node.op_type for node in model.graph})
        actual = onnx.reference.ReferenceEvaluator(ir.to_proto(model)).run(None, inputs)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
        onnx.checker.check_model(ir.to_proto(model), full_check=True)

    @parameterized.parameterized.expand(
        [("input", "image"), ("weight", "weight_full"), ("bias", "bias_full")]
    )
    def test_unknown_shape_is_not_fused(self, _, name):
        model, _ = _make_model()
        value = model.graph.inputs[0] if name == "image" else model.graph.initializers[name]
        value.shape = None

        count = instance_norm_to_onnx_group_norm_rule.apply_to_model(model)

        self.assertEqual(count, 0)

    @parameterized.parameterized.expand([(1, 4, 1, 1), (4, 4), (1, 1, 1)])
    def test_other_affine_layouts_are_not_fused(self, *affine_shape):
        model, inputs = _make_model(affine_shape=affine_shape)
        expected = ort_run("original_group_norm", model, inputs)

        count = instance_norm_to_onnx_group_norm_rule.apply_to_model(model)

        self.assertEqual(count, 0)
        assert_allclose(ort_run("optimized_group_norm", model, inputs), expected)

    @parameterized.parameterized.expand(["float", "float16"])
    def test_dynamic_affine_parameters_are_preserved(self, dtype):
        model, inputs = _make_model(dtype=dtype)
        for name in ("weight_full", "bias_full"):
            value = model.graph.initializers.pop(name)
            inputs[name] = value.const_value.numpy()
            value.const_value = None
            model.graph.inputs.append(value)
        expected = ort_run("original_group_norm", model, inputs)

        count = instance_norm_to_onnx_group_norm_rule.apply_to_model(model)

        self.assertEqual(count, 1)
        self.assertNotIn("Transpose", {node.op_type for node in model.graph})
        self.assertNotIn("Cast", {node.op_type for node in model.graph})
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
        assert_allclose(ort_run("optimized_group_norm", model, inputs), expected)

    @parameterized.parameterized.expand(
        [
            (groups, channels, opset, dtype)
            for groups, channels in ((32, 320), (32, 32), (8, 40))
            for opset in (17, 21)
            for dtype in ("float", "float16")
        ]
    )
    @unittest.skipUnless(
        "CUDAExecutionProvider" in ort.get_available_providers(),
        "CUDAExecutionProvider is unavailable",
    )
    def test_reported_shapes_on_cuda(self, groups, channels, opset, dtype):
        model, inputs = _make_model(opset, groups, channels, dtype)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        def run():
            session = ort.InferenceSession(
                ir.to_proto(model).SerializeToString(),
                options,
                providers=["CUDAExecutionProvider"],
            )
            return session.run(None, inputs)

        expected = run()
        optimize_for_ort(model)
        assert_allclose(run(), expected)


if __name__ == "__main__":
    unittest.main()
