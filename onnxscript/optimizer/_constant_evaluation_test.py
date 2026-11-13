# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import contextlib
import dataclasses
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
import onnx_ir as ir
import parameterized

from onnxscript import optimizer


def _model(version: int = 1) -> ir.Model:
    return ir.from_onnx_text(
        f"""
        <ir_version: 9, opset_import: ["test.custom": {version}]>
        graph () => (float[2] y)
        <float[2] x = {{1.0, 2.0}}>
        {{
            y = test.custom.Double <scale=2.0>(x)
        }}
        """
    )


class ConstantEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.evaluators = optimizer.ConstantEvaluatorRegistry()
        self.callback = mock.Mock(
            side_effect=lambda inputs, attrs, ctx: [
                ir.tensor(inputs[0].numpy() * attrs["scale"].as_float())
            ]
        )
        self.evaluators.register("Double", domain="test.custom", version=1)(self.callback)

    def _fold(self, model=None, **kwargs):
        if model is None:
            model = _model()
        optimizer.fold_constants(model, evaluators=self.evaluators, **kwargs)
        onnx.checker.check_model(ir.to_proto(model))
        return model

    def test_constant_inputs_attributes_and_context(self):
        model = self._fold()
        self.assertFalse(any(n.op_type == "Double" for n in model.graph))
        output = model.graph.outputs[0]
        self.assertEqual(output.name, "y")
        np.testing.assert_array_equal(output.const_value.numpy(), [2.0, 4.0])
        self.assertIn("x", output.meta[optimizer.FOLDED_FROM_KEY])
        inputs, attrs, context = self.callback.call_args.args
        self.assertEqual(len(inputs), 1)
        self.assertEqual(attrs["scale"].as_float(), 2.0)
        self.assertEqual(
            context, optimizer.ConstantEvaluationContext("test.custom", "Double", 1, 1)
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            context.version = 2
        with self.assertRaises(TypeError):
            attrs["other"] = ir.AttrInt64("other", 1)

    @parameterized.parameterized.expand(["fold_constants", "optimize", "optimize_ir"])
    def test_public_entrypoints(self, entrypoint):
        model = _model()
        getattr(optimizer, entrypoint)(
            model, evaluators=self.evaluators, onnx_shape_inference=False
        )
        np.testing.assert_array_equal(model.graph.outputs[0].const_value.numpy(), [2, 4])
        onnx.checker.check_model(ir.to_proto(model))

    @parameterized.parameterized.expand(["fold_constants", "optimize"])
    def test_model_proto(self, entrypoint):
        model = ir.to_proto(_model())
        result = getattr(optimizer, entrypoint)(
            model, evaluators=self.evaluators, onnx_shape_inference=False
        )
        if entrypoint == "optimize":
            model = result
        onnx.checker.check_model(model)
        np.testing.assert_array_equal(
            ir.from_proto(model).graph.outputs[0].const_value.numpy(), [2, 4]
        )

    def test_no_global_registration(self):
        custom = self._fold()
        default = _model()
        optimizer.fold_constants(default)
        self.assertEqual(default.graph.node(0).op_type, "Double")
        other = optimizer.ConstantEvaluatorRegistry()
        other.register("Double", domain="test.custom")(
            lambda inputs, attrs, ctx: [ir.tensor(np.array([5, 6], dtype=np.float32))]
        )
        optimizer.fold_constants(default, evaluators=other)
        np.testing.assert_array_equal(default.graph.outputs[0].const_value.numpy(), [5, 6])
        np.testing.assert_array_equal(custom.graph.outputs[0].const_value.numpy(), [2, 4])

    @parameterized.parameterized.expand([("version",), ("should_fold",), ("input_size",)])
    def test_policy_prevents_callback(self, policy):
        model = _model(version=2 if policy == "version" else 1)
        kwargs = {}
        if policy == "should_fold":
            kwargs["should_fold"] = lambda node: False
        elif policy == "input_size":
            kwargs["input_size_limit"] = 1
        self._fold(model, **kwargs)
        self.callback.assert_not_called()
        self.assertEqual(model.graph.node(0).op_type, "Double")

    def test_nonconstant_input_is_not_evaluated(self):
        model = _model()
        model.graph.initializers.pop("x")
        model.graph.inputs.append(model.graph.node(0).inputs[0])
        self._fold(model)
        self.callback.assert_not_called()

    def test_graph_input_initializer_is_preserved(self):
        model = _model()
        model.graph.inputs.append(model.graph.initializers["x"])
        self._fold(model)
        self.callback.assert_not_called()

    @parameterized.parameterized.expand(
        [
            ("missing_ir", False, False, False),
            ("missing_proto", False, True, False),
            ("missing_forced", False, False, True),
            ("readable_ir", True, False, False),
            ("readable_proto", True, True, False),
        ]
    )
    def test_external_input_availability(self, _, readable, as_proto, force):
        model = _model()

        def release_inputs():
            for call in self.callback.call_args_list:
                for tensor in call.args[0]:
                    if isinstance(tensor, ir.ExternalTensor):
                        tensor.release()

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as cleanup:
            cleanup.callback(release_inputs)
            location = os.path.join(directory, "input.bin")
            if readable:
                np.array([3, 5], dtype=np.float32).tofile(location)
            tensor = ir.ExternalTensor(
                location=location,
                offset=0,
                length=8,
                dtype=ir.DataType.FLOAT,
                shape=ir.Shape([2]),
                name="x",
            )
            cleanup.callback(tensor.release)
            model.graph.initializers["x"].const_value = tensor
            node = model.graph.node(0)
            if as_proto:
                model = ir.to_proto(model)
            kwargs = {"should_fold": lambda node: True} if force else {}
            if readable:
                optimizer.fold_constants(model, evaluators=self.evaluators, **kwargs)
            else:
                with self.assertLogs("onnxscript.optimizer", level="WARNING") as logs:
                    optimizer.fold_constants(model, evaluators=self.evaluators, **kwargs)
                self.assertTrue(any("x" in message for message in logs.output))
            if as_proto:
                model = ir.from_proto(model)
            if readable:
                self.callback.assert_called_once()
                np.testing.assert_array_equal(
                    model.graph.outputs[0].const_value.numpy(), [6, 10]
                )
                onnx.checker.check_model(ir.to_proto(model))
            else:
                self.callback.assert_not_called()
                self.assertEqual(model.graph.node(0).op_type, "Double")
                if not as_proto:
                    self.assertIs(model.graph.node(0), node)
                preserved = model.graph.initializers["x"].const_value
                self.assertIsInstance(preserved, ir.ExternalTensor)
                self.assertEqual(preserved.location, location)

    def test_decline_preserves_node(self):
        self.callback.side_effect = None
        self.callback.return_value = None
        model = self._fold()
        self.assertEqual(model.graph.node(0).op_type, "Double")

    def test_control_flow_is_not_evaluated(self):
        model = _model()
        model.graph.node(0).attributes["body"] = ir.AttrGraph(
            "body", ir.Graph([], [], nodes=[])
        )
        self._fold(model)
        self.callback.assert_not_called()

    def test_decline_falls_back_to_reference_evaluation(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 9, opset_import: ["": 18]>
            graph () => (float[2] y)
            <float[2] x = {1.0, 2.0}> {
                y = Neg(x)
            }
            """
        )
        callback = mock.Mock(return_value=None)
        self.evaluators.register("Neg", domain="ai.onnx")(callback)
        self._fold(model)
        callback.assert_called_once()
        np.testing.assert_array_equal(model.graph.outputs[0].const_value.numpy(), [-1, -2])

    def test_builtin_partial_simplification_has_precedence(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 9, opset_import: ["": 18]>
            graph () => (float[2] y)
            <float[2] x = {1.0, 2.0}> {
                y = Concat <axis=0>(x)
            }
            """
        )
        callback = mock.Mock(return_value=None)
        self.evaluators.register("Concat")(callback)
        self._fold(model)
        callback.assert_not_called()

    def test_registrations_are_snapshotted_per_pass(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 9, opset_import: ["test.custom": 1]>
            graph () => (float[2] y)
            <float[2] x = {1.0, 2.0}> {
                mid = test.custom.Double <scale=2.0>(x)
                y = test.custom.Late(mid)
            }
            """
        )
        late = mock.Mock(side_effect=lambda inputs, attrs, ctx: [inputs[0]])

        def double(inputs, attributes, context):
            self.evaluators.register("Late", domain="test.custom")(late)
            return [ir.tensor(inputs[0].numpy() * 2)]

        self.callback.side_effect = double
        self._fold(model)
        late.assert_not_called()
        self.assertEqual(model.graph.node(0).op_type, "Late")
        self._fold(model)
        late.assert_called_once()
        np.testing.assert_array_equal(model.graph.outputs[0].const_value.numpy(), [2, 4])

    @parameterized.parameterized.expand(
        [
            ("invalid_attribute", ValueError("bad attribute")),
            ("callback_file", FileNotFoundError("callback configuration file is missing")),
        ]
    )
    def test_callback_failure_has_context(self, _, failure):
        self.callback.side_effect = failure
        with self.assertRaisesRegex(RuntimeError, r"test.custom::Double, opset 1") as error:
            self._fold()
        self.assertIs(error.exception.__cause__, failure)

    @parameterized.parameterized.expand(
        [
            ("arity", [], ValueError, "sequence of 1 tensors"),
            ("not_sequence", np.array([2, 4]), ValueError, "sequence"),
            ("tensor_type", [np.array([2, 4])], TypeError, "ir.TensorProtocol"),
            ("dtype", [ir.tensor(np.array([2, 4], dtype=np.int64))], ValueError, "dtype"),
            ("shape", [ir.tensor(np.array([2], dtype=np.float32))], ValueError, "shape"),
        ]
    )
    def test_invalid_result(self, _, result, error, message):
        self.callback.side_effect = None
        self.callback.return_value = result
        model = _model()
        with self.assertRaisesRegex(error, message):
            self._fold(model)
        self.assertEqual(model.graph.node(0).op_type, "Double")
        self.assertEqual(set(model.graph.initializers), {"x"})

    def test_output_size_limit(self):
        model = _model()
        model.graph.outputs[0].shape = ir.Shape([3])
        self.callback.side_effect = None
        self.callback.return_value = [ir.tensor(np.ones(3, dtype=np.float32))]
        self._fold(model, output_size_limit=2)
        self.assertEqual(model.graph.node(0).op_type, "Double")

    def test_no_growth_can_exceed_output_limit(self):
        model = self._fold(output_size_limit=1)
        np.testing.assert_array_equal(model.graph.outputs[0].const_value.numpy(), [2, 4])

    def test_optional_inputs_and_multiple_outputs(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 9, opset_import: ["test.custom": 1]>
            graph () => (int64 a, int64 b) {
                a, b = test.custom.Pair(, )
            }
            """
        )
        callback = mock.Mock(return_value=[ir.tensor(np.array(3)), ir.tensor(np.array(4))])
        self.evaluators.register("Pair", domain="test.custom")(callback)
        self._fold(model)
        self.assertTrue(all(v.const_value is not None for v in model.graph.outputs))
        self.assertEqual(callback.call_args.args[2].output_count, 2)
        self.assertTrue(all(value is None for value in callback.call_args.args[0]))
        np.testing.assert_array_equal(model.graph.outputs[0].const_value.numpy(), 3)
        np.testing.assert_array_equal(model.graph.outputs[1].const_value.numpy(), 4)

    def test_multi_output_size_limit_counts_removed_inputs_once(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 9, opset_import: ["test.custom": 1]>
            graph () => (float[2] y, float[2] z)
            <float[2] x = {1.0, 2.0}> {
                y, z = test.custom.Pair(x)
            }
            """
        )
        self.evaluators.register("Pair", domain="test.custom")(
            lambda inputs, attrs, ctx: [
                ir.tensor(np.ones(2, dtype=np.float32)),
                ir.tensor(np.ones(2, dtype=np.float32)),
            ]
        )
        self._fold(model, output_size_limit=1)
        self.assertEqual(model.graph.node(0).op_type, "Pair")

    def test_local_function_uses_own_version_and_adds_constant_opset(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 9, opset_import: ["test.custom": 2, "local": 1]>
            graph () => (float[2] y) {
                y = local.F()
            }
            <domain: "local", opset_import: ["test.custom": 1]>
            F () => (y) {
                y = test.custom.Value()
            }
            """
        )
        callback = mock.Mock(return_value=[ir.tensor(np.array([7, 8], dtype=np.float32))])
        self.evaluators.register("Value", domain="test.custom", version=1)(callback)
        self._fold(model)
        function = next(iter(model.functions.values()))
        self.assertEqual(function.graph.node(0).op_type, "Constant")
        self.assertIn("", function.opset_imports)
        self.assertEqual(callback.call_args.args[2].version, 1)

    @parameterized.parameterized.expand(["RandomUniform", "ConstantOfShape"])
    def test_nondeterminism_and_blacklist(self, op_type):
        node = ir.Node("", op_type, [], num_outputs=1)
        model = ir.Model(
            ir.Graph([], list(node.outputs), nodes=[node], opset_imports={"": 18}),
            ir_version=9,
        )
        callback = mock.Mock(return_value=[ir.tensor(np.array(1, dtype=np.float32))])
        self.evaluators.register(op_type)(callback)
        optimizer.fold_constants(model, evaluators=self.evaluators)
        callback.assert_not_called()


class ConstantEvaluatorRegistryTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("all", None, [1, 2, 3, 20]),
            ("exact", 2, [2]),
            ("range", (2, 3), [2, 3]),
            ("lower", (2, None), [2, 3, 20]),
            ("upper", (None, 2), [1, 2]),
        ]
    )
    def test_versions(self, _, version, selected):
        registry = optimizer.ConstantEvaluatorRegistry()

        def callback(inputs, attrs, ctx):
            return None

        self.assertIs(registry.register("Double", "test.custom", version)(callback), callback)
        for opset in (1, 2, 3, 20):
            self.assertEqual(
                registry._lookup("test.custom", "Double", opset) is callback,
                opset in selected,
            )

    @parameterized.parameterized.expand(
        [
            ("string", "1", TypeError),
            ("bool", True, TypeError),
            ("zero", 0, ValueError),
            ("negative", -1, ValueError),
            ("short_tuple", (1,), TypeError),
            ("long_tuple", (1, 2, 3), TypeError),
            ("float_endpoint", (1, 2.0), ValueError),
            ("bool_endpoint", (True, 2), ValueError),
            ("inverted", (3, 2), ValueError),
        ]
    )
    def test_invalid_version(self, _, version, error):
        registry = optimizer.ConstantEvaluatorRegistry()
        with self.assertRaisesRegex(error, "version"):
            registry.register("Double", version=version)

    def test_domain_normalization_and_overlap(self):
        registry = optimizer.ConstantEvaluatorRegistry()

        def callback(inputs, attrs, ctx):
            return None

        registry.register("Foo", domain="ai.onnx", version=(1, 3))(callback)
        self.assertIs(registry._lookup("", "Foo", 2), callback)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            registry.register("Foo", version=(3, None))(callback)
        registry.register("Foo", version=(4, None))(callback)
        self.assertIs(registry._lookup("ai.onnx", "Foo", 4), callback)

    @parameterized.parameterized.expand(
        [
            ("empty", "", "", ValueError),
            ("whitespace", " ", "", ValueError),
            ("name_type", 1, "", TypeError),
            ("domain_type", "Foo", None, TypeError),
        ]
    )
    def test_invalid_names(self, _, name, domain, error):
        with self.assertRaises(error):
            optimizer.ConstantEvaluatorRegistry().register(name, domain)

    def test_noncallable(self):
        with self.assertRaisesRegex(TypeError, "callable"):
            optimizer.ConstantEvaluatorRegistry().register("Foo")(42)


if __name__ == "__main__":
    unittest.main()
