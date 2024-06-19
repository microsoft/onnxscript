# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import contextlib
import io
import os
import unittest

import numpy as np
import onnx
import onnx.parser
import onnx.reference
import onnxruntime as ort

from onnxscript import ir
from onnxscript.rewriter import generic_pattern, pattern

FLOAT = onnx.TensorProto.FLOAT


class GenericPatternTest(unittest.TestCase):
    def _range(self, *shape, bias: float | None = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_graph_pattern_builder(self):
        """Test replacing Add + Add by AddAdd."""

        def match_pattern(op, x, y, z):
            """Builds the pattern to match."""
            tmp = op.Add(x, y)
            return op.Add(tmp, z)

        def apply_pattern(op, x, y, z, **_):
            """Builds the replacement graph."""
            return op.AddAdd(x, y, z, domain="ZZZ")

        def validate_mapping(context, x, y, z, **_) -> bool:
            """Validates the mapping."""
            del context
            return True

        rule = pattern.RewriteRule(
            match_pattern,
            apply_pattern,
            validate_mapping,
            generic_pattern.GenericPatternMatcher,
        )

        class AddAdd(onnx.reference.op_run.OpRun):
            op_domain = "ZZZ"

            def _run(self, x, y, z):
                return (x + y + z,)

        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node("Add", ["x", "y"], ["gggg"]),
                    onnx.helper.make_node("Add", ["gggg", "z"], ["final"]),
                ],
                "dummy",
                [
                    onnx.helper.make_tensor_value_info("x", FLOAT, [None, None]),
                    onnx.helper.make_tensor_value_info("y", FLOAT, [None, None]),
                    onnx.helper.make_tensor_value_info("z", FLOAT, [None, None]),
                ],
                [onnx.helper.make_tensor_value_info("final", FLOAT, [None, None])],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        )
        onnx.checker.check_model(model)

        model = onnx.shape_inference.infer_shapes(model)
        ir_model = ir.serde.deserialize_model(model)

        rule.apply_to_model(ir_model)
        self.assertEqual(
            ["AddAdd"],
            [n.op_type for n in ir_model.graph],
        )
        # TODO: do that in pattern.py.
        ir_model.opset_imports["ZZZ"] = 1
        rewriten_model = ir.serde.serialize_model(ir_model)
        self.assertEqual(
            ["AddAdd"],
            [n.op_type for n in rewriten_model.graph.node],
        )

        feeds = {
            "x": self._range(5, 6),
            "y": self._range(5, 6),
            "z": self._range(5, 6),
        }
        ref1 = onnx.reference.ReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(rewriten_model.graph.initializer))
        opsets = {v.domain: v.version for v in rewriten_model.opset_import}
        self.assertIn("ZZZ", opsets)
        self.assertEqual(opsets["ZZZ"], 1)

        ref2 = onnx.reference.ReferenceEvaluator(rewriten_model, new_ops=[AddAdd])
        got = ref2.run(None, feeds)
        np.testing.assert_almost_equal(expected[0], got[0])

    def test_graph_pattern_builder_multi_outputs(self):
        def match_pattern(op, x, y, w, z):
            """Builds the pattern to match."""
            tmp = op.Add(x, y)
            tmp2 = op.Add(tmp, w)
            r1 = op.Add(tmp, z)
            return tmp2, r1

        def apply_pattern(op, x, y, w, z, **_):
            """Builds the pattern to match."""
            return op.AddAddAddAdd(x, y, w, z, domain="ZZZ", outputs=2)

        def validate_mapping(context, **_) -> bool:
            return True

        rule = pattern.RewriteRule(
            match_pattern,
            apply_pattern,
            validate_mapping,
            generic_pattern.GenericPatternMatcher,
            verbose=10,
        )

        class AddAddAddAdd(onnx.reference.op_run.OpRun):
            op_domain = "ZZZ"

            def _run(self, x, y, w, z):
                return (x + y + w, x + y + z)

        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node("Add", ["x", "y"], ["gggg"]),
                    onnx.helper.make_node("Add", ["gggg", "w"], ["f1"]),
                    onnx.helper.make_node("Add", ["gggg", "z"], ["f2"]),
                ],
                "dummy",
                [
                    onnx.helper.make_tensor_value_info("x", FLOAT, [None, None]),
                    onnx.helper.make_tensor_value_info("y", FLOAT, [None, None]),
                    onnx.helper.make_tensor_value_info("z", FLOAT, [None, None]),
                    onnx.helper.make_tensor_value_info("w", FLOAT, [None, None]),
                ],
                [
                    onnx.helper.make_tensor_value_info("f1", FLOAT, [None, None]),
                    onnx.helper.make_tensor_value_info("f2", FLOAT, [None, None]),
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        )
        onnx.checker.check_model(model)

        model = onnx.shape_inference.infer_shapes(model)
        ir_model = ir.serde.deserialize_model(model)

        rule.apply_to_model(ir_model)
        self.assertEqual(
            ["AddAddAddAdd"],
            [n.op_type for n in ir_model.graph],
        )
        # TODO: do that in pattern.py.
        ir_model.opset_imports["ZZZ"] = 1

        rewriten_model = ir.serde.serialize_model(ir_model)

        self.assertEqual(
            ["AddAddAddAdd"],
            [n.op_type for n in rewriten_model.graph.node],
        )

        feeds = {
            "x": self._range(5, 6),
            "y": self._range(5, 6),
            "w": self._range(5, 6),
            "z": self._range(5, 6),
        }
        ref1 = onnx.reference.ReferenceEvaluator(model)
        expected = ref1.run(None, feeds)

        self.assertEqual(0, len(rewriten_model.graph.initializer))
        opsets = {v.domain: v.version for v in rewriten_model.opset_import}
        self.assertIn("ZZZ", opsets)
        self.assertEqual(opsets["ZZZ"], 1)

        ref2 = onnx.reference.ReferenceEvaluator(rewriten_model, new_ops=[AddAddAddAdd])
        got = ref2.run(None, feeds)
        np.testing.assert_almost_equal(expected[0], got[0])

    def check_with_ort(self, model: onnx.ModelProto, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if isinstance(model, onnx.ModelProto):
            model = model.SerializeToString()
        session = ort.InferenceSession(model, providers=providers)
        return session

    def get_rotary_model(self):
        inputs = [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64, shape=[]),
            onnx.helper.make_tensor_value_info("pos_ids", FLOAT, shape=[]),
            onnx.helper.make_tensor_value_info("axis", onnx.TensorProto.INT64, shape=[]),
        ]
        nodes = [
            onnx.helper.make_node("Unsqueeze", ["x", "axis"], ["_onx_unsqueeze0"]),
            onnx.helper.make_node("Cast", ["_onx_unsqueeze0"], ["_onx_cast0"], to=1),
            onnx.helper.make_node("MatMul", ["pos_ids", "_onx_cast0"], ["_onx_matmul0"]),
            onnx.helper.make_node("Transpose", ["_onx_matmul0"], ["_onx_transpose0"]),
            onnx.helper.make_node(
                "ConcatTraining",
                ["_onx_transpose0", "_onx_transpose0"],
                ["_onx_concattraining0", "_onx_concattraining1"],
                domain="com.microsoft",
            ),
            onnx.helper.make_node("Sin", ["_onx_concattraining0"], ["_onx_sin0"]),
            onnx.helper.make_node("Cast", ["_onx_sin0"], ["_onx_cast02"], to=1),
            onnx.helper.make_node("Cos", ["_onx_concattraining0"], ["_onx_cos0"]),
            onnx.helper.make_node("Cast", ["_onx_cos0"], ["_onx_cast03"], to=1),
        ]
        outputs = [
            onnx.helper.make_tensor_value_info("_onx_cast02", onnx.TensorProto.UNDEFINED, []),
            onnx.helper.make_tensor_value_info("_onx_cast03", onnx.TensorProto.UNDEFINED, []),
        ]
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes,
                "experiment",
                inputs,
                outputs,
            ),
            opset_imports=[
                onnx.helper.make_opsetid("", 18),
                onnx.helper.make_opsetid("com.microsoft", 18),
            ],
        )
        return model

    def test_shared_root_value_test(self):
        def match_pattern(op, x):
            t1 = op.Sin(x)
            t2 = op.Cos(x)
            return t1, t2

        def apply_pattern(op, x, **_):
            return op.SinCos(x, domain="com.microsoft", outputs=2)

        rule = pattern.RewriteRule(
            match_pattern,
            apply_pattern,
            matcher=generic_pattern.GenericPatternMatcher,
        )
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] y) => (float[N] z)
            {
                temp1 = Sin(y)
                temp2 = Cos(y)
                z = Add(temp1, temp2)
            }
        """
        )
        onnx.checker.check_model(model_proto)
        model = onnx.shape_inference.infer_shapes(model_proto)
        ir_model = ir.serde.deserialize_model(model)
        rule.apply_to_model(ir_model)
        rewritten_model = ir.serde.serialize_model(ir_model)
        graph = rewritten_model.graph
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0].op_type, "SinCos")

    def test_rotary_embedding(self):
        # The test work on a model if it has the expected name.
        # A dummy model is used if not present (not implemented yet).

        def match_pattern(op, x, pos_ids, axis):
            # original code: the code does verifies the constant yet
            # unsqueeze = op.Unsqueeze(x, [1])

            unsqueeze = op.Unsqueeze(x, axis)
            cast = op.Cast(unsqueeze, to=FLOAT)

            matmul = op.MatMul(pos_ids, cast)
            transpose = op.Transpose(matmul)
            output, _length = op.ConcatTraining(
                transpose,
                transpose,
                domain="com.microsoft",
                outputs=2,
            )

            sin = op.Sin(output)
            cast1 = op.Cast(sin, to=FLOAT)
            cos = op.Cos(output)
            cast2 = op.Cast(cos, to=FLOAT)
            return cast1, cast2

        def validate_mapping(match_result, **_) -> bool:
            del match_result
            return True

        def apply_pattern(op, x, pos_ids, axis, **_):
            del axis
            cos_cache = op.Constant(
                value=onnx.numpy_helper.from_array(np.random.rand(256, 256).astype(np.float16))
            )
            sin_cache = op.Constant(
                value=onnx.numpy_helper.from_array(np.random.rand(256, 256).astype(np.float16))
            )
            return op.RotaryEmbedding(
                x,
                pos_ids,
                cos_cache,
                sin_cache,
                domain="com.microsoft",
                outputs=2,
            )

        rule = pattern.RewriteRule(
            match_pattern,
            apply_pattern,
            validate_mapping,
            generic_pattern.GenericPatternMatcher,
            verbose=10,
        )

        model = self.get_rotary_model()

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            # back to ir
            model = onnx.shape_inference.infer_shapes(model)
            ir_model = ir.serde.deserialize_model(model)

            # starts matching
            rule.apply_to_model(ir_model)
            ir_model.opset_imports["com.microsoft"] = 1

            rewriten_model = ir.serde.serialize_model(ir_model)

        expected = ["Constant", "Constant", "RotaryEmbedding"]
        self.assertEqual(expected, [n.op_type for n in rewriten_model.graph.node])
        out = buffer.getvalue()
        # TODO(Rama): What is this assertion testing? Is it to check that `verbose` is working?
        self.assertIn("[GenericPatternMatcher.match", out)

    def test_rotary_embedding_onnxscript(self):
        # The test work on a model if it has the expected name.
        # A dummy model is used if not present (not implemented yet).

        def rotary_match_pattern(op, x, pos_ids, axis):
            unsqueeze = op.Unsqueeze(x, axis)
            cast = op.Cast(unsqueeze, to=FLOAT)

            matmul = op.MatMul(pos_ids, cast)
            transpose = op.Transpose(matmul)
            output, _length = op.ConcatTraining(
                transpose, transpose, domain="com.microsoft", outputs=2
            )

            sin = op.Sin(output)
            cast1 = op.Cast(sin, to=FLOAT)
            cos = op.Cos(output)
            cast2 = op.Cast(cos, to=FLOAT)
            return cast1, cast2

        def validate_rotary_mapping(match_result, **_) -> bool:
            # If some pattern needs to be rejected.
            del match_result
            return True

        def rotary_apply_pattern(op, x, pos_ids, axis, **_):
            cos_cache = op.Constant(
                value=onnx.numpy_helper.from_array(np.random.rand(256, 256).astype(np.float16))
            )
            sin_cache = op.Constant(
                value=onnx.numpy_helper.from_array(np.random.rand(256, 256).astype(np.float16))
            )
            part1, part2 = op.RotaryEmbedding(
                x, pos_ids, cos_cache, sin_cache, domain="com.microsoft", outputs=2
            )
            return part1, part2

        rule = pattern.RewriteRule(
            rotary_match_pattern,
            rotary_apply_pattern,
            validate_rotary_mapping,
            generic_pattern.GenericPatternMatcher,
            verbose=10,
        )

        model = self.get_rotary_model()

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            # back to ir
            model = onnx.shape_inference.infer_shapes(model)
            ir_model = ir.serde.deserialize_model(model)

            # starts matching
            rule.apply_to_model(ir_model)
            ir_model.opset_imports["com.microsoft"] = 1

            rewriten_model = ir.serde.serialize_model(ir_model)

        expected = ["Constant", "Constant", "RotaryEmbedding"]
        self.assertEqual(expected, [n.op_type for n in rewriten_model.graph.node])
        out = buffer.getvalue()
        # TODO(justinchuby): Remove this assert - capturing stdout is not robust
        self.assertIn("[GenericPatternMatcher.match", out)

    def test_rotary_emb_file_onnxscript(self):
        # The test work on a model if it has the expected name.
        # A dummy model is used if not present (not implemented yet).

        def rotary_match_pattern(op, x, pos_ids, axis):
            unsqueeze = op.Unsqueeze(x, axis)
            cast = op.Cast(unsqueeze, to=FLOAT)

            matmul = op.MatMul(pos_ids, cast)
            transpose = op.Transpose(matmul)
            output, _length = op.ConcatTraining(
                transpose, transpose, domain="com.microsoft", outputs=2
            )

            sin = op.Sin(output)
            cast1 = op.Cast(sin, to=FLOAT)
            cos = op.Cos(output)
            cast2 = op.Cast(cos, to=FLOAT)
            return cast1, cast2

        def validate_rotary_mapping(match_result, **_) -> bool:
            # If some pattern needs to be rejected.
            del match_result
            return True

        def rotary_apply_pattern(op, x, pos_ids, axis):
            cos_cache = op.Constant(
                value=onnx.numpy_helper.from_array(np.random.rand(256, 256).astype(np.float16))
            )
            sin_cache = op.Constant(
                value=onnx.numpy_helper.from_array(np.random.rand(256, 256).astype(np.float16))
            )
            part1, part2 = op.RotaryEmbedding(
                x, pos_ids, cos_cache, sin_cache, domain="com.microsoft", outputs=2
            )
            return part1, part2

        model_path = "gemma_optimized_pre_grad_training_2.onnx"
        if not os.path.exists(model_path):
            raise unittest.SkipTest(f"{model_path!r} is missing")
        model = onnx.load(model_path)
        model = onnx.shape_inference.infer_shapes(model)
        ir_model = ir.serde.deserialize_model(model)

        rule = pattern.RewriteRule(
            rotary_match_pattern,
            rotary_apply_pattern,
            validate_rotary_mapping,
            generic_pattern.GenericPatternMatcher,
            verbose=10,
        )

        rule.apply_to_model(ir_model)
        # TODO: do that in pattern.py.
        ir_model.opset_imports["ZZZ"] = 1

        rewriten_model = ir.serde.serialize_model(ir_model)

        buffer = rewriten_model.SerializeToString()
        with open(f"{model}.opt.onnx", "wb") as f:
            f.write(buffer)
        self.check_with_ort(rewriten_model)

    def test_transpose_transpose_onnxscript(self):
        # TODO(rama): Attribute-parameters not yet supported in multi-output matching.
        # def transpose_transpose_pattern(op, X, perm0, perm1):
        #     xt = op.Transpose(X, perm=perm0)
        #     Y = op.Transpose(xt, perm=perm1)
        #     return Y

        def transpose_transpose_pattern(op, X):
            XT = op.Transpose(X, outputs=["XT"])
            Y = op.Transpose(XT, outputs=["Y"])
            return Y

        def transpose_transpose_mapping(perm0, perm1):
            new_perm = [0 for p in perm0]
            for i, p in enumerate(perm1):
                new_perm[i] = perm0[p]
            # replace by return [perm0[p] for p in perm1] ?
            return new_perm

        def transpose_transpose_check(op, **_) -> bool:
            return True

        def transpose_transpose_apply_pattern(op, X, XT: ir.Value, Y, **_):
            perm0 = XT.producer().attributes.get("perm")
            if perm0 is not None:
                perm0 = perm0.value  # TODO(rama): handle RefAttr
            perm1 = Y.producer().attributes.get("perm")
            if perm1 is not None:
                perm1 = perm1.value  # TODO(rama): handle RefAttr
            if perm0 is None and perm1 is None:
                return op.Identity(X)
            if perm0 is None:
                perm0 = range(len(perm1) - 1, -1, -1)
            if perm1 is None:
                perm1 = range(len(perm0) - 1, -1, -1)
            composed_perm = transpose_transpose_mapping(perm0, perm1)
            return op.Transpose(X, perm=composed_perm)

        rule = pattern.RewriteRule(
            transpose_transpose_pattern,
            transpose_transpose_apply_pattern,
            transpose_transpose_check,
            generic_pattern.GenericPatternMatcher,
            verbose=0,
        )

        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node("Transpose", ["X"], ["xt"], perm=[1, 2, 0]),
                    onnx.helper.make_node("Transpose", ["xt"], ["Y"], perm=[1, 2, 0]),
                ],
                "name",
                [onnx.helper.make_tensor_value_info("X", FLOAT, [None, None, None])],
                [onnx.helper.make_tensor_value_info("Y", FLOAT, [None, None, None])],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

        # back to ir
        ir_model = ir.serde.deserialize_model(model)

        # starts matching

        rule.apply_to_model(ir_model)
        rewriten_model = ir.serde.serialize_model(ir_model)

        expected = ["Transpose"]
        self.assertEqual(expected, [n.op_type for n in rewriten_model.graph.node])
        node = rewriten_model.graph.node[0]
        self.assertEqual(len(node.attribute), 1)
        att = node.attribute[0]
        self.assertEqual(att.name, "perm")
        self.assertEqual(list(att.ints), [2, 0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
