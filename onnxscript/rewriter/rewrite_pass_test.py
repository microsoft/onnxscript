# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import contextlib
import io
import unittest

import onnx_ir as ir

from onnxscript import rewriter


class RewritePassTest(unittest.TestCase):
    def rule(self, *, verbose=0):
        def target(op, x):
            return op.Neg(op.Neg(x))

        def replacement(op, x):
            return op.Identity(x)

        return rewriter.RewriteRule(target, replacement, name="DoubleNeg", verbose=verbose)

    def model(self, *, matches=True):
        first_op = "Neg" if matches else "Abs"
        return ir.from_onnx_text(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            graph (float[N] x) => (float[N] z)
            {{
                t = {first_op}(x)
                z = Neg(t)
            }}
            """
        )

    def test_default_is_quiet_and_keeps_summary_log(self):
        model = self.model()
        buffer = io.StringIO()
        with self.assertLogs("onnxscript.rewriter") as logs, contextlib.redirect_stdout(
            buffer
        ):
            result = rewriter.RewritePass([self.rule()])(model)

        self.assertEqual(buffer.getvalue(), "")
        self.assertEqual(
            logs.output,
            ["INFO:onnxscript.rewriter:Applied 1 of general pattern rewrite rules."],
        )
        self.assertTrue(result.modified)
        self.assertIs(result.model, model)
        self.assertEqual([node.op_type for node in model.graph], ["Identity"])

    def test_verbose_for_sequence_and_rule_set(self):
        for use_rule_set in (False, True):
            with self.subTest(use_rule_set=use_rule_set):
                rules = [self.rule()]
                if use_rule_set:
                    rules = rewriter.RewriteRuleSet(rules)
                model = self.model()
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    result = ir.passes.PassManager([rewriter.RewritePass(rules, verbose=1)])(
                        model
                    )

                self.assertIn("DoubleNeg: Matched Nodes", buffer.getvalue())
                self.assertIn("Replacement Nodes", buffer.getvalue())
                self.assertTrue(result.modified)
                self.assertIs(result.model, model)
                self.assertEqual([node.op_type for node in model.graph], ["Identity"])

    def test_none_preserves_rule_verbosity_and_zero_overrides_it(self):
        for verbose in (None, 0):
            with self.subTest(verbose=verbose):
                model = self.model()
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    result = rewriter.RewritePass([self.rule(verbose=1)], verbose=verbose)(
                        model
                    )

                self.assertEqual(bool(buffer.getvalue()), verbose is None)
                self.assertTrue(result.modified)
                self.assertEqual([node.op_type for node in model.graph], ["Identity"])

    def test_failed_match_reports_reason_without_rewriting(self):
        model = self.model(matches=False)
        before = ir.serde.serialize_model(model).SerializeToString()
        buffer = io.StringIO()
        with self.assertNoLogs("onnxscript.rewriter"), contextlib.redirect_stdout(buffer):
            result = rewriter.RewritePass([self.rule()], verbose=3)(model)

        self.assertIn("DoubleNeg", buffer.getvalue())
        self.assertIn("OpType mismatch: expected Neg, got Abs", buffer.getvalue())
        self.assertFalse(result.modified)
        self.assertEqual(ir.serde.serialize_model(model).SerializeToString(), before)


if __name__ == "__main__":
    unittest.main()
