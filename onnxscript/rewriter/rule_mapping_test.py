import unittest

import onnx.parser

from onnxscript._legacy_ir import irbuilder
from onnxscript.rewriter.pattern import RewriteRuleSet
from onnxscript.rewriter import broadcast_to_matmul, gemm_to_matmul_add


class RuleMappingTest(unittest.TestCase):
    def test_rule_mapping_for_reshape(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 4, 512, 512] input_x, float[4, 512, 64] input_y, float[4, 512, 64] input_z) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )

        ir = irbuilder.build_ir(model)
        new_rules = broadcast_to_matmul.rules.rules + [gemm_to_matmul_add.rule]
        rule_set = RewriteRuleSet(new_rules)
        count = rule_set.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.graph.nodes), 4)


if __name__ == "__main__":
    unittest.main()
