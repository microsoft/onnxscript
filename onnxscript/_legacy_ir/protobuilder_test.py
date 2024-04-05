import unittest

import numpy as np
import onnx.checker
import onnx.parser

from onnxscript._legacy_ir import irbuilder, protobuilder
from onnxscript.rewriter import pattern
from onnxscript.rewriter.onnxruntime import instance_to_group_normalization

op = pattern.onnxop


class ConcatSerializeTest(unittest.TestCase):
    def rule(self) -> pattern.RewriteRule:
        def concat_pattern(x, y, axis):
            seq = op.SequenceConstruct(x, y)
            return op.ConcatFromSequence(seq, axis=axis)

        def concat(x, y, axis):
            return op.Concat(x, y, axis=axis)

        return pattern.RewriteRule(concat_pattern, concat)

    def test_concat_serialize(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[M] z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = self.rule().apply_to_model(ir)
        # Tests related to IR
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.graph.nodes), 1)
        # Tests related to serialization to ModelProto
        model_proto = protobuilder.build_model_proto(ir)
        onnx.checker.check_model(model_proto)

    def test_concat_in_function_serialize(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[N] x, float[M] y) => (float[Z] z)
            {
                z = pkg.custom.afunction (x, y)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (x, y) => (z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = self.rule().apply_to_model(ir)
        # Tests related to IR
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.functions), 1)
        self.assertEqual(len(ir.functions[0].nodes), 1)
        self.assertEqual(ir.functions[0].nodes[0].op_type, "Concat")
        # Tests related to serialization to ModelProto
        model_proto = protobuilder.build_model_proto(ir)
        onnx.checker.check_model(model_proto)

    def test_concat_in_nested_function_serialize(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[N] x, float[M] y) => (float[Z] z)
            {
                z = pkg.custom.afunction (x, y)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17, "pkg.custom": 1]>
            afunction (x, y) => (z)
            {
                z = pkg.custom.nestedfunction(x, y)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            nestedfunction (x, y) => (z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = self.rule().apply_to_model(ir)
        # Tests related to IR
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.functions), 2)
        self.assertEqual(len(ir.functions[0].nodes), 1)
        self.assertEqual(len(ir.functions[1].nodes), 1)
        self.assertEqual(ir.functions[0].nodes[0].op_type, "nestedfunction")
        self.assertEqual(ir.functions[1].nodes[0].op_type, "Concat")
        # Tests related to serialization to ModelProto
        model_proto = protobuilder.build_model_proto(ir)
        onnx.checker.check_model(model_proto)


class ControlFlowSerializeTest(unittest.TestCase):
    def test_conditional_serialize(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 16, "local" : 1 ]>
            agraph (float[N] x) => (float[N] y)
            {
                f = Constant <value = bool {0}> ()
                t = Constant <value = bool {1}> ()
                y1 = local.myfun (f, x)
                y = local.myfun (t, y1)
            }
            <opset_import: [ "" : 16 ], domain: "local">
            myfun (b, lx) => (ly)
            {
                ly = If (b) <
                    then_branch = g1 () => (float[N] z_then)
                    {
                        two = Constant <value = float[1] {2.0}> ()
                        z_then =  Mul (lx, two)
                    },
                    else_branch = g2 () => (float[N] z_else)
                    {
                        three = Constant <value = float[1] {3.0}> ()
                        z_else =  Mul (lx, three)
                    }
                    >
            }
        """
        )
        ir = irbuilder.build_ir(model)
        # Tests related to serialization to ModelProto
        model_proto = protobuilder.build_model_proto(ir)
        onnx.checker.check_model(model_proto)

    def test_function_attribute_serialize(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 16, "local" : 1 ]>
            agraph (float[N] x) => (float[N] y)
            {
                f = Constant <value = bool {0}> ()
                t = Constant <value = bool {1}> ()
                y1 = local.myfun<a: int =1, b: int =2> (f, x)
                y = local.myfun<a: int =2> (t, y1)
            }
            <opset_import: [ "" : 16 ], domain: "local">
            myfun <a, b:int = 1>(l, lx) => (ly)
            {
                ly = Mul (l, lx)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        model_proto = protobuilder.build_model_proto(ir)
        onnx.checker.check_model(model_proto)
        function_proto = model_proto.functions[0]
        self.assertEqual(function_proto.attribute, ["a"])
        self.assertEqual(len(function_proto.attribute_proto), 1)
        b_attr_proto = function_proto.attribute_proto[0]
        self.assertEqual(b_attr_proto.name, "b")
        self.assertEqual(b_attr_proto.type, onnx.AttributeProto.INT)
        self.assertEqual(b_attr_proto.i, 1)

    def test_com_microsoft_opset_is_supported_in_protobuilder(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, shape_a)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, scale, B)
                shape_b = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, shape_b)
                mul_output = Mul (instance_norm_reshape, weight)
                output = Add (mul_output, bias)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight = np.random.rand(320, 1, 1).astype(np.float16)
        bias = np.random.rand(320, 1, 1).astype(np.float16)
        model.graph.initializer.extend(
            [
                onnx.helper.make_tensor(
                    "scale",
                    onnx.TensorProto.FLOAT16,
                    [32],
                    np.ones(32, dtype=np.float16),
                ),
                onnx.helper.make_tensor(
                    "B", onnx.TensorProto.FLOAT16, [32], np.zeros(32, dtype=np.float16)
                ),
                onnx.helper.make_tensor(
                    "weight", onnx.TensorProto.FLOAT16, [320, 1, 1], weight
                ),
                onnx.helper.make_tensor("bias", onnx.TensorProto.FLOAT16, [320, 1, 1], bias),
            ]
        )
        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 1)
        model_proto = protobuilder.build_model_proto(ir)
        onnx.checker.check_model(model_proto)


if __name__ == "__main__":
    unittest.main()
