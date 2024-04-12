import unittest

import onnx.checker
import onnx.parser

from onnxscript._legacy_ir import irbuilder, protobuilder
from onnxscript.rewriter import pattern

op = pattern.onnxop


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


if __name__ == "__main__":
    unittest.main()
