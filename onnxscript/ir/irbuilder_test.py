import unittest

import onnx.parser

from onnxscript.ir import irbuilder


class IRBuilderTest(unittest.TestCase):
    def test_irbuilder(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                three = Constant <value_int=3>()
                x_cube = Pow(x, three)
                B = Constant <value_float=0.044715>()
                x_cube_mul_B = Mul(x_cube, B)
                sum = Add(x, x_cube_mul_B)
                C = Constant <value_float=0.79788>()
                C_times_sum = Mul(C, sum)
                tanh = Tanh(C_times_sum)
                one = Constant <value_float=1.0> ()
                one_plus_tanh = Add(one, tanh)
                half = Constant <value_float=0.5> ()
                half_x = Mul(half, x)
                z = Mul(one_plus_tanh, half_x)
            }
            """
        )
        irbuilder.build_ir(model)

    def test_shape_is_accessible_for_graph_value_with_value_info(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            <float[N] t>
            {
                t = Add (x, y)
                z = Add (t, x)
            }
        """
        )
        irmodel = irbuilder.build_ir(model)
        self.assertEqual(
            irmodel.graph.nodes[0].outputs[0].tensor_shape_proto(),
            onnx.TensorShapeProto(dim=[onnx.TensorShapeProto.Dimension(dim_param="N")]),
        )

    def test_shape_is_accessible_for_function_value_with_experimental_value_info(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (x, y) => (z)
            {
                o = MatMul (x, y)
                shape = Constant <value_ints=[1, 1, 0, 0]> ()
                z = Reshape (o, shape)
            }
        """
        )
        # Hack to put value_info in since parser does not support this experimental naming format
        model.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/o", onnx.TensorProto.FLOAT, ["N", "K"]
            )
        )
        irmodel = irbuilder.build_ir(model)
        self.assertEqual(
            irmodel.functions[0].nodes[0].outputs[0].tensor_shape_proto(),
            onnx.TensorShapeProto(
                dim=[
                    onnx.TensorShapeProto.Dimension(dim_param="N"),
                    onnx.TensorShapeProto.Dimension(dim_param="K"),
                ]
            ),
        )

    def test_function_input_is_correctly_linked_with_subnodes_in_function_when_shape_is_missing(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[N] x, float[M] y) => (float[Z] z)
            {
                z = afunction (x, y)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (x, y) => (z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
        """
        )
        irmodel = irbuilder.build_ir(model)
        self.assertIsNotNone(irmodel.functions[0].nodes[0].inputs[0])
        self.assertIsNotNone(irmodel.functions[0].nodes[0].inputs[1])
        self.assertEqual(
            irmodel.functions[0].nodes[0].inputs[0], irmodel.functions[0].values["x"]
        )
        self.assertEqual(
            irmodel.functions[0].nodes[0].inputs[1], irmodel.functions[0].values["y"]
        )

    def test_function_input_is_correctly_linked_with_subnodes_in_function_when_shape_is_present(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[N] x, float[M] y) => (float[Z] z)
            {
                z = afunction (x, y)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (x, y) => (z)
            {
                t = SequenceConstruct (x, y)
                z = ConcatFromSequence <axis=0> (t)
            }
        """
        )
        # Hack to put value_info in since parser does not support this experimental naming format
        model.graph.value_info.extend(
            [
                onnx.helper.make_tensor_value_info(
                    "pkg.custom::afunction/x", onnx.TensorProto.FLOAT, ["N"]
                ),
                onnx.helper.make_tensor_value_info(
                    "pkg.custom::afunction/y", onnx.TensorProto.FLOAT, ["M"]
                ),
            ]
        )
        irmodel = irbuilder.build_ir(model)
        self.assertIsNotNone(irmodel.functions[0].nodes[0].inputs[0])
        self.assertIsNotNone(irmodel.functions[0].nodes[0].inputs[1])
        self.assertEqual(
            irmodel.functions[0].nodes[0].inputs[0], irmodel.functions[0].values["x"]
        )
        self.assertEqual(
            irmodel.functions[0].nodes[0].inputs[1], irmodel.functions[0].values["y"]
        )
        self.assertEqual(
            irmodel.functions[0].nodes[0].inputs[0].tensor_shape_proto(),
            onnx.TensorShapeProto(
                dim=[
                    onnx.TensorShapeProto.Dimension(dim_param="N"),
                ]
            ),
        )
        self.assertEqual(
            irmodel.functions[0].nodes[0].inputs[1].tensor_shape_proto(),
            onnx.TensorShapeProto(
                dim=[
                    onnx.TensorShapeProto.Dimension(dim_param="M"),
                ]
            ),
        )

    def test_out_of_context_value_reference_is_correct(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[16, 16] x, bool cond) => (float[16, 16] z) {
                two = Constant <value_float=2.0> ()
                z = If (cond) <
                    then_branch = then_graph () => (then_z) {
                        three = Constant <value_float=3.0> ()
                        temp = Add (two, three)
                        then_z = Mul (temp, x)
                    },
                    else_branch = else_graph () => (else_z) {
                        four = Constant <value_float=4.0> ()
                        temp = Add (two, four)
                        else_z = Mul (temp, x)
                    }
                >
            }
        """
        )
        irmodel = irbuilder.build_ir(model)
        then_graph = irmodel.graph.nodes[1].attributes["then_branch"]
        self.assertIsNotNone(then_graph.nodes[2].inputs[1])
        else_graph = irmodel.graph.nodes[1].attributes["else_branch"]
        self.assertIsNotNone(else_graph.nodes[2].inputs[1])


if __name__ == "__main__":
    unittest.main()
