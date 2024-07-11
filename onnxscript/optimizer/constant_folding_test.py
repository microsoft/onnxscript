# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnx
import parameterized
import pytest

import onnxscript.optimizer as optimizer
from onnxscript.ir import serde
from onnxscript.optimizer import _constant_folding, constant_folding


@parameterized.parameterized_class(("using_ir",), [(False,), (True,)])
class FoldConstantsTest(unittest.TestCase):
    def _fold(self, model: onnx.ModelProto, onnx_shape_inference=False):
        if self.using_ir:
            ir_model = serde.deserialize_model(model)
            _constant_folding.fold_constants(
                ir_model, onnx_shape_inference=onnx_shape_inference
            )
            optimizer.remove_unused_nodes(ir_model)
            return serde.serialize_model(ir_model)
        else:
            constant_folding.fold_constants(model, onnx_shape_inference=onnx_shape_inference)
            optimizer.remove_unused_nodes(model)
            return model

    def test_fold_add(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, four)
            }
        """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 2)
        self.assertEqual(optimized.graph.node[0].output[0], "four")

    def test_fold_cast_like(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_int=2> ()
                two_float = CastLike(two, x)
                four = Add(two_float, two_float)
                z = Mul(x, four)
            }
        """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 2)
        self.assertEqual(optimized.graph.node[0].output[0], "four")

    def test_fold_shape(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[16, 16] x) => (float[16, 16] z) {
                shape = Shape(x)
                rank = Size(shape)
                two_float = CastLike(rank, x)
                four = Add(two_float, two_float)
                z = Mul(x, four)
            }
        """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 2)
        self.assertEqual(optimized.graph.node[0].output[0], "four")

    def test_fold_shape_slice(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[M, N, 16, 16] x) => (float[M, N, 16, 16] z) {
                shape = Shape <start:int = 2>(x)
                two = Size(shape)
                two_float = CastLike(two, x)
                four = Add(two_float, two_float)
                z = Mul(x, four)
            }
        """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 2)
        self.assertEqual(optimized.graph.node[0].output[0], "four")

    def test_fold_if_cond(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[16, 16] x) => (float[16, 16] z) {
                shape = Shape(x)
                rank = Size(shape)
                zero = Constant <value_int=0> ()
                zero_cast = CastLike (zero, rank)
                is_scalar = Equal(zero_cast, rank)
                z = If (is_scalar) <
                    then_branch = then_graph () => (then_z) { then_z = Add (x, x) },
                    else_branch = else_graph () => (else_z) { else_z = Mul (x, x) }
                >
            }
        """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 1)
        self.assertEqual(optimized.graph.node[0].output[0], "z")
        self.assertEqual(optimized.graph.node[0].op_type, "Mul")

    def test_fold_inside_if_branch(self):
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
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 1)
        then_graph = onnx.helper.get_node_attr_value(optimized.graph.node[0], "then_branch")
        self.assertEqual(len(then_graph.node), 2)
        else_graph = onnx.helper.get_node_attr_value(optimized.graph.node[0], "else_branch")
        self.assertEqual(len(else_graph.node), 2)

    def test_fold_if_propagate(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[16, 16] x) => (float[16, 16] z) {
                shape = Shape(x)
                rank = Size(shape)
                zero = Constant <value_int=0> ()
                two = Constant <value_float=2.0> ()
                zero_cast = CastLike (zero, rank)
                is_scalar = Equal(zero_cast, rank)
                m = If (is_scalar) <
                    then_branch = then_graph () => (then_z) { then_z = Add (x, x) },
                    else_branch = else_graph () => (else_z) { else_z = Mul (two, two) }
                >
                m_square = Mul (m, m)
                z = Mul (x, m_square)
            }
        """
        )
        optimized = self._fold(model)
        print(onnx.printer.to_text(optimized))
        self.assertEqual(len(optimized.graph.node), 2)
        self.assertEqual(optimized.graph.node[0].output[0], "m_square")
        self.assertEqual(optimized.graph.node[0].op_type, "Constant")

    def test_fold_redundant_cast(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float: float = 2.0> ()
                x_cast = CastLike(x, two)
                z = Mul(x_cast, two)
            }
        """
        )
        optimized = self._fold(model, onnx_shape_inference=True)
        self.assertEqual(len(optimized.graph.node), 2)

    def test_fold_redundant_cast2(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float: float = 2.0> ()
                z = CastLike(x, two)
            }
        """
        )
        optimized = self._fold(model, onnx_shape_inference=True)
        self.assertEqual(len(optimized.graph.node), 1)
        self.assertEqual(optimized.graph.node[0].op_type, "Identity")
        self.assertEqual(optimized.graph.node[0].output[0], "z")
        self.assertEqual(optimized.graph.node[0].input[0], "x")

    @pytest.mark.skip(reason="Feature removed to catch errors early")
    def test_fold_undefined_vars(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                four = Add(two, two)
                y = Shape(t1)
                w = CastLike(x, t2)
                w2 = CastLike(t3, t4)
                w3 = Size(t5)
                z = Sum (four, y, w, w2, w3)
            }
        """
        )
        # No optimizations expected. Just make sure it doesn't crash.
        optimized = self._fold(model, onnx_shape_inference=False)
        self.assertEqual(len(optimized.graph.node), 6)

    def test_shape_inference(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[64] x) => (int64[N] z) {
                one = Constant <value_int=1> ()
                cond = Equal(one, one)
                temp = If (cond) <
                    then_branch = then_graph () => (then_z) {
                        shape1 = Constant <value_ints=[8, 8]> ()
                        then_z = Reshape(x, shape1)
                    },
                    else_branch = else_graph () => (else_z) {
                        shape2 = Constant <value_ints=[64]> ()
                        else_z = Reshape(x, shape2)
                    }>
                shape = Shape(temp)   # shape = [8, 8] or [64], but [8, 8] after constant propagation
                rank = Size(shape)    # rank = 2 or 1, but 2 after constant propagation
                C = Add (rank, rank)
                z = Mul(x, C)
            }
        """
        )
        optimized = self._fold(model, onnx_shape_inference=True)
        print(onnx.printer.to_text(optimized))
        self.assertEqual(len(optimized.graph.node), 2)
        self.assertEqual(optimized.graph.node[0].output[0], "C")

    def test_static_split_to_sequence_with_scalar_split_and_squence_at_is_folded_as_split(
        self,
    ):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,512] x) => ( return_val) {
   int64_128 = Constant <value: tensor = int64 int64_128 {128}> ()
   splits = SplitToSequence <axis: int = 1> (x, int64_128)
   int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
   split_0 = SequenceAt (splits, int64_0)
   int64_1 = Constant <value: tensor = int64 int64_1 {1}> ()
   split_1 = SequenceAt (splits, int64_1)
   int64_2 = Constant <value: tensor = int64 int64_2 {2}> ()
   split_2 = SequenceAt (splits, int64_2)
   int64_3 = Constant <value: tensor = int64 int64_3 {3}> ()
   split_3 = SequenceAt (splits, int64_3)
   return_val = Concat <axis: int = 1> (split_0, split_1, split_2, split_3)
}
            """
        )

        # TODO: There is an unrelated limitation that `symbolic_value` is not
        # utilized when the value is only referenced by graph output.
        # E.g., the following test model will not have this optimization
        # applied.
        """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,512] x) => ( split_0,  split_1,  split_2,  split_3) {
   int64_128 = Constant <value: tensor = int64 int64_128 {128}> ()
   splits = SplitToSequence <axis: int = 1> (x, int64_128)
   int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
   split_0 = SequenceAt (splits, int64_0)
   int64_1 = Constant <value: tensor = int64 int64_1 {1}> ()
   split_1 = SequenceAt (splits, int64_1)
   int64_2 = Constant <value: tensor = int64 int64_2 {2}> ()
   split_2 = SequenceAt (splits, int64_2)
   int64_3 = Constant <value: tensor = int64 int64_3 {3}> ()
   split_3 = SequenceAt (splits, int64_3)
}
        """
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 2)
        self.assertEqual(len(optimized.graph.node[-2].output), 4)
        self.assertEqual(optimized.graph.node[-2].op_type, "Split")

    def test_static_split_to_sequence_with_list_split_and_squence_at_is_folded_as_split(
        self,
    ):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,512] x) => ( return_val) {
   const = Constant <value: tensor = int64[3] const {256,128,128}> ()
   splits = SplitToSequence <axis: int = 1> (x, const)
   int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
   split_0 = SequenceAt (splits, int64_0)
   int64_1 = Constant <value: tensor = int64 int64_1 {1}> ()
   split_1 = SequenceAt (splits, int64_1)
   int64_2 = Constant <value: tensor = int64 int64_2 {2}> ()
   split_2 = SequenceAt (splits, int64_2)
   return_val = Concat <axis: int = 1> (split_0, split_1, split_2)
}
            """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 3)
        self.assertEqual(len(optimized.graph.node[-2].output), 3)
        self.assertEqual(optimized.graph.node[-2].op_type, "Split")

    def test_static_split_to_sequence_with_list_split_no_keepdims_and_squence_at_is_folded_as_split_with_squeeze(
        self,
    ):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,3] x) => ( return_val) {
   const = Constant <value: tensor = int64[3] const {1,1,1}> ()
   splits = SplitToSequence <axis: int = 1, keepdims: int = 0> (x, const)
   int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
   split_0 = SequenceAt (splits, int64_0)
   int64_1 = Constant <value: tensor = int64 int64_1 {1}> ()
   split_1 = SequenceAt (splits, int64_1)
   int64_2 = Constant <value: tensor = int64 int64_2 {2}> ()
   split_2 = SequenceAt (splits, int64_2)
   return_val = Concat <axis: int = 1> (split_0, split_1, split_2)
}
            """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 7)
        self.assertEqual(len(optimized.graph.node[1].output), 3)
        self.assertEqual(optimized.graph.node[1].op_type, "Split")
        self.assertEqual(len([n for n in optimized.graph.node if n.op_type == "Squeeze"]), 3)

    def test_split_to_sequence_and_concat_from_sequence_with_new_axis_0(
        self,
    ):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,3] x) => (float[1,3] return_val) {
   const = Constant <value: tensor = int64[3] const {1,1,1}> ()
   splits = SplitToSequence <axis: int = 1> (x, const)
   return_val = ConcatFromSequence <axis: int = 1, new_axis: int = 0> (splits)
}
            """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 3)
        self.assertEqual(optimized.graph.node[2].op_type, "Concat")
        onnx.checker.check_model(optimized)

    def test_split_to_sequence_and_concat_from_sequence_with_new_axis_1(
        self,
    ):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,3] x) => (float[1,3] return_val) {
   const = Constant <value: tensor = int64[3] const {1,1,1}> ()
   splits = SplitToSequence <axis: int = 1> (x, const)
   return_val = ConcatFromSequence <axis: int = 1, new_axis: int = 1> (splits)
}
            """
        )
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph.node), 7)
        self.assertEqual(optimized.graph.node[6].op_type, "Concat")
        onnx.checker.check_model(optimized)


if __name__ == "__main__":
    unittest.main()
