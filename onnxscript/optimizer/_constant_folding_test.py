# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import parameterized

import onnxscript.optimizer as optimizer
from onnxscript import ir
from onnxscript.optimizer import _constant_folding


def _create_model(model_text: str) -> ir.Model:
    """Create a model from the given text."""
    model = onnx.parser.parse_model(model_text)
    return ir.serde.deserialize_model(model)


class FoldConstantsTest(unittest.TestCase):
    def _fold(self, model: ir.Model | str, onnx_shape_inference=False, **kwargs):
        if isinstance(model, str):
            model = _create_model(model)
        _constant_folding.fold_constants(
            model, onnx_shape_inference=onnx_shape_inference, **kwargs
        )
        optimizer.remove_unused_nodes(model)
        # Ensure the model is valid after optimization
        onnx.checker.check_model(ir.serde.serialize_model(model))
        return model

    def test_fold_add(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, four)
            }
        """

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual(optimized.graph[0].outputs[0].name, "four")

    def test_fold_cast_like(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_int=2> ()
                two_float = CastLike(two, x)
                four = Add(two_float, two_float)
                z = Mul(x, four)
            }
        """

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual(optimized.graph[0].outputs[0].name, "four")

    def test_fold_shape(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[16, 16] x) => (float[16, 16] z) {
                shape = Shape(x)
                rank = Size(shape)
                two_float = CastLike(rank, x)
                four = Add(two_float, two_float)
                z = Mul(x, four)
            }
        """

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual(optimized.graph[0].outputs[0].name, "four")

    def test_fold_shape_slice(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[M, N, 16, 16] x) => (float[M, N, 16, 16] z) {
                shape = Shape <start:int = 2>(x)
                two = Size(shape)
                two_float = CastLike(two, x)
                four = Add(two_float, two_float)
                z = Mul(x, four)
            }
        """

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual(optimized.graph[0].outputs[0].name, "four")

    def test_fold_if_cond(self):
        model = """
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

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 1)
        self.assertEqual(optimized.graph[0].outputs[0].name, "z")
        self.assertEqual(optimized.graph[0].op_type, "Mul")

    def test_fold_inside_if_branch(self):
        model = """
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

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 1)
        then_graph = optimized.graph[0].attributes["then_branch"].as_graph()
        self.assertEqual(len(then_graph), 2)
        else_graph = optimized.graph[0].attributes["else_branch"].as_graph()
        self.assertEqual(len(else_graph), 2)

    def test_fold_if_propagate(self):
        model = """
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

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual(optimized.graph[0].outputs[0].name, "m_square")
        self.assertEqual(optimized.graph[0].op_type, "Constant")

    def test_fold_redundant_cast(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float: float = 2.0> ()
                x_cast = CastLike(x, two)
                z = Mul(x_cast, two)
            }
        """

        optimized = self._fold(model, onnx_shape_inference=True)
        self.assertEqual(len(optimized.graph), 2)

    def test_fold_redundant_cast2(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float: float = 2.0> ()
                z = CastLike(x, two)
            }
        """

        optimized = self._fold(model, onnx_shape_inference=True)
        self.assertEqual(len(optimized.graph), 1)
        self.assertEqual(optimized.graph[0].op_type, "Identity")
        self.assertEqual(optimized.graph[0].outputs[0].name, "z")
        self.assertEqual(optimized.graph[0].inputs[0].name, "x")

    def test_shape_inference(self):
        model = """
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

        optimized = self._fold(model, onnx_shape_inference=True)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual(optimized.graph[0].outputs[0].name, "C")

    def test_static_split_to_sequence_with_scalar_split_and_squence_at_is_folded_as_split(
        self,
    ):
        model = """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,512] x) => (float[1,512] return_val) {
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
}"""

        # TODO: There is an unrelated limitation that `symbolic_value` is not
        # utilized when the value is only referenced by graph output.
        # E.g., the following test model will not have this optimization
        # applied.
        #
        # <
        #    ir_version: 8,
        #    opset_import: ["" : 18]
        # >
        # func (float[1,512] x) => ( split_0,  split_1,  split_2,  split_3) {
        #    int64_128 = Constant <value: tensor = int64 int64_128 {128}> ()
        #    splits = SplitToSequence <axis: int = 1> (x, int64_128)
        #    int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
        #    split_0 = SequenceAt (splits, int64_0)
        #    int64_1 = Constant <value: tensor = int64 int64_1 {1}> ()
        #    split_1 = SequenceAt (splits, int64_1)
        #    int64_2 = Constant <value: tensor = int64 int64_2 {2}> ()
        #    split_2 = SequenceAt (splits, int64_2)
        #    int64_3 = Constant <value: tensor = int64 int64_3 {3}> ()
        #    split_3 = SequenceAt (splits, int64_3)
        # }
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual(len(optimized.graph[-2].outputs), 4)
        self.assertEqual(optimized.graph[-2].op_type, "Split")

    def test_static_split_to_sequence_with_list_split_and_squence_at_is_folded_as_split(
        self,
    ):
        model = """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,512] x) => (float[1,N] return_val) {
   const = Constant <value: tensor = int64[3] const {256,128,128}> ()
   splits = SplitToSequence <axis: int = 1> (x, const)
   int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
   split_0 = SequenceAt (splits, int64_0)
   int64_1 = Constant <value: tensor = int64 int64_1 {1}> ()
   split_1 = SequenceAt (splits, int64_1)
   int64_2 = Constant <value: tensor = int64 int64_2 {2}> ()
   split_2 = SequenceAt (splits, int64_2)
   return_val = Concat <axis: int = 1> (split_0, split_1, split_2)
}"""

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 3)
        self.assertEqual(len(optimized.graph[-2].outputs), 3)
        self.assertEqual(optimized.graph[-2].op_type, "Split")

    def test_static_split_to_sequence_with_list_split_no_keepdims_and_squence_at_is_folded_as_split_with_squeeze(
        self,
    ):
        model = """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,3] x) => (float[1,3] return_val) {
   const = Constant <value: tensor = int64[3] const {1,1,1}> ()
   splits = SplitToSequence <axis: int = 1, keepdims: int = 0> (x, const)
   int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
   split_0 = SequenceAt (splits, int64_0)
   int64_1 = Constant <value: tensor = int64 int64_1 {1}> ()
   split_1 = SequenceAt (splits, int64_1)
   int64_2 = Constant <value: tensor = int64 int64_2 {2}> ()
   split_2 = SequenceAt (splits, int64_2)
   return_val = Concat <axis: int = 1> (split_0, split_1, split_2)
}"""
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 7)
        self.assertEqual(len(optimized.graph[1].outputs), 3)
        self.assertEqual(optimized.graph[1].op_type, "Split")
        self.assertEqual(len([n for n in optimized.graph if n.op_type == "Squeeze"]), 3)

    def test_split_to_sequence_and_concat_from_sequence_with_new_axis_0(
        self,
    ):
        model = """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,3] x) => (float[1,3] return_val) {
   const = Constant <value: tensor = int64[3] const {1,1,1}> ()
   splits = SplitToSequence <axis: int = 1> (x, const)
   return_val = ConcatFromSequence <axis: int = 1, new_axis: int = 0> (splits)
}"""

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 3)
        self.assertEqual(optimized.graph[2].op_type, "Concat")

    def test_split_to_sequence_and_concat_from_sequence_with_new_axis_1(
        self,
    ):
        model = """
<
   ir_version: 8,
   opset_import: ["" : 18]
>
func (float[1,3] x) => (float[1,3] return_val) {
   const = Constant <value: tensor = int64[3] const {1,1,1}> ()
   splits = SplitToSequence <axis: int = 1> (x, const)
   return_val = ConcatFromSequence <axis: int = 1, new_axis: int = 1> (splits)
}"""

        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 7)
        self.assertEqual(optimized.graph[6].op_type, "Concat")

    def test_initializer_input_not_folded(self):
        model_text = """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[N] x, float[1] c = {1.0} ) => (float[N] z)
            {
                # c is not a constant, and following should not be folded.
                two_c = Add (c, c)
                z = Mul (x, two_c)
            }"""
        optimized = self._fold(model_text)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual(optimized.graph.node(0).op_type, "Add")

    @parameterized.parameterized.expand(
        [
            ("output = Dropout(input)",),
            ("output = Dropout(input, zero, true)",),
            ("output = Dropout(input, half)",),
            ("output = Dropout(input, half, false)",),
        ]
    )
    def test_dropout_identity(self, dropout_node: str):
        model = f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] input) => (float[N] output)
            <float zero = {{0.0}}, float half = {{0.5}}, bool true = {{1}}, bool false = {{0}}>
            {{
                {dropout_node}
            }}
        """
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 1)
        self.assertEqual(optimized.graph.node(0).op_type, "Identity")

    @parameterized.parameterized.expand(
        [
            ("output, mask = Dropout(input)",),
            ("output, mask = Dropout(input, zero, true)",),
            ("output, mask = Dropout(input, half)",),
            ("output, mask = Dropout(input, half, false)",),
        ]
    )
    def test_dropout_identity_mask(self, dropout_node: str):
        model = f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] input) => (float[N] output, bool[N] mask)
            <float zero = {{0.0}}, float half = {{0.5}}, bool true = {{1}}, bool false = {{0}}>
            {{
                {dropout_node}
            }}
        """
        optimized = self._fold(model)
        nodes = list(optimized.graph)
        self.assertEqual(len(nodes), 3)
        ops = [node.op_type for node in nodes]
        self.assertEqual(ops, ["Identity", "Shape", "ConstantOfShape"])

    def test_concat_identity(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z)
            {
                z = Concat <axis=-1> (x)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 1)
        self.assertEqual(optimized.graph.node(0).op_type, "Identity")

    def test_concat_zero_length(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N, 128] x1, float[N, 0] x2, float[N, 128] x3) => (float[N, M] z)
            {
                z = Concat <axis=-1> (x1, x2, x3)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 1)
        self.assertEqual([x.name for x in optimized.graph.node(0).inputs], ["x1", "x3"])

    def test_concat_zero_length_identity(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N, 0] x1, float[N, 128] x2, float[N, 0] x3) => (float[N, M] z)
            {
                z = Concat <axis=-1> (x1, x2, x3)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 1)
        self.assertEqual(optimized.graph.node(0).op_type, "Identity")
        self.assertEqual([x.name for x in optimized.graph.node(0).inputs], ["x2"])

    def test_concat_zero_length_output(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N, 0] x1, float[N, 0] x2, float[N, 0] x3) => (float[N, M] z)
            {
                z = Concat <axis=-1> (x1, x2, x3)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 1)
        self.assertEqual(optimized.graph.node(0).op_type, "Identity")
        self.assertEqual([x.name for x in optimized.graph.node(0).inputs], ["x1"])

    def test_expand_identity(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[128, 256] x) => (float[128, 256] z)
            {
                shape = Constant <value_ints=[128, 256]> ()
                z = Expand (x, shape)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(optimized.graph.node(-1).op_type, "Identity")

    def test_expand_identity_symdim(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[B, 256] x) => (float[B, 256] z)
            {
                b = Shape <start=0, end=1> (x)
                const_256 = Constant <value_ints=[256]> ()
                shape = Concat <axis=0> (b, const_256)
                z = Expand (x, shape)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(optimized.graph.node(-1).op_type, "Identity")

    def test_abs_symdim(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[B, 256] x) => (float[B, 256] z)
            {
                b = Shape <start=0, end=1> (x)
                const_256 = Constant <value_ints=[256]> ()
                b_256 = Concat <axis=0> (b, const_256)
                shape = Abs (b_256)
                z = Expand (x, shape)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(optimized.graph.node(-1).op_type, "Identity")

    def test_reshape_identity(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[128, 256] x) => (float[128, 256] z)
            {
                shape = Constant <value_ints=[128, 256]> ()
                z = Reshape (x, shape)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(optimized.graph.node(-1).op_type, "Identity")

    def test_reshape_identity_symdim(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[B, 256] x, float[B, 128] y) => (float[B, 256] z)
            {
                b = Shape <start=0, end=1> (y)
                const_256 = Constant <value_ints=[256]> ()
                shape = Concat <axis=0> (b, const_256)
                z = Reshape (x, shape)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(optimized.graph.node(-1).op_type, "Identity")

    def test_gather_symdim(self):
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[B, 256] x, float[B, 128] y) => (float[B, 256] z)
            {
                b_128 = Shape (y)
                index_0 = Constant <value_ints=[0]> ()
                b = Gather <axis=0> (b_128, index_0)
                const_256 = Constant <value_ints=[256]> ()
                shape = Concat <axis=0> (b, const_256)
                z = Reshape (x, shape)
            }
        """
        optimized = self._fold(model)
        self.assertEqual(optimized.graph.node(-1).op_type, "Identity")

    def test_large_transpose(self):
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[M, 256] x) => (float[M, 512] z)
            <float[1, 1] w = {1.0}> # placeholder for large initializer of shape [512, 256]
            {
                wt = Transpose (w)
                z = MatMul (x, wt)
            }
        """
        model = _create_model(model_text)
        w = model.graph.initializers["w"]
        w.shape = ir.Shape([512, 256])
        w.const_value = ir.tensor(np.random.random((512, 256)).astype(np.float32))

        # Input size limit will prevent folding of Transpose op
        optimized = self._fold(model, input_size_limit=3 * 512 * 256)
        ops = [node.op_type for node in optimized.graph]
        self.assertEqual(ops, ["Transpose", "MatMul"])

        # Input size limit will allow folding of Transpose op
        # Since there is no increase in model-size, output-size is not a concern.
        optimized = self._fold(model, input_size_limit=4 * 512 * 256)
        ops = [node.op_type for node in optimized.graph]
        self.assertEqual(ops, ["Constant", "MatMul"])

    def test_multi_graph_identity_output_preserves_output_name(self):
        model = """
            <ir_version: 10, opset_import: ["" : 20]>
            agraph (float[N] x) => (float[N] graph_output1, float[N] graph_output2) {
                t = Identity(x)
                graph_output1 = Identity(t)
                graph_output2 = Identity(t)
            }"""
        optimized = self._fold(model)
        self.assertEqual(len(optimized.graph), 2)
        self.assertEqual([n.op_type for n in optimized.graph], ["Identity", "Identity"])
        self.assertEqual(
            [n.outputs[0].name for n in optimized.graph], ["graph_output1", "graph_output2"]
        )
        self.assertEqual([input.name for input in optimized.graph.inputs], ["x"])


if __name__ == "__main__":
    unittest.main()
