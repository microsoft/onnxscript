# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import onnx
import onnxscript.optimizer as optimizer



class OptimizerTest(unittest.TestCase):
        def test_static_split_to_sequence_with_uneven_split(self):
            model = onnx.parser.parse_model(
                """
                <
                    ir_version: 8,
                    opset_import: ["pkg.onnxscript.torch_lib" : 1, "" : 18, "pkg.onnxscript.torch_lib.common" : 1],
                    producer_name: "pytorch",
                    producer_version: "2.2.0"
                >
                main_graph (float[3,5] l_tensor_x_) => (float[3,5] return_val)
                < _val_2, float[3,5] l_tensor_x_, float[2,5] getitem, float[1,5] getitem_1>
                {
                    _val_1 = Constant <value: tensor = int64 {2}> ()
                    _val_2 = pkg.onnxscript.torch_lib.aten_split <dim: int = 0> (l_tensor_x_, _val_1)
                    _val_3 = Constant <value: tensor = int64 {0}> ()
                    getitem = pkg.onnxscript.torch_lib.aten_getitem (_val_2, _val_3)
                    _val_5 = Constant <value: tensor = int64 {1}> ()
                    getitem_1 = pkg.onnxscript.torch_lib.aten_getitem (_val_2, _val_5)
                    return_val = Concat <axis: int = 0> (getitem_1, getitem)
                }

                <domain: "pkg.onnxscript.torch_lib", opset_import: ["" : 18]>
                aten_split (self, split_size) => (return_val)
                {
                    return_val = SplitToSequence <axis: int = @dim> (self, split_size)
                }

                <domain: "pkg.onnxscript.torch_lib", opset_import: ["" : 18]>
                aten_getitem (self, i) => (return_val)
                {
                    return_val = SequenceAt (self, i)
                }

                <domain: "pkg.onnxscript.torch_lib.common", opset_import: ["" : 18]>
                Rank (input) => (return_val)
                {
                    tmp = Shape (input)
                    return_val = Size (tmp)
                }

                <domain: "pkg.onnxscript.torch_lib.common", opset_import: ["" : 18]>
                IsScalar (input) => (return_val)
                {
                    tmp = Shape (input)
                    tmp_0 = Size (tmp)
                    tmp_1 = Constant <value_int: int = 0> ()
                    return_val = Equal (tmp_0, tmp_1)
                }
                """
            )
            optimized = optimizer.optimize(model, num_iterations=1, onnx_shape_inference=False)
            self.assertEqual(len(optimized.graph.node), 2)
            self.assertEqual(len(optimized.graph.node[0].output), 2)
            self.assertEqual(optimized.graph.node[0].op_type, "Split")

if __name__ == "__main__":
    unittest.main()
