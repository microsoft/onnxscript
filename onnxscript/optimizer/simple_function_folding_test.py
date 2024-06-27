# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx

from onnxscript.optimizer import remove_unused_function, simple_function_folding


class SingleNodeFunctionFoldingTest(unittest.TestCase):
    def test_fold_single_node_function(self):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["this" : 1, "" : 18]
>
func ( x,  y) => ( return_val) {
   tmp = this.foldable (x)
   return_val = Add (tmp, y)
}
<
  domain: "this",
  opset_import: ["" : 18]
>
foldable (x) => (return_val)
{
   return_val = Identity (x)
}
            """
        )

        simple_function_folding.inline_simple_functions(model)
        model = remove_unused_function.remove_unused_functions(model)

        self.assertEqual(len(model.functions), 0)

    def test_fold_single_node_function_ref_attr(self):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["this" : 1, "" : 18]
>
func ( x,  y,  z) => ( return_val) {
   tmp = this.foldable <dim = 0> (x, y)
   return_val = Add (tmp, z)
}
<
  domain: "this",
  opset_import: ["" : 18]
>
foldable <dim>(x, y) => (return_val)
{
   return_val = Concat <axis: int = @dim> (x, y)
}
            """
        )

        simple_function_folding.inline_simple_functions(model)
        model = remove_unused_function.remove_unused_functions(model)

        self.assertEqual(len(model.functions), 0)
        self.assertFalse(model.graph.node[0].attribute[0].ref_attr_name)
        self.assertEqual(model.graph.node[0].attribute[0].name, "axis")

    def test_fold_single_node_function_nested(self):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["this" : 1, "" : 18]
>
func ( x,  y,  z) => ( return_val) {
   tmp = this.non_foldable <axis = 0> (x, y)
   return_val = Add (tmp, z)
}
<
  domain: "this",
  opset_import: ["" : 18]
>
foldable <axis>(x, y) => (return_val)
{
   return_val = Concat <axis: int = @axis> (x, y)
}
<
  domain: "this",
  opset_import: ["this" : 1,"" : 18]
>
non_foldable <axis>(x, y) => (return_val)
{
   tmp = this.foldable <axis: int = @axis> (x, y)
   tmp_0 = this.foldable <axis: int = @axis> (x, y)
   return_val = Add (tmp, tmp_0)
}
            """
        )

        simple_function_folding.inline_simple_functions(model)
        model = remove_unused_function.remove_unused_functions(model)

        self.assertEqual(len(model.functions), 1)
        self.assertEqual(model.functions[0].node[0].op_type, "Concat")
        self.assertEqual(model.functions[0].node[1].op_type, "Concat")

    def test_fold_single_node_function_create_new_nodes_with_correct_attributes(self):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 9,
   opset_import: ["this" : 1, "" : 21]
>
func (float[1,512] x) => ( a,  b,  c) {
   a = this.prim_cast <to: int = 10> (x)
   b = this.prim_cast <to: int = 6> (x)
   c = this.prim_cast <to: int = 7> (x)
}
<
  domain: "this",
  opset_import: ["" : 18]
>
prim_cast <to>(x) => (return_val)
{
   return_val = Cast <to: int = @to> (x)
}
            """
        )
        simple_function_folding.inline_simple_functions(model)
        model = remove_unused_function.remove_unused_functions(model)
        self.assertEqual(len(model.functions), 0)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[0].attribute[0].i, 10)
        self.assertEqual(model.graph.node[1].attribute[0].i, 6)
        self.assertEqual(model.graph.node[2].attribute[0].i, 7)

    def test_fold_nested_if_function_succeeds(self):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 9,
   opset_import: ["this" : 1, "" : 21]
>
func (float[1,512] x, float[1,512] y) => ( out) {
   out = this.foldable_func (x, y)
}
<
  domain: "this",
  opset_import: ["" : 18]
>
foldable_func (x, y) => (z_6)
{
   cond = Constant <value: tensor = bool cond {1}> ()
   z_6 = If (cond) <then_branch: graph = thenGraph_4 () => ( z_2) {
      cond_0 = Not (cond)
      z_2 = If (cond_0) <then_branch: graph = thenGraph_5 () => ( z) {
         z = Add (x, x)
      }, else_branch: graph = elseGraph_5 () => ( z_1) {
         z_1 = Identity (x)
      }>
   }, else_branch: graph = elseGraph_4 () => ( z_5) {
      z_5 = If (cond) <then_branch: graph = thenGraph_10 () => ( z_3) {
         z_3 = Add (y, y)
      }, else_branch: graph = elseGraph_10 () => ( z_4) {
         z_4 = Add (x, y)
      }>
   }>
}
             """
        )

        simple_function_folding.inline_simple_functions(model)
        model = remove_unused_function.remove_unused_functions(model)

        self.assertEqual(len(model.functions), 0)
        self.assertEqual(len(model.graph.node), 2)
        self.assertEqual(model.graph.node[1].op_type, "If")

    def test_fold_function_with_unused_output(self):
        model = onnx.parser.parse_model(
            """
<
   ir_version: 8,
   opset_import: ["this" : 1, "" : 18]
>
func ( x,  y,  z) => ( return_val) {
   tmp = this.non_foldable <axis = 0> (x, y)
   return_val = Add (tmp, z)
}
<
  domain: "this",
  opset_import: ["" : 18]
>
foldable <axis>(x, y) => (return_val, unused, unused1)
{
   return_val = Concat <axis: int = @axis> (x, y)
   unused = Identity (x)
   unused1 = Identity (y)
}
<
  domain: "this",
  opset_import: ["this" : 1,"" : 18]
>
non_foldable <axis>(x, y) => (return_val)
{
   tmp, unused, unused1 = this.foldable <axis: int = @axis> (x, y)
   tmp_0, unused2, unused3 = this.foldable <axis: int = @axis> (x, y)
   return_val = Add (tmp, tmp_0)
}
            """
        )

        simple_function_folding.inline_functions_with_unused_outputs(model)
        model = remove_unused_function.remove_unused_functions(model)
        self.assertEqual(len(model.functions), 1)


if __name__ == "__main__":
    unittest.main()
