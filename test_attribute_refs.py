#!/usr/bin/env python3

"""Test for attribute references in rewrite rules."""

import unittest
import onnx
import onnxscript.ir as ir
import onnxscript.rewriter.pattern as pattern


class TestAttributeRefs(unittest.TestCase):
    def test_rewrite_rule_with_attribute_ref_fails_in_copy(self):
        """Test that rewrite rules fail when trying to extract function with RefAttr."""
        
        # Create a pattern that matches Transpose
        def transpose_pattern(op, x):
            return op.Transpose(x, _outputs=["result"])
        
        def replacement(op, x, result: ir.Value):
            return op.Identity(x)
        
        # This will trigger the _copy_for_function issue when as_function=True
        rule = pattern.RewriteRule(transpose_pattern, replacement, as_function=True)
        
        # Create a simple model manually using the IR
        input_val = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT))
        output_val = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT))
        
        # Create a transpose node with a ref attribute  
        perm_ref_attr = ir.RefAttr("perm", "axis_param", ir.AttributeType.INTS)
        
        transpose_node = ir.Node(
            domain="",
            op_type="Transpose", 
            inputs=[input_val],
            outputs=[output_val],
            attributes={"perm": perm_ref_attr}
        )
        
        # Create graph
        graph = ir.Graph(
            inputs=[input_val],
            outputs=[output_val],
            nodes=[transpose_node]
        )
        
        # Create model
        model = ir.Model(
            graph=graph,
            ir_version=8
        )
        
        print("Graph nodes:")
        for node in model.graph:
            print(f"  Node: {node.op_type}")
            for attr_name, attr in node.attributes.items():
                print(f"    Attribute {attr_name}: {attr}, is_ref: {attr.is_ref()}")
                if attr.is_ref():
                    print(f"      References: {attr.ref_attr_name}")
        
        try:
            # This should trigger the NotImplementedError in _copy_for_function
            count = rule.apply_to_model(model)
            print(f"Unexpected success: Rewrite applied {count} times")
            return False
        except NotImplementedError as e:
            print(f"Expected NotImplementedError: {e}")
            # This confirms the issue exists
            return True
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    test = TestAttributeRefs()
    success = test.test_rewrite_rule_with_attribute_ref_fails_in_copy()
    print(f"Test result: {'PASS' if success else 'FAIL'}")