# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Examples demonstrating e-graph based pattern matching benefits.

This module provides practical examples showing how e-graph based pattern
matching improves upon traditional pattern matching approaches.
"""

import onnx
import onnx.helper as oh
import numpy as np

from onnxscript import ir
from onnxscript.rewriter import pattern
from onnxscript.rewriter.egraph import build_egraph_from_ir
from onnxscript.rewriter.egraph_pattern import EGraphPatternMatcher


def create_commutative_example_model():
    """Create a model that demonstrates commutative pattern matching challenges."""
    # Create model with equivalent expressions in different orders
    model_proto = oh.make_model(
        oh.make_graph(
            [
                # Pattern 1: Add(a, b) -> Mul(result, c)
                oh.make_node("Add", ["a", "b"], ["sum1"]),
                oh.make_node("Mul", ["sum1", "c"], ["result1"]),
                
                # Pattern 2: Add(b, a) -> Mul(c, result) - same computation, different order
                oh.make_node("Add", ["b", "a"], ["sum2"]),
                oh.make_node("Mul", ["c", "sum2"], ["result2"]),
                
                # Pattern 3: More complex - nested commutative operations
                oh.make_node("Mul", ["a", "b"], ["prod1"]),
                oh.make_node("Add", ["prod1", "c"], ["sum3"]),
                oh.make_node("Add", ["c", "prod1"], ["sum4"]),  # Equivalent to sum3
            ],
            "commutative_example",
            [
                oh.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 3]),
                oh.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [2, 3]),
                oh.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [2, 3]),
            ],
            [
                oh.make_tensor_value_info("result1", onnx.TensorProto.FLOAT, [2, 3]),
                oh.make_tensor_value_info("result2", onnx.TensorProto.FLOAT, [2, 3]),
                oh.make_tensor_value_info("sum3", onnx.TensorProto.FLOAT, [2, 3]),
                oh.make_tensor_value_info("sum4", onnx.TensorProto.FLOAT, [2, 3]),
            ]
        ),
        opset_imports=[oh.make_opsetid("", 17)]
    )
    return model_proto


def traditional_pattern_matching_example():
    """Demonstrate traditional pattern matching challenges with commutative operations."""
    print("=== Traditional Pattern Matching Challenges ===")
    
    model_proto = create_commutative_example_model()
    model_ir = ir.serde.deserialize_model(model_proto)
    
    print(f"Original model has {len(list(model_ir.graph))} nodes")
    
    # Traditional approach needs multiple patterns for commutative matching
    def pattern1(op, x, y, z):
        sum_result = op.Add(x, y)
        return op.Mul(sum_result, z)
    
    def pattern2(op, x, y, z):
        sum_result = op.Add(y, x)  # Swapped inputs
        return op.Mul(z, sum_result)  # Swapped inputs
    
    def pattern3(op, x, y, z):
        sum_result = op.Add(x, y)
        return op.Mul(z, sum_result)  # Different Mul order
    
    def pattern4(op, x, y, z):
        sum_result = op.Add(y, x)  # Both swapped
        return op.Mul(sum_result, z)
    
    def replacement(op, x, y, z):
        return op.FusedAddMul(x, y, z, domain="custom")
    
    # Would need multiple rules to catch all combinations
    rules = [
        pattern.RewriteRule(pattern1, replacement, name="AddMul_1"),
        pattern.RewriteRule(pattern2, replacement, name="AddMul_2"), 
        pattern.RewriteRule(pattern3, replacement, name="AddMul_3"),
        pattern.RewriteRule(pattern4, replacement, name="AddMul_4"),
    ]
    
    print(f"Traditional approach needs {len(rules)} separate rules for commutative matching")
    print("This grows exponentially with the number of commutative operations!")


def egraph_pattern_matching_example():
    """Demonstrate e-graph based pattern matching benefits."""
    print("\n=== E-Graph Pattern Matching Benefits ===")
    
    model_proto = create_commutative_example_model()
    model_ir = ir.serde.deserialize_model(model_proto)
    
    # Build e-graph
    egraph, value_to_eclass = build_egraph_from_ir(model_ir.graph)
    
    print(f"Original graph: {len(list(model_ir.graph))} nodes")
    print(f"E-graph: {len(egraph.eclasses)} equivalence classes")
    
    # Show how equivalent operations are grouped
    add_operations = egraph.find_nodes_by_op("Add")
    mul_operations = egraph.find_nodes_by_op("Mul")
    
    print(f"\nAdd operations found: {len(add_operations)}")
    add_eclasses = set()
    for eclass_id, node in add_operations:
        canonical_id = egraph._find(eclass_id)
        add_eclasses.add(canonical_id)
        print(f"  E-class {canonical_id}: Add({node.children})")
    
    print(f"\nMul operations found: {len(mul_operations)}")
    mul_eclasses = set()
    for eclass_id, node in mul_operations:
        canonical_id = egraph._find(eclass_id)
        mul_eclasses.add(canonical_id)
        print(f"  E-class {canonical_id}: Mul({node.children})")
    
    print(f"\nEquivalent Add operations grouped into {len(add_eclasses)} e-classes")
    print(f"Equivalent Mul operations grouped into {len(mul_eclasses)} e-classes")
    print("\nWith e-graphs:")
    print("- Only ONE pattern needed for each operation type")
    print("- Commutative matching happens automatically")
    print("- Pattern matching is order-independent")
    print("- Exponential explosion of rules is avoided")


def demonstrate_pattern_complexity():
    """Show how pattern complexity grows with traditional vs e-graph approaches."""
    print("\n=== Pattern Complexity Comparison ===")
    
    def calculate_traditional_patterns(num_commutative_ops):
        """Calculate number of patterns needed for traditional matching."""
        # Each commutative binary operation can be in 2 orders
        # For a pattern with n commutative ops, need 2^n patterns
        return 2 ** num_commutative_ops
    
    def calculate_egraph_patterns(num_commutative_ops):
        """Calculate number of patterns needed for e-graph matching."""
        # E-graphs handle commutativity automatically - always just 1 pattern
        return 1
    
    print("Number of patterns needed for different complexities:")
    print("Commutative Ops | Traditional | E-Graph | Reduction Factor")
    print("----------------|-------------|---------|------------------")
    
    for n in range(1, 8):
        traditional = calculate_traditional_patterns(n)
        egraph = calculate_egraph_patterns(n)
        reduction = traditional / egraph
        print(f"{n:14d} | {traditional:11d} | {egraph:7d} | {reduction:16.0f}x")
    
    print("\nAs you can see, traditional approach grows exponentially!")
    print("E-graph approach stays constant at 1 pattern regardless of complexity.")


def main():
    """Run all examples."""
    print("E-Graph Pattern Matching Examples")
    print("=" * 50)
    
    traditional_pattern_matching_example()
    egraph_pattern_matching_example()
    demonstrate_pattern_complexity()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("- E-graphs automatically handle commutative operations")
    print("- Reduce pattern explosion from exponential to constant")
    print("- Enable order-independent pattern matching")
    print("- Provide more robust and efficient rewriting")


if __name__ == "__main__":
    main()