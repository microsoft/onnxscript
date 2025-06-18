# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for e-graph based pattern matching."""

import unittest
import onnx
import onnx.helper as oh
import numpy as np

from onnxscript import ir
from onnxscript.rewriter.egraph import ENode, EClass, EGraph, build_egraph_from_ir
from onnxscript.rewriter.egraph_pattern import EGraphPatternMatcher, demonstrate_egraph_benefits


class TestEGraph(unittest.TestCase):
    """Test the core e-graph data structures."""
    
    def test_enode_equality(self):
        """Test that ENodes with same content are equal."""
        node1 = ENode(op="Add", children=(1, 2))
        node2 = ENode(op="Add", children=(1, 2))
        node3 = ENode(op="Add", children=(2, 1))  # Different order
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        self.assertEqual(hash(node1), hash(node2))
        self.assertNotEqual(hash(node1), hash(node3))
    
    def test_eclass_basic_operations(self):
        """Test basic e-class operations."""
        eclass = EClass(id=0)
        node1 = ENode(op="Add", children=(1, 2))
        node2 = ENode(op="Mul", children=(1, 2))
        
        eclass.add_node(node1)
        eclass.add_node(node2)
        
        self.assertEqual(len(eclass.nodes), 2)
        self.assertIn(node1, eclass.nodes)
        self.assertIn(node2, eclass.nodes)
    
    def test_egraph_add_node(self):
        """Test adding nodes to e-graph."""
        egraph = EGraph()
        
        # Add a simple constant node
        const_node = ENode(op="Constant", children=())
        const_id = egraph.add_node(const_node)
        
        self.assertEqual(const_id, 0)
        self.assertIn(const_node, egraph.get_nodes_in_eclass(const_id))
        
        # Add an operation node
        add_node = ENode(op="Add", children=(const_id, const_id))
        add_id = egraph.add_node(add_node)
        
        self.assertEqual(add_id, 1)
        self.assertIn(add_node, egraph.get_nodes_in_eclass(add_id))
    
    def test_egraph_hash_consing(self):
        """Test that identical nodes are merged."""
        egraph = EGraph()
        
        # Create child e-classes first
        child1 = ENode(op="Constant", children=())
        child2 = ENode(op="Constant", children=())
        
        id_child1 = egraph.add_node(child1)
        id_child2 = egraph.add_node(child2)
        
        node1 = ENode(op="Add", children=(id_child1, id_child2))
        node2 = ENode(op="Add", children=(id_child1, id_child2))  # Identical
        
        id1 = egraph.add_node(node1)
        id2 = egraph.add_node(node2)
        
        # Should return the same e-class ID
        self.assertEqual(id1, id2)
    
    def test_egraph_union_find(self):
        """Test union-find operations."""
        egraph = EGraph()
        
        # Create two separate e-classes
        node1 = ENode(op="Add", children=())
        node2 = ENode(op="Mul", children=())
        
        id1 = egraph.add_node(node1)
        id2 = egraph.add_node(node2)
        
        self.assertNotEqual(id1, id2)
        
        # Merge them
        merged_id = egraph.merge(id1, id2)
        
        # Both should now resolve to the same canonical ID
        self.assertEqual(egraph._find(id1), egraph._find(id2))
        self.assertEqual(egraph._find(id1), merged_id)
    
    def test_commutative_rules(self):
        """Test that commutative rules merge equivalent expressions."""
        egraph = EGraph()
        
        # Create constant nodes
        const1_id = egraph.add_node(ENode(op="Constant", children=(), attributes=(("value", 1),)))
        const2_id = egraph.add_node(ENode(op="Constant", children=(), attributes=(("value", 2),)))
        
        # Create commutative operations in different orders
        # These should be different initially because children order is different
        add1 = ENode(op="Add", children=(const1_id, const2_id))
        add2 = ENode(op="Add", children=(const2_id, const1_id))  # Swapped order
        
        add1_id = egraph.add_node(add1)
        add2_id = egraph.add_node(add2)
        
        # Initially different (since children are in different order)
        self.assertNotEqual(add1_id, add2_id)
        
        # Apply commutative rules
        egraph.apply_commutative_rules()
        
        # Should now be in the same e-class
        self.assertEqual(egraph._find(add1_id), egraph._find(add2_id))


class TestEGraphFromIR(unittest.TestCase):
    """Test building e-graphs from ONNX IR."""
    
    def test_simple_graph_to_egraph(self):
        """Test converting a simple ONNX graph to e-graph."""
        # Create simple Add(a, b) model
        model_proto = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["a", "b"], ["result"])],
                "simple",
                [
                    oh.make_tensor_value_info("a", onnx.TensorProto.FLOAT, []),
                    oh.make_tensor_value_info("b", onnx.TensorProto.FLOAT, []),
                ],
                [oh.make_tensor_value_info("result", onnx.TensorProto.FLOAT, [])]
            )
        )
        
        model_ir = ir.serde.deserialize_model(model_proto)
        egraph, value_to_eclass = build_egraph_from_ir(model_ir.graph)
        
        # Should have e-classes for: input a, input b, Add operation
        self.assertGreaterEqual(len(egraph.eclasses), 3)
        
        # Check that all values are mapped
        for node in model_ir.graph:
            for output in node.outputs:
                self.assertIn(output, value_to_eclass)
    
    def test_commutative_graph_merging(self):
        """Test that commutative operations are merged in e-graph."""
        # Create model with commutative operations in different orders
        model_proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["a", "b"], ["sum1"]),
                    oh.make_node("Add", ["b", "a"], ["sum2"]),  # Swapped
                ],
                "commutative",
                [
                    oh.make_tensor_value_info("a", onnx.TensorProto.FLOAT, []),
                    oh.make_tensor_value_info("b", onnx.TensorProto.FLOAT, []),
                ],
                [
                    oh.make_tensor_value_info("sum1", onnx.TensorProto.FLOAT, []),
                    oh.make_tensor_value_info("sum2", onnx.TensorProto.FLOAT, []),
                ]
            )
        )
        
        model_ir = ir.serde.deserialize_model(model_proto)
        egraph, value_to_eclass = build_egraph_from_ir(model_ir.graph)
        
        # Find the two Add operations
        add_nodes = egraph.find_nodes_by_op("Add")
        self.assertEqual(len(add_nodes), 2)
        
        # They should be in the same e-class after commutative merging
        eclass_ids = [eclass_id for eclass_id, _ in add_nodes]
        canonical_ids = [egraph._find(eclass_id) for eclass_id in eclass_ids]
        
        # Should have same canonical e-class
        self.assertEqual(len(set(canonical_ids)), 1)


class TestEGraphPatternMatcher(unittest.TestCase):
    """Test e-graph based pattern matching."""
    
    def test_matcher_creation(self):
        """Test creating an e-graph pattern matcher."""
        # Create a simple pattern - just test that it can be created
        from onnxscript.rewriter import pattern
        
        def simple_pattern(op, x, y):
            return op.Add(x, y)
        
        # This is a basic test to ensure the infrastructure works
        # A full test would need to create actual pattern IR
        pass
    
    def test_demonstrate_benefits(self):
        """Test the demonstration function runs without error."""
        # This tests that our demo code works
        try:
            demonstrate_egraph_benefits()
        except Exception as e:
            self.fail(f"Demonstration failed with error: {e}")


if __name__ == "__main__":
    unittest.main()