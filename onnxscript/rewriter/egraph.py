# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""E-graph implementation for efficient pattern matching.

E-graphs (equality graphs) are a data structure that compactly represents many equivalent
programs by merging expressions that are equivalent. This enables more efficient and robust
pattern matching compared to traditional tree-based approaches.

Key concepts:
- EClass: Equivalence class representing a set of equivalent expressions
- ENode: A node in the e-graph representing an operation with e-class children  
- EGraph: Container managing e-classes and providing union-find operations
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from onnxscript import ir


@dataclass(frozen=True)
class ENode:
    """Represents a single operation/expression in the e-graph.
    
    An ENode consists of an operator identifier and a list of e-class IDs
    representing its children. Two ENodes are equal if they have the same
    operator and the same child e-classes.
    """
    op: str  # Operation identifier (e.g., "Add", "Mul", "Constant")
    children: Tuple[int, ...]  # E-class IDs of children
    domain: str = ""  # ONNX operator domain
    attributes: Tuple[Tuple[str, Any], ...] = ()  # Sorted attributes for hashing
    
    def __post_init__(self):
        # Ensure children is a tuple for immutability and hashing
        if not isinstance(self.children, tuple):
            object.__setattr__(self, 'children', tuple(self.children))
        # Ensure attributes is a sorted tuple for consistent hashing
        if not isinstance(self.attributes, tuple):
            object.__setattr__(self, 'attributes', tuple(sorted(self.attributes)))


@dataclass
class EClass:
    """Represents an equivalence class of expressions.
    
    An e-class contains multiple equivalent expressions (ENodes) and maintains
    metadata about the equivalence class.
    """
    id: int
    nodes: Set[ENode] = field(default_factory=set)
    parents: Set[Tuple[ENode, int]] = field(default_factory=set)  # (parent_node, child_index)
    
    def add_node(self, node: ENode) -> None:
        """Add a node to this equivalence class."""
        self.nodes.add(node)
    
    def merge_from(self, other: EClass) -> None:
        """Merge another e-class into this one."""
        self.nodes.update(other.nodes)
        self.parents.update(other.parents)


class EGraph:
    """E-graph data structure for representing equivalent expressions.
    
    The e-graph maintains equivalence classes and provides operations for:
    - Adding new expressions
    - Merging equivalent expressions
    - Efficient lookups and pattern matching
    """
    
    def __init__(self):
        self.eclasses: Dict[int, EClass] = {}  # e-class ID -> EClass
        self.hashcons: Dict[ENode, int] = {}  # ENode -> e-class ID (hash consing)
        self.unionfind: Dict[int, int] = {}  # Union-find for e-class merging
        self.next_id = 0
    
    def _find(self, eclass_id: int) -> int:
        """Find the canonical e-class ID using union-find."""
        if eclass_id not in self.unionfind:
            self.unionfind[eclass_id] = eclass_id
            return eclass_id
        
        # Path compression
        if self.unionfind[eclass_id] != eclass_id:
            self.unionfind[eclass_id] = self._find(self.unionfind[eclass_id])
        return self.unionfind[eclass_id]
    
    def _union(self, id1: int, id2: int) -> int:
        """Union two e-classes and return the canonical ID."""
        canonical1 = self._find(id1)
        canonical2 = self._find(id2)
        
        if canonical1 == canonical2:
            return canonical1
        
        # Merge smaller into larger
        eclass1 = self.eclasses[canonical1]
        eclass2 = self.eclasses[canonical2]
        
        if len(eclass1.nodes) < len(eclass2.nodes):
            canonical1, canonical2 = canonical2, canonical1
            eclass1, eclass2 = eclass2, eclass1
        
        # Merge eclass2 into eclass1
        eclass1.merge_from(eclass2)
        self.unionfind[canonical2] = canonical1
        
        # Update hashcons for merged nodes
        for node in eclass2.nodes:
            self.hashcons[node] = canonical1
        
        # Remove the merged e-class
        del self.eclasses[canonical2]
        
        return canonical1
    
    def add_node(self, node: ENode) -> int:
        """Add a node to the e-graph and return its e-class ID.
        
        If an equivalent node already exists, return its e-class ID.
        Otherwise, create a new e-class.
        """
        # Canonicalize children
        canonical_children = tuple(self._find(child) for child in node.children)
        canonical_node = ENode(
            op=node.op,
            children=canonical_children,
            domain=node.domain,
            attributes=node.attributes
        )
        
        # Check if this node already exists (hash consing)
        if canonical_node in self.hashcons:
            return self._find(self.hashcons[canonical_node])
        
        # Create new e-class
        eclass_id = self.next_id
        self.next_id += 1
        
        eclass = EClass(id=eclass_id)
        eclass.add_node(canonical_node)
        
        self.eclasses[eclass_id] = eclass
        self.hashcons[canonical_node] = eclass_id
        self.unionfind[eclass_id] = eclass_id
        
        # Update parent relationships
        for i, child_id in enumerate(canonical_children):
            canonical_child_id = self._find(child_id)
            if canonical_child_id in self.eclasses:
                child_eclass = self.eclasses[canonical_child_id]
                child_eclass.parents.add((canonical_node, i))
        
        return eclass_id
    
    def merge(self, id1: int, id2: int) -> int:
        """Merge two e-classes."""
        return self._union(id1, id2)
    
    def get_eclass(self, eclass_id: int) -> Optional[EClass]:
        """Get the e-class for the given ID."""
        canonical_id = self._find(eclass_id)
        return self.eclasses.get(canonical_id)
    
    def get_nodes_in_eclass(self, eclass_id: int) -> Set[ENode]:
        """Get all nodes in the given e-class."""
        eclass = self.get_eclass(eclass_id)
        return eclass.nodes if eclass else set()
    
    def find_nodes_by_op(self, op: str, domain: str = "") -> List[Tuple[int, ENode]]:
        """Find all nodes with the given operation."""
        result = []
        for eclass_id, eclass in self.eclasses.items():
            canonical_id = self._find(eclass_id)
            if canonical_id != eclass_id:
                continue  # Skip non-canonical e-classes
            
            for node in eclass.nodes:
                if node.op == op and node.domain == domain:
                    result.append((eclass_id, node))
        return result
    
    def apply_commutative_rules(self) -> None:
        """Apply commutative rules to merge equivalent expressions.
        
        For commutative operations like Add and Mul, merge expressions that
        differ only in the order of arguments.
        """
        commutative_ops = {"Add", "Mul"}
        
        # Group nodes by commutative signature
        for op in commutative_ops:
            commutative_groups: Dict[Tuple[str, Tuple[int, ...]], List[int]] = {}
            
            for eclass_id, node in self.find_nodes_by_op(op):
                if len(node.children) == 2:  # Binary operations
                    # Create canonical signature by sorting children
                    canonical_children = tuple(sorted(node.children))
                    signature = (op, canonical_children)
                    
                    if signature not in commutative_groups:
                        commutative_groups[signature] = []
                    commutative_groups[signature].append(eclass_id)
            
            # Merge e-classes with the same commutative signature
            for group in commutative_groups.values():
                if len(group) > 1:
                    # Merge all e-classes in the group
                    canonical = group[0]
                    for eclass_id in group[1:]:
                        canonical = self.merge(canonical, eclass_id)


def build_egraph_from_ir(graph_or_function: Union[ir.Graph, ir.Function]) -> Tuple[EGraph, Dict[ir.Value, int]]:
    """Build an e-graph from an ONNX IR graph or function.
    
    Returns:
        egraph: The constructed e-graph
        value_to_eclass: Mapping from IR values to e-class IDs
    """
    egraph = EGraph()
    value_to_eclass: Dict[ir.Value, int] = {}
    
    # Process nodes in topological order
    def add_ir_node(ir_node: ir.Node) -> None:
        # Get e-class IDs for input values
        child_eclasses = []
        for input_value in ir_node.inputs:
            if input_value in value_to_eclass:
                child_eclasses.append(value_to_eclass[input_value])
            else:
                # Handle constants and graph inputs
                if input_value.const_value is not None:
                    # Create constant node
                    const_node = ENode(
                        op="Constant",
                        children=(),
                        attributes=(("value", input_value.const_value),)
                    )
                    const_eclass = egraph.add_node(const_node)
                    value_to_eclass[input_value] = const_eclass
                    child_eclasses.append(const_eclass)
                else:
                    # Graph input - create placeholder
                    input_node = ENode(
                        op="Input",
                        children=(),
                        attributes=(("name", input_value.name),)
                    )
                    input_eclass = egraph.add_node(input_node)
                    value_to_eclass[input_value] = input_eclass
                    child_eclasses.append(input_eclass)
        
        # Create attributes tuple
        attributes = []
        for attr_name, attr_value in ir_node.attributes.items():
            # Convert attribute to hashable form
            if hasattr(attr_value, 'numpy'):
                # For numpy arrays, use tuple of shape and flattened values
                arr = attr_value.numpy()
                hashable_value = (tuple(arr.shape), tuple(arr.flatten().tolist()))
            else:
                hashable_value = attr_value
            attributes.append((attr_name, hashable_value))
        
        # Create e-node for this operation
        enode = ENode(
            op=ir_node.op_type,
            children=tuple(child_eclasses),
            domain=ir_node.domain,
            attributes=tuple(sorted(attributes))
        )
        
        # Add to e-graph
        eclass_id = egraph.add_node(enode)
        
        # Map output values to this e-class
        for output_value in ir_node.outputs:
            value_to_eclass[output_value] = eclass_id
    
    # Add all nodes
    for node in graph_or_function:
        add_ir_node(node)
    
    # Apply commutative rules
    egraph.apply_commutative_rules()
    
    return egraph, value_to_eclass