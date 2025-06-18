# E-Graph Based Pattern Matching

E-graphs (equality graphs) provide a more efficient and robust approach to pattern matching compared to traditional tree-based methods. This document explains how to use e-graph based pattern matching in ONNX Script.

## Overview

E-graphs represent equivalent expressions in equivalence classes (e-classes), enabling:

- **Order-independent matching**: Commutative operations are automatically handled
- **Efficient pattern matching**: Match once per equivalence class instead of per node
- **Reduced pattern explosion**: Exponential growth of patterns becomes constant
- **Robust rewriting**: Less sensitive to graph structure variations

## Basic Usage

### Using E-Graph Pattern Matching

```python
from onnxscript.rewriter import egraph, egraph_pattern
from onnxscript import ir

# Convert your ONNX model to IR
model_ir = ir.serde.deserialize_model(onnx_model)

# Build e-graph from the model
graph_egraph, value_to_eclass = egraph.build_egraph_from_ir(model_ir.graph)

# The e-graph automatically groups equivalent expressions
print(f"Original graph: {len(list(model_ir.graph))} nodes")
print(f"E-graph: {len(graph_egraph.eclasses)} equivalence classes")
```

### Viewing E-Graph Structure

```python
# Find operations by type
add_operations = graph_egraph.find_nodes_by_op("Add")
mul_operations = graph_egraph.find_nodes_by_op("Mul")

print("Add operations:")
for eclass_id, node in add_operations:
    canonical_id = graph_egraph._find(eclass_id)
    print(f"  E-class {canonical_id}: Add with children {node.children}")
```

## Commutative Operation Handling

One of the key benefits of e-graphs is automatic handling of commutative operations:

### Traditional Approach Problem

```python
# Traditional pattern matching needs multiple rules for commutative operations:

def pattern1(op, x, y, z):
    sum_result = op.Add(x, y)
    return op.Mul(sum_result, z)

def pattern2(op, x, y, z):  
    sum_result = op.Add(y, x)  # Swapped Add inputs
    return op.Mul(sum_result, z)

def pattern3(op, x, y, z):
    sum_result = op.Add(x, y)
    return op.Mul(z, sum_result)  # Swapped Mul inputs

def pattern4(op, x, y, z):
    sum_result = op.Add(y, x)  # Both operations swapped
    return op.Mul(z, sum_result)

# Need 4 separate rules for 2 commutative operations!
# This grows as 2^n for n commutative operations
```

### E-Graph Approach Solution

```python
# With e-graphs, only ONE pattern needed:

def egraph_pattern(op, x, y, z):
    sum_result = op.Add(x, y)  # Order doesn't matter!
    return op.Mul(sum_result, z)  # Order doesn't matter!

# E-graph automatically handles all commutative variations
# Same pattern matches Add(x,y) and Add(y,x)
# Same pattern matches Mul(a,b) and Mul(b,a)
```

## Pattern Complexity Comparison

The benefits become dramatic as pattern complexity increases:

| Commutative Ops | Traditional Rules | E-Graph Rules | Reduction |
|-----------------|-------------------|---------------|-----------|
| 1               | 2                 | 1             | 2x        |
| 2               | 4                 | 1             | 4x        |
| 3               | 8                 | 1             | 8x        |
| 4               | 16                | 1             | 16x       |
| 5               | 32                | 1             | 32x       |
| 7               | 128               | 1             | 128x      |

## Advanced Features

### Custom Equivalence Rules

E-graphs can be extended with custom equivalence rules beyond commutativity:

```python
# Example: Custom associativity rules could be added
egraph.apply_associative_rules()  # Future extension

# Example: Custom algebraic rules
egraph.apply_algebraic_rules([
    ("Add(x, 0)", "x"),  # x + 0 = x
    ("Mul(x, 1)", "x"),  # x * 1 = x
    ("Mul(x, 0)", "0"),  # x * 0 = 0
])  # Future extension
```

### E-Graph Analysis

```python
# Analyze e-graph structure
def analyze_egraph(egraph):
    print(f"Total e-classes: {len(egraph.eclasses)}")
    
    # Count operations by type
    op_counts = {}
    for eclass in egraph.eclasses.values():
        for node in eclass.nodes:
            op_counts[node.op] = op_counts.get(node.op, 0) + 1
    
    print("Operations by type:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")
```

## Integration with Existing Rewriter

The e-graph approach integrates with the existing rewriter infrastructure:

```python
from onnxscript.rewriter import pattern
from onnxscript.rewriter.egraph_pattern import EGraphPatternMatcher

# Create pattern using existing API
def my_pattern(op, x, y):
    return op.Add(x, y)

def my_replacement(op, x, y):
    return op.CustomOp(x, y, domain="my_domain")

# Use e-graph matcher instead of traditional matcher
rule = pattern.RewriteRule(
    my_pattern, 
    my_replacement,
    matcher=EGraphPatternMatcher  # Use e-graph based matching
)

# Apply as usual
rule.apply_to_model(model_ir)
```

## Performance Benefits

E-graph based pattern matching provides several performance benefits:

1. **Reduced Pattern Matching Complexity**: O(e-classes) instead of O(nodes)
2. **Automatic Commutative Handling**: No manual enumeration of argument orders
3. **Global Optimization View**: Can find globally optimal rewrite sequences
4. **Caching Benefits**: Equivalent expressions computed once

## Limitations and Future Work

Current limitations of the e-graph implementation:

1. **Pattern Complexity**: Currently supports basic structural patterns
2. **Attribute Matching**: Limited attribute pattern support
3. **Rewrite Integration**: Basic integration with existing rewrite rules

Future extensions could include:

- Associativity rules for operations like Add and Mul
- Algebraic simplification rules (x + 0 = x, x * 1 = x, etc.)
- Advanced pattern matching with constraints
- Integration with cost models for optimal rewriting

## Examples

See `onnxscript/rewriter/egraph_examples.py` for complete working examples that demonstrate:

- Traditional vs e-graph pattern matching comparison
- Commutative operation handling
- Pattern complexity analysis
- Performance benefits demonstration

Run the examples with:

```bash
python onnxscript/rewriter/egraph_examples.py
```

## Conclusion

E-graph based pattern matching represents a significant improvement over traditional approaches, especially for patterns involving commutative operations. The automatic handling of equivalent expressions reduces pattern complexity from exponential to constant, making it practical to write complex rewrite rules without pattern explosion.