(heading-target-checkers)=
# Node and Value Level Checkers

The pattern matching infrastructure supports custom validation logic at both the node and value levels through checker functions. These checkers allow for more sophisticated pattern matching by enabling additional constraints beyond basic operator and structure matching.

## Value-Level Checkers

Value-level checkers validate properties of specific values in the pattern. They are particularly useful for checking constants, shapes, or other value-specific properties.

### Basic Usage

A value checker is a function that takes a `MatchContext` and an `ir.Value`, and returns either a boolean or a `MatchResult`:

```python
def is_positive_constant(context, value: ir.Value):
    """Check if a value is a positive constant."""
    if value.const_value is not None:
        # Get the numpy array from const_value
        numpy_array = value.const_value.numpy()

        # Check if it represents a single value and is positive
        if numpy_array.size != 1:
            return False

        return float(numpy_array.item()) > 0

    return False
```

You can use this checker directly in your pattern by passing the callable as an input:

```python
def add_pattern(op, x, y):
    # Use callable as input to create ValuePattern with checker
    return op.Add(is_positive_constant, y)
```

This pattern will only match `Add` operations where the first input is a positive constant value.

### Example Usage

```python
from onnxscript.rewriter import pattern
from onnxscript import ir, optimizer
import onnx

# Create a model with different Add operations
model_proto = onnx.parser.parse_model("""
    <ir_version: 7, opset_import: [ "" : 17]>
    agraph (float[N] x, float[N] y) => (float[N] z1, float[N] z2, float[N] z3)
    {
        pos_const = Constant <value_float = 2.5> ()
        neg_const = Constant <value_float = -1.5> ()
        z1 = Add(x, y)           # non-constant first parameter
        z2 = Add(pos_const, y)   # positive constant first parameter
        z3 = Add(neg_const, y)   # negative constant first parameter
    }
""")
model = ir.serde.deserialize_model(model_proto)

# Apply constant propagation to set const_value fields
optimizer.basic_constant_propagation(model.graph.all_nodes())

# Create the pattern with value checker
rule_pattern = pattern.Pattern(add_pattern)

# Test matching against different Add nodes
add_nodes = [node for node in model.graph if node.op_type == "Add"]

# Non-constant first parameter - will not match
match_result = rule_pattern.match(model, model.graph, add_nodes[0])
print(f"Non-constant: {bool(match_result)}")  # False

# Positive constant first parameter - will match
match_result = rule_pattern.match(model, model.graph, add_nodes[1])
print(f"Positive constant: {bool(match_result)}")  # True

# Negative constant first parameter - will not match
match_result = rule_pattern.match(model, model.graph, add_nodes[2])
print(f"Negative constant: {bool(match_result)}")  # False
```

## Node-Level Checkers

Node-level checkers validate properties of the operation nodes themselves, such as attributes, operation types, or other node-specific properties.

### Basic Usage

A node checker is a function that takes a `MatchContext` and an `ir.Node`, and returns either a boolean or a `MatchResult`:

```python
def shape_node_checker(context, node):
    """Check if a Shape operation has start attribute equal to 0."""
    return node.attributes.get_int("start", 0) == 0
```

You can use this checker by passing it to the `_check` parameter of an operation:

```python
def shape_pattern(op, x):
    return op.Shape(x, _check=shape_node_checker)
```

This pattern will only match `Shape` operations where the `start` attribute is 0 (or not present, as the default is 0).

### Example Usage

```python
from onnxscript.rewriter import pattern
from onnxscript import ir
import onnx

# Create a model with different Shape operations
model_proto = onnx.parser.parse_model("""
    <ir_version: 7, opset_import: [ "" : 17]>
    agraph (float[N, M] x) => (int64[2] z1, int64[2] z2, int64[1] z3)
    {
        z1 = Shape(x)
        z2 = Shape <start: int = 0>(x)
        z3 = Shape <start: int = 1>(x)
    }
""")
model = ir.serde.deserialize_model(model_proto)

# Create the pattern with node checker
rule_pattern = pattern.Pattern(shape_pattern)

# Test matching against different Shape nodes
nodes = list(model.graph)
shape_nodes = [node for node in nodes if node.op_type == "Shape"]

# Shape without start attribute (default 0) - will match
match_result = rule_pattern.match(model, model.graph, shape_nodes[0])
print(f"No start attr: {bool(match_result)}")  # True

# Shape with start=0 - will match
match_result = rule_pattern.match(model, model.graph, shape_nodes[1])
print(f"Start=0: {bool(match_result)}")  # True

# Shape with start=1 - will not match
match_result = rule_pattern.match(model, model.graph, shape_nodes[2])
print(f"Start=1: {bool(match_result)}")  # False
```

## Combining Checkers

You can combine both node-level and value-level checkers in the same pattern for more sophisticated matching:

```python
def complex_pattern(op, x, y):
    # Value-level checker for first input
    validated_x = is_positive_constant
    # Node-level checker for the operation
    return op.Add(validated_x, y, _check=lambda ctx, node: len(node.attributes) == 0)
```

This pattern will only match `Add` operations where:
1. The first input is a positive constant (value-level check)
2. The node has no custom attributes (node-level check)

## Error Handling

Checkers can return either:
- `True`: Check passed, continue matching
- `False`: Check failed, pattern does not match
- `MatchResult`: More detailed result with potential failure reasons

If a checker raises an exception, it will be caught and treated as a match failure, allowing patterns to fail gracefully when encountering unexpected conditions.
