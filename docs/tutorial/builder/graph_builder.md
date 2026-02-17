# GraphBuilder Tutorial

The `GraphBuilder` and `OpBuilder` classes provide a programmatic API for
constructing ONNX IR graphs. Instead of writing `@script`-decorated functions
and converting them to ONNX, you build graphs imperatively — adding operations,
managing naming hierarchies, and letting the builder handle constant promotion,
type casting, and shape inference automatically.

This tutorial covers all the main features with runnable examples.

## Setup

Every `GraphBuilder` wraps an `ir.Graph` that already declares its opset imports.
The builder reads the default opset version from the graph and creates an
`OpBuilder` for the standard ONNX domain (`""`).

```python
import onnx_ir as ir
import onnxscript

graph = ir.Graph(
    name="my_graph",
    inputs=[],
    outputs=[],
    nodes=[],
    opset_imports={"": 23},      # ONNX opset 23
)
builder = onnxscript.GraphBuilder(graph)
op = builder.op                  # OpBuilder for the default domain
```

`op` is the primary interface for adding nodes. Any attribute access on `op` is
interpreted as an ONNX op type: `op.Add(...)`, `op.Mul(...)`, `op.Relu(...)`, etc.

## Adding Operations

### Basic graph construction

Define graph inputs as `ir.Value` objects with type and shape, then call ops
through `op`:

```python
x = ir.Value(
    name="x",
    type=ir.TensorType(ir.DataType.FLOAT),
    shape=ir.Shape([3, 4]),
)
y = ir.Value(
    name="y",
    type=ir.TensorType(ir.DataType.FLOAT),
    shape=ir.Shape([3, 4]),
)
graph.inputs.extend([x, y])

t1 = op.Add(x, y)      # Add node
t2 = op.Mul(x, y)      # Mul node
z  = op.Add(t1, t2)    # final Add

graph.outputs.append(z)
```

Each `op.<OpType>(...)` call:

1. Creates an `ir.Node` with the given inputs.
2. Appends it to the graph.
3. Runs basic constant propagation and shape inference.
4. Returns the output `ir.Value` (or a tuple for multi-output ops).

### Shape inference

The builder automatically runs shape inference after adding each node. If the
inputs have known types and shapes, the outputs will too:

```python
result = op.Add(x, y)
print(result.type.dtype)     # DataType.FLOAT
print(list(result.shape))   # [3, 4]
```

## Naming

### Default output names

By default, output values are named `{OpType}_output`:

```python
t = op.Add(x, y)
print(t.name)  # "Add_output"
```

Node names include a sequential counter for uniqueness:

```python
print(t.producer().name)  # "Add_node_0"
```

### Custom output names

Pass `_outputs` to specify output names explicitly — either as strings or
pre-created `ir.Value` objects:

```python
t = op.Add(x, y, _outputs=["my_sum"])
print(t.name)  # "my_sum"

out_val = ir.Value(name="result")
t = op.Mul(x, y, _outputs=[out_val])
assert t is out_val
print(t.name)  # "result"
```

### Hierarchical naming (module context)

When building graphs from nested modules (e.g. layers of a neural network),
use `push_module` / `pop_module` to add dot-separated prefixes to all names
generated within that context:

```python
builder.push_module("layer1")
t1 = op.Add(x, y)
print(t1.name)                  # "layer1.Add_output"
print(t1.producer().name)       # "layer1.Add_node_2"

builder.push_module("attention")
t2 = op.Mul(t1, y)
print(t2.name)                  # "layer1.attention.Mul_output"

builder.pop_module()             # back to "layer1"
builder.pop_module()             # back to root

t3 = op.Add(t2, x)
print(t3.name)                  # "Add_output"  (no prefix)
```

This makes it easy to trace which layer produced each value when inspecting
or debugging a large graph.

## Constant Promotion and Auto-Casting

One of the most useful features of the builder is automatic promotion of
Python scalars and sequences to ONNX tensor constants.

### Scalars

You can pass Python `int`, `float`, `bool`, or `str` values directly as
inputs to ops. The builder creates a named initializer automatically:

```python
x = ir.Value(name="x", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([3]))
graph.inputs.append(x)

result = op.Add(x, 1)
# The constant "1" becomes an initializer named "const_1_i64"
```

### Schema-aware type casting

When the ONNX schema requires inputs to share the same type, the builder
casts the constant to match. For example, `Add` requires both inputs to have
the same type (`T`). If `x` is a `FLOAT` tensor:

```python
x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3]))
result = op.Add(x, 1)
# The int "1" is converted to a FLOAT tensor, named "const_1_f32"
```

The builder uses a two-pass approach:

1. **First pass:** Scan inputs to bind type variables (e.g. `T → FLOAT`).
2. **Second pass:** Cast constants to match the bound types.

This correctly handles cases like `op.Add(1, x)` where the constant appears
before the typed tensor.

### Dynamic CastLike for unknown types

When the input type is unknown at graph-construction time (e.g. the graph
input has no type annotation), the builder falls back to inserting a
`CastLike` node that resolves the type at runtime:

```python
x = ir.Value(name="x", shape=ir.Shape([3]))   # no type specified
result = op.Add(x, 1)
# Produces two nodes:
#   CastLike(const_1_i64, x)  → cast constant to match x's runtime type
#   Add(x, cast_result)
```

### Constant caching

Constants are cached by `(value, dtype)` so that identical constants are shared
across operations and across layers. The cache key includes the target dtype,
so `1` cast to `INT64` and `1` cast to `FLOAT` produce separate initializers:

```python
r1 = op.Add(x, 1)     # creates "const_1_i64"
r2 = op.Add(x, 1)     # reuses the same ir.Value — no duplicate initializer
```

Constants are *not* qualified with the hierarchical context prefix (they use
`qualify=False`), so they are naturally shared across different modules/layers.

### Sequences

Lists and tuples of homogeneous scalars are also promoted and cached:

```python
result = op.Add(x, [1, 2, 3])
# Creates initializer "const_[1,2,3]_i64"
```

For longer sequences, the name is abbreviated:

```python
result = op.Add(x, [10, 20, 30, 40, 50])
# Creates initializer "const_[10,20,...]_i64"
```

## Custom Domains

### Inline domain override

Pass `_domain` (and optionally `_version`) to call ops from non-standard domains:

```python
result = op.CustomOp(x, y, _domain="com.microsoft", _version=1)
node = result.producer()
print(node.domain)    # "com.microsoft"
print(node.version)   # 1
```

### OpBuilder for a custom domain

For repeated use of a custom domain, create a dedicated `OpBuilder`:

```python
ms_op = builder.opset("com.microsoft", 1)
t1 = ms_op.CustomOp(x, y)
t2 = ms_op.AnotherOp(t1, x)

# All nodes automatically get domain="com.microsoft", version=1
```

You can freely mix operations from different domain builders in the same graph:

```python
t1 = op.Add(x, y)                    # standard domain
t2 = ms_op.FusedOp(t1, y)            # com.microsoft domain
t3 = op.Relu(t2)                     # back to standard domain
```

## Initializers

Besides automatic constant promotion, you can create initializers explicitly
using `ir.tensor()` and the builder's `initializer` method:

```python
import numpy as np

weights = ir.tensor(np.random.randn(3, 4).astype(np.float32), name="weights")
w = builder.initializer(weights)
# w is an ir.Value registered in the graph, named with the current context prefix
```

The `qualify` parameter controls whether the hierarchical context prefix is
applied. It defaults to `True` for explicit initializers and is set to `False`
internally for shared constants:

```python
builder.push_module("encoder")
w = builder.initializer(weights, name="W")
print(w.name)  # "encoder.W"
builder.pop_module()
```

## Putting It All Together

Here is a complete example that builds a small model with two layers:

```python
import onnx_ir as ir
import onnxscript

# Create graph with opset 23
graph = ir.Graph(
    name="two_layer_model",
    inputs=[],
    outputs=[],
    nodes=[],
    opset_imports={"": 23},
)

# Define input
x = ir.Value(
    name="input",
    type=ir.TensorType(ir.DataType.FLOAT),
    shape=ir.Shape([1, 784]),
)
graph.inputs.append(x)

builder = onnxscript.GraphBuilder(graph)
op = builder.op

# Layer 1
builder.push_module("layer1")
w1 = builder.initializer(
    ir.tensor([[0.1] * 784] * 128, dtype=ir.DataType.FLOAT, name="weight")
)
t = op.MatMul(x, w1)
t = op.Add(t, 0.0)          # bias as a scalar constant — auto-promoted to f32
t = op.Relu(t)
builder.pop_module()

# Layer 2
builder.push_module("layer2")
w2 = builder.initializer(
    ir.tensor([[0.1] * 128] * 10, dtype=ir.DataType.FLOAT, name="weight")
)
t = op.MatMul(t, w2)
t = op.Add(t, 0.0)
builder.pop_module()

graph.outputs.append(t)

# Wrap in a model
model = ir.Model(graph=graph, ir_version=10)
```

Node and value names will reflect the module hierarchy:

- `layer1.weight`, `layer1.MatMul_node_0`, `layer1.Relu_output`
- `layer2.weight`, `layer2.MatMul_node_4`, `layer2.Add_output`

While scalar constants like `0.0` are shared across layers as `const_0.0_f32`.

## Recovering the Builder from OpBuilder

The `OpBuilder` keeps a reference back to its parent `GraphBuilder` via the
`builder` property. This means helper functions only need to accept `op` as a
parameter — they can always recover the full builder when they need features
like `push_module`, `initializer`, or `opset`:

```python
def build_linear(op, x, weight, bias_value):
    """A reusable helper that only takes `op`."""
    builder = op.builder
    builder.push_module("linear")
    t = op.MatMul(x, weight)
    t = op.Add(t, bias_value)   # scalar auto-promoted
    builder.pop_module()
    return t
```

This pattern keeps function signatures simple while preserving access to the
full builder API when needed.
