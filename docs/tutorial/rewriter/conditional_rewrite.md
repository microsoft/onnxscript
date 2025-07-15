# Using the `match_condition` parameter for pattern-matching

This section talks about how to utilize the `match_condition` parameter. The `match_condition` parameter checks if the pattern matches the target pattern with certain constraints in consideration.

Let us consider a model which consists of the following pattern.

![target_pattern](examples/img/broadcast_01.png){align=center}

Based on the [ONNX Matmul spec](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul), onnx `Matmul` behaves like `numpy.matmul` and also follows numpy broadcasting. So in this particular pattern if matmul broadcasting is enough, then we don't need the reshapes. To validate this, we need to check the following:

1. Input shapes check: `input_a` and `input_b` should be broadcastable
2. Output shape check: `shape_c` should be the same as the output shape from the `matmul(input_a, input_b)`

If the above are true, then we don't need the reshapes and we can eliminate them using a pattern based rewrite.

First, write a target pattern and replacement pattern in a similar way to the first example.

```{literalinclude} examples/broadcast_matmul.py
:pyobject: two_reshapes_matmul_reshape_pattern
```

```{literalinclude} examples/broadcast_matmul.py
:pyobject: matmul_pattern
```

:::{note}
:name: omitting inputs in signature

The target pattern in this case has 5 inputs `input_a`, `input_b`, `shape_a`, `shape_b`, `shape_c`. However, the replacement pattern only utilizes `input_a` and `input_b`. To avoid referencing all the unused parameters in the replacement pattern signature, pass only `input_a` and `input_b` and use `**_` to represent all the unused parameters.

Similarly for writing the condition checking function, we require only `input_a`, `input_b` and `shape_c`. Use `**_` to represent all the unused parameters in the condition matching function signature.
:::

In order to validate whether matmul broadcast is sufficient, we write a condition checking function as below.
Note that the relevant inputs passed to the check function are all instances of {py:class}`onnx_ir.Value`. These represent
the values in the input graph IR that matched against the corresponding _pattern variables_ in the target
pattern. Please see documentation of the [IR API](https://onnx.ai/ir-py/) for more details on how to use it, for example to identify
the type or shape or rank of these values.

```{literalinclude} examples/broadcast_matmul.py
:pyobject: check_if_not_need_reshape
```

With all the necessary components in place, the pattern rewrite rule with the `match_condition` function is created and then the `rewriter.rewrite` is called to apply the rewrite.

```{literalinclude} examples/broadcast_matmul.py
:pyobject: apply_rewrite
```

The final graph with the applied rewrite looks as follows:

![broadcast_rewrite](examples/img/broadcast_02.png){align=center}

# Using PatternMatchContext for Advanced Condition Checking

The `context` parameter passed to condition functions is an instance of {py:class}`onnxscript.rewriter.PatternMatchContext`, which provides access to additional information about the pattern match that can be useful for sophisticated condition checking.

## PatternMatchContext Properties

The PatternMatchContext provides the following read-only properties:

- `model`: The entire ONNX model being matched
- `graph_or_function`: The specific graph or function being matched
- `main_root_node`: The primary root node of the matching subgraph
- `output_values`: The output values of the matching subgraph
- `nodes`: All nodes that are part of the matching subgraph

## Example Usage

Here's an example showing how to use the PatternMatchContext to implement more sophisticated condition checking:

```python
def advanced_condition_check(context, x, y, **_):
    """Example condition function using PatternMatchContext."""
    
    # Access the main node of the pattern match
    main_node = context.main_root_node
    
    # Check that the main_node does not have an attribute called "alpha"
    for attr in main_node.attribute:
        if attr.name == "alpha":
            return False
    
    # Access the broader graph context and check that x occurs as a graph-input
    model = context.model
    input_names = [input.name for input in model.graph.input]
    if x not in input_names:
        return False
    
    # You can inspect the matched nodes for advanced validation
    for node in context.nodes:
        if node.op_type == "Constant":
            # Check properties of constant nodes in the match
            pass
    
    # Access output values for shape/type validation
    outputs = context.output_values
    if len(outputs) > 0 and outputs[0].shape is not None:
        # Validate output shapes
        pass
    
    return True
```

This context information enables condition functions to make decisions based on the broader graph structure, the specific nodes involved in the match, and relationships between matched patterns and the rest of the model.
