# Building an ONNX Graph with `onnxscript.ir`

This tutorial will guide you through the process of creating an ONNX graph using the `onnxscript.ir` module. We will explore two methods: using `onnxscript.ir.tape.Tape` methods and using `ir.node` constructors. The focus will be on graph construction.

## Method 1: Using `onnxscript.ir.tape.Tape` Methods

The `Tape` class is a recorder that collects nodes and initializers created during the construction of a graph or function. It supports creating nodes with single or multiple outputs and registering initializers.

### Step-by-Step Instructions

1. **Import the necessary modules:**

    ```python
    import onnx
    from onnxscript import ir
    ```

2. **Create a `Tape` object:**

    ```python
    tape = ir.tape.Tape()
    ```

3. **Define the inputs:**

    ```python
    input_a = ir.Value(name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2)))
    input_b = ir.Value(name="input_b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2)))
    ```

4. **Create nodes using the `op` method:**

    ```python
    add_node = tape.op("Add", inputs=[input_a, input_b])
    mul_node = tape.op("Mul", inputs=[add_node, input_b])
    ```

5. **Define the outputs:**

    ```python
    output = ir.Value(name="output", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2)))
    ```

6. **Using attributes and creating subgraphs:**

    ```python
    # Create a node with attributes
    relu_node = tape.op("Relu", inputs=[add_node], attributes={"alpha": 0.1})

    # Create a subgraph
    subgraph = ir.Graph(
        inputs=[input_a],
        outputs=[relu_node],
        nodes=[relu_node],
        name="subgraph"
    )

    # Add the subgraph to the main graph
    graph.nodes.append(subgraph)
    ```

7. **Create the graph:**

    ```python
    graph = ir.Graph(
        inputs=[input_a, input_b],
        outputs=[output],
        nodes=tape.nodes,
        initializers=tape.initializers,
        name="example_graph"
    )
    ```

8. **Convert the graph to `ModelProto` and save it to a file:**

    ```python
    model = ir.Model(graph, ir_version=7)
    model_proto = ir.to_proto(model)
    ir.save(model, "example_model.onnx")
    ```

9. **Examining the objects created using print():**

    ```python
    print("Graph:", graph)
    print("Subgraph:", subgraph)
    print("Nodes in the main graph:", graph.nodes)
    print("Nodes in the subgraph:", subgraph.nodes)
    ```

## Method 2: Using `ir.node` Constructors

The `ir.node` constructors provide a more direct way to create nodes and connect them in a graph.

### Step-by-Step Instructions

1. **Import the necessary modules:**

    ```python
    import onnx
    from onnxscript import ir
    ```

2. **Define the inputs:**

    ```python
    input_a = ir.Input("input_a", shape=ir.Shape([1, 2]), type=ir.TensorType(ir.DataType.FLOAT))
    input_b = ir.Input("input_b", shape=ir.Shape([1, 2]), type=ir.TensorType(ir.DataType.FLOAT))
    ```

3. **Create nodes using the `ir.node` constructor:**

    ```python
    add_node = ir.node("Add", inputs=[input_a, input_b])
    mul_node = ir.node("Mul", inputs=[add_node.outputs[0], input_b])
    ```

4. **Define the outputs:**

    ```python
    output = ir.Value(name="output", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2)))
    ```

5. **Using attributes and creating subgraphs:**

    ```python
    # Create a node with attributes
    relu_node = ir.node("Relu", inputs=[add_node.outputs[0]], attributes={"alpha": 0.1})

    # Create a subgraph
    subgraph = ir.Graph(
        inputs=[input_a],
        outputs=[relu_node.outputs[0]],
        nodes=[relu_node],
        name="subgraph"
    )

    # Add the subgraph to the main graph
    graph.nodes.append(subgraph)
    ```

6. **Create the graph:**

    ```python
    graph = ir.Graph(
        inputs=[input_a, input_b],
        outputs=[output],
        nodes=[add_node, mul_node],
        name="example_graph"
    )
    ```

7. **Convert the graph to `ModelProto` and save it to a file:**

    ```python
    model = ir.Model(graph, ir_version=7)
    model_proto = ir.to_proto(model)
    ir.save(model, "example_model.onnx")
    ```

8. **Examining the objects created using print():**

    ```python
    print("Graph:", graph)
    print("Subgraph:", subgraph)
    print("Nodes in the main graph:", graph.nodes)
    print("Nodes in the subgraph:", subgraph.nodes)
    ```

By following these steps, you can create an ONNX graph using both `onnxscript.ir.tape.Tape` methods and `ir.node` constructors. This tutorial focused on graph construction. A separate tutorial can be made for manipulating the graph.
