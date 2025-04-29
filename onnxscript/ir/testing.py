from __future__ import annotations

from onnxscript import ir


def assert_same_graph(graph1: ir.Graph, graph2: ir.Graph):
    """
    Compares two ir.Graph objects to determine if they are the same.
    Collects all differences and raises an error with a summary if any differences are found.
    """
    differences: list[str] = []

    def compare_values(value1: ir.Value, value2: ir.Value, context: str):
        if value1.name != value2.name:
            differences.append(f"{context} name differs: {value1.name} vs {value2.name}")
        if value1.type != value2.type:
            differences.append(f"{context} type differs: {value1.type} vs {value2.type}")
        if value1.shape != value2.shape:
            differences.append(f"{context} shape differs: {value1.shape} vs {value2.shape}")

    # Compare inputs
    if len(graph1.inputs) != len(graph2.inputs):
        differences.append(
            f"Different number of inputs: {len(graph1.inputs)} vs {len(graph2.inputs)}"
        )
    else:
        for i, (input1, input2) in enumerate(zip(graph1.inputs, graph2.inputs)):
            compare_values(input1, input2, f"Input {i}")

    # Compare outputs
    if len(graph1.outputs) != len(graph2.outputs):
        differences.append(
            f"Different number of outputs: {len(graph1.outputs)} vs {len(graph2.outputs)}"
        )
    else:
        for i, (output1, output2) in enumerate(zip(graph1.outputs, graph2.outputs)):
            compare_values(output1, output2, f"Output {i}")

    # Compare initializers
    if graph1.initializers.keys() != graph2.initializers.keys():
        differences.append(
            f"Initializer keys differ: {graph1.initializers.keys()} vs {graph2.initializers.keys()}"
        )
    else:
        for key in graph1.initializers:
            compare_values(
                graph1.initializers[key], graph2.initializers[key], f"Initializer '{key}'"
            )

    # Compare nodes
    if len(graph1) != len(graph2):
        differences.append(f"Different number of nodes: {len(graph1)} vs {len(graph2)}")
    else:
        for i, (node1, node2) in enumerate(zip(graph1, graph2)):
            if node1.op_type != node2.op_type:
                differences.append(
                    f"Node {i} op_type differs: {node1.op_type} vs {node2.op_type}"
                )
            if len(node1.inputs) != len(node2.inputs):
                differences.append(
                    f"Node {i} has different number of inputs: {len(node1.inputs)} vs {len(node2.inputs)}"
                )
            if len(node1.outputs) != len(node2.outputs):
                differences.append(
                    f"Node {i} has different number of outputs: {len(node1.outputs)} vs {len(node2.outputs)}"
                )
            for j, (input1, input2) in enumerate(zip(node1.inputs, node2.inputs)):
                if input1 and input2:  # Ensure inputs are not None
                    compare_values(input1, input2, f"Node {i} input {j}")
            for j, (output1, output2) in enumerate(zip(node1.outputs, node2.outputs)):
                compare_values(output1, output2, f"Node {i} output {j}")

    # Raise error if differences are found
    if differences:
        error_message = "\n".join(differences)
        raise AssertionError(f"Graphs are not the same:\n{error_message}")
