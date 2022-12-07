from os_graph_builder import GraphBuilder
from aten_graph import IOIdentity, ATENGraph, ATENNode, create_example_graph

def to_onnxscript_io_format(aten_io_identity):
    """
    We should update the inputs to match the ONNXScript GraphBuilder
    requirement.
    """
    return aten_io_identity

def aten_add(gb, aten_node):
    """
    Here, we need to create an ONNX node accordign to the definition in ONNX Script
    to map the given aten node.
    Some inputs may need to be transformed to attributes of an ONNX node.
    Data type is also need to be transformed to accetable ones of ONNX.

    """
    onnx_node = aten_node
    gb.add_node(onnx_node)

    print("====== aten_add works. ======")

func_mapping = {
    "aten::add":aten_add,}

aten_graph : ATENGraph = create_example_graph()
aten_graph.print()

gb = GraphBuilder()

for input in aten_graph.inputs:
    gb.add_input(to_onnxscript_io_format(input))

for output in aten_graph.outputs:
    gb.add_output(to_onnxscript_io_format(output))

for node in aten_graph.nodes:
    if node.op in func_mapping:
        func_mapping[node.op](gb, node)

model_name = "os_test_model.onnx"
onnx_model = gb.make_model(model_name)
