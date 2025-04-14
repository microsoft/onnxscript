import pytest

from onnxscript import script
from typing import List, Tuple


from onnxscript.utils import graph_view_utils as gvu
from onnxscript import ir



tape = ir._tape.Tape()


def build_metadata(instance_hierarchy, class_hierarchy):
    return {
        "pkg.torch.onnx.class_hierarchy": str(class_hierarchy),
        "pkg.torch.onnx.name_scopes": str(instance_hierarchy)
    }

class HierarchyBuilder(ir._tape.Tape):
    def __init__(self, graph_like: ir.Graph | ir.Function | None = None) -> None:
        super().__init__(graph_like)

    def add_hierarchical_node(self, hierarchy: List[Tuple[str, str]]):
        # Hierarchy is a list of tuples, where each tuple contains (instance name, module type)

        instance_hierarchy = []
        class_hierarchy = []
        for instance_name, module_type in hierarchy:
            instance_hierarchy.append(instance_name)
            class_hierarchy.append(module_type)

        self.op(
            op_type="HierarchyNode",
            inputs=[],
            metadata_props=build_metadata(instance_hierarchy, class_hierarchy),
        )

# add a basic test to make sure the test file is working
def test_onenode():

    B = HierarchyBuilder()
    B.add_hierarchical_node([("", "class0")])

    P = gvu.PytorchHierarchyNode()
    print("\nadding bnode")
    P.add_node(
        B.nodes[0]
    )
    print("\nadded bnode")
    nodes = P.get_nodes([""])
    assert len(nodes) == 1
    assert nodes[0] is B.nodes[0]

def test_twonodes():
    B = HierarchyBuilder()
    B.add_hierarchical_node([("", "class0")])
    B.add_hierarchical_node([("", "class0")])

    P = gvu.PytorchHierarchyNode()
    P.add_nodes(B.nodes)

    nodes = P.get_nodes([""])
    assert len(nodes) == 2
    assert nodes[0] is B.nodes[0]
    assert nodes[1] is B.nodes[1]

def test_twonodes_one_with_hierarchy():
    B = HierarchyBuilder()
    B.add_hierarchical_node([("", "class0")])
    B.add_hierarchical_node([("", "class0"), ("a", "class1")])


    P = gvu.PytorchHierarchyNode()
    P.add_nodes(B.nodes)

    print("Printing hierarchy")
    import pdb
    pdb.set_trace()
    P.print_hierarchy()

    print("test 1")
    nodes = P.get_nodes([""])
    assert len(nodes) == 2
    assert nodes[0] in B.nodes
    assert nodes[1] in B.nodes
    assert nodes[0] is not nodes[1]

    print("test 2")
    nodes = P.get_nodes(["", "a"])
    assert len(nodes) == 1
    assert nodes[0] is B.nodes[1]
