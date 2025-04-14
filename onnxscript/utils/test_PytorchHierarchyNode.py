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

    def add_non_hierarchical_node(self):
        # Non-hierarchical node is a single instance of a module
        self.op(
            op_type="NonHierarchicalNode",
            inputs=[]
        )

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


def add_node_expect_success(P, node):
    assert P.add_node(node) is True

def add_node_expect_failure(P, node):
    assert P.add_node(node) is False

# add a basic test to make sure the test file is working
def test_onenode():

    B = HierarchyBuilder()
    B.add_hierarchical_node([("", "class0")])

    P = gvu.PytorchHierarchyNode()

    print("\nadding bnode")

    add_node_expect_success(P, B.nodes[0])

    print("\nadded bnode")

    nodes = P.get_nodes([""])
    assert len(nodes) == 1
    assert nodes[0] is B.nodes[0]

def test_twonodes():
    B = HierarchyBuilder()
    B.add_hierarchical_node([("", "class0")])
    B.add_hierarchical_node([("", "class0")])

    P = gvu.PytorchHierarchyNode()
    add_node_expect_success(P, B.nodes[0])
    add_node_expect_success(P, B.nodes[1])

    nodes = P.get_nodes([""])
    assert len(nodes) == 2
    assert nodes[0] is B.nodes[0]
    assert nodes[1] is B.nodes[1]

def test_twonodes_one_with_hierarchy():
    B = HierarchyBuilder()
    B.add_hierarchical_node([("", "class0")])
    B.add_hierarchical_node([("", "class0"), ("a", "class1")])


    P = gvu.PytorchHierarchyNode()
    add_node_expect_success(P, B.nodes[0])
    add_node_expect_success(P, B.nodes[1])

    print("Printing hierarchy")
    P.print_hierarchy()

    nodes = P.get_nodes([""])
    assert len(nodes) == 2
    assert nodes[0] in B.nodes
    assert nodes[1] in B.nodes
    assert nodes[0] is not nodes[1]

    nodes = P.get_nodes(["", "a"])
    assert len(nodes) == 1
    assert nodes[0] is B.nodes[1]

def test_three_levels_of_hierarchy():
    B = HierarchyBuilder()
    B.add_hierarchical_node([("", "class0"), ("a", "class1"), ("b", "class2")])
    B.add_hierarchical_node([("", "class0"), ("a", "class1"), ("b", "class2")])
    B.add_hierarchical_node([("", "class0"), ("a", "class1"), ("b", "class2")])
    B.add_hierarchical_node([("", "class0"), ("a", "class1"), ("b", "class2")])

    P = gvu.PytorchHierarchyNode()
    add_node_expect_success(P, B.nodes[0])
    add_node_expect_success(P, B.nodes[1])
    add_node_expect_success(P, B.nodes[2])
    add_node_expect_success(P, B.nodes[3])

    nodes = P.get_nodes(["", "a", "b"])
    assert len(nodes) == 4
    assert nodes[0] in B.nodes
    assert nodes[1] in B.nodes
    assert nodes[2] in B.nodes
    assert nodes[3] in B.nodes


def test_non_hierarchical_node():
    B = HierarchyBuilder()
    B.add_non_hierarchical_node()

    P = gvu.PytorchHierarchyNode()
    add_node_expect_failure(P, B.nodes[0])

    assert len(P.get_nodes([""])) == 0
    assert len(P.children) == 0
