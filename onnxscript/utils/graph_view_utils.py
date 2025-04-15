from onnxscript import ir

import ast


def is_initializer(value):
    return input.producer() == None

def gather_initializers(inputs):
    inits = set()
    for input in inputs:
        if is_initializer(input):
            inits.add(input)

def is_constant(value):
    return value.producer.op_type == 'Constant'

def gather_constants(inputs):
    consts = set()
    for input in inputs:
        if is_constant(input):
            consts.add(input)


def has_internal_usage(usage):
    return "INTERNAL" in usage

def has_external_usage(usage):
    return "EXTERNAL" in usage

def classify_usage(value, nodes):
    usage = set()
    for use in value.uses():
        user_node = use[0]
        if user_node in nodes:
            usage.add("INTERNAL")
        else:
            usage.add("EXTERNAL")
    return usage

def find_subgraph_inputs(nodes):
    inputs = set()
    initializers = set()
    for node in nodes:
        for ninput in node.inputs:
            if ninput in node.graph.inputs:
                inputs.add(ninput)
            elif any(ninput is init for init in node.graph.initializers):
                initializers.add(ninput)
            elif ninput.producer() == None:
                inputs.add(ninput)
            elif ninput.producer() not in nodes:
                inputs.add(ninput)

    return inputs, initializers

def find_subgraph_outputs(nodes):
    output = set()
    used_output = set()
    for node in nodes:
        for noutput in node.outputs:
            usage = classify_usage(noutput, nodes)
            if has_external_usage(usage):
                if has_internal_usage(usage):
                    used_output.add(noutput)
                else:
                    output.add(noutput)
    return [output, used_output]


def bGraphView(name, nodes):


    [view_inputs,  view_initializers] = find_subgraph_inputs(nodes)
    [view_outputs, used_outputs] = find_subgraph_outputs(nodes)

    for used_output in used_outputs:
        producer_node = used_output.producer()
        nodes.remove(producer_node)
        for output in producer_node.outputs:
            usage = classify_usage(output,nodes)
            if has_internal_usage(usage):
                view_inputs.add(output)
            if has_external_usage(usage):
                if output in view_outputs:
                    view_outputs.remove(output)

    return ir.GraphView(name=name,
                        inputs=view_inputs,
                        outputs=view_outputs,
                        nodes=nodes,
                        initializers=view_initializers)

########################################
# rebuild_pytorch_dynamo_instance_code #
########################################

from typing import List


class PytorchMetadataNode:
    def __init__(self, node):
        self._node = node

        if self.check_node_metadata_exists():
            self.instance_metadata = ast.literal_eval(self._node.metadata_props['pkg.torch.onnx.name_scopes'])
            self.class_metadata = ast.literal_eval(self._node.metadata_props['pkg.torch.onnx.class_hierarchy'])
            print(f'self.node.metadata_props: {self._node.metadata_props}')

    def check_node_metadata_exists(self):
        if 'pkg.torch.onnx.name_scopes' in self._node.metadata_props and \
           'pkg.torch.onnx.class_hierarchy' in self._node.metadata_props:
            return True
        else:
            return False

    def is_last_level(self, level):
        if len(self.instance_metadata) - 1 == level:
            return True
        else:
            return False

    def get_instance_name(self, depth=0):
        if depth >= len(self.instance_metadata):
            return None
        else:
            return self.instance_metadata[depth]

    def get_class_name(self, depth=0):
        if depth >= len(self.instance_metadata):
            return None
        else:
            return self.class_metadata[depth]

class PytorchHierarchyNode:
    def __init__(self):
        self.instance_name = None
        self.module_type   = None
        self.children      = []
        self.nodes         = []

    def print_hierarchy(self, instance_hierarchy: List[str] = None):
        if instance_hierarchy is None:
            instance_hierarchy = []
        instance_hierarchy.append(self.instance_name)

        for child in self.children:
            child.print_hierarchy(list(instance_hierarchy))

        for node in self.nodes:
            print(f"Node: {node._node.name}, Instance: {'/'.join(instance_hierarchy)}, Module: {self.module_type}")


    def get_unwrapped_nodes(self):
        # Return _node from self._nodes
        return [node._node for node in self.nodes]

    # Checks if the search hierarchy matches the instance hierarchy
    def hierarchy_matches(self, search_hierarchy: List[str], instance_hierarchy: List[str] = []):
        search_length = min(len(search_hierarchy), len(instance_hierarchy))
        for i in range(search_length):
            if search_hierarchy[i] != instance_hierarchy[i]:
                return False
        return True

    # Return all nodes from the given name hierarchy on down
    def get_nodes(self, search_hierarchy: List[str], instance_hierarchy: List[str] = None):
        if instance_hierarchy is None:
            instance_hierarchy = []

        nodes_to_return = []
        # base case for recursion
        # 1 - search_hierarchy does not match instance_hierarchy
        instance_hierarchy.append(self.instance_name)
        #print(f"search_hierarchy: {search_hierarchy}")
        #print(f"instance_hierarchy: {instance_hierarchy}")

        if not self.hierarchy_matches(search_hierarchy, instance_hierarchy):
            return []

        for child in self.children:
                child_nodes = child.get_nodes(search_hierarchy, list(instance_hierarchy))
                nodes_to_return.extend(child_nodes)

        if len(instance_hierarchy) >= len(search_hierarchy):
            nodes_to_return.extend(self.get_unwrapped_nodes())

        return nodes_to_return

    # def add_nodes(self, nodes):
    #     for node in nodes:
    #         self.add_node(node)
    def add_node(self, node, level=0):

        # if node.name == 'node_Constant_2153':
        #     import pdb
        #     pdb.set_trace()

        print("calling add_node")
        if not isinstance(node, PytorchMetadataNode):
            node = PytorchMetadataNode(node)
            if node.check_node_metadata_exists() is False:
                return False

        if self.instance_name is None:
            print(f"setting instance name to {node.get_instance_name(level)}")
            self.instance_name = node.get_instance_name(level)
        if self.module_type is None:
            self.module_type = node.get_class_name(level)

        # check that instance name and module type match
        if self.instance_name != node.get_instance_name(level):
            #raise ValueError(f"Instance name mismatch: {self.instance_name} != {node.get_instance_name(level)}")
            return False
        if self.module_type   != node.get_class_name(level):
            #raise ValueError(f"Module type mismatch: {self.module_type} != {node.get_class_name(level)}")
            return False
        # if this is the last level of the hierarchy, add the node to this node
        # otherwise find the child node that matches the next level of the hierarchy
        # and add the node to that child
        if node.is_last_level(level):
            print(f"Adding node {node} to {self.instance_name}")
            self.nodes.append(node)
            return True
        else:
            for child in self.children:
                if child.instance_name == node.get_instance_name(level + 1):
                    return child.add_node(node, level + 1)

            # if no child matches the next level of the hierarchy, create a new child node
            new_child = PytorchHierarchyNode()
            new_child.instance_name = node.get_instance_name(level + 1)
            new_child.module_type   = node.get_class_name(level + 1)
            self.children.append(new_child)
            return new_child.add_node(node, level + 1)

def add_metadata_to_unannotated_constant_nodes(graph):
    for node in graph._nodes:
        if node.op_type == 'Constant' and not node.metadata_props:
            # search all of the uses to determine which hierarhcy to add
            # to the constant node
            # if all users have the same hierarchy, add that hierarchy to the constant node
            # if the users have different hierarchies, use the one level above the highest
            # level in the hierarchy
            metadata = set()
            for output in node.outputs:
                for user in output.uses():
                    user_node = user[0]
                    if user_node.metadata_props:
                        metadata.add((user_node.metadata_props['pkg.torch.onnx.name_scopes'],
                                      user_node.metadata_props['pkg.torch.onnx.class_hierarchy']))

            if len(metadata) == 1:
                name, class_hier = metadata.pop()
                node.metadata_props['pkg.torch.onnx.name_scopes'] = name
                node.metadata_props['pkg.torch.onnx.class_hierarchy'] = class_hier
            else:
                # convert the metadata_namescope set to a list of lists
                metadata_list = [(ast.literal_eval(x[0]),ast.literal_eval(x[1])) for x in list(metadata)]

                # find the index of namescope_list with the shortest length
                min_index = min(range(len(metadata_list)), key=lambda i: len(metadata_list[i][0]))

                # get the shortest namescope
                shortest_hierarchy = metadata_list[min_index]

                # remove the last level of the hierarchy
                target_name  = shortest_hierarchy[0][:len(shortest_hierarchy[0]) - 1]
                target_class = shortest_hierarchy[1][:len(shortest_hierarchy[1]) - 1]

                # convert the target_hierarchy to a string
                target_name_str = str(target_name)
                target_class_str = str(target_class)

                # add the target_hierarchy to the node
                node.metadata_props['pkg.torch.onnx.name_scopes'] = target_name_str
                node.metadata_props['pkg.torch.onnx.class_hierarchy'] = target_class_str
    return graph
