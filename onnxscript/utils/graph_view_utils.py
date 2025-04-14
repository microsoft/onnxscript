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

##################################################
## TODO (JSM): encapsulte this into a function  ##
##################################################

# model = onnx.load('mistral.onnx')

# model_ir = ir.serde.deserialize_model(model)

# layer_dict = {}

# no_name_scopes = set()
# for node in ir.traversal.RecursiveGraphIterator(model_ir.graph):
#     if 'pkg.torch.onnx.name_scopes' in node.metadata_props:
#         name_scopes = ast.literal_eval(node.metadata_props['pkg.torch.onnx.name_scopes'])
#         if name_scopes[1].startswith('layer'):
#             if name_scopes[1] not in layer_dict:
#                 layer_dict[name_scopes[1]] = []
#             layer_dict[name_scopes[1]].append(node)
#         else:
#             print(node)
#     else:
#         no_name_scopes.add(node)

# scoped_nodes = set()
# stop = False
# for node in no_name_scopes:
#     #input('pause for enter')
#     print(node)
#     layer_usage = set()
#     for value in node.outputs:
#         print(f"\t{value}")
#         print(f"\t{node.name}")
#         if value.name == 'val_39':
#             stop = True
#             input('found val_39')
#         for use in value.uses():
#             used_node = use[0]
#             print(f"\t\t{used_node}")
#             if 'pkg.torch.onnx.name_scopes' in used_node.metadata_props:
#                 print(f"\t\t\t{used_node.metadata_props['pkg.torch.onnx.name_scopes']}")
#                 name_scopes = ast.literal_eval(used_node.metadata_props['pkg.torch.onnx.name_scopes'])
#                 layer_usage.add(name_scopes[1])
#             else:
#                 print(f"\t\t\tno scope")
#                 layer_usage.add("")
#         print(f'\t\tlayer usage {layer_usage}')
#     if len(layer_usage) == 1:
#         scope = next(iter(layer_usage))
#         print(scope)
#         print(layer_dict.keys())
#         print(node)
#         if scope in layer_dict:
#             layer_dict[scope].insert(0, node)
#             scoped_nodes.add(node)
#         if stop == True:
#             pass

# for key in layer_dict:
#     print(key+"********")
#     layer = bGraphView(key, layer_dict[key])

#     print(key+"STARTIO********")
#     print(f"inputs: {layer.inputs}")
#     print(f"outputs: {layer.outputs}")
#     print(key+"ENDIO********")
#     print("\n\n")

# #exit(1)
# #layer0_gv = bGraphView('layers.0', layer_dict['layers.0'])
# layer1_gv = bGraphView('layers_1',layer_dict['layers.1'])

# print(f"inputs: {layer1_gv.inputs}")
# print(f"outputs: {layer1_gv.outputs}")
# #print(f"initializers: {.initializers}")

# #exit(1)

# layer0 = layer_dict['layers.0']
# layer1 = layer_dict['layers.1']

# d = {}

# for a in layer0:
#     d[a] = {a}
#     for b in layer1:
#         if ir_node__eq__(a, b):
#             d[a].add(b)

# for node in d:
#     if len(d[node]) == 1:
#         print(f"single node set: {node}")
#         print(f"value: {node.outputs[0]}")
#         #print(f"uses: {node.outputs[0].uses()}")
#         print(f"len: {len(node.outputs[0].uses())}")
#         for use in node.outputs[0].uses():
#             print(str(use)+'\n\n')
#     #if len(d[node]) > 2:
#     #    print(f"set with {len(d[node])} nodes")
#     #    for n in d[node]:
#     #        print(f'\t{n}')
#     #    input('press n to continue')

# #print(model.graph)

# layer = bGraphView('layers_0', layer_dict['layers.0'])

# for node in layer._nodes:
#      if node.name == 'node_Constant_2143':
#         print(node)
#         input('found it!')


# model = ir.Model(layer, ir_version=10)
# proto = ir.serde.serialize_model(model)
# print('saving')
# onnx.save(proto, 'layer0.onnx')


# print('done')

