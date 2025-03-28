import copy

import numpy as np
import onnx

import pdb

from onnxscript import ir
from onnxscript import rewriter
from onnxscript.rewriter.pattern import (
    MatchResult, ValuePattern, GraphPattern, OpsetPatternBuilder, pattern_builder, NodeOutputPattern, ReplacementSubgraph

)



from collections.abc import Iterable
from onnxscript.utils.graph_view_utils import bGraphView

#print("**************************************")
#print("********* Pattern Builder ************")
#print("**************************************")


def direct_convert_ir_graph_to_pattern(graph):


    # Transform IR values to ValuePatterns
    vmap = {}
    for input in graph.inputs:
        vmap[input] = ValuePattern(input.name)

    for init in graph.initializers:
        vmap[init] = ValuePattern(init.name)


    for node in graph._nodes:
        if node.op_type == 'Constant':
            vmap[node.outputs[0]] = ValuePattern(node.outputs[0].name)

    builder = OpsetPatternBuilder("", record=True)

    with pattern_builder(builder):
        for node in graph._nodes:
            ninputs = []
            for ninput in node.inputs:
                ninputs.append(vmap[ninput])

            #if len(node.outputs) > 1:
            vp_outputs = builder.__getattr__(node.op_type)(*ninputs,_domain=node.domain, _outputs=len(node.outputs))
            #else:
            #    vp_outputs = builder.__getattr__(node.op_type)(*ninputs)


            if isinstance(vp_outputs,NodeOutputPattern):
                vp_outputs = [vp_outputs]

            for vp_output in iter(vp_outputs):
                vmap[node.outputs[vp_output.output_index]] = vp_output


    pinputs = []
    for input in graph.inputs:
        pinputs.append(vmap[input])

    # build graph outputs
    poutputs = []
    for output in graph.outputs:
        poutputs.append(vmap[output])

    return GraphPattern(inputs=pinputs, outputs=poutputs, nodes=builder.nodes())

from enum import Enum

def remove_input_from_node(node, inp):
    node._inputs = [x for x in node._inputs if x is not inp]
    inp._remove_usage(node)


class LoopBodyInputType(Enum):
    UNDEFINED = 0
    ACTIVATION = 1
    CONSTANT   = 2
    PARAMETER  = 3
    ITERATOR   = 4
    CONDITION  = 5

    def __str__(self):
        return self.name

class LoopBodyTemplate:
    def __init__(self, filename):
        self.load(filename)
        self.pattern  = direct_convert_ir_graph_to_pattern(self._ir_graph)
        self.function = self._build_ir_function()
        self.function_replace = self._build_function_replace_pattern()
        self.signature = [LoopBodyInputType.UNDEFINED] * len(self._ir_graph.inputs)

    def _build_ir_function(self):
        return ir.Function(domain='loop',
                           name='fn_' + self._ir_graph.name,
                           graph = self._ir_graph,
                           attributes=[])

    def _build_function_replace_pattern(self):

        inputs  = [vdisconnect(copy.copy(x)) for x in self._ir_graph.inputs]
        outputs = [vdisconnect(copy.copy(x)) for x in self._ir_graph.outputs]

        node = ir.Node(domain=self.function.domain,
                       version=0,
                       op_type=self.function.name,
                       inputs=inputs, outputs=outputs)

        g = ir.Graph(inputs=inputs, outputs=outputs, nodes=[node])

        return ReplacementPatternGraph(g)

    def get_iterator_index(self):
        for i in range(len(self.signature)):
            if self.signature[i] == LoopBodyInputType.ITERATOR:
                return i

    def insert_gather_nodes(self, loop_iterations):
        for index,LoopInputType in enumerate(self.signature):
            if LoopInputType == LoopBodyInputType.PARAMETER:
                # The Current Input Value will be the output of the gather node
                gather_index = self.function.inputs[self.get_iterator_index()]
                squeeze_out   = self.function.inputs[index]

                gather_in    = ir.Value(name=squeeze_out.name+"_gather_in",
                                        shape=ir.Shape([loop_iterations, *squeeze_out.shape.dims]),
                                        type=squeeze_out.type)
                gather_out   = ir.Value(name=squeeze_out.name+"_gather_out",
                                        shape=ir.Shape([1, *squeeze_out.shape.dims]),
                                        type=squeeze_out.type)
                for usage in squeeze_out.uses():
                    if usage.node.op_type == "Identity":
                        usage.node.replace_input_with(usage.idx, gather_in)
                        usage.node.outputs[0].shape = copy.copy(gather_in.shape)

                self.function.inputs[index] = gather_in

                self.function.append(ir.Node(domain='',
                                             op_type='Gather',
                                             inputs = [gather_in, gather_index],
                                             outputs = [gather_out],
                                             num_outputs = 1
                                             )
                                     )
                squeeze_out.name += "_squeeze_out"
                squeeze_axis = build_constant_from_tensor(f'{gather_out.name}_squeeze_axis', ir.Tensor(np.array([0])))
                self.function.append(squeeze_axis)
                self.function.append(ir.Node(domain='',
                                             op_type='Squeeze',
                                             inputs=[gather_out, squeeze_axis.outputs[0]],
                                             outputs=[squeeze_out],
                                             num_outputs= 1,
                                             version = 13
                                             )
                                     )

                self.function.sort()


    def build_function_match_pattern(self, graph):
        graph.sort()
        nodes  = find_nodes_of_optype(graph, self.function.name)
        nodes.insert(0,graph.node('iteration_ext'))
        nodes.insert(0,graph.node('condition_ext'))

        ir_model = ir.Model(bGraphView('inlined_pipe_pattern', nodes), ir_version=10)

        model = ir.serde.serialize_model(ir_model)
        onnx.save(model, 'pipeline_match_pattern.onnx')

        pattern  = direct_convert_ir_graph_to_pattern(ir_model.graph)

        return (pattern, nodes)


    def load(self, filename):
        self._model_proto = onnx.load(filename)
        self._ir_model    = ir.serde.deserialize_model(self._model_proto)
        self._ir_graph    = self._ir_model.graph

    def update(self):
        self._ir_model     = ir.Model(self._ir_graph, ir_version = 10)
        self._model_proto  = ir.serde.serialize_model(self._ir_model)

    def save(self, filename):
        self.update()
        onnx.save(self._model_proto, filename)

    def set_signature_index(self, index, stype):
        self.signature[index] = stype

    @property
    def output_signature(self):
        # The output signature is the same as the input signature but without the iteration input
        return self.signature[1:]


def same(input_list):
    return len(set(input_list)) == 1

def append_output_to_node(node, output):
    output._producer = node
    output._index = node.outputs[-1]._index + 1
    node._outputs = (*node._outputs, output)
    node._num_outputs = len(node._outputs)

def prepend_output_to_node(node, output):
    output._producer = node
    output._index = 0
    for outp in node._outputs:
        outp._index += 1
    node._outputs = (output, *node._outputs)
    node._num_outputs = len(node._outputs)

def prepend_input_to_node(node, input):
    input._add_usage(node, 0)
    # increment the index for all uses on this node
    for i,inp in enumerate(node._inputs):
        inp._remove_usage(node, i)
        inp._add_usage(node, i+1)

    node._inputs = (input, *node._inputs)

def normalize_io_for_loop_rolling(graph, LoopBody):

    # This takes a graph that has consecutive identical nodes
    # and normalizes the i/o indicies prior to the loop rolling
    # optimization. Specifically, this function identifies output-
    # to-input pairs and permutes the indices of the input to match
    # the previous output.

    # The ONNX loop node requires that interloop dependencies
    # have identical input and output indices.

    # Run a topological sort to ensure the layers are in order.
    graph.sort()

    # get the consecutive node layers
    # TODO: write a check to ensure that there is only one
    #       set of consecutive nodes.
    nodes = find_nodes_of_optype(graph, LoopBody.function.name)

    # Loop through all the nodes (execept the last one) and
    # identify the input to output pairs
    input_swaps = []
    for i in range(len(nodes)-1):
        a_node = nodes[i]
        b_node = nodes[i+1]

        for a_out in a_node.outputs:
            # Require that outputs of a have a single use of b_node
            assert(len(a_out.uses()) == 1)
            assert(a_out.uses()[0][0] is b_node)

            a_use_index = a_out.uses()[0][1]
            input_swap = (a_out.index(), a_use_index)
            if i == 0:
                # add swaps from the first node
                input_swaps.append(input_swap)
            else:
                # check that they are the same in the rest
                assert(input_swap in input_swaps)

    # apply the input swaps to each nodes
    for node in nodes:
        for swap in input_swaps:
            a = node.inputs[swap[0]]
            b = node.inputs[swap[1]]
            node.replace_input_with(swap[0], b)
            node.replace_input_with(swap[1], a)

    # apply the input swaps to the function graph
    # mark swapped nodes as activations
    activations = 0
    for swap in input_swaps:
        a = LoopBody.function.inputs[swap[0]]
        b = LoopBody.function.inputs[swap[1]]
        LoopBody.function.inputs[swap[0]] = b
        LoopBody.function.inputs[swap[1]] = a
        LoopBody.signature[swap[0]] = LoopBodyInputType.ACTIVATION
        activations+=1

    # Next Inputs according to how they are produced.
    # Indexable inputs will have different constant or none producers
    # Constant values broadcast to all nodes will have the same producer
    # Skip the (all) Activation inputs (have been swapped to beginning of the list)
    for index in range(activations, len(nodes[0].inputs)):
        inputs    = []
        producers = []
        for node in nodes:
            cinput = node.inputs[index]
            inputs.append(cinput)
        if same(inputs):
            # Constant with Respect to Loop
            LoopBody.signature[index] = LoopBodyInputType.CONSTANT
        else:
            # Must be Indexed
            LoopBody.signature[index] = LoopBodyInputType.PARAMETER


    # Match Output Signature to Input Signature
    for index,LoopInputType in enumerate(LoopBody.signature):
        cinput  = LoopBody.function.inputs[index]
        noutput = vdisconnect(copy.copy(cinput))
        noutput._uses = {}
        update_node_outputs = False

        if LoopInputType == LoopBodyInputType.CONSTANT or \
           LoopInputType == LoopBodyInputType.PARAMETER:
            # Update Names and Add Output
            cinput.name += "_" + str(LoopInputType) + "_in"
            noutput.name += "_" + str(LoopInputType) + "_out"

            LoopBody.function.outputs.append(noutput)

            # Add Identify to Pass Inputs to new Output
            Ident = ir.Node(domain='',
                            op_type='Identity',
                            inputs = [cinput],
                            outputs = [noutput],
                            num_outputs =1)
            LoopBody.function.append(Ident)

            #Add Output to Function Call Nodes
            for i,node in enumerate(nodes):
                output_copy = copy.copy(noutput)

                #preserve single_assignment
                output_copy.name += f'_{i}'
                append_output_to_node(node,output_copy)

    # Add Iterator and Condition Inputs (Leave Unconnected within function for now)
    iteration = ir.Value(name='iteration', shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.INT64))
    condition = ir.Value(name='cond', shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.BOOL))

    LoopBody.signature.insert(0, LoopBodyInputType.CONDITION)
    LoopBody.function.inputs.insert(0,condition)

    LoopBody.signature.insert(0, LoopBodyInputType.ITERATOR)
    LoopBody.function.inputs.insert(0,iteration)


    iteration_ext = build_constant_from_tensor('iteration_ext', ir.Tensor(np.array([len(nodes)])))
    condition_ext = build_constant_from_tensor('condition_ext', ir.Tensor(np.array([True])))
    #iteration_ext = ir.Value(name='iteration', shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.INT64))
    #condition_ext = ir.Value(name='cond_ext', shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.BOOL))

    # add these nodes to the graph before the rest to maintain topological sorted-ness
    nodes[0].prepend([iteration_ext,condition_ext])

    for node in nodes:
        prepend_input_to_node(node, condition_ext.outputs[0])
        prepend_input_to_node(node, iteration_ext.outputs[0])

    # Add Identity Node for Condition
    condition_out   = ir.Value(name='cond_out', type=ir.TensorType(ir.DataType.BOOL))
    condition_ident = ir.Node(domain='',
                              op_type='Identity',
                              inputs  = [condition],
                              outputs = [condition_out],
                              num_outputs =1)
    LoopBody.function.append(condition_ident)
    LoopBody.function.outputs.insert(0,condition_out)

    # Add New Condition Output to Node
    for i,node in enumerate(nodes):
        noutput = ir.Value(name=f'cond_out_{i}', type=ir.TensorType(ir.DataType.BOOL))
        prepend_output_to_node(node,noutput)

    graph.sort()
    return graph


import copy

def vdisconnect(value):
    value._uses = {}
    value._producer = None
    value._index = None
    return value


class ReplacementPatternGraph(rewriter.pattern.ReplacementPatternFunction):
    def __init__(self, ir_graph):
        self._graph = ir_graph


    def get_replacement(self, match: MatchResult) -> ReplacementSubgraph | None:

        context = rewriter.RewriterContext()
        # match.bindings is dictionary of value_name (str) in replacement subgraph pattern (i.e. ir_graph -> IR Value in actual graph)
        vvmap = {} # Dictionary mapping values in replacement subgraph pattern -> values in the replacement subgraph

        for value in self._graph.inputs:
            if value.name in match.bindings:
                vvmap[value] = match.bindings[value.name]
            else:
                vvmap[value] = value

        for node in self._graph._nodes:
            ninputs = []
            for ninput in node.inputs:
                ninputs.append(vvmap[ninput])


            coutput = context.__getattr__(node.op_type)(*ninputs, **node.attributes, _outputs=len(node.outputs), _domain=node.domain, _version=node.version)
            if not isinstance(coutput,Iterable):
                coutput = [coutput]

            for i, cout in enumerate(coutput):
                vvmap[node.outputs[cout.index()]] = cout

        new_outputs = [vvmap[x] for x in self._graph.outputs]
        return rewriter.ReplacementSubgraph(
            match, new_outputs, context.nodes, context.initializers, context.used_opsets
        )


def convert_graph_to_function_call_pattern(graph):

    inputs  = [vdisconnect(copy.copy(x)) for x in graph.inputs]
    outputs = [vdisconnect(copy.copy(x)) for x in graph.outputs]

    node = ir.Node('', graph.name+'_fcall', inputs, outputs=outputs)

    g = ir.Graph(inputs=inputs, outputs=outputs, nodes=[node])


    return ReplacementPatternGraph(g)


def find_nodes_of_optype(graph, layername):
    nodes = []
    for node in ir.traversal.RecursiveGraphIterator(graph):
        if node.op_type == layername:
            nodes.append(node)
    return nodes


def build_layer_pipeline_pattern(graph, layername):

    nodes    = find_nodes_of_optype(graph, layername)
    ir_model = ir.Model(bGraphView('inlined_pipe_pattern', nodes), ir_version=10)

    model = ir.serde.serialize_model(ir_model)
    onnx.save(model, 'pipeline_match_pattern.onnx')

    pattern  = direct_convert_ir_graph_to_pattern(ir_model.graph)

    return (pattern, nodes)

def build_constant_from_tensor(name, tensor):
    value_attribute = ir.Attr(name='value', type=ir.AttributeType.TENSOR, value=tensor)
    ir_value_out    = ir.Value(name=name+'_out', type=ir.TensorType(tensor.dtype))
    return ir.Node('', 'Constant', name=name, inputs=[], outputs=[ir_value_out], attributes=[value_attribute])

def build_concat_node_from_inputs(inputs):

    axis  = ir.Attr(name='axis', type=ir.AttributeType.INT, value=0)
    ndim  = len(inputs) * inputs[0].shape.dims[0]
    output_shape = ir.Shape([ndim, *inputs[0].shape.dims[1:]])
    output       = ir.Value(name=f'{inputs[0].name}_concat', shape=output_shape, type=inputs[0].type)
    return ir.Node('', 'Concat', inputs=inputs, attributes=[axis], outputs=[output])

def build_reshape_node(inp, reshape_shape):
    reshape_out  = ir.Value(name=f'{inp.name}_reshape',  type=inp.type)
    return ir.Node('', 'Reshape', inputs=[inp, reshape_shape], outputs=[reshape_out])



def build_loop_replace_pattern(graph, LoopBody):

    nodes = find_nodes_of_optype(graph, LoopBody.function.name)

    graph_nodes = []
    loop_inputs = []

    # Build max_iteration and condition constants
    M = build_constant_from_tensor('M', ir.Tensor(np.array([len(nodes)])))
    cond = build_constant_from_tensor('cond', ir.Tensor(np.array([True])))

    graph_nodes.append(M)
    graph_nodes.append(cond)

    loop_inputs.append(M.outputs[0])
    loop_inputs.append(cond.outputs[0])

    graph_inputs  = []
    for i, LoopInputType in enumerate(LoopBody.signature):

        if LoopInputType == LoopBodyInputType.PARAMETER:
            # Build Concat Node
            concat_inputs  = []
            for node in nodes:
                nvalue = vdisconnect(copy.copy(node.inputs[i]))
                graph_inputs.append(nvalue)
                concat_inputs.append(nvalue)

            concat_node = build_concat_node_from_inputs(concat_inputs)
            graph_nodes.append(concat_node)

            # Build Reshape Node
            reshape_shape_const = build_constant_from_tensor(f'reshape_shape_const_{i}', ir.Tensor(np.array([len(nodes),*concat_inputs[0].shape.dims])))
            graph_nodes.append(reshape_shape_const)

            reshape_node = build_reshape_node(concat_node.outputs[0], reshape_shape_const.outputs[0])
            graph_nodes.append(reshape_node)
            loop_inputs.append(reshape_node.outputs[0])
        elif LoopInputType == LoopBodyInputType.CONSTANT:
            constant_input = nodes[0].inputs[i]
            constant_node  = constant_input.producer()
            constant_value  = constant_node.attributes['value'].value.numpy()
            n_constant_node = build_constant_from_tensor(constant_input.name+"_const_val", ir.Tensor(constant_value))
            graph_nodes.append(n_constant_node)
            loop_inputs.append(n_constant_node.outputs[0])
        elif LoopInputType == LoopBodyInputType.ACTIVATION:
            cinp = vdisconnect(copy.copy(LoopBody.function.inputs[i]))
            graph_inputs.append(cinp)
            loop_inputs.append(cinp)



    loop_outputs = []
    graph_outputs = []
    for i, LoopOutputType in enumerate(LoopBody.output_signature):
        output = vdisconnect(copy.copy(LoopBody.function.outputs[i]))
        if LoopOutputType != LoopBodyInputType.CONDITION:
            loop_outputs.append(output)
        if LoopOutputType == LoopBodyInputType.ACTIVATION:
            graph_outputs.append(output)

    LoopBody.insert_gather_nodes(len(nodes))
    body_attr = ir.Attr(name='body', type=ir.AttributeType.GRAPH, value=LoopBody.function._graph)
    graph_nodes.append(ir.Node('', 'Loop', inputs=loop_inputs, attributes = [body_attr], outputs=loop_outputs, graph=None))

    graph = ir.Graph(name='loop_replace',nodes=graph_nodes, inputs=graph_inputs, outputs=graph_outputs)

    graph.sort()

    model = ir.serde.serialize_model(ir.Model(graph, ir_version=8))
    onnx.save(model, 'replacementgraph.onnx')

    return ReplacementPatternGraph(graph)
