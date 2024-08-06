
import onnx
import onnx.inliner
import onnxscript.optimizer
import onnxscript.rewriter
from onnxscript.rewriter import onnxruntime as ort_rewriter
from onnx import NodeProto, TensorProto, helper, external_data_helper, numpy_helper
import numpy as np
from onnxscript import ir
from onnxscript.ir import convenience as ir_convenience
from logging import getLogger
from typing import List, Optional, Tuple, Union



#move out the directory

#     #original
# def export_to_onnx(input_model_path: str, output_model_path: str) -> onnx.ModelProto:
#     """
#     Export a model to ONNX.
#     Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite, 
#     and onnx.inliner.inline_local_functions.
#     """
#     # Load the ONNX model
#     onnx_model = onnx.load(input_model_path, load_external_data=False)
#     # # # Apply the onnx optimizer
#     # onnx_model = onnxscript.optimizer.optimize(onnx_model) #-- this one kinda messes up for now, causes that symbolicdim error even without kunal's changes
    

#     # #apply the onnx rewriter
#     onnx_model = onnxscript.rewriter.rewrite(onnx_model)
    

#     # apply the onnxruntime rewriter
#     onnx_model = ort_rewriter.rewrite(onnx_model) # keep

#     # apply the onnx inliner
#     # onnx_model = onnx.inliner.inline_local_functions(onnx_model)

#     # Save the ONNX model
#     # save_onnx_model(onnx_model, output_model_path, "myrules.onnx.data")
#     # onnx_model = onnx.load(output_model_path, load_external_data=False)
    
#     onnx_model = onnx.inliner.inline_local_functions(onnx_model)
    
#     # ir should be done after inlining --> ir optimization function
#     testing_model = ir.serde.deserialize_model(onnx_model) 
    
#     transpose_initializer_subgraphs(testing_model.graph)
#     onnx_model = ir.serde.serialize_model(testing_model)


    
    
#     save_onnx_model(onnx_model, output_model_path, "llama3b.onnx.data")


# def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
#     """
#     Save the ONNX model to disk, supporting external data.
#     """
#     onnx.save(
#         onnx_model,
#         output_path,
#         save_as_external_data=True,
#         all_tensors_to_one_file=True,
#         location=data_path,  # Must be a relative path
#         size_threshold=0,
#         convert_attribute=False,
#     )
#     print(f"Model saved with external data to {output_path}")
    
# #uncomment for llama2, make sure youre in the right directory.
# # input_model_path = "/home/t-assumange/llama2-7b_Dynamo_transformers4.41/rank_0_Llama-2-7b-hf_decoder_with_past_model_fp32.onnx"
# # output_model_path = "/home/t-assumange/llama2-7b_Dynamo_transformers4.41/llama2.onnx"

# #uncomment for llama3, make sure youre in the right directory.
# input_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/rank_0_Meta-Llama-3-8B_decoder_with_past_model_fp32.onnx"
# output_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/llama3b.onnx"





# def transpose_initializer_subgraphs(graph):
#     transpose_nodes = []
    
#   #traverse the graph to find Transpose nodes
#     for node in graph:
#         if node.op_type == 'Transpose':
#             transpose_nodes.append(node)
    
#     #process each Transpose node
#     for node in transpose_nodes:
        
#         # make an assertion that every operator is a transpose.
#         assert node.op_type == 'Transpose'
#         transpose_input = node.inputs[0]
#         transpose_input_name = transpose_input.name
#         permutation = node.attributes['perm'].value
        
#         if transpose_input_name not in graph.initializers:
#             continue
#         if len(transpose_input.uses()) !=1:
#             continue

#         # call to update initializer
#         update_initializer(transpose_input, permutation)
#         num_users = len(node.outputs[0].uses())
#         ir_convenience.replace_all_uses_with(node.outputs[0], [transpose_input]*num_users) 

#         graph.remove(node, safe=True)

   
        

# def transpose_tensor(tensor : np.ndarray, permutation : list[int]) -> np.ndarray:
#     return np.transpose(tensor, axes=permutation)

# def update_initializer(initializer : ir.Value, permutation: list[int]):

#     #get original initializer and replace it
#     array = initializer.const_value.numpy()
#     transposed_initializer = transpose_tensor(array, permutation)
#     tensor = ir.tensor(transposed_initializer)
#     initializer.const_value = tensor
#     initializer.shape = tensor.shape

# def rotary_embedding_caches(inv_freq, max_seq_len):

   
#     t = np.arange(max_seq_len, dtype=np.int64).type_as(inv_freq)

#     freqs = np.outer(t, inv_freq)
#     emb = np.cat((freqs, freqs), dim=-1)
#     cos_cache, sin_cache = emb.cos(), emb.sin()
#     return cos_cache, sin_cache



    

# export_to_onnx(input_model_path, output_model_path)






def export_to_onnx(input_model_path: str, output_model_path: str):
    """
    Export a model to ONNX.
    Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite, 
    and onnx.inliner.inline_local_functions.
    """
    # Load the ONNX model
    onnx_model = onnx.load(input_model_path, load_external_data=False)
    

    # Apply the onnx rewriter
    # onnx_model = onnxscript.rewriter.rewrite(onnx_model)
    
    # Apply the onnxruntime rewriter
    # onnx_model = ort_rewriter.rewrite(onnx_model)
    
    onnx_model = onnx.inliner.inline_local_functions(onnx_model)
    
    testing_model = ir.serde.deserialize_model(onnx_model)

    fuse_add_and_layernorm(testing_model.graph)

    
    transpose_initializer_subgraphs(testing_model.graph)
    # update_group_query_attention(testing_model.graph, 8192)  # Adding this line to update GroupQueryAttention nodes
    
    onnx_model = ir.serde.serialize_model(testing_model)
    
    save_onnx_model(onnx_model, output_model_path, "llama3bxyz.onnx.data")



def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
    """
    Save the ONNX model to disk, supporting external data.
    """
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,  # Must be a relative path
        size_threshold=0,
        convert_attribute=False,
    )
    print(f"Model saved with external data to {output_path}")


from onnxscript.ir import Node, AttrFloat32, Value



# def fuse_add_and_layernorm(graph: ir.Graph):
#     nodes_to_remove = []
#     nodes_to_add = []

#     for node in graph._nodes:
#         if node.op_type == 'Add':
#             add_node = node
#             add_output = add_node.outputs[0]

#             for user, _ in add_output.uses():
#                 if user.op_type == 'SimplifiedLayerNormalization':
#                     layernorm_node = user

#                     # Correctly extract the epsilon attribute as an Attr instance
#                     epsilon_attr = layernorm_node.attributes['epsilon']

#                     # Create SkipSimplifiedLayerNormalization node
#                     skip_layernorm_node = Node(
#                         op_type='SkipSimplifiedLayerNormalization',
#                         domain='com.microsoft',
#                         name="Skip" + layernorm_node.name,
#                         inputs=[
#                             add_node.inputs[0],  # input
#                             add_node.inputs[1],  # skip
#                             layernorm_node.inputs[1]  # gamma (weight)
#                         ],
#                         outputs=[
#                             Value(name=layernorm_node.outputs[0].name),
#                             Value(name=""),
#                             Value(name=""),
#                             Value(name=add_node.outputs[0].name)
#                         ],
#                         attributes= [epsilon_attr]
                        
#                     )

#                     nodes_to_add.append(skip_layernorm_node)
#                     nodes_to_remove.append(add_node)
#                     nodes_to_remove.append(layernorm_node)
#                     break

#     for node in nodes_to_remove:
#         graph.remove(node)

#     for node in nodes_to_add:
#         graph.append(node)

from onnxscript.ir import Node, AttrFloat32, Value, Graph

def fuse_add_and_layernorm(graph: Graph):
    nodes_to_remove = []
    nodes_to_add = []
    reconnections: list[tuple[Node, int, Value]] = []  # List to hold reconnections to be made after iteration

    for node in graph:
        if node.op_type == 'Add':
            add_node = node
            add_output = add_node.outputs[0]

            assert len(add_output.uses()) == 1

            for user, _ in add_output.uses():
                if user.op_type == 'SimplifiedLayerNormalization':
                    layernorm_node = user

                    # Extract the epsilon attribute correctly
                    epsilon_attr = layernorm_node.attributes["epsilon"]

                    # Create SkipSimplifiedLayerNormalization node
                    skip_layernorm_node = Node(
                        op_type='SkipSimplifiedLayerNormalization',
                        domain='com.microsoft',
                        # name="Skip" + layernorm_node.name,
                        inputs=[
                            add_node.inputs[0],  # input
                            add_node.inputs[1],  # skip
                            layernorm_node.inputs[1]  # gamma (weight)
                        ],
                        outputs=[
                            Value(name=layernorm_node.outputs[0].name),
                            Value(name=""),
                            Value(name=""),
                            Value(name=add_node.outputs[0].name)
                        ],
                        attributes=[epsilon_attr]
                    )

                    nodes_to_add.append(skip_layernorm_node)
                    nodes_to_remove.append(add_node)
                    nodes_to_remove.append(layernorm_node)

                    # Collect reconnections
                    for output in layernorm_node.outputs:
                        for user, idx in output.uses():
                            reconnections.append((user, idx, skip_layernorm_node.outputs[0]))

                    break

    # Perform reconnections
    for user, idx, new_output in reconnections:
        user.replace_input_with(idx, new_output)

    # Ensure nodes to be removed are no longer used by other nodes
    graph.remove(nodes_to_remove)

    for node in nodes_to_add:
        graph.append(node)




def transpose_initializer_subgraphs(graph):
    transpose_nodes = []

    # Traverse the graph to find Transpose nodes
    for node in graph:
        if node.op_type == 'Transpose':
            transpose_nodes.append(node)
    
    # Process each Transpose node
    for node in transpose_nodes:
        # Make an assertion that every operator is a transpose
        assert node.op_type == 'Transpose'
        transpose_input = node.inputs[0]
        transpose_input_name = transpose_input.name
        permutation = node.attributes['perm'].value
        
        if transpose_input_name not in graph.initializers:
            continue
        if len(transpose_input.uses()) != 1:
            continue

        # Call to update initializer
        update_initializer(transpose_input, permutation)
        num_users = len(node.outputs[0].uses())
        ir_convenience.replace_all_uses_with(node.outputs[0], [transpose_input] * num_users)

        graph.remove(node, safe=True)



def transpose_initializer_subgraphs(graph):
    transpose_nodes = []
    
  #traverse the graph to find Transpose nodes
    for node in graph:
        if node.op_type == 'Transpose':
            transpose_nodes.append(node)
    
    #process each Transpose node
    for node in transpose_nodes:
        
        # make an assertion that every operator is a transpose.
        assert node.op_type == 'Transpose'
        transpose_input = node.inputs[0]
        transpose_input_name = transpose_input.name
        permutation = node.attributes['perm'].value
        
        if transpose_input_name not in graph.initializers:
            continue
        if len(transpose_input.uses()) !=1:
            continue

        # call to update initializer
        update_initializer(transpose_input, permutation)
        num_users = len(node.outputs[0].uses())
        ir_convenience.replace_all_uses_with(node.outputs[0], [transpose_input]*num_users) 

        graph.remove(node, safe=True)

   
        

def transpose_tensor(tensor : np.ndarray, permutation : list[int]) -> np.ndarray:
    return np.transpose(tensor, axes=permutation)

def update_initializer(initializer : ir.Value, permutation: list[int]):

    #get original initializer and replace it
    array = initializer.const_value.numpy()
    transposed_initializer = transpose_tensor(array, permutation)
    tensor = ir.tensor(transposed_initializer)
    initializer.const_value = tensor
    initializer.shape = tensor.shape

def rotary_embedding_caches(inv_freq, max_seq_len):

   
    t = np.arange(max_seq_len, dtype=np.int64).astype(inv_freq.dtype)

    freqs = np.outer(t, inv_freq)
    emb = np.concatenate((freqs, freqs), dim=-1)
    cos_cache, sin_cache = emb.cos(), emb.sin()
    return cos_cache, sin_cache




input_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/llama3bxyz.onnx"
output_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/llama3bxyz.onnx"

export_to_onnx(input_model_path, output_model_path)
# python /home/t-assumange/onnxscript/initializer.py








""""

def fuse_add_and_layernorm(graph: ir.Graph):
    nodes_to_remove = []
    nodes_to_add = []

    for node in graph._nodes:
        if node.op_type == 'Add':
            add_node = node
            add_output = add_node.outputs[0]

            for user, _ in add_output.uses():  
                if user.op_type == 'SimplifiedLayerNormalization':
                    layernorm_node = user

                    

                    # Create SkipSimplifiedLayerNormalization node
                    print(layernorm_node.attributes['epsilon'], "layernorm_node.attributes")
                    skip_layernorm_node = ir.Node(
                        op_type='SkipSimplifiedLayerNormalization',
                        domain='com.microsoft',
                        name = "Skip" + layernorm_node.name,
                        inputs=[
                            add_node.inputs[0],  # input
                            add_node.inputs[1],  # skip
                            layernorm_node.inputs[1]  # gamma (weight)
                        ],
                        outputs=[layernorm_node.outputs[0],
                                 "",
                                 "",
                                 add_node.outputs[0], # skip bias
                                 ], 

                                 #dictionary of attributes
                                
                        attributes={"epsilon": layernorm_node.attributes["epsilon"]}
                    )

                    nodes_to_add.append(skip_layernorm_node)
                    nodes_to_remove.append(add_node)
                    nodes_to_remove.append(layernorm_node)
                    break

    for node in nodes_to_remove:
        graph.erase_node(node)

    for node in nodes_to_add:
        graph.add_node(node)
"""