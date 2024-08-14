
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



def export_to_onnx(input_model_path: str, output_model_path: str):
    """
    Export a model to ONNX.
    Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite,
    and onnx.inliner.inline_local_functions.
    """
    try:
        ######but what if you run the optimizer after everything else? i mean after the rewriter and inliner? downwards
        # Load the ONNX model
        onnx_model = onnx.load(input_model_path, load_external_data=False)

        # Apply the onnx rewriter
        onnx_model = onnxscript.rewriter.rewrite(onnx_model)

        # Apply the onnxruntime rewriter
        onnx_model = ort_rewriter.rewrite(onnx_model)

        #save the model
        # save_onnx_model(onnx_model, output_model_path, "llama3bcons+rew.onnx.data")

        # #load with external data
        # onnx_model = onnx.load(output_model_path, load_external_data=True)


        # # Apply the onnx optimizer
        # onnx_model = onnxscript.optimizer.optimize(onnx_model)

        # #save again
        # save_onnx_model(onnx_model, output_model_path, "llama3bcons+rew.onnx.data")

        #load with no external data
        # onnx_model = onnx.load(output_model_path, load_external_data=False)

        onnx_model = onnx.inliner.inline_local_functions(onnx_model)

        testing_model = ir.serde.deserialize_model(onnx_model)

        print("fuse_add_+_layernorm...")
        fuse_add_and_layernorm(testing_model.graph)

        print("transpose_initializer_subgraphs...")
        transpose_initializer_subgraphs(testing_model.graph)
        # update_rotary_embedding(testing_model.graph, 8192)



        print("pack_qkv_weights...")
        pack_qkv_weights_ir(testing_model.graph)



        onnx_model = ir.serde.serialize_model(testing_model)

        save_onnx_model(onnx_model, output_model_path, "llama3bcons+rew+in.onnx.data")
    except Exception as e:
        print(f"An error occurred: {e}")



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


from onnxscript.ir import Node, AttrFloat32, Value, Graph
from onnxscript.ir import convenience as ir_convenience

# def pack_qkv_weights(graph: ir.Graph):
#     # Create a dictionary to map node names to node objects for O(1) access
#     node_map = {node.name: node for node in graph._nodes}
#     for node in list(graph._nodes):  # Use list() to avoid modifying during iteration
#         if node.op_type == 'SkipSimplifiedLayerNormalization':
#             uses = tuple(node.outputs[0].uses())
#             if len(uses) < 3:
#                 continue
#             # Find the three MatMul nodes following SkipSimplifiedLayerNormalization
#             q_matmul_name = tuple(node.outputs[0].uses())[0][0].name
#             k_matmul_name = tuple(node.outputs[0].uses())[1][0].name
#             v_matmul_name = tuple(node.outputs[0].uses())[2][0].name

#             q_matmul = node_map[q_matmul_name]
#             k_matmul = node_map[k_matmul_name]
#             v_matmul = node_map[v_matmul_name]

#             if q_matmul.op_type != 'MatMul' or k_matmul.op_type != 'MatMul' or v_matmul.op_type != 'MatMul':
#                 continue

#             # Get the weight initializers
#             q_weight = q_matmul.inputs[1]
#             k_weight = k_matmul.inputs[1]
#             v_weight = v_matmul.inputs[1]

#             # Load the initializers as numpy arrays
#             q_array = q_weight.const_value.numpy()
#             k_array = k_weight.const_value.numpy()
#             v_array = v_weight.const_value.numpy()

#             # Ensure the shapes are compatible for concatenation
#             if q_array.shape[0] == k_array.shape[0] == v_array.shape[0]:
#                 common_dim = q_array.shape[0]
#                 q_array = q_array.reshape(common_dim, -1)
#                 k_array = k_array.reshape(common_dim, -1)
#                 v_array = v_array.reshape(common_dim, -1)
#                 packed_weights = np.concatenate([q_array, k_array, v_array], axis=1)
#             else:
#                 print("Incompatible shapes for concatenation")
#                 continue

#             # Generate a unique name for the packed weights
#             unique_layer_id = node.name.split('_')[-1]
#             packed_weight_name = f"model.layers.{unique_layer_id}.attn.qkv_proj.MatMul.weight"

#             # Create a new initializer for the packed weights
#             packed_initializer = ir.tensor(packed_weights)
#             packed_weight = ir.Value(name=packed_weight_name, const_value=packed_initializer)

#             # Create a new MatMul node
#             new_matmul = ir.Node(
#                 op_type='MatMul',
#                 domain='com.microsoft',
#                 name=f"/model/layers.{unique_layer_id}/attn/qkv_proj/MatMul",
#                 inputs=[node.outputs[0], packed_weight],
#                 outputs=[ir.Value(name=f"/model/layers.{unique_layer_id}/attn/qkv_proj/MatMul/output_0")],
#             )

#             # Insert the new MatMul node and remove the old ones
#             graph.insert_after(v_matmul, new_matmul)
#             graph.remove([q_matmul, k_matmul, v_matmul])

#             # Find the GroupQueryAttention node
#             gqa_node = next((n for n in graph._nodes if n.op_type == 'GroupQueryAttention' and q_matmul.outputs[0] in n.inputs), None)

#             if gqa_node:
#                 # Update the GroupQueryAttention node
#                 gqa_node.replace_input_with(gqa_node.inputs.index(q_matmul.outputs[0]), new_matmul.outputs[0])
#                 gqa_node.replace_input_with(gqa_node.inputs.index(k_matmul.outputs[0]), None)
#                 gqa_node.replace_input_with(gqa_node.inputs.index(v_matmul.outputs[0]), None)

#             # Update the uses of the outputs
#             ir_convenience.replace_all_uses_with(q_matmul.outputs[0], new_matmul.outputs[0])
#             ir_convenience.replace_all_uses_with(k_matmul.outputs[0], new_matmul.outputs[0])
#             ir_convenience.replace_all_uses_with(v_matmul.outputs[0], new_matmul.outputs[0])

def pack_qkv_weights_ir(graph: Graph):
    nodes_to_remove = []
    for node in list(graph._nodes):
        if node.op_type not in ['SimplifiedLayerNormalization', 'SkipSimplifiedLayerNormalization']:
            continue

        # Find the three MatMul nodes following the normalization
        matmul_nodes = []
        for use in node.outputs[0].uses():
            if use[0].op_type == 'MatMul':
                matmul_nodes.append(use[0])
                if len(matmul_nodes) == 3:
                    break

        if len(matmul_nodes) != 3:
            continue

        q_matmul, k_matmul, v_matmul = matmul_nodes

        # Find the GroupQueryAttention node
        gqa_node = next((n for n in graph._nodes if n.op_type == 'GroupQueryAttention' and
                         any(output in n.inputs for output in [q_matmul.outputs[0], k_matmul.outputs[0], v_matmul.outputs[0]])), None)

        if not gqa_node:
            continue

        # Get the weight initializers
        q_weight = graph.initializers[q_matmul.inputs[1].name].const_value.numpy()
        k_weight = graph.initializers[k_matmul.inputs[1].name].const_value.numpy()
        v_weight = graph.initializers[v_matmul.inputs[1].name].const_value.numpy()

        # Pack the weights
        packed_weights = np.concatenate([q_weight, k_weight, v_weight], axis=1)

        # Create a new initializer for the packed weights
        layer_name = node.name.split('.layers.')[1].split('.')[0] if 'layers' in node.name else "unknown"
        packed_initializer = ir.tensor(packed_weights)
        packed_weight = ir.Value(name=f"model.layers.{layer_name}.attn.qkv_proj.MatMul.weight", const_value=packed_initializer)

        # Add the new initializer to the graph
        graph.initializers[packed_weight.name] = packed_weight

        # Create a new MatMul node
        new_matmul = Node(
            domain='com.microsoft',
            op_type='MatMul',
            inputs=[node.outputs[0], packed_weight],
            outputs=[Value(name=f"/model/layers.{layer_name}/attn/qkv_proj/MatMul/output_0")],
            name=f"/model/layers.{layer_name}/attn/qkv_proj/MatMul"
        )

        # Insert the new MatMul node
        graph.insert_after(node, new_matmul)

        # Prepare new inputs for the GQA node
        new_gqa_inputs = list(gqa_node.inputs)
        new_gqa_inputs[0] = new_matmul.outputs[0]  # query
        new_gqa_inputs[1] = None  # key (removed)
        new_gqa_inputs[2] = None  # value (removed)

        # Create new output Values for the new GQA node
        new_gqa_outputs = [
            Value(name=f"/model/layers.{layer_name}/attn/GroupQueryAttention/output_0"),
            Value(name=f"present.{layer_name}.key"),
            Value(name=f"present.{layer_name}.value")
        ]

        # Create a new GQA node with updated inputs, new outputs, and correct attributes
        new_gqa_node = Node(
            domain='com.microsoft',
            op_type='GroupQueryAttention',
            inputs=new_gqa_inputs,
            outputs=new_gqa_outputs,
            name=gqa_node.name,
            attributes=list(gqa_node.attributes.values())
        )

        # Replace the old GQA node with the new one
        graph.insert_after(gqa_node, new_gqa_node)

        # Update all uses of the old GQA node's outputs to use the new GQA node's outputs
        for old_output, new_output in zip(gqa_node.outputs, new_gqa_node.outputs):
            ir_convenience.replace_all_uses_with(old_output, new_output)

        # Mark nodes for removal
        nodes_to_remove.extend([q_matmul, k_matmul, v_matmul, gqa_node])

        # Update any other uses of the old MatMul outputs
        ir_convenience.replace_all_uses_with(q_matmul.outputs[0], new_matmul.outputs[0])
        ir_convenience.replace_all_uses_with(k_matmul.outputs[0], new_matmul.outputs[0])
        ir_convenience.replace_all_uses_with(v_matmul.outputs[0], new_matmul.outputs[0])

    # Remove all marked nodes at once
    graph.remove(nodes_to_remove)

    # Cleanup any disconnected nodes
    graph.remove([node for node in graph._nodes if not node.inputs and not node.outputs])



    # Cleanup any disconnected nodes
def fuse_add_and_layernorm(graph: Graph):

    for node in graph:
        if node.op_type != 'Add':
            continue

        # Only check nodes that are connected to an Add node
        add_node = node
        add_output = add_node.outputs[0]
        users_is_simplified_layer_norm = [user for user, _ in add_output.uses() if user.op_type == 'SimplifiedLayerNormalization']
        if not users_is_simplified_layer_norm:
            continue
        if len(users_is_simplified_layer_norm) > 1:
            raise ValueError("Add node connected to multiple SimplifiedLayerNormalization nodes")

        skip_layernorm_add_output = Value(name=add_node.outputs[0].name)
        nodes_to_remove = []
        for user, i in tuple(add_output.uses()):
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
                        skip_layernorm_add_output
                    ],
                    attributes=[epsilon_attr]
                )

                graph.insert_after(add_node, skip_layernorm_node)
                nodes_to_remove.append(add_node)
                nodes_to_remove.append(layernorm_node)

                # Collect reconnections
                for output in layernorm_node.outputs:
                    ir_convenience.replace_all_uses_with(output, skip_layernorm_node.outputs[0])
            else:
                # Reconnect the Add output to the user
                user.replace_input_with(i, skip_layernorm_add_output)

        # Ensure nodes to be removed are no longer used by other nodes
        graph.remove(nodes_to_remove, safe=True)





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


# def rotary_embedding_caches(inv_freq: np.ndarray, max_seq_len: int):


#     t = np.arange(max_seq_len, dtype=np.int64).astype(inv_freq.dtype)

#     freqs = np.outer(t, inv_freq)
#     emb = np.concatenate((freqs, freqs), dim=-1)
#     cos_cache, sin_cache = emb.cos(), emb.sin()
#     # Reshape cos/sin cache from (M, H) to (M, H/2)
#     head_size = cos_cache.shape[1]
#     cos_cache = cos_cache[:, : (head_size // 2)]
#     sin_cache = sin_cache[:, : (head_size // 2)]
#     return cos_cache, sin_cache

def rotary_embedding_caches(inv_freq: np.ndarray, max_seq_len: int):
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos_cache = np.cos(emb).astype(np.float16)
    sin_cache = np.sin(emb).astype(np.float16)
    return cos_cache, sin_cache

MAX_SEQ_LEN = 8192
HEAD_DIM = 64


def update_rotary_embedding(graph: ir.Graph, max_seq_len: int):
    gqa_nodes = [node for node in graph if node.op_type == 'GroupQueryAttention']
    print(f"Found {len(gqa_nodes)} GroupQueryAttention nodes")

    inv_freq = 1.0 / (10000 ** (np.arange(0, HEAD_DIM, 2).astype(np.float32) / HEAD_DIM))

    for i, gqa_node in enumerate(gqa_nodes):
        print(f"Processing GroupQueryAttention node {i+1}")

        new_cos_cache, new_sin_cache = rotary_embedding_caches(inv_freq, max_seq_len=MAX_SEQ_LEN)

        # Create new initializers
        cos_cache_name = f"cos_cache_{i}"
        sin_cache_name = f"sin_cache_{i}"

        cos_cache_initializer = ir.tensor(new_cos_cache)
        sin_cache_initializer = ir.tensor(new_sin_cache)

        cos_cache_value = ir.Value(name=cos_cache_name, const_value=cos_cache_initializer) # because the is no const value set for the caches
        sin_cache_value = ir.Value(name=sin_cache_name, const_value=sin_cache_initializer)

        # Add initializers to the graph
        graph.initializers[cos_cache_name] = cos_cache_value
        graph.initializers[sin_cache_name] = sin_cache_value

        # Update GQA node inputs
        # Assuming cos_cache is the 8th input and sin_cache is the 9th input
        gqa_node.replace_input_with(7, cos_cache_value)
        gqa_node.replace_input_with(8, sin_cache_value)

        print(f"Successfully updated rotary embedding for GQA node {i+1}")
        print(f"New cos_cache shape: {new_cos_cache.shape}, New sin_cache shape: {new_sin_cache.shape}")




input_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/rank_0_Meta-Llama-3-8B_decoder_with_past_model_fp32.onnx"
output_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/llama3bcons+rew+in.onnx"

export_to_onnx(input_model_path, output_model_path)
# python /home/t-assumange/onnxscript/initializer.py