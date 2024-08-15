
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
from onnxscript.ir import Node, AttrFloat32, Value, Graph

def export_to_onnx(input_model_path: str, output_model_path: str):
    """
    Export a model to ONNX.
    Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite,
    and onnx.inliner.inline_local_functions.
    The order in which you make these calls matters.
    Suggested workflow:
    - Load without external data and apply the rewriters; onnxscript.rewriter.rewrite(), ort_rewriter.rewrite() ...
    - Save the model and load it again with external data set to True, so we can apply the optimizer successfully
    - To apply the inliner, we need to load it with no external data after we save it with the optimizer.
    - Finally apply the rest of the initializer calls
    """
    try:
        # Load the ONNX model
        onnx_model = onnx.load(input_model_path, load_external_data=False)

        # Apply the onnx rewriter
        onnx_model = onnxscript.rewriter.rewrite(onnx_model)

        # Apply the onnxruntime rewriter
        onnx_model = ort_rewriter.rewrite(onnx_model)

        #save the model
        save_onnx_model(onnx_model, output_model_path, "llama3bcons+rew.onnx.data")

        # #load with external data
        onnx_model = onnx.load(output_model_path, load_external_data=True)


        # # Apply the onnx optimizer
        onnx_model = onnxscript.optimizer.optimize(onnx_model)

        # #save again
        save_onnx_model(onnx_model, output_model_path, "llama3bcons+rew.onnx.data")

        #load with no external data
        onnx_model = onnx.load(output_model_path, load_external_data=False)

        # Apply the inliner
        onnx_model = onnx.inliner.inline_local_functions(onnx_model)

        # serialize

        testing_model = ir.serde.deserialize_model(onnx_model)

        print("fuse_add_+_layernorm...")
        fuse_add_and_layernorm(testing_model.graph)

        print("transpose_initializer_subgraphs...")
        transpose_initializer_subgraphs(testing_model.graph)
        update_rotary_embedding(testing_model.graph, 8192)

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
        gqa_node.replace_input_with(7, cos_cache_value)
        gqa_node.replace_input_with(8, sin_cache_value)

        print(f"Successfully updated rotary embedding for GQA node {i+1}")
        print(f"New cos_cache shape: {new_cos_cache.shape}, New sin_cache shape: {new_sin_cache.shape}")


# Supports both LLaMA2 and 3

input_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/rank_0_Meta-Llama-3-8B_decoder_with_past_model_fp32.onnx"
output_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/llama3bcons+rew+in.onnx"
export_to_onnx(input_model_path, output_model_path)