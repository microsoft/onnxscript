
import onnx
import onnx.inliner
import onnxscript.optimizer
import onnxscript.rewriter
from onnxscript.rewriter import onnxruntime as ort_rewriter
from onnx import NodeProto, TensorProto, helper, numpy_helper
import numpy as np
from onnxscript import ir
from onnxscript.ir import convenience as ir_convenience




    
def export_to_onnx(input_model_path: str, output_model_path: str) -> onnx.ModelProto:
    """
    Export a model to ONNX.
    Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite, 
    and onnx.inliner.inline_local_functions.
    """
    # Load the ONNX model
    onnx_model = onnx.load(input_model_path, load_external_data=False)
    # # # Apply the onnx optimizer
    # onnx_model = onnxscript.optimizer.optimize(onnx_model) #-- this one kinda messes up for now, causes that symbolicdim error even without kunal's changes
    

    # #apply the onnx rewriter
    onnx_model = onnxscript.rewriter.rewrite(onnx_model)
    

    # apply the onnxruntime rewriter
    onnx_model = ort_rewriter.rewrite(onnx_model) # keep

    # apply the onnx inliner
    # onnx_model = onnx.inliner.inline_local_functions(onnx_model)

    # Save the ONNX model
    # save_onnx_model(onnx_model, output_model_path, "myrules.onnx.data")
    # onnx_model = onnx.load(output_model_path, load_external_data=False)
    
    onnx_model = onnx.inliner.inline_local_functions(onnx_model)
    
    # ir should be done after inlining --> ir optimization function
    testing_model = ir.serde.deserialize_model(onnx_model) 
    
    transpose_initializer_subgraphs(testing_model.graph)
    onnx_model = ir.serde.serialize_model(testing_model)
    
    
    save_onnx_model(onnx_model, output_model_path, "mynewrules.onnx.data")


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
    

input_model_path = "/home/t-assumange/llama2-7b_Dynamo_transformers4.41/rank_0_Llama-2-7b-hf_decoder_with_past_model_fp32.onnx"
output_model_path = "/home/t-assumange/llama2-7b_Dynamo_transformers4.41/mynewrules.onnx"

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
    

export_to_onnx(input_model_path, output_model_path)

