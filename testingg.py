
    
import onnx
import onnx.inliner
import onnxscript.optimizer
import onnxscript.rewriter
from onnxscript.rewriter import onnxruntime as ort_rewriter

from onnxscript.rewriter.onnxruntime.transformers import (
    biassplitgelu,
    fastgelu,
    layernorm,
    multihead_attention,
)
    
def export_to_onnx(input_model_path: str, output_model_path: str) -> onnx.ModelProto:
    """
    Export a model to ONNX.
    Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite, 
    and onnx.inliner.inline_local_functions.
    """

    # Load the ONNX model
    try:
        onnx_model = onnx.load(input_model_path)
    except FileNotFoundError:
        print(f"Error: The file {input_model_path} was not found.")
        return
    # # Apply the onnx optimizer
    # onnx_model = onnxscript.optimizer.optimize(onnx_model)

    #apply the onnx rewriter
    # i, onnx_model = multihead_attention.MHALlama2RewriteRule().apply_to_model(onnx_model)
     # Apply the ONNX rewriter
    rewriter_rule = multihead_attention.MHALlama2RewriteRule()

    # Iterate over the functions directly since `functions` is a list-like container
    new_functions = []
    for function in onnx_model.functions:
        # print(f"Function: {function}")
        rewrite_or_none = rewriter_rule.try_rewrite_function(function)
        if rewrite_or_none is not None:
            new_functions.append(rewrite_or_none[1])
        else:
            new_functions.append(function)

    # Replace functions in the model
    del onnx_model.functions[:]
    onnx_model.functions.extend(new_functions)

    print("ONNX rewriter applied.")
    # # apply the onnxruntime rewriter
    # onnx_model = ort_rewriter.rewrite(onnx_model)

    # # apply the onnx inliner
    # onnx_model = onnx.inliner.inline_local_functions(onnx_model)
    # print("printing the byte size after opt" ,onnx_model.ByteSize())
    # Save the ONNX model
    save_onnx_model(onnx_model, output_model_path, "rank_0_Llama-2-7b-hf_decoder_with_past_model_fp32.onnx.data")
    

def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
    """
    Save the ONNX model to disk, supporting external data.
    """
    onnx.save(
       onnx_model,
       output_path,
       save_as_external_data=True,
       all_tensors_to_one_file=True,
       location=data_path,
       size_threshold=0,
       convert_attribute=False,

    )
   
    return onnx_model

# usage
input_model_path = "/home/t-assumange/llama2-7b/rank_0_Llama-2-7b-hf_decoder_with_past_model_fp32.onnx"
output_model_path = "/home/t-assumange/llama2-7b/testing2.onnx"
export_to_onnx(input_model_path, output_model_path)
