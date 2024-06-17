
    
import onnx
import onnx.inliner
import onnxscript.optimizer
import onnxscript.rewriter
from onnxscript.rewriter import onnxruntime as ort_rewriter
    
def export_to_onnx(input_model_path: str, output_model_path: str) -> onnx.ModelProto:
    """
    Export a model to ONNX.
    Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite, 
    and onnx.inliner.inline_local_functions.
    """

    # Load the ONNX model
    onnx_model = onnx.load(input_model_path, load_external_data=False)
    print("printing the byte size before opt" ,onnx_model.ByteSize())
    # Apply the onnx optimizer
    onnx_model = onnxscript.optimizer.optimize(onnx_model)

    #apply the onnx rewriter
    onnx_model = onnxscript.rewriter.rewrite(onnx_model)

    # apply the onnxruntime rewriter
    onnx_model = ort_rewriter.rewrite(onnx_model)

    # apply the onnx inliner
    onnx_model = onnx.inliner.inline_local_functions(onnx_model)
    print("printing the byte size after opt" ,onnx_model.ByteSize())
    # Save the ONNX model
    save_onnx_model(onnx_model, output_model_path, "llama2-7b/rank_0_Llama-2-7b-hf_decoder_with_past_model_fp32.onnx.data")
    

def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
    """
    Save the ONNX model to disk, supporting external data.
    """
    onnx.save(
        onnx_model,
        output_path,
        # save_as_external_data=False,
        # all_tensors_to_one_file=True,
        # location=data_path,  # Must be a relative path
        # size_threshold=0,
        # convert_attribute=False,
    )
    # print(f"Model saved with external data to {data_path}")
    return onnx_model

# Example usage
input_model_path = "/home/t-assumange/llama2-7b/rank_0_Llama-2-7b-hf_decoder_with_past_model_fp32.onnx"
output_model_path = "/home/t-assumange/llama2-7b/optimize_model_llama222.onnx"
export_to_onnx(input_model_path, output_model_path)
