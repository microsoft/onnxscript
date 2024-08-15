
import onnx
import onnx.inliner
import onnxscript.optimizer
import onnxscript.rewriter
from onnxscript.rewriter import onnxruntime as ort_rewriter
from onnxscript.ir import convenience as ir_convenience

def export_to_onnx(input_model_path: str, output_model_path: str):
    """
    Export a model to ONNX. This is for testing your rewrite rules without accessing the initializers
    Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite,
    and onnx.inliner.inline_local_functions.
    The order in which you make these calls matters depending on how the rewrite rules are implemented
    Suggested workflow:
    - Load without external data and apply the rewriters; onnxscript.rewriter.rewrite(), ort_rewriter.rewrite() ...
    - Save the model and load it again with external data set to True, so we can apply the optimizer successfully
    - To apply the inliner, we need to load it with no external data after we save it with the optimizer.
    """
     # Load the ONNX model
    onnx_model = onnx.load(input_model_path, load_external_data=False)

    # Apply the onnx rewriter
    onnx_model = onnxscript.rewriter.rewrite(onnx_model)

    # Apply the onnxruntime rewriter
    onnx_model = ort_rewriter.rewrite(onnx_model)

    #save the model
    save_onnx_model(onnx_model, output_model_path, "llamatest.onnx.data")

    # #load with external data
    onnx_model = onnx.load(output_model_path, load_external_data=True)

    # # Apply the onnx optimizer
    onnx_model = onnxscript.optimizer.optimize(onnx_model)

    # #save again
    save_onnx_model(onnx_model, output_model_path, "llamatest.onnx.data")

    #load with no external data
    onnx_model = onnx.load(output_model_path, load_external_data=False)

    # Apply the inliner
    onnx_model = onnx.inliner.inline_local_functions(onnx_model)


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

# uncomment for llama2, make sure youre in the right directory.
# input_model_path = "/home/t-assumange/llama2-7b_Dynamo_transformers4.41/rank_0_Llama-2-7b-hf_decoder_with_past_model_fp32.onnx"
# output_model_path = "/home/t-assumange/llama2-7b_Dynamo_transformers4.41/llama2.onnx"

# uncomment for llama3, make sure youre in the right directory.
input_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/rank_0_Meta-Llama-3-8B_decoder_with_past_model_fp32.onnx"
output_model_path = "/home/t-assumange/llama3-8-Bdynamo_transformers4.41/llamatest.onnx"

export_to_onnx(input_model_path, output_model_path)