import onnx
import onnx.inliner
import torch
 
import onnxscript.optimizer
import onnxscript.rewriter
 
def export_to_onnx() -> onnx.ModelProto:
# def export_to_onnx(model: Any, *args: Sequence[Any], optimize: bool = True) -> onnx.ModelProto:
 
    """
    Export a model to ONNX.
    If optimize is True, it calls *onnxscript.optimizer.optimize*,
    *onnxscript.rewriter.rewriter*, *onnx.inliner.inline_local_functions*.
    """
    input_model_path = "/home/t-assumange/onnxruntime/onnxruntime/python/tools/transformers/llama3-8Bdynamo/rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx"
    output_model_path = "/home/t-assumange/"
    onnx_model = onnx.load(input_model_path)
    model_proto = onnxscript.rewriter.rewrite(onnx_model)
    model_proto = onnx.inliner.inline_local_functions(model_proto)

    save_onnx_model(model_proto, output_model_path, "/home/t-assumange/onnxruntime/onnxruntime/python/tools/transformers/llama3-8Bdynamo/rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx.data")

    
    
def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="/home/t-assumange/onnxruntime/onnxruntime/python/tools/transformers/llama3-8Bdynamo/rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx.data, # add the .onnx.data file path",
        size_threshold=0,
        convert_attribute=False,
    )
    
    return onnx_model
    