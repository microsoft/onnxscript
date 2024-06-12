# import onnx
# import onnx.inliner
# import torch
 
# import onnxscript.optimizer
# import onnxscript.rewriter
 
# def export_to_onnx() -> onnx.ModelProto:
# # def export_to_onnx(model: Any, *args: Sequence[Any], optimize: bool = True) -> onnx.ModelProto:
 
#     """
#     Export a model to ONNX.
#     If optimize is True, it calls *onnxscript.optimizer.optimize*,
#     *onnxscript.rewriter.rewriter*, *onnx.inliner.inline_local_functions*.
#     """
#     input_model_path = "/home/t-assumange/onnxruntime/onnxruntime/python/tools/transformers/llama3-8Bdynamo/rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx"
#     output_model_path = "/home/t-assumange/onnxruntime/"
#     onnx_model = onnx.load(input_model_path)
#     model_proto = onnxscript.rewriter.rewrite(onnx_model)
#     model_proto = onnx.inliner.inline_local_functions(model_proto)


#     save_onnx_model(model_proto, output_model_path, "onnxruntime/onnxruntime/python/tools/transformers/llama3-8Bdynamo/rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx.data")

    
# def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
#     onnx.save(
#         onnx_model,
#         output_path,
#         save_as_external_data=True,
#         all_tensors_to_one_file=True,
#         location="onnxruntime/onnxruntime/python/tools/transformers/llama3-8Bdynamo/rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx.data",
#         size_threshold=0,
#         convert_attribute=False,
#     )
    
#     return onnx_model
# export_to_onnx()
# save_onnx_model(onnx_model=onnx_model, output_path="/home/t-assumange/onnxruntime/", data_path="/home/t-assumange/onnxruntime/onnxruntime/python/tools/transformers/llama3-8Bdynamo/rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx.data")

    
import onnx
import onnx.inliner
import onnxscript.optimizer
import onnxscript.rewriter

def export_to_onnx(input_model_path: str, output_model_path: str) -> onnx.ModelProto:
    """
    Export a model to ONNX.
    Applies onnxscript.optimizer.optimize, onnxscript.rewriter.rewrite, 
    and onnx.inliner.inline_local_functions.
    """
    # Load the ONNX model
    onnx_model = onnx.load(input_model_path)
    print(f"Loaded model from {input_model_path}")

    # Print initial model graph
    print("Initial model graph:")
    print(onnx.helper.printable_graph(onnx_model.graph))

    # Rewrite the model
    model_proto = onnxscript.rewriter.rewrite(onnx_model)
    print("Rewrite step completed.")
    print("Model graph after rewrite:")
    print(onnx.helper.printable_graph(model_proto.graph))

    if model_proto.graph.node == []:
        print("Graph is empty after rewrite.")
        return model_proto

    # Inline local functions
    model_proto = onnx.inliner.inline_local_functions(model_proto)
    print("Inlining step completed.")
    print("Model graph after inlining:")
    print(onnx.helper.printable_graph(model_proto.graph))

    if model_proto.graph.node == []:
        print("Graph is empty after inlining.")
        return model_proto

    # Optimize the model if the optimizer is available
    if onnxscript.optimizer:
        model_proto = onnxscript.optimizer.optimize(model_proto)
        print("Optimization step completed.")
        print("Model graph after optimization:")
        print(onnx.helper.printable_graph(model_proto.graph))

    if model_proto.graph.node == []:
        print("Graph is empty after optimization.")
        return model_proto
        
    # Save the optimized model to disk
    save_onnx_model(model_proto, output_model_path, "rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx.data")
    print(f"Optimized model saved to {output_model_path}")

    return model_proto

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
    print(f"Model saved with external data to {data_path}")
    return onnx_model

# Example usage
input_model_path = "/home/t-assumange/onnxruntime/onnxruntime/python/tools/transformers/llama3-8Bdynamo/rank_0_Meta-Llama-3-8B_decoder_model_fp32.onnx"
output_model_path = "/home/t-assumange/onnxruntime/optimized_model.onnx"
export_to_onnx(input_model_path, output_model_path)
