from onnxscript.ir.passes.common import graph_extration
from onnxscript import ir

model = ir.load("d2s_refiner_compliant_v2.all.quant.opt.linux.onnx")
input_names = [
    "Transpose_out0",
    "input_3:0_scale",
    "StatefulPartitionedCall/model_2/tf.nn.space_to_depth/SpaceToDepth:0_zero_point",
    "StatefulPartitionedCall/model_2/conv2d_9/Conv2D/ReadVariableOp:0_quantized",
    "StatefulPartitionedCall/model_2/conv2d_9/Conv2D/ReadVariableOp:0_scale",
    "StatefulPartitionedCall/model_2/conv2d_9/Conv2D/ReadVariableOp:0_zero_point",
    "StatefulPartitionedCall/model_2/conv2d_9/BiasAdd:0_scale",
    "StatefulPartitionedCall/model_2/conv2d_9/BiasAdd:0_zero_point",
    "StatefulPartitionedCall/model_2/conv2d_9/BiasAdd/ReadVariableOp:0_quantized"
]
output_names = ["StatefulPartitionedCall/model_2/conv2d_9/BiasAdd"]
result = graph_extration.ExtractGraphPass(input_names, output_names)(model)
print(result.model)
ir.save(
    result.model,
    "extracted_model.onnx",
)
