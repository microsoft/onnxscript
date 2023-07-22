from onnxscript import opset15 as op
from onnxscript import script
import onnx

true = onnx.helper.make_tensor("true", onnx.TensorProto.BOOL, [], [1])

@script()
def Dropout(data, ratio, training_mode, seed: float):
    if training_mode:
        rand = op.RandomUniformLike(data, dtype=1, seed=seed)
        mask = rand >= ratio
        output = op.Where(mask, data, 0) / (1.0 - ratio)
    else:
        mask = op.ConstantOfShape(op.Shape(data), value=true)
        output = data
    return (output, mask)
