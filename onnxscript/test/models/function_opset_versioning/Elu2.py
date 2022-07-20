
from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset16 as op16
from onnxscript.test.models.function_opset_versioning.Selu2 import Selu

# onnx-script function, version 1, using onnx opset15
@script(Opset('this', 1))
def Elu(X: FLOAT[None], beta: FLOAT[1]=1.0) -> FLOAT[None]:
    alpha = op16.Constant(value_float=1.)
    return Selu(X, alpha, beta)
