
from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset15 as op15
from onnxscript.test.models.function_opset_versioning.Selu1 import Selu

# onnx-script function, version 1, using onnx opset15
@script(Opset('this', 1))
def Elu(X, alpha: float = 1.0):
    gamma = op15.Constant(value_float=1.)
    return Selu(X, alpha, gamma)
