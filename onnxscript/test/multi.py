
from onnxscript.types import FLOAT

def multi (A : FLOAT["N"]) -> FLOAT["N"] :
    x, y = onnx.Foo (A)
    x, y = onnx.Bar (A)
    return x + y