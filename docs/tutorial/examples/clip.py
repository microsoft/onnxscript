from onnxscript import opset15 as op
from onnxscript import script


@script()
def Clip(input, min, max):
    # The inputs min and max are optional inputs
    result = input
    # OptionalHasElement is used to check if an optional input is not None
    if op.OptionalHasElement(min):
        result = op.Where(result < min, min, result)
    if op.OptionalHasElement(max):
        result = op.Where(result > max, max, result)
    return result
