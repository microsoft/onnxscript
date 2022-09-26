"""
Model Local Functions
=====================

A model in ONNX may contain model-local functions. When converting an *onnxscript*
function to a ModelProto, the default behavior is to include function-definitions
for all transitively called function-ops as model-local functions in the generated
model (for which an *onnxscript* function definition has been seen). Callers can
override this behavior by explicitly providing the list of FunctionProtos to be
included in the generated model.

"""

#%%
# First, let us define an ONNXScript function that calls other ONNXScript functions.

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT
from onnxscript.utils import proto2text
from onnxscript.values import Opset

# A dummy opset used for model-local functions
local = Opset("local", 1)


@script(local)
def diff_square(x, y):
    diff = x - y
    return diff * diff


@script(local)
def sum(z):
    return op.ReduceSum(z, keepdims=1)


@script()
def l2norm(x: FLOAT["N"], y: FLOAT["N"]) -> FLOAT[1]:  # noqa: F821
    return op.sqrt(sum(diff_square(x, y)))


#%%
# Let's see what the generated model looks like by default:

model = l2norm.to_model_proto()
print(proto2text(model))

#%%
# Let's now explicitly specify which functions to include.
# First, generate a model with no model-local functions:

model = l2norm.to_model_proto(functions=[])
print(proto2text(model))

#%%
# Now, generate a model with one model-local function:

model = l2norm.to_model_proto(functions=[sum])
print(proto2text(model))
