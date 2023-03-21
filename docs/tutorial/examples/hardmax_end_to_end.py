import onnx
from onnxscript import script

# We use ONNX opset 15 to define the function below.
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT

# We use the script decorator to indicate that
# this is meant to be translated to ONNX.
@script()
def onnx_hardmax(X: FLOAT[...], axis: int = 0) -> FLOAT[...]:
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""

    # The type annotation on X indicates that it is a float tensor of
    # unknown rank. The type annotation on axis indicates that it will
    # be treated as an int attribute in ONNX.
    #
    # Invoke ONNX opset 15 op ArgMax.
    # Use unnamed arguments for ONNX input parameters, and named
    # arguments for ONNX attribute parameters.
    argmax = op.ArgMax(X, axis=axis, keepdims=False)
    xshape = op.Shape(X, start=axis)
    # use the Constant operator to create constant tensors
    zero = op.Constant(value_ints=[0])
    depth = op.GatherElements(xshape, zero)
    empty_shape = op.Constant(value_ints=[0])
    depth = op.Reshape(depth, empty_shape)
    values = op.Constant(value_ints=[0, 1])
    cast_values = op.CastLike(values, X)
    return op.OneHot(argmax, depth, cast_values, axis=axis)

# onnx_model is an in-memory ModelProto
onnx_model = onnx_hardmax.to_model_proto()

# Save the ONNX model at a given path
onnx.save(onnx_model, "hardmax.onnx")
