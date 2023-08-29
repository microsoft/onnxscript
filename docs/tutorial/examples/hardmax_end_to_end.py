import onnx

# We use ONNX opset 15 to define the function below.
from onnxscript import FLOAT
from onnxscript import opset15 as op
from onnxscript import script


# We use the script decorator to indicate that
# this is meant to be translated to ONNX.
@script()
def onnx_hardmax(X, axis: int):
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


# We use the script decorator to indicate that
# this is meant to be translated to ONNX.
@script()
def sample_model(X: FLOAT[64, 128], Wt: FLOAT[128, 10], Bias: FLOAT[10]) -> FLOAT[64, 10]:
    matmul = op.MatMul(X, Wt) + Bias
    return onnx_hardmax(matmul, axis=1)


# onnx_model is an in-memory ModelProto
onnx_model = sample_model.to_model_proto()

# Save the ONNX model at a given path
onnx.save(onnx_model, "sample_model.onnx")

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print(f"The model is invalid: {e}")
else:
    print("The model is valid!")
