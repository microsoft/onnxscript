
import onnxscript.onnx.opset15 as op

# TODO: Need to verify Reduction definitions below.
# Behavior for integral types is not explicitly described in ONNX spec.
# Need to verify if any issues exist with numeric precision.

def ReduceSumSquare(data, axes: List[int],  keepdims: int):
    # Note: attribute input is promoted to input when calling ReduceSum
    return op.ReduceSum (data * data, axes, keepdims=keepdims)

def ReduceL1(data, axes: List[int],  keepdims: int):
    return op.ReduceSum (op.Abs(data), axes, keepdims=keepdims)

def ReduceL2(data, axes: List[int],  keepdims: int):
    # TODO: ONNX spec is unclear about behavior for integral types!
    sum_square = op.ReduceSum (data*data axes, keepdims=keepdims)
    # TODO: must cast for integral types
    return op.Sqrt(sum_square)

def ReduceLogSum(data, axes: List[int],  keepdims: int):
    return op.Log (op.ReduceSum (data, axes, keepdims=keepdims))

def ReduceLogSumExp(data, axes: List[int],  keepdims: int):
    return op.Log (op.ReduceSum (op.Exp(data), axes, keepdims=keepdims))

def DepthToSpace():
    ...

def SpaceToDepth():
    ...

def Hardmax():
    ...


