from onnxscript import script
from onnxscript.onnx import opset15 as op
# from onnxscript.onnx_types import FLOAT

# Issues relating to optional output

# Use Case 1: In this scenario, outputs are always well-defined, and
# optional outputs serve only as an optimization hint to the kernel whether
# some output is required. In the function-implementation, we can simply
# choose to compute the output and rely on function-inlining and subsequent
# optimizations to eliminate unnecessary computation. No special support
# is required in OnnxScript.
# Example: LayerNormalization

# Toy version of LayerNormalization-like example:
# Second output is optional. But in the function-definition we always return it.
@script()
def MeanDiff (x) :
    mean = op.ReduceMean(x)
    diff = x - mean
    # Caller context may have second output "missing".
    # It is the inliner's responsibility to handle this correctly.
    return (diff, mean)

# Use Case 2: In this scenario, the inputs/attributes determine which outputs
# are computed. In particular, the op may return different numbers of outputs
# in different situations. An example is the BatchNormalization op. This
# introduces some challenges (in situations where use-case 1's approach is
# not feasible).
