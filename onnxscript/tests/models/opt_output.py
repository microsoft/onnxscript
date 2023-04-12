from onnxscript import script
from onnxscript.onnx_opset import opset15 as op

# from onnxscript.onnx_types import FLOAT

# Issues relating to optional output

# Use Case 1: In this scenario, outputs are always well-defined, and
# optional outputs serve only as an optimization hint to the kernel whether
# some output is required. In the function-implementation, we can simply
# choose to compute the output and rely on function-inlining and subsequent
# optimizations to eliminate unnecessary computation. No special support
# is required in OnnxScript.
# Example: LayerNormalization


# Here is a toy version of a LayerNormalization-like example:
# Here, the second output is optional. But this impacts only the caller's
# code, and not the function-definition. In the function definition, we always
# return it.
@script()
def MeanDiff(x):
    mean = op.ReduceMean(x)
    diff = x - mean
    # Caller context may have second output "missing".
    # It is the inliner's responsibility to handle this correctly.
    return (diff, mean)


# A call to a function with an optional output:
@script()
def MeanDiffCaller(x):
    diff, _ = MeanDiff(x)
    return diff * diff


# Use Case 2: In this scenario, the inputs/attributes determine which outputs
# are computed. In particular, the op may return different numbers of outputs
# in different situations. An example is the BatchNormalization op. This
# introduces some challenges (in situations where use-case 1's approach is
# not feasible).

# BatchNorm can still be defined as a function, as in use-case 1, by computing
# the extra outputs even in the case where it is unused. This can lead to inefficiency
# due to redundant computation, but that can be eliminated by optimization subsequent
# to inlining. However, the optimization may not be feasible if the entire
# computation graph is unavailable: eg., if we extract subgraphs (from a Pytorch
# program) and execute it subgraph at-a-time.

# However, the question here is whether it is useful to support examples such as
# the one below:


@script()
def ConditionalOptOutput(x, y, flag: bool):
    if flag:
        z1 = x + y
        # Challenge: Should we support this kind of usage?
        # If so, how do we handle this?
        # Specifically, the ONNX representation does not enable us to capture
        # conditionals that return different number of outputs in different branches.
        # We would need some way to represent a dummy "None" value.
        z2 = None
    else:
        z1 = x + y
        z2 = x / y
    return z1, z2
