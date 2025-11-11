# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""A utility and an example showing how onnxscript functions can be used to define function expansions
and be used with the inliner to replace calls to the custom function with an expanded subgraph.
This is useful to perform certain classes of graph surgery easily.
"""

import onnx

import onnxscript
import onnxscript.utils.replace as replace

script = onnxscript.script
FLOAT = onnxscript.FLOAT
op = onnxscript.values.opset22
local = onnxscript.values.Opset("local", 1)


# Example Model: Actual models can come from ModelBuilder or Exporter or any other source.
# Models can contain calls to custom operations (from a custom domain like 'local' here or
# even "com.microsoft" etc.)
@script()
def model_script(X: FLOAT["N"], Y: FLOAT["N"]) -> FLOAT["N"]:
    DoubleX = op.Add(X, X)
    YSquare = op.Mul(Y, Y)
    # Example call to a custom operation
    Temp1 = local.CustomOp1(DoubleX, YSquare)
    # Another call to a custom operation with an attribute
    Temp2 = local.CustomOp2(Temp1, alp=0.9)
    return Temp2


# Define expansions for custom operations as onnxscript functions
@script(opset=local)
def CustomOp1(X: FLOAT["N"], Y: FLOAT["N"]) -> FLOAT["N"]:
    Temp1 = op.Sub(X, Y)
    return op.Div(Temp1, X)


@script(opset=local)
def CustomOp2(X: FLOAT["N"], alp: float) -> FLOAT["N"]:
    Temp2 = op.Elu(X, alpha=alp)
    return op.Mul(Temp2, Temp2)


# Now, we can replace the custom operations in the model with their expansions:

functions = [CustomOp1.to_function_proto(), CustomOp2.to_function_proto()]

model = model_script.to_model_proto()

print("Original Model with custom operations:")
print(onnx.printer.to_text(model))


updated_model = replace.replace_functions(model, functions)

print("\nUpdated Model after replacing custom operations with their expansions:")
print(onnx.printer.to_text(updated_model))
