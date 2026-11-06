# Values

Import function objects and operators from `onnxscript.values`; the top-level
`onnxscript.OnnxFunction` is the same public class. Evaluator selection is
available from `onnxscript.evaluator`. Do not import compiler implementation
details from `onnxscript._internal`.

## Graph ownership and model export

`function.graph()` returns the **live graph owned by the function**, not a copy.
Do not modify it: changes can affect subsequent exports. Clone it with
`function.graph().clone()` before making independent transformations.
`function.to_model_proto()` exports a separate model without modifying the
function's graph.

Model export is deliberately **permissive**: missing type annotations can produce
an incomplete model that fails the ONNX checker. Export does not run validation
or shape inference automatically. Functions with required attributes cannot be
exported as models.

Use the existing ONNX checker explicitly when a valid model is required:

```python
import onnx
from onnxscript import FLOAT, opset18, script

@script()
def identity(x):
    return opset18.Identity(x)

# Without overrides, this unannotated function exports an incomplete model.
incomplete_model = identity.to_model_proto()
# onnx.checker.check_model(incomplete_model) raises onnx.checker.ValidationError.

model = identity.to_model_proto(
    input_types=[FLOAT["N"]], output_types=[FLOAT["N"]]
)
onnx.checker.check_model(model)  # Validation errors are not suppressed.
```

`input_types` and `output_types`, when supplied, must each contain exactly one
type per model input or output. Attributes are not model inputs. An empty
sequence is valid only when the corresponding model signature has zero values.
These sequences replace existing types, including annotations. In contrast,
`io_types` only fills missing input/output types. Keyword arguments passed to
`to_model_proto` take precedence over those passed to `@script`, including an
explicit `None` that disables a decorator-supplied override.

## Library export status

{func}`onnxscript.export_onnx_lib` is retained for compatibility but is a legacy,
weakly supported helper. It saves local functions in a `ModelProto` with an empty
graph and placeholder opset imports. This is neither an executable model nor a
standardized ONNX library format, and the helper does not validate it. Prefer
`function.to_function_proto()` for an individual ONNX function, or
`function.to_model_proto()` with explicit checker validation for an executable
model.

```{eval-rst}
.. autofunction:: onnxscript.export_onnx_lib
```

## Reference

```{eval-rst}
.. autoclass:: onnxscript.values.Opset
    :members:
```

```{eval-rst}
.. autoclass:: onnxscript.values.Op
    :members:
```

```{eval-rst}
.. autoclass:: onnxscript.values.OnnxFunction
    :members:
```
