# onnxscript.version_converter

The native converter supports **upgrading standard-domain opsets 18 through 28**.
Opset 29 is not supported. Both `onnx_ir.Model` and `onnx.ModelProto` inputs are
updated in place, including their opset imports.

Downgrades and conversions outside the native range require `fallback=True`,
which delegates to ONNX's converter when available. The installed ONNX package
must provide the schemas needed to check or execute the converted model.

Conversion traverses nested graphs (including graph attributes of custom
operators) and respects each local function's own opset imports. The public pass
inlines local-function calls before conversion, resolving referenced attributes.
The native converter also preserves reference attributes when the adapter can
translate them directly.

Legacy GroupNormalization is decomposed into its original arithmetic to preserve
per-group scale and bias, epsilon, and accumulation precision without requiring
static intermediate shapes or dtypes. DFT axis attributes are materialized as
inputs, and GridSample interpolation modes are translated when crossing opset 20.
Other revisions through opset 28 are upward-compatible for existing valid inputs.

Individual IR nodes may record an operator's schema revision rather than the
enclosing scope's imported opset (for example, MatMul-13 in an opset-18 model).
Such revisions below 18 are accepted only when the same operator schema is still
active at opset 18; they do not enable conversion of pre-18 model imports or
operators whose schemas changed before opset 18.

Missing source versions, unsupported native source/target ranges, and invalid
required adapter inputs or attributes raise errors instead of silently marking
an unconverted model with the target opset. An IR model may be partially modified
if conversion raises; do not use that model as a successfully converted result.

```{eval-rst}
.. automodule::onnxscript.version_converter
.. currentmodule:: onnxscript
```

## Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: functiontemplate.rst
    :nosignatures:

    version_converter.convert_version
```

## IR passes

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    version_converter.ConvertVersionPass
```
