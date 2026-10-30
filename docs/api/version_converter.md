# onnxscript.version_converter

The native converter supports **upgrading standard-domain opsets 18 through 28**.
Opset 29 is not supported. Both `onnx_ir.Model` and `onnx.ModelProto` inputs are
updated in place, including their opset imports.

Downgrades and conversions outside the native range require `fallback=True`,
which delegates to ONNX's converter when available. The installed ONNX package
must provide the schemas needed to check or execute the converted model.

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
