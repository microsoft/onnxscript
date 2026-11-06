# Opsets and Types

(l-api-opsets)=

## Opsets

Standard-domain opsets 1 through 28 are available from both `onnxscript` and
`onnxscript.onnx_opset`, for example `from onnxscript import opset28`.
Use an ONNX installation that provides the schemas for the operators you call.
Unreleased opset 29 is not exposed.

```{eval-rst}
.. automodule:: onnxscript.onnx_opset

    .. autodata:: opset28
        :annotation: All operators defined in opset 28 for the main domain.

    .. autodata:: opset27
        :annotation: All operators defined in opset 27 for the main domain.

    .. autodata:: opset26
        :annotation: All operators defined in opset 26 for the main domain.

    .. autodata:: opset25
        :annotation: All operators defined in opset 25 for the main domain.

    .. autodata:: opset24
        :annotation: All operators defined in opset 24 for the main domain.

    .. autodata:: opset23
        :annotation: All operators defined in opset 23 for the main domain.

    .. autodata:: opset22
        :annotation: All operators defined in opset 22 for the main domain.

    .. autodata:: opset21
        :annotation: All operators defined in opset 21 for the main domain.

    .. autodata:: opset20
        :annotation: All operators defined in opset 20 for the main domain.

    .. autodata:: opset19
        :annotation: All operators defined in opset 19 for the main domain.

    .. autodata:: opset18
        :annotation: All operators defined in opset 18 for the main domain.

    .. autodata:: opset17
        :annotation: All operators defined in opset 17 for the main domain.

    .. autodata:: opset16
        :annotation: All operators defined in opset 16 for the main domain.

    .. autodata:: opset15
        :annotation: All operators defined in opset 15 for the main domain.
```

## Types

Tensor annotations are also exported directly from `onnxscript`. Low-bit types
include `INT2`, `UINT2`, `INT4`, `UINT4`, `FLOAT4E2M1`, `FLOAT6E2M3`, `FLOAT6E3M2`,
and the float8 formats. Float6 annotations require ONNX IR 1.1 or later.
An operator's ONNX schema determines which dtypes it accepts; exporting a dtype
does not make it valid for every operator.

```{eval-rst}
.. automodule:: onnxscript.onnx_types

    .. autodata:: BFLOAT16

    .. autodata:: BOOL

    .. autodata:: COMPLEX64

    .. autodata:: COMPLEX128

    .. autodata:: DOUBLE

    .. autodata:: FLOAT

    .. autodata:: FLOAT16

    .. autodata:: FLOAT4E2M1

    .. autodata:: FLOAT6E2M3

    .. autodata:: FLOAT6E3M2

    .. autodata:: FLOAT8E4M3FN

    .. autodata:: FLOAT8E4M3FNUZ

    .. autodata:: FLOAT8E5M2

    .. autodata:: FLOAT8E5M2FNUZ

    .. autodata:: FLOAT8E8M0

    .. autodata:: INT2

    .. autodata:: INT4

    .. autodata:: INT8

    .. autodata:: INT16

    .. autodata:: INT32

    .. autodata:: INT64

    .. autodata:: STRING

    .. autodata:: UINT8

    .. autodata:: UINT2

    .. autodata:: UINT4

    .. autodata:: UINT16

    .. autodata:: UINT32

    .. autodata:: UINT64
```
