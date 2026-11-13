# Generator for onnx_opset

Use this module the generate onnx_opset implementations when new opsets are introduced with new ONNX versions.

## Generate

```sh
python opgen
```

When generating with an ONNX development build that includes unreleased opset 29,
exclude it to keep the public API capped at opset 28:

```sh
python opgen -x 29
```

After generation, update the public exports in `onnxscript/__init__.py` and verify
schema and dtype coverage with `onnxscript/onnx_opset_test.py` and
`onnxscript/onnx_types_test.py`.

Run

```sh
python opgen -h
```

for more information.
