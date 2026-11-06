# Compiler implementation

Use the public {func}`onnxscript.script` decorator to compile supported Python
functions into {class}`onnxscript.values.OnnxFunction` objects. See
[Decorator](decorator.md) and [Values](values.md) for the supported entry points.

The AST compiler is an implementation detail in `onnxscript._internal`.
Its classes, signatures, and intermediate representations are unsupported and
may change without public API compatibility guarantees. There is no public
`onnxscript.converter.Converter` API.
