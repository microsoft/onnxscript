# onnx-script roadmap

onnx-script is being actively developed. Its features and specs may change as a result.

* Improve error-checking and error-handling to identify use of unsupported
Python features/constructs as early as possible and report it.

* Improve documentation and tutorials

* Features
  - Support nested functions, for use as sub-graphs in ops like Scan
  - Generalize support for _break_ statements in loops
  - Support indexing notation for slicing/gather/scatter operations
  - Improve type-annotation support, especially for tensor shapes
  - Improve support for non-tensor types (sequences, maps, and optional)
  - Improve checking in eager-mode to ensure its semantics is aligned with ONNX semantics
  - Support for variadic inputs/outputs