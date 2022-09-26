# ONNX Script Roadmap

<blockquote>
<table>
<tr>
<td>⚠️</td>
<td>
<strong>NOTE:</strong> ONNX Script is in <strong>very early and
active development</strong> and the team anticipates <strong>breaking
changes</strong> as the project evolves. ONNX Script is <strong>not
ready for production</strong>, but early feedback is welcome.
</td>
<td>⚠️</td>
</tr>
</table>
</blockquote>

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