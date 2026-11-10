# Introduction

The ONNX Rewriter tool provides the user with the functionality to replace certain patterns in an ONNX graph with another pattern based on conditional rewrite rules provided by the user.

# Usage

There are three main components needed when rewriting patterns in the graph:

1. `target_pattern` : Original pattern to match against. This pattern is written as a function using ONNXScript-like operators.
2. `replacement_pattern` : Pattern to replace the original pattern with. This pattern is also written as a function using ONNXScript-like operators.
3. `match_condition` (optional) : Pattern rewrite will occur only if the match condition is satisfied.

## Pattern Options

When defining patterns, you can use several special options to control how patterns match and what they produce:

- `_allow_other_attributes`: Controls whether the pattern allows additional attributes not specified in the pattern (default: True)
- `_allow_other_inputs`: Controls whether the pattern allows additional inputs beyond those specified (default: False)
- `_domain`: Specifies the operator domain for matching or creating operations
- `_outputs`: Specifies the number and optionally names of outputs from an operation

These options are documented in detail in the following sections.

```{include} simple_example.md
```

```{include} attributes.md
```

```{include} allow_other_inputs.md
```

```{include} domain_option.md
```

```{include} outputs_option.md
```

```{include} conditional_rewrite.md
```

```{include} or_pattern.md
```

```{include} commute.md
```

```{include} node_value_checkers.md
```

## Debugging rewrite rules

To see matcher progress and applied replacements while running a pass, set
`verbose` explicitly:

```python
from onnxscript import rewriter

rewrite_pass = rewriter.RewritePass([rule], verbose=1)
result = rewrite_pass(model)  # model is an onnx_ir.Model
print(result.modified)  # Whether any rewrite was applied
```

Verbose diagnostics go to stdout; they do not configure Python logging.
`RewritePass` also logs an application count at INFO level on the
`onnxscript.rewriter` logger when rewrites occur. With the default `verbose=None`,
each rule retains its own verbosity setting. Use `verbose=0` to suppress verbose
output or a higher level, such as `verbose=3`, for more matcher detail.
Verbose mode still applies rewrites.

To investigate why a rule did not apply, use the existing `MatchingTracer` instead:

```python
tracer = rewriter.MatchingTracer()
rule.apply_to_model(model, tracer=tracer)
tracer.report()
```

This collects diagnostics from the same matcher and condition checks used during
rewriting, without applying replacements. The report shows the best match across
the traced rules, its status, and any recorded failure reason and source
nodes/values. For example, an operator mismatch reports
`OpType mismatch: expected Abs, got Neg`; check functions can provide specific
reasons using the failure conventions described above. If there are no partial
matches, the report prints `No matches found.`; unrelated starting operator types
are skipped rather than reported individually.

Use a fresh tracer to inspect a particular candidate root node:

```python
tracer = rewriter.MatchingTracer()
rule.match(model, model.graph, node, tracer=tracer)
tracer.report()
```

`rule.match` checks the pattern and its conditions, but does not invoke the
replacement function. Its `SUCCESS` status means only those checks passed.
Whole-model tracing also constructs replacements, reporting `REPLACEMENT_FAILED`
with any recorded reason if construction fails. Only attempts with a successfully
constructed replacement are reported as `SUCCESS` in this mode. Tracing can
invoke rule visitors and perform analysis/cleanup, so use `model.clone()` when
the original model must be preserved exactly.
