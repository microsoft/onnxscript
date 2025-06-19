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
