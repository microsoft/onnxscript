# OR Patterns

*Note* : This feature is work-in-progress.

Consider the following pattern:

```{literalinclude} examples/or_pattern.py
:pyobject: scaled_matmul
```

This pattern will successfully match against the sequence "MatMul => Mul => Relu" as
well as the sequence "MatMul => Div => Relu". The matcher will bind the variable
specified in `tag_var` (`op_type` in the above example) to a value from those
listed in `tag_values` to indicate which of the alternatives was used for a
successful match. We can use this in the rewrite function to determine how
we want to rewrite the matched sub-graph, as illustrated by the following code:

```{literalinclude} examples/or_pattern.py
:pyobject: scaled_matmul_replacement
```