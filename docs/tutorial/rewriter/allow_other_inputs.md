# Specifying variable inputs in the pattern

This section demonstrates the use of the `_allow_other_inputs` option in pattern-based rewriting.
The `_allow_other_inputs` option allows the pattern to match nodes that have additional inputs
beyond those specified in the pattern. If it is set to `False` (the default), then the node must
have exactly the specified inputs for a successful match. If set to `True`, the pattern will
match nodes that have the specified inputs plus any number of additional inputs.

This is particularly useful when matching operations like `Conv` that can have optional inputs
(such as bias), or when creating generic patterns that should work with various input configurations.

```{literalinclude} examples/allow_other_inputs.py
:pyobject: conv_pattern
```

```{literalinclude} examples/allow_other_inputs.py
:pyobject: conv_replacement
```

```{literalinclude} examples/allow_other_inputs.py
:pyobject: apply_rewrite
```

In this example, the pattern matches `Conv` operations with any number of inputs. A `Conv` operation
might have 2 inputs (input and weight) or 3 inputs (input, weight, and bias). By setting
`_allow_other_inputs=True`, our pattern will match both cases even though we only specify 2 inputs
in the pattern definition.
