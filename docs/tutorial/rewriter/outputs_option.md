# Specifying outputs in the pattern

This section demonstrates the use of the `_outputs` option in pattern-based rewriting.
The `_outputs` option allows you to specify the number of outputs an operation produces
and optionally assign names to those outputs for easier reference in replacement patterns.

The `_outputs` option can be specified in two ways:
- As an integer: `_outputs=2` specifies that the operation produces 2 unnamed outputs
- As a list of strings/None: `_outputs=["first", "second"]` specifies 2 named outputs

## Matching operations with multiple outputs

```{literalinclude} examples/outputs_option.py
:pyobject: split_pattern
```

This pattern matches `Split` operations that produce exactly 2 outputs. The `_outputs=2`
specification ensures the pattern only matches operations with this specific output count.

## Creating replacement operations with named outputs

```{literalinclude} examples/outputs_option.py
:pyobject: custom_split_replacement
```

In the replacement, `_outputs=["first_half", "second_half"]` creates two outputs with
descriptive names. This can make the replacement pattern more readable and maintainable.

**Important**: The number of outputs in the replacement pattern must match the number of
outputs in the target pattern. Since the pattern specifies `_outputs=2`, the replacement
must also produce exactly 2 outputs.

## Complete rewrite example

```{literalinclude} examples/outputs_option.py
:pyobject: apply_rewrite
```

The `_outputs` option is particularly important when:
- Working with operations that have variable numbers of outputs (like `Split`)
- Creating custom operations that need specific output configurations
- Ensuring pattern matching precision by specifying exact output counts
- Improving code readability by naming outputs in replacement patterns
