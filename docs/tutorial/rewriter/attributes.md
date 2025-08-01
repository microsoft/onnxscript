# Specifying attributes in the pattern

This section demonstrates the use of attribute values in pattern-based rewriting.
First, write a target pattern and replacement pattern in a similar way to the previous examples.
The example pattern below will match successfully only against Dropout nodes with the
attribute value `training_mode` set to `False`.

The `_allow_other_attributes` option allows the pattern to match nodes that have additional attributes
not specified in the pattern. If it is set to `False`, then the node must have only the specified
attribute values, and no other attributes, for a successful match. The default value for this
option is `True`.

```{literalinclude} examples/allow_other_attributes.py
:pyobject: add_pattern
```

```{literalinclude} examples/allow_other_attributes.py
:pyobject: add_replacement
```

```{literalinclude} examples/allow_other_attributes.py
:pyobject: apply_rewrite
```
