# Specifying domains in the pattern

This section demonstrates the use of the `_domain` option in pattern-based rewriting.
The `_domain` option allows you to specify which operator domain the pattern should match against,
and also allows you to create replacement operations in specific domains.

ONNX operators can belong to different domains:
- The default ONNX domain (empty string or "ai.onnx")
- Custom domains like "com.microsoft" for Microsoft-specific operations
- User-defined domains for custom operations

## Matching operations from a specific domain

```{literalinclude} examples/domain_option.py
:pyobject: custom_relu_pattern
```

In this pattern, `_domain="custom.domain"` ensures that only `Relu` operations from the 
"custom.domain" domain will be matched, not standard ONNX `Relu` operations.

## Creating replacement operations in a specific domain

```{literalinclude} examples/domain_option.py
:pyobject: microsoft_relu_replacement
```

Here, the replacement operation is created in the "com.microsoft" domain, which might
provide optimized implementations of standard operations.

## Complete rewrite example

```{literalinclude} examples/domain_option.py
:pyobject: apply_rewrite
```

This example shows how domain-specific pattern matching can be used to migrate operations
between different operator domains, such as replacing custom domain operations with
standard ONNX operations or vice versa.