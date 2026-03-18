---
name: writing-rewrite-rules
description: >
  How to write ONNX rewrite rules using onnxscript.rewriter for the
  mobius package. Covers the RewriteRuleClassBase API, pattern
  matching, check/rewrite methods, file organization, and testing conventions.
  Use this skill when creating rules that transform parts of an ONNX model
  (e.g., replacing standard ops with custom/fused ops).
---

# Skill: Writing Rewrite Rules

## When to use

Use this skill when creating rules that transform parts of an ONNX model graph
— for example, replacing standard ops with custom or fused ops for better
runtime performance.  Rewrite rules live in `src/mobius/rewrite_rules/`
and are applied **after** model export.

## API overview

Rewrite rules use `RewriteRuleClassBase` from `onnxscript.rewriter`:

```python
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet
from onnxscript.rewriter import rewrite
```

1. **Subclass** `RewriteRuleClassBase` with `pattern()`, `check()`, and
   `rewrite()` methods.
2. Call `.rule()` on an instance to create a `RewriteRule`.
3. Wrap one or more rules in a `RewriteRuleSet`.
4. Apply via `rewrite(model, pattern_rewrite_rules=rule_set)`.

> **Important:** The keyword argument is `pattern_rewrite_rules`, **not** `rules`.

```python
class MyRule(RewriteRuleClassBase):
    def pattern(self, op, ...): ...
    def check(self, context, ...): ...
    def rewrite(self, op, ...): ...

def my_rules() -> RewriteRuleSet:
    return RewriteRuleSet([MyRule().rule()])

# Apply
rewrite(model, pattern_rewrite_rules=my_rules())
```

## Pattern function

`pattern(self, op, ...)` defines the ONNX subgraph to match.  The positional
parameters after `op` become the matched inputs.

```python
def pattern(self, op, q, k, v, attn_bias_2d):
    q_4d = op.Unsqueeze(q, [0])
    k_4d = op.Unsqueeze(k, [0])
    v_4d = op.Unsqueeze(v, [0])
    attn_bias_4d = op.Unsqueeze(attn_bias_2d, [0, 1])
    attn_out = op.Attention(
        q_4d, k_4d, v_4d, attn_bias_4d,
        _allow_other_attributes=True,
        _outputs=["attn_out"],
    )
    return op.Squeeze(attn_out, [0])
```

Key options:

- **`_outputs=["name"]`** — Capture an intermediate value so it can be
  referenced in `check()` and `rewrite()` by that keyword name.
- **`_allow_other_attributes=True`** — Match an op even if it has extra
  attributes not listed in the pattern (e.g. `scale`, `num_heads`).

## Check function

`check(self, context, **kwargs)` validates structural requirements that the
pattern alone cannot express.  Matched inputs and captured outputs arrive as
keyword arguments.

Return `MatchResult()` (no arguments) for success.  Call `.fail("reason")` to
reject the match:

```python
def check(self, context, attn_bias_2d, attn_out, **_):
    result = MatchResult()

    # Walk the producer chain to verify structure
    where = attn_bias_2d.producer()
    if where is None or where.op_type != "Where":
        return result.fail("Expected Where producing attention bias")

    # Access attributes on matched nodes
    attn = attn_out.producer()
    if attn.attributes.get_float("scale", None) is None:
        return result.fail("Missing scale attribute on Attention")
    if attn.attributes.get_int("q_num_heads", None) is None:
        return result.fail("Missing q_num_heads attribute on Attention")

    return result
```

Common attribute accessors:

- `node.attributes.get_float("name")` / `node.attributes.get_float("name", default)`
- `node.attributes.get_int("name")` / `node.attributes.get_int("name", default)`

## Rewrite function

`rewrite(self, op, **kwargs)` builds the replacement subgraph.  The `op`
parameter is an **IR tape builder** (not the same as `onnxscript`'s
`OpBuilder`).  Matched inputs and captured outputs arrive as keyword arguments.

```python
def rewrite(self, op, q, k, v, attn_bias_2d, attn_out, **_):
    attn = attn_out.producer()
    scale = attn.attributes.get_float("scale")
    num_heads = attn.attributes.get_int("q_num_heads")

    cu_seqlens = self._trace_cu_seqlens(attn_bias_2d)
    cu_seqlens_i32 = op.Cast(cu_seqlens, to=6)

    return op.op(
        "PackedMultiHeadAttention",
        inputs=[q, k, v, None, token_offset, cu_seqlens_i32],
        domain="com.microsoft",
        attributes={"scale": scale, "num_heads": num_heads},
    )
```

### Critical: constant tensors in rewrite

Raw Python lists **cannot** be used as inputs in the rewrite function.
Always create constants explicitly:

```python
# GOOD — explicit Constant node
axes_0 = op.Constant(value_ints=[0])
neg_one = op.Constant(value_ints=[-1])
result = op.Squeeze(x, axes_0)

# BAD — raw list (will fail)
result = op.Squeeze(x, [0])
```

### Custom / domain-specific ops

Use `op.op(...)` to emit single-output ops from non-default domains:

```python
op.op(
    "PackedMultiHeadAttention",
    inputs=[q, k, v, None, token_offset, cu_seqlens],
    domain="com.microsoft",
    attributes={"scale": scale, "num_heads": num_heads},
)
```

Pass `None` in the inputs list for optional inputs that should be left empty.

### Multi-output custom ops

Use `op.op_multi_out(...)` for ops with multiple outputs.
**`op.op()` returns a single `ir.Value`; `op.op_multi_out()` returns
`Sequence[ir.Value]`.**

```python
outputs = op.op_multi_out(
    "GroupQueryAttention",
    inputs=[q, k, v, past_key, past_value, seqlens_k, total_seq_len],
    domain="com.microsoft",
    attributes={"num_heads": num_heads, "kv_num_heads": kv_num_heads},
    num_outputs=3,
)
attn_out, present_key, present_value = outputs[0], outputs[1], outputs[2]
```

### Matching patterns with shared intermediate values

The rewriter will **not** match a pattern if an intermediate node's output
has consumers outside the matched subgraph.  For example, matching
`Add → RMSNorm` will fail if the `Add` output is also used by a downstream
residual connection.

**Workaround:** Match only the end node, then trace back in `check()`:

```python
def pattern(self, op, add_out, weight):
    # Only match RMSNorm — don't include Add in the pattern
    return op.RMSNormalization(add_out, weight, _allow_other_attributes=True)

def check(self, context, add_out, **_):
    result = MatchResult()
    producer = add_out.producer()
    if producer is None or producer.op_type != "Add":
        return result.fail("Input is not from Add")
    if len(list(add_out.uses())) < 2:
        return result.fail("Add has only 1 consumer")
    return result

def rewrite(self, op, add_out, weight, **_):
    add_node = add_out.producer()
    input_a, input_b = add_node.inputs[0], add_node.inputs[1]
    # Create fused op and reroute the shared output
    outputs = op.op_multi_out("FusedOp", inputs=[input_a, input_b, weight], ...)
    add_out.replace_all_uses_with(outputs[1])  # reroute skip connection
    return outputs[0]  # return the primary output
```

## File organization

```
src/mobius/rewrite_rules/
├── __init__.py                          # Public exports
├── _packed_attention.py                 # Rule implementation (private module)
├── _packed_attention_test.py            # Unit tests (next to source)
├── _group_query_attention.py            # Attention → GQA rule
├── _group_query_attention_test.py       # GQA rule tests
├── _skip_norm.py                        # Add+RMSNorm → SkipNorm rule
└── _skip_norm_test.py                   # SkipNorm rule tests
```

### Conventions

- Rule files are **private modules**: `_rule_name.py`.
- Unit tests go **next to the source file**: `_rule_name_test.py`.
- Export the public factory function from `__init__.py`:

```python
# __init__.py
__all__ = ["packed_attention_rules"]
from mobius.rewrite_rules._packed_attention import packed_attention_rules
```

- Each rule module should provide a factory function (e.g.
  `packed_attention_rules()`) that returns a `RewriteRuleSet`.

## Testing

Write unit tests that:

1. Build a model containing the target pattern (either a tiny model from
   the model library or a synthetic graph).
2. Count ops before applying the rule.
3. Apply the rule set via `rewrite(model, pattern_rewrite_rules=rules)`.
4. Count ops after and assert the expected replacements occurred.
5. Verify that non-matching subgraphs are **not** affected.

```python
from collections import Counter
from onnxscript.rewriter import rewrite
from mobius.rewrite_rules import packed_attention_rules


def _count_ops(model) -> Counter:
    return Counter(node.op_type for node in model.graph)


class TestPackedAttentionRules:
    def test_rule_replaces_vision_attention(self):
        model = build_model_with_pattern(...)
        counts_before = _count_ops(model)
        assert counts_before["Attention"] == 4

        rewrite(model, pattern_rewrite_rules=packed_attention_rules())

        counts_after = _count_ops(model)
        assert counts_after["PackedMultiHeadAttention"] == 2
        assert counts_after["Attention"] == 2  # text decoder untouched

    def test_rule_preserves_non_matching_model(self):
        """Models without the pattern are not affected."""
        model = build_text_only_model(...)
        counts_before = _count_ops(model)

        rewrite(model, pattern_rewrite_rules=packed_attention_rules())

        counts_after = _count_ops(model)
        assert counts_after["Attention"] == counts_before["Attention"]
        assert counts_after.get("PackedMultiHeadAttention", 0) == 0

    def test_rules_returns_rule_set(self):
        from onnxscript.rewriter._rewrite_rule import RewriteRuleSet
        rules = packed_attention_rules()
        assert isinstance(rules, RewriteRuleSet)
```

## Reference files

- **Full rule implementations:**
  - `src/mobius/rewrite_rules/_packed_attention.py` — Block-diagonal → PackedMHA
  - `src/mobius/rewrite_rules/_group_query_attention.py` — Attention → GQA
  - `src/mobius/rewrite_rules/_skip_norm.py` — Add+RMSNorm → SkipNorm
- **Test examples:**
  - `src/mobius/rewrite_rules/_packed_attention_test.py`
  - `src/mobius/rewrite_rules/_group_query_attention_test.py`
  - `src/mobius/rewrite_rules/_skip_norm_test.py`
- **Exports:**
  `src/mobius/rewrite_rules/__init__.py`
