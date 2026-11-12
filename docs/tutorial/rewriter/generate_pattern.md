# Generating a rewrite rule from an existing model

Writing the `pattern` of a rule by hand means transcribing part of a model into `op.Xxx(...)` calls.
`onnxscript.rewriter.pattern_codegen` does the transcription for you. Name the nodes of interest
(`ir.Node.name`) and it generates a `RewriteRuleClassBase` subclass whose `pattern` reproduces them.
It requires [libcst](https://libcst.readthedocs.io) (`pip install libcst`).

```bash
python -m onnxscript.rewriter.pattern_codegen model.onnx --nodes LeakyRelu Split -o rule.py
```

or from Python:

```python
import onnx_ir as ir
from onnxscript.rewriter import pattern_codegen

model = ir.load("model.onnx")
print(pattern_codegen.generate_rewrite_rule(model, ["LeakyRelu", "Split"]))
```

```python
class LeakyReluSplitRule(pattern.RewriteRuleClassBase):
    """Rewrite rule generated from nodes: LeakyRelu, Split"""

    def pattern(self, op, a):
        b = op.LeakyRelu(a, alpha=0.10000000149011612)
        c, d = op.Split(b, axis=-1, num_outputs=2, _outputs=2)
        return c, d

    def rewrite(self, op, a: ir.Value):
        raise NotImplementedError("Write the replacement here.")
```

How the selection is turned into a pattern:

- Values consumed by the selected nodes but produced elsewhere (graph inputs, initializers, other
  nodes) become the pattern's inputs.
- Values produced by the selected nodes that are used by other nodes or are graph outputs become
  the pattern's outputs. Matching requires every other intermediate value to stay inside the match.
- The selection must be convex: no path may leave it and come back in.
- Integer, float, string and list attributes are kept. Other attributes (tensors, graphs) are
  skipped and noted in a comment.

The generated pattern is as specific as the model. Float attributes are written at full precision
because attribute matching uses exact equality. Generalize by hand, for example by replacing an
attribute value with a pattern variable, and then write the `rewrite` method.
