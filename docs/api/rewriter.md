# onnxscript.rewriter

```{eval-rst}
.. automodule::onnxscript.rewriter
.. currentmodule:: onnxscript
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: functiontemplate.rst
    :nosignatures:

    rewriter.rewrite
```

## IR passes

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    rewriter.RewritePass
```

## ONNX Runtime-specific fusions

The opt-in rules in `onnxscript.rewriter.ort_fusions` target ONNX Runtime contrib
operators rather than general ONNX runtimes.

GQA fusion recognizes supported causal-mask constructions, including the legacy
extra-column-and-slice form. It declines structurally different masks and
sliding-window masks: the replacement does not set `local_window_size`. The
supported `attention_mask` wrapper retains the existing ORT GenAI assumption
that this input only identifies current tokens, not arbitrary attention masking.

FusedMatMul division fusion requires a singleton, finite, nonzero divisor and
matching floating-point tensor types supported by the target operator. It leaves
zero (including negative zero), NaN, and infinite divisors as `Div` operations.
In particular, division by infinity is not replaced by multiplication by zero,
which can behave differently for infinite matrix products. Scaling must remain
finite and normal in both the float32 `alpha` attribute and the tensor type;
overflow, underflow, and subnormal scales are conservatively declined. Ordinary
floating-point reciprocal rounding is retained for float16, bfloat16, and float32.
For double tensors, the computed scale must be exactly representable by the
float32 attribute to avoid introducing a lower-precision scale.
