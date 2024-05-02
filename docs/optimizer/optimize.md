# Optimizing a Model using the Optimizer

## Introduction

The ONNX Script `Optimizer` tool provides the user with the functionality to optimize an ONNX model by performing optimizations and clean-ups such as constant folding, dead code elimination, etc.

## Usage

In order to utilize the optimizer tool,

```python
    import onnxscript

    onnxscript.optimizer.optimize(model)
```

### optimize API
The `onnxscript.optimizer.optimize` call takes in several optional parameters that allows the caller to further fine-tune the process of optimization.

```{eval-rst}
.. autofunction:: onnxscript.optimizer.optimize
    :noindex:
```

## Description of optimizations applied by `onnxscript.optimizer.optimize`

:::{table}
:widths: auto
:align: center

| Optimization 'onnxscript.optimizer.` + .. | Description |
| - | - |
| **Constant folding** <br>`constant_folding.fold_constants` | Applies constant folding optimization to the model. |
| **Constant propagation** <br>`constant_folding.fold_constants` | Applies constant propagation optimization to the model. Applied as part of the constant folding optimization. |
| **Sequence simplification** <br>`constant_folding.fold_constants` | Simplifies Sequence based ops (SequenceConstruct, ConcatFromSequence) present in the model. Applied as part of the constant folding optimization. |
| **Remove unused nodes** <br>`remove_unused.remove_unused_nodes` | Removes unused nodes from the model. |
| **Remove unused functions** <br>`remove_unused_function.remove_unused_functions` | Removes unused function protos from the model. |
| **Inline functions with unused outputs** <br>`simple_function_folding.inline_functions_with_unused_outputs` | Inlines function nodes that have unused outputs. |
| **Inline simple functions** <br>`simple_function_folding.inline_simple_functions` | Inlines simple functions based on a node count threshold. |
:::

## List of pattern rewrite rules applied by `onnxscript.optimizer.optimize`

```{eval-rst}
.. autosummary::
    :nosignatures:

    onnxscript.rewriter.broadcast_to_matmul
    onnxscript.rewriter.cast_constant_of_shape
    onnxscript.rewriter.gemm_to_matmul_add
    onnxscript.rewriter.no_op

```
