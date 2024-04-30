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

```{eval-rst}
.. autosummary::
    :nosignatures:

    onnxscript.optimizer.constant_folding.fold_constants
    onnxscript.optimizer.copy_propagation.do_copy_propagation
    onnxscript.optimizer.copy_propagation.do_sequence_simplification
    onnxscript.optimizer.remove_unused.remove_unused_nodes
    onnxscript.optimizer.remove_unused_function.remove_unused_functions
    onnxscript.optimizer.simple_function_folding.inline_functions_with_unused_outputs
    onnxscript.optimizer.simple_function_folding.inline_simple_functions

```

## List of pattern rewrite rules applied by `onnxscript.optimizer.optimize`

```{eval-rst}
.. autosummary::
    :nosignatures:

    onnxscript.rewriter.broadcast_to_matmul
    onnxscript.rewriter.cast_constant_of_shape
    onnxscript.rewriter.gemm_to_matmul_add
    onnxscript.rewriter.no_op

```
