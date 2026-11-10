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

## Custom constant evaluation

Use an explicit `ConstantEvaluatorRegistry` to evaluate custom operators whose
inputs are constant. Registrations affect only calls receiving that registry;
they do not modify global optimizer behavior.

```python
from typing import Mapping, Sequence

import onnx_ir as ir
from onnxscript import optimizer

evaluators = optimizer.ConstantEvaluatorRegistry()


@optimizer.register_constant_folder(
    "Double", domain="example", version=(1, None), registry=evaluators
)
def double(
    inputs: Sequence[ir.TensorProtocol | None],
    attributes: Mapping[str, ir.Attr],
    context: optimizer.ConstantEvaluationContext,
) -> Sequence[ir.TensorProtocol] | None:
    if inputs[0] is None:
        return None
    return [ir.tensor(inputs[0].numpy() * 2)]


optimizer.optimize(model, evaluators=evaluators)
# Or: optimizer.fold_constants(model, evaluators=evaluators)
```

The equivalent decorator is `@evaluators.register("Double", domain="example",
version=(1, None))`. An integer version matches exactly; a tuple specifies
inclusive bounds, and `None` means no bound. Overlapping registrations for the
same domain/operator are rejected, and `ai.onnx` is normalized to the standard
empty domain.

Callbacks receive constant input tensors (`None` for omitted optional inputs),
a read-only attribute mapping, and immutable operator information. Return one
tensor per output, or `None` to defer to ONNX reference evaluation. Callbacks
must be deterministic and must not mutate input tensors or attributes.
Invalid output counts, types, dtypes, and known shapes raise actionable errors.

Custom evaluation follows the normal constant-input, nondeterminism, blacklist,
input-size, and output-size policies. As with reference evaluation,
`should_fold=True` can override the default blacklist and input-size limit,
but not the constant-input or nondeterminism requirements. Built-in partial
simplifications run first and are not overridden by this API. Use the
[rewriter](../../api/rewriter.md) for algebraic rewrites of nonconstant inputs,
not the optimizer's private partial-evaluator registry.

## Description of optimizations applied by `onnxscript.optimizer.optimize`

| Optimization | Description |
|-------------|-------------|
| **Constant folding** <br>`constant_folding.fold_constants` | Applies constant folding optimization to the model. |
| **Constant propagation** <br>`constant_folding.fold_constants` | Applies constant propagation optimization to the model. Applied as part of the constant folding optimization. |
| **Sequence simplification** <br>`constant_folding.fold_constants` | Simplifies Sequence based ops (SequenceConstruct, ConcatFromSequence) present in the model. Applied as part of the constant folding optimization. |
| **Remove unused nodes** <br>`remove_unused.remove_unused_nodes` | Removes unused nodes from the model. |
| **Remove unused functions** <br>`remove_unused_function.remove_unused_functions` | Removes unused function protos from the model. |
| **Inline functions with unused outputs** <br>`simple_function_folding.inline_functions_with_unused_outputs` | Inlines function nodes that have unused outputs. |
| **Inline simple functions** <br>`simple_function_folding.inline_simple_functions` | Inlines simple functions based on a node count threshold. |
