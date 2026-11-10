# onnxscript.optimizer

```{eval-rst}
.. automodule::onnxscript.optimizer
.. currentmodule:: onnxscript
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: functiontemplate.rst
    :nosignatures:

    optimizer.optimize
    optimizer.inline
    optimizer.basic_constant_propagation
    optimizer.fold_constants
    optimizer.register_constant_folder
```

```{eval-rst}
.. autoclass:: onnxscript.optimizer.ConstantEvaluatorRegistry
    :members: register

.. autoclass:: onnxscript.optimizer.ConstantEvaluationContext
    :members:
```

`optimizer.ConstantEvaluator` is the callback type:
`Callable[[Sequence[ir.TensorProtocol | None], Mapping[str, ir.Attr],
ConstantEvaluationContext], Sequence[ir.TensorProtocol] | None]`.
