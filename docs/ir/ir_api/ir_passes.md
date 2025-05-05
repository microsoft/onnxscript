# ir.passes

```{eval-rst}
.. automodule::onnxscript.ir.passes
.. currentmodule:: onnxscript
```

## Use built-in passes

Common, reusable passes are implemented in `ir.passes.common`. You can use {py:class}`ir.passes.Sequential <onnxscript.ir.passes.Sequential>` to chain passes or use {py:class}`ir.passes.PassManager <onnxscript.ir.passes.PassManager>` which supports early stopping if no changes are made.

## Pass infrastructure

Inherent {py:class}`ir.passes.InPlacePass <onnxscript.ir.passes.InPlacePass>` or {py:class}`ir.passes.FunctionalPass <onnxscript.ir.passes.FunctionalPass>` to define a pass. You will need to implement the `call` method which returns a {py:class}`ir.passes.PassResult <onnxscript.ir.passes.PassResult>`.

Alternatively, inherent the base class `ir.passes.PassBase <onnxscript.ir.passes.PassBase>` and override the two properties `changes_input` and `in_place` to set properties of the pass.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: classtemplate.rst
    :nosignatures:

    ir.passes.PassBase
    ir.passes.InPlacePass
    ir.passes.FunctionalPass
    ir.passes.Sequential
    ir.passes.PassResult
    ir.passes.PassManager
```

## Errors

```{eval-rst}
.. autoexception:: onnxscript.ir.passes.InvariantError
.. autoexception:: onnxscript.ir.passes.PreconditionError
.. autoexception:: onnxscript.ir.passes.PostconditionError
.. autoexception:: onnxscript.ir.passes.PassError
```
