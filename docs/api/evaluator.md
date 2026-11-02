# Eager evaluator contexts

Calling an `@script()` function executes its original Python body in eager mode.
ONNX operators and calls to other scripted functions dispatch through
`onnxscript.evaluator.default()`. The default backend is ONNX Runtime;
`OnnxReferenceRuntimeEvaluator` provides the ONNX reference backend.

## Global defaults and temporary overrides

`set_default(evaluator)` changes the **process-global fallback**. It remains
available for application-wide configuration. A context without a temporary
override observes the latest global default, not a snapshot. Concurrent global
setters are not isolated; use `default_as` for temporary backend selection.

```python
from onnxscript import evaluator

with evaluator.default_as(evaluator.OnnxReferenceRuntimeEvaluator()):
    result = scripted_function(x)
```

`default_as(evaluator)` installs a **context-local override**, without changing
the global fallback. Nested overrides restore the previous override on exit,
including on exceptions. A global `set_default` during an override does not
replace that override; when the outermost override exits, the latest global
fallback becomes visible.

Selection follows Python's `contextvars` semantics:

- An asyncio task inherits the context at task creation. Later overrides in
  either the parent or child do not change the other's selection.
- An empty `contextvars.Context()` has no override and uses the global fallback.
  `copy_context().run(...)` propagates the current override, with subsequent
  context changes isolated from the source.
- Thread inheritance follows Python's thread/context configuration. Ordinary
  threads on conventional CPython builds start with an empty context; Python
  3.14 can configure inheritance with `Thread(context=...)` and
  `sys.flags.thread_inherit_context`. Explicitly run work in an empty or copied
  context when inheritance must be independent of those settings.
- `asyncio.to_thread` propagates the caller's context. A thread-pool submission
  does not itself copy the submitter's context.

Contexts hold evaluator **references**, not copies. Sharing an evaluator
instance between threads or tasks does not make that instance or its mutable
state concurrency-safe. These APIs select the backend only; they do not change
scalar promotion, array/tensor wrapping, or the `Evaluator` protocol.

## Graph attributes during eager execution

`@graph()` associates a directly nested Python function with the graph compiled
when its enclosing `@script()` function was decorated. Eager calls track the
active `OnnxFunction` explicitly, independently of wrapper stack depth.
Nested calls to other scripted functions restore the enclosing function's
context, including after exceptions. Closure values are read from the enclosing
Python invocation, so separate invocations do not share captured tensor values.

Ordinary Python wrappers around scripted calls and custom evaluators that
implement `eval_op` and `eval_function` remain supported. An evaluator dispatched
through `OnnxFunction.__call__` may invoke `function.function` with additional
helper frames; no specific evaluator inheritance or stack depth is required.
`BaseEvaluator.eval_function` also establishes the function context when called
directly.

Only graph functions defined **directly inside the active scripted function**
are supported. Graphs nested inside graph bodies are not supported; this does
not extend the compiler's accepted graph syntax. Calling the raw
`scripted_function.function(...)` outside evaluator dispatch bypasses the
required function context and raises a diagnostic at `@graph()`. Executing a
graph definition from a different Python function likewise raises a diagnostic
instead of accidentally selecting a graph by name from an unrelated invocation.

```{eval-rst}
.. automodule:: onnxscript.evaluator
   :members:
```
