# Supported Python subset

`@script()` translates a subset of Python into ONNX when the decorator runs.
Calling the resulting function runs its original Python body eagerly, with
ONNX operators evaluated by the selected evaluator. A construct that runs
eagerly is not necessarily exportable.

This page lists the supported forms. See the [tutorial](index.md) for type
annotations, attributes, constants, and complete examples.

## Syntax and execution boundary

| Construct | Compiled ONNX | Eager execution / boundary |
| --- | --- | --- |
| Tensor arithmetic and comparisons | Expressions such as `2 * x + 1.0`, `x < 3`, and `(0 < x) & (x < 3)` map to ONNX operators. Scalar literals are cast to the dtype of the tensor operand when the operator schema requires it. | Tensor wrappers follow ONNX operator semantics, not NumPy promotion rules. Arbitrary Python or NumPy calls are not translated. |
| Chained comparisons | `0 < x < 3` is rejected. Combine separate comparisons with `&`. | Python evaluates chained comparisons with truth testing, not elementwise conjunction. |
| Returns | Return a value, or a tuple of output values, at function-body level, after any control flow. A nested graph function has its own function-body return and return annotation. | Eager execution preserves a Python tuple of outputs. Exported model runners instead return an ordered list of graph outputs. |
| Return inside control flow | A `return` inside an `if`, `for`, or `while` is rejected, including inside a statically selected branch. Assign branch/loop results to variables and return afterwards. | Early returns run eagerly but cannot be exported. |
| Conditional statements | `if condition: ... else: ...` becomes ONNX `If` for a runtime condition. Variables used afterwards must be defined on every path. | Use a scalar Boolean tensor as the condition. Both branches must produce matching ONNX types. |
| Loops | `for i in range(count)` (one argument) and `while condition` (a condition variable updated by the body) become ONNX `Loop`. Initialize loop-carried values before the loop. | Eager mode runs the Python loop. `range(start, stop, step)` and arbitrary iterables are not supported by the compiler. |
| Indexing and slicing | Scalar indices and slices such as `x[1:]`, `x[::-1]`, and `x[start:stop:step]` map to ONNX indexing operators. Negative constant steps can use omitted bounds. | If the step is a runtime value, both bounds must be explicit: `x[::step]` is rejected because the default bounds depend on the step's sign. Constant slice components must be integers. Ellipsis and `newaxis` are not supported. |
| Nested graph functions | Define `@graph()` functions directly inside a scripted function and pass them as graph-valued attributes, for example `op.Scan(..., body=body)`. They can read captured outer values. | Eager evaluation requires the `@graph()` decorator; a plain nested `def` compiles but is rejected by the eager evaluators. |
| Captured graph values | A captured value must not be reassigned between graph definition and graph use. `global` and `nonlocal` are not supported. | Pass state through explicit graph inputs and outputs. |

For example, use a single return after the conditional:

```python
from onnxscript import BOOL, FLOAT, opset17 as op, script

@script()
def choose(x: FLOAT["N"], condition: BOOL) -> FLOAT["N"]:
    if condition:
        result = op.Add(x, 1)
    else:
        result = op.Sub(x, 1)
    return result
```

See the tutorial's [control-flow and graph-attribute examples](index.md#control-flow)
for complete loops and nested graphs. Script-time constant attribute
expressions follow the rules in the
[script constants section](index.md#semantics-script-constants).

## Printing is a debugging aid

A standalone `print(...)` statement is allowed but omitted from the compiled
graph, including its arguments. It runs in eager mode, so it can be used to
inspect eager values. Other expression statements and Python I/O are not
exported.

## Custom operators are a separate execution contract

An operator call qualified by an `Opset` can target a custom domain:

```python
from onnxscript import script, values

custom = values.Opset("example.custom", 1)

@script()
def transform(x):
    return custom.Transform(x, mode="fast")
```

The compiler emits this node even without a registered operator schema.
Without a schema, positional arguments become ONNX inputs and keyword
arguments become ONNX attributes; schema-based argument checks and scalar
casts are skipped. The compiler does not verify that the operator exists, so
a misspelled operator name may not be caught at decoration time.

Running the node requires a runtime implementation (or an included ONNX
function). The built-in eager evaluators require an operator schema.

## Compiler/eager parity and return conventions

The paired tests in `tests/eager_mode_test.py` export models, run them, and
compare the results with eager execution of the original Python bodies. They
check values, dtypes, shapes, output counts, and output types for:

| Boundary | Representative coverage |
| --- | --- |
| Optional operator inputs | `Clip` with omitted trailing inputs, explicit `None`, an omitted middle input, and named inputs. |
| Function arguments | Default and explicit scalar attribute arguments, and out-of-order keyword arguments. |
| Scalar promotion | Literals mixed with `float32`, `float64`, and `int64` tensors; standalone Python integer, float, and Boolean inputs. |
| Output wrapping | Rank-zero and vector arrays, tensor wrappers, mixed array/tensor inputs, single outputs, and tuples of outputs. |
| Control flow and graph attributes | Both conditional branches, zero and nonzero loop counts, forward/reverse/dynamic slices, separate comparisons, and a directly nested `Scan` graph with a captured value. |

Eager input and output conventions: Python integer inputs become `int64`
tensors and Python float inputs use NumPy's default floating dtype. If any
tensor input is a NumPy array (including one in an input sequence), eager
tensor results are returned as NumPy arrays. With only `onnxscript.tensor.Tensor`
or Python scalar inputs, results are returned as `Tensor` objects. Tuple and
list containers are preserved, and a single output is not wrapped in a tuple.
Attribute arguments do not affect output wrapping.

Tests run under both ONNX Runtime and ONNX's reference evaluator, with these
reference-evaluator exceptions: local-function calls with omitted attribute
defaults and counted loops with an omitted initial condition run the exported
model with ONNX Runtime only; the captured-value `Scan` case is skipped for the
reference evaluator when its `Scan` implementation does not pass outer-scope
values to the body.

Negative syntax and custom-domain/printing boundaries are covered in
`tests/language_support_test.py`.
