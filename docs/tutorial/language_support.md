# Supported Python subset

`@script()` translates a **subset of Python** into ONNX when the decorator runs.
Calling the resulting function eagerly executes its original Python body, with
ONNX operators evaluated by the selected evaluator. Successful eager execution
alone is therefore not evidence that a construct can be exported, or that a
particular ONNX backend implements it.

This page describes the supported forms, not a promise to accept arbitrary
Python. Start with the [tutorial](index.md) for type annotations, attributes,
constants, and complete examples.

## Syntax and execution boundary

| Construct | Compiled ONNX | Eager execution / boundary |
| --- | --- | --- |
| Tensor arithmetic and comparisons | Expressions such as `2 * x + 1.0`, `x < 3`, and `(0 < x) & (x < 3)` map to ONNX operators. Scalar literals are cast to a compatible tensor operand's dtype when the operator schema requires it. | Tensor wrappers use ONNX operators, not general NumPy promotion rules. Do not assume arbitrary Python or NumPy calls are translatable. |
| Chained comparisons | `0 < x < 3` is rejected. Write separate comparisons combined with `&` for elementwise conjunction. | Python's chained-comparison truth testing is not elementwise tensor conjunction; it is not part of the portable subset. |
| Returns | Return a value, or a tuple of output values, at function-body level, after any control flow. A nested graph function has its own function-body return and return annotation. | Eager execution preserves a Python tuple of outputs. Exported model runners instead return an ordered list of graph outputs. |
| Return inside control flow | A `return` inside an `if`, `for`, or `while` is rejected, including inside a statically selected branch. Assign branch/loop results to variables and return afterwards. | Python could perform an early return, but this is not an exportable control-flow form. |
| Conditional statements | `if condition: ... else: ...` becomes ONNX `If` for a runtime condition. Variables used afterwards must be defined on every path. | The condition must have an unambiguous truth value; use a scalar Boolean tensor for portable runtime conditions. Both branches must satisfy ONNX type requirements. |
| Loops | `for i in range(count)` (one argument) and `while condition` (a condition variable updated by the body) become ONNX `Loop`. Initialize loop-carried values before the loop. | Eager mode executes Python control flow, evaluating operators in the body. `range(start, stop, step)` and arbitrary iterables are not supported compiler forms. |
| Indexing and slicing | Scalar indices and slices such as `x[1:]`, `x[::-1]`, and `x[start:stop:step]` map to ONNX indexing operators. Negative constant steps can use omitted bounds. | Tensor indexing uses ONNX operators. If the step is a runtime value, **both bounds must be explicit**: `x[::step]` is rejected by the compiler because default bounds depend on its sign. Constant slice components must be integers. Ellipsis and `newaxis` are not supported. |
| Nested graph functions | Define `@graph()` functions directly inside a scripted function and pass them as graph-valued attributes, for example `op.Scan(..., body=body)`. They can read captured outer values. | Use `@graph()` for eager execution as well. A plain nested `def` may compile as a graph attribute, but the built-in eager evaluators reject it without the decorator. Ordinary Python helper functions and arbitrarily deep graph nesting are not covered by this contract. |
| Captured graph values | A captured value must not be reassigned between graph definition and graph use. `global` and `nonlocal` statements are not supported ways to mutate captures. | Use explicit graph inputs/outputs for state, rather than Python closure mutation. |

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
for complete loops and nested graphs. These restrictions apply to scripted
tensor computation; script-time constant attribute expressions have the
separate evaluation rules described in the
[script constants section](index.md#semantics-script-constants).

## Printing is a debugging aid

A standalone `print(...)` statement is permitted but **omitted from the compiled
graph**. Its arguments are not emitted as graph computations. The original
Python statement runs in eager mode, so it can help inspect eager values.
Do not use printing, or side effects in its arguments, as model semantics.
This exception does not make arbitrary expression statements or Python I/O
exportable.

## Custom operators are a separate execution contract

An operator call qualified by an `Opset` can target a custom domain:

```python
from onnxscript import script, values

custom = values.Opset("example.custom", 1)

@script()
def transform(x):
    return custom.Transform(x, mode="fast")
```

The compiler can preserve this node even without a registered operator schema.
Without a schema, positional arguments are treated as ONNX inputs and keyword
arguments as ONNX attributes; schema-directed argument checks and scalar casts
are unavailable. This is intentional custom-operator support, **not** a blanket
rejection of unknown domains. Conversely, emitting a node is not validation
that an operator exists: do not rely on every misspelled standard operator
being rejected at decoration time.

Execution requires a suitable schema/evaluator and a runtime implementation
(or an included ONNX function). The built-in eager evaluators require operator
signatures and cannot run an arbitrary schema-less custom node merely because
it compiled.

## Compiler/eager parity and return conventions

The bounded paired tests in `tests/eager_mode_test.py` export models and execute
them independently from the original Python bodies. They compare values,
dtypes, shapes, output counts, and representations for:

| Boundary | Representative coverage |
| --- | --- |
| Optional operator inputs | `Clip` with omitted trailing inputs, explicit `None`, an omitted middle input, and named inputs. |
| Function arguments | Default and explicit scalar attribute arguments, and out-of-order keyword arguments. |
| Scalar promotion | Literals mixed with `float32`, `float64`, and `int64` tensors; standalone Python integer, float, and Boolean inputs. |
| Output wrapping | Rank-zero and vector arrays, tensor wrappers, mixed array/tensor inputs, single outputs, and tuples of outputs. |
| Control flow and graph attributes | Both conditional branches, zero and nonzero loop counts, forward/reverse/dynamic slices, separate comparisons, and a directly nested `Scan` graph with a captured value. |

The existing conventions are deliberate: Python integer inputs become `int64`
tensors and Python float inputs use NumPy's default floating dtype. If **any
tensor input is a NumPy array** (including one in an input sequence), eager
tensor results are unwrapped to NumPy arrays. With only `onnxscript.tensor.Tensor`
or Python scalar inputs, tensor results retain their `Tensor` wrappers.
Unwrapping preserves tuple/list containers; it does not turn a single output
into a one-element tuple. Attribute arguments do not select array wrapping.

Tests use ONNX Runtime and ONNX's reference evaluator where supported. Two
reference-model limitations are kept explicit: local-function calls with
omitted attribute defaults, and counted loops with an omitted initial
condition, are compared against an **ONNX Runtime model**, while eager
execution is checked under **both** evaluators. The remaining paired cases
execute both the exported model and eager operators on each backend. This is
not a claim of universal backend coverage.

Negative syntax and custom-domain/printing boundaries are covered in
`tests/language_support_test.py`.
