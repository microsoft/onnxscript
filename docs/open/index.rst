Proposed Features
=================

The features described below are proposals, but they have not been implemented yet.

**Functions as Graph Attributes**

ONNX allows graph-valued attributes. This is the mechanism used to define (quasi)
higher-order ops, such as *If*, *Loop*, *Scan*, and *SequenceMap*.
While we use Python control-flow to encode *If* and *Loop*, it is useful
to have a generic mechanism to support graph-valued attributes.
We propose to allow users to use Python function-definitions to represent
graph-valued attributes, as shown in the example below:

::

    from onnxscript import script
    from onnxscript.onnx_opset import opset15 as op

    @script()
    def CumulativeSum(X):
        def Sum(sum_in, next):
            sum_out = sum_in + next
            return sum_out, sum_out
        all_sum, cumulative_sum = op.Scan (0, X, body=Sum, num_scan_inputs=1)
        return cumulative_sum

Specifically, the function-definition of *Sum* is converted into a graph and used
as the attribute-value when invoking the *Scan* op.

Function definitions used as graph-attributes must satisfy some constraints.
They cannot update outer-scope variables, but may reference them.
(Specifically, the functions cannot use *global* or *nonlocal* declarations.)
They are also restricted from using local-variables with the same name
as outer-scope variables (no shadowing).

There is also an interaction between SSA-renaming and the use of outer-scope
variables inside a function-definition. The following code is invalid, since
the function *CumulativeSum* references the global *g*, which is updated
in between the function-definition and function-use. Note that, from an
ONNX perspective, the two assignments to *g* represent two distinct tensors
*g1* and *g2*.

::

    from onnxscript import script
    from onnxscript.onnx_opset import opset15 as op

    @script()
    def CumulativeSum(X):
        g = op.Constant(value=0);
        def Sum(sum_in, next):
            sum_out = sum_in + next + g
            return sum_out, sum_out
        g = op.Constant(value=1)
        all_sum, cumulative_sum = op.Scan (0, X, body=Sum, num_scan_inputs=1)
        return cumulative_sum


Limitations and Open Issues
===========================

The following are some of the limitations and open issues of the current implementation.

* Discrepancies between eager-mode execution and the translated ONNX.
    Such discrepancies are possible due to various convenience (syntactic-sugar)
    features supported by the translator. For example, a construct like `1+X`
    where `X` is a tensor will fail under standard Python execution, but is
    translated into a tensor addition operation, with the constant `1` being
    promoted to a tensor.
