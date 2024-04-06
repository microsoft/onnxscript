# Pattern Matching Based Rewrite Rules

This design document describes a proposed API for specifying rewrite-rules for
optimizing and transforming ONNX models.

In its simple form, a rewrite-rule contains a source-pattern and a target. The
source-pattern describes a sub-graph that is to be replaced by the specified
target. In an extended form, the rewrite-rule can also specify a _condition_
to be evaluated for every matching sub-graph to decide whether the replacement
should happen or not. These are referred to as _conditional rewrite rules_.

Here is a simple example of a _pattern_ that looks for a subgraph that is
gemm-like:
```py
def gemm_pattern(alpha, beta, A, B, C):
    term1 = alpha * op.MatMul (A, B)
    term2 = beta * C
    return term1 + term2
```

A successful match of this pattern against a subgraph in a given model can
be represented by a dictionary mapping the inputs of this pattern (such
as `alpha` and `A`) to _values_ in the graph IR. (Note that a value
in the IR represents either an input of a function/graph or a computed
value, that is, an output of some node). More generally, the dictionary
can also store the values bound to intermediate variables such as `term1`.
We refer to this dictionary as a _match\_binding_.

This allows us to encode extra-conditions that we wish to check after
a successful match. For example, to replace this subgraph by a single
call to `Gemm` requires some additional conditions: we may require
`alpha` to be a scalar constant (so that it can be replaced by an
attribute in the call to `Gemm`), and we may need to check that there
is no broadcasting in the `MatMul` since `Gemm` does not support broadcast.

We can encode such conditions using a guard function as below:
```py
def gemm_condition(alpha: ir.Value, beta: ir.Value, A, B, C, **other_bindings):
    alpha_value = alpha.constant_value
    if (alpha_value is None):
        # alpha is not a static constant. Do not proceed with this match
        return False 
```
We can call this function from the rewriter-engine via as `gemm_condition(**bindings)`
where `bindings` is the dictionary produced by the successful match above.

We can also construct the replacement subgraph using the results of a
successful match. This can also be defined as a function as below:
```py
def gemm_replacement(alpha, beta, A, B, C):
    return op.Gemm(A, B, C, ...)
```

Note that we can also potentially combine the logic for the guard function and
the replacement function into one function. (Currently they are different.)

We can then define a rewrite rule as below:
```py
gemm_fusion_rule = RewriteRule (gemm_patttern, gemm_replacement, gemm_condition)
```

We have a couple of design choices with regards to some of the details of
the above API.

1. Using a _FunctionProto_ or _ONNX Script_ : We can represent a pattern
such as the above one using a _FunctionProto_, authored using _ONNX Script_.

2. We can use a custom pattern authoring language, similar to ONNX Script.
The current implementation supports one version of such a language.

Option 1 has the advantage that we reuse the existing implementation,
and users already familiar with ONNX Script can directly use it.
However, this approach has some limitations. For example, in the
pattern we may not care about a required-attribute's value. If
ONNX Script complains about a missing required-attribute, that would
be a problem when specifying a pattern.

Option 2 allows us to have a more extensible mechanism, customized to
the case of patterns. For example, it is easy to allow users to specify
that we are looking for attribute `alpha` to have a value of `0.16325`
within some error tolerance limits. (Note that we can push such conditions
into the guard/condition function too. So, this specific example is more
of a syntactic-sugar convenience.) Another interesting use-case, which
comes up in practice, is when we want to do some wild-card matching against
a op/function name ... eg., if we want to match against a call to a
function (produced by the pytorch exporter) with a name that has `Silu` as
a prefix.

## IR and Builder for Patterns and Rewrite-Rules

We already have a couple of approaches to pattern-matching based rewrite-rules:
the original one (restricted to single-output patterns, but supporting other
features such as a mechanism to bind variables to specific nodes of the
pattern, and using them to check for extra-conditions before applying a
rewrite-rule, and allowing for more efficient match-and-replace), and a
newer extension to support multiple-output patterns, but not integrated
with the other mechanisms.

We can integrate and simplify these approaches, including the different options
for describing patterns mentioned above, by unifying the classes defining
the patterns and rewrite-rules (into a pattern IR) and by providing different
factory (constructor) methods for patterns and rewrite-rules. Specifically,
these methods will be the developer-facing APIs for expressing rewrite-rules.
Hence, it is important for us to consolidate these APIs to avoid potential
confusion among users writing rewrite-rules. (The implementations of the
pattern-matching can vary, depending on the complexity of the pattern,
such as whether it is single-output or multiple-output, etc.)

Proposal:

* The rewrite-rule constructor takes two parameters, one to specify the
pattern, and one to specify the replacement. We can provide a version with
an extra condition parameter (as a convenience feature).

* The pattern can be either a FunctionProto (or, equivalently, an ONNX Script
function) or a function that constructs a pattern-graph or a pattern-graph.
In the first two cases, they are converted into a pattern-graph.

* The replacement is a function that accepts an expanded match-bindings
(that is, `**match_bindings`) and returns either `None` (in case of failure)
or a representation of the modification to the graph (in terms of removed
and added nodes/values).

* We may benefit from using a trace-mode onnxscript function above. In
particular, we may be able to improve upon the replacement function by
integrating the graph-modification cleanly into a generalization of the
trace-mode onnxscript.



