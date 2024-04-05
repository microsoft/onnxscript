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

```py
@overload
def MakeRewriteRule (target: onnx.FunctionProto, replacement: onnx.FunctionProto) -> RewriteRule:
    ...

@overload
def MakeRewriteRule (
    target: TargetPatternFunction,
    replacement: ReplacementPatternFunction
    ) -> RewriteRule:
    ...

# Other overloads, including support for extra validation conditions
```

## The Replacement Pattern and Conditions

In a simplified setting, the replacement pattern can be thought of as a function (FunctionProto)
that has the same signature as the target pattern: so, they have the same number of inputs
and outputs, of the same type and kind (eg., attribute parameters and input parameters).

In a conditional rewrite-rule, we can specify an extra predicate-function as a parameter
when constructing the rewrite-rule. The predicate is evaluated for every successful match
to decide whether the rewrite-transformation should be applied. For example, we may wish
to apply the transformation only if certain shape-conditions are met. The predicate
function gets information about the match as its input parameter. In general, the result
of a successful match is represented by its _match-bindings_, which is a dictionary that
binds various variables used in the target-pattern to the values (and attributes) in the
graph that are matched against it. For example, for the gemm-like pattern presented above,
the match-bindings will provide the values bound to "A" as well as "term1", allowing the
predicate to check the type and shape of these values if desired (as well as any other
information available from the underlying graph IR). (Note: some of this is yet to be
implemented. Currently, the dictionary doesn't yet include bindings for intermediate
variables.)

This leads to the following overloads:
```py
@overload
def MakeRewriteRule (
    target: onnx.FunctionProto,
    replacement: onnx.FunctionProto,
    condition: MatchPredicate,
    ) -> RewriteRule:
    ...

@overload
def MakeRewriteRule (
    target: TargetPatternFunction,
    replacement: ReplacementPatternFunction,
    condition: MatchPredicate,
    ) -> RewriteRule:
    ...
```



