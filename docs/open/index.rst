Limitations and Open Issues
===========================

The following are some of the limitations and open issues of the current implementation.

* Discrepancies between eager-mode execution and the translated ONNX.
    Such discrepancies are possible due to various convenience (syntactic-sugar)
    features supported by the translator. For example, a construct like `1+X`
    where `X` is a tensor will fail under standard Python execution, but is
    translated into a tensor addition operation, with the constant `1` being
    promoted to a tensor.
