# Function authoring guide for the ATen Library in ONNX Script

## ATen operators

The ATen library is "The foundational tensor and mathematical operation library"[^1] in PyTorch. It defines the Tensor operations in PyTorch. The Python interfaces in PyTorch expose a set of similar, but not necessarily identical operators that most developers are familiar with, like `torch.add`. The ATen operators are exposed under the namespace `torch.ops.aten`.
The _ATen Library in ONNX Script_ (`atenlib`) is a library of ONNX decomposition of the corresponding ATen operators as ONNX functions. Each function (`ATen op`) has only two purposes: (1) decompose the ATen op’s logic into ONNX, and (2) specify the requirements for inputs and guarantees of the outputs in its signature.

## An anatomy of an ATen op

An example of an ATen op, `add`, is the following:

[function_libs/torch_aten/core.py](https://github.com/microsoft/onnx-script/blob/2952f41d9a76e48be378f100fe1623d744fe1943/onnxscript/function_libs/torch_aten/ops/core.py#L58-L63)

```python
@torch_op("aten::add")  # (1)
def aten_add(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:  # (2)
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""  # (3)
    alpha = op.CastLike(alpha, other)
    other = op.Mul(other, alpha)
    return op.Add(self, other)  # (4)
```

The ATen `add` operator defines an `alpha` parameter that will scale the second operand. In the example above, we see

1.


### Function signature

### `trace_only` ops

## Common patterns

### Inputs and attributes

### Optional inputs

### Scalar inputs

### Transforming attributes

### DType dependent logic

## Style guide

## Checklist

[^1]: https://pytorch.org/cppdocs/#aten
