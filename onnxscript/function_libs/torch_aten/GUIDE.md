# Function authoring guide for the ATen Library in ONNX Script

## ATen operators

The ATen library is "The foundational tensor and mathematical operation library"[^1] in PyTorch. It defines the Tensor operations in PyTorch. The Python interfaces in PyTorch expose a set of similar, but not necessarily identical operators that most developers are familiar with, like `torch.add`. The ATen operators are exposed under the namespace `torch.ops.aten`.
The _ATen Library in ONNX Script_ (`atenlib`) is a library of ONNX decomposition of the corresponding ATen operators as ONNX functions. Each function (`ATen op`) has only two purposes: (1) decompose the ATen op’s logic into ONNX, and (2) specify the requirements for inputs and guarantees of the outputs in its signature.

## An anatomy of an ATen op

An example of an ATen op, `add`, is the following:

[function_libs/torch_aten/core.py](https://github.com/microsoft/onnx-script/blob/2952f41d9a76e48be378f100fe1623d744fe1943/onnxscript/function_libs/torch_aten/ops/core.py#L58-L63)

```python
@torch_op("aten::add")  # (1)
def aten_add(  # (2)
    self: TReal,  # (3)
    other: TReal,
    alpha: float = 1.0
) -> TReal:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""  # (4)
    alpha = op.CastLike(alpha, other)
    other = op.Mul(other, alpha)
    return op.Add(self, other)  # (5)
```

The ATen `add` operator defines an `alpha` parameter that will scale the second operand. In the example above, we

1. Register the function with the library using the `torch_op` decorator
2. Declares the function `aten_add`. The naming rule is `f"aten_{op_name}"`. The name is generated.
3. Declares the arguments to the function. The types are the most compatible types with ONNX the function can accept (See Function signature). ONNX `Input`s are annotated as tensors (`TReal` in this case). ONNX `Attribute`s are annotated using native Python types, following the ONNX Script syntax.

Every function template in the `atenlib` has a reference signature from PyTorch [`native_functions.yaml`](https://github.com/pytorch/pytorch/blob/44d8e6c2aa80dbeb2afc1e4471dc1b66bf47779a/aten/src/ATen/native/native_functions.yaml#L497) (4). `atenlib` functions should typically follow this signature.


## Writing a new op

Now that we have an idea of what a function should look like, let's see how to implement a new op. Consider `gelu` as an example:

First, we find the operator in `onnxscript/function_libs/torch_aten/ops/nn.py`.

```python
def aten_gelu(self: TensorType, approximate: str = "none") -> TensorType:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    raise NotImplementedError()
```

Then we write the function body:

```
    self = op.Cast(self, to=FLOAT.dtype)

    if approximate == "tanh":
        result = _aten_gelu_approximate_tanh(self)
    else:
        result = _aten_gelu_approximate_none(self)
    return result


@torch_op("aten::gelu", overload=True)
def _aten_gelu_approximate_none(self: TReal) -> TReal:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    self = op.Cast(self, to=FLOAT.dtype)
    # GELU(x) = 0.5 * x * [1 + ERF(x/sqrt(2)]
    inner = op.Div(self, 1.4142135623730951)
    erf = op.Erf(inner)
    inner = op.Add(erf, 1)
    inner = op.Mul(self, inner)
    result = op.Mul(0.5, inner)
    return result


@torch_op("aten::gelu", overload=True)
def _aten_gelu_approximate_tanh(self: TReal) -> TReal:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    self = op.Cast(self, to=FLOAT.dtype)
    # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
    cubed = op.Pow(self, 3)
    inner = op.Mul(0.044715, cubed)
    inner = op.Add(self, inner)
    # Prefer explicit graph construction over precomputed constants for clarity.
    inner = op.Mul(op.Sqrt(op.Div(2.0, _MATH_PI)), inner)
    inner = op.Tanh(inner)
    inner = op.Add(inner, 1)
    inner = op.Mul(self, inner)
    result = op.Mul(0.5, inner)
    return result
```


### Function signature

### `trace_only` ops

## Common patterns

### Inputs and attributes

### Optional inputs

### Scalar inputs

### Transforming attributes

### DType dependent logic

## Style guide

### Robustness

### Simplicity

## Testing

## Checklist

[^1]: https://pytorch.org/cppdocs/#aten
