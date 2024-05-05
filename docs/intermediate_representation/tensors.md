# Tensor Representation in the IR

The ONNX IR offers the {py:class}`ir.TensorProtocol <onnxscript.ir.TensorProtocol>` interface for usings different data structures as backing data for tensors. Besides the traditional {py:class}`onnx.TensorProto`, you can also use {py:class}`np.ndarray`, {py:class}`torch.Tensor`, {py:class}`jax.Array`, and virtually anything else to represent tensors in the graph. This allows for them to be accessed and serialized via the same `TensorProtocol` interface, without incurring additional copies at initialization.

## The `TensorProtocol`

{py:class}`ir.TensorProtocol <onnxscript.ir.TensorProtocol>` defines a read-only interface for representing tensors. A tensor class implementing the interface has attributes like `name`, `shape`, `dtype`, `size`, `nbytes` and `metadata_props` to describe basic properties of the tensor. Additionally, it implements two methods {py:meth}`numpy <onnxscript.ir.TensorProtocol.numpy>` and {py:meth}`__array__  <onnxscript.ir.TensorProtocol.__array__>` which will produce equivalent NumPy arrays from the backing data.

:::{note}
When interacting with initializers, constant values and tensor attributes, it is best to assume `TensorProtocol` and only use `isinstance` to check for concrete classes when there is a need.
:::

## Tensor Classes

### ir.TensorProtoTensor

The ONNX spec defines [different ways](https://github.com/onnx/onnx/blob/d6f87121ba256ac6cc4d1da0463c300c278339d2/onnx/onnx.proto#L567-L654) for storing tensor data as an {py:class}`onnx.TensorProto <onnx.ir.TensorProtocol>` protocol buffer message. The IR has corresponding classes for each of these data storage methods.

We use the {py:class}`ir.TensorProtoTensor <onnxscript.ir.TensorProtoTensor>` as a wrapper around the proto to implement the `ir.TensorProtocol` interface. You can access `shape`, `dtype` etc. as usual. A copy is incurred only when `numpy()` is called.

```{eval-rst}
.. exec_code::
    import onnx
    from onnxscript import ir

    tensor_proto = onnx.helper.make_tensor("tensor", onnx.TensorProto.INT16, (3,), [1, 2, 3])
    tensor = ir.TensorProtoTensor(tensor_proto)
    print(tensor: ", tensor)  # TensorProtoTensor<INT16,[3]>(name='tensor')
    print("shape: ", tensor.shape)  # ir.Shape([3])
    print("dtype: ", tensor.dtype)  # ir.DataType.INT16
    print(tensor.raw == tensor_proto)  # The raw field is the exact tensor_proto provided at initialization
    print("tobytes: ", tensor.tobytes())  # b'\x01\x00\x02\x00\x03\x00'
    print("numpy: ", tensor.numpy())  # array([1, 2, 3], dtype=int16)
```

### ir.ExternalTensor

Tensor data stored externally in the disk are typically large and will take up memory when loaded. The {py:class}`ir.ExternalTensor <onnxscript.ir.ExternalTensor>` class uses memory mapping to avoid loading the tensor into memory. You are able to use the tensor as a normal NumPy array with minimal memory usage.

Refer to {py:func}`ir.serde.deserialize_tensor <onnxscript.ir.serde.deserialize_tensor>` to find an example on converting an `onnx.TensorProto` to an {py:class}`ir.ExternalTensor <onnxscript.ir.ExternalTensor>`.

### ir.Tensor

{py:class}`ir.Tensor <onnxscript.ir.Tensor>` is a wrapper around NumPy array compatible array objects like {py:class}`np.ndarray` and {py:class}`torch.Tensor`. It is best for creating in-memory tensors without converting it to a `TensorProto` to reduce the conversion overhead.

:::{tip}
An array object is compatible if it defines the `__array__` method.
:::

To create a tensor from an array, simply initialize it with an NumPy array

```python
tensor = ir.Tensor(np.random.rand(1, 2))
```

The initializer will obtain dtype and shape information from the array.

To create a tensor from objects other than NumPy array, you need to specify the dtype:

```{eval-rst}
.. exec_code::
    import torch
    from onnxscript import ir


    torch_tensor = torch.tensor([1, 2, 3], dtype=torch.float16)
    tensor = ir.Tensor(torch_tensor, dtype=ir.DataType.FLOAT16)
    print(tensor.numpy())  # array([1., 2., 3.], dtype=float16)
```

### Subclass ir.Tensor for More Efficient Access and Broader dtype Support

{py:class}`ir.Tensor` internally converts any array compatible objects into NumPy arrays to produce the byte representation in `tobytes()`. This can be inefficient due to the additional conversion. It also limits support for dtypes not supported by NumPy like bfloat16, because the `__array__` method would fail.

To fully support arrays from other frameworks, it is usually a good idea to create specialized classes to handle them. The `TorchTensor` class below demonstrates how you can subclass `ir.Tensor` to handle PyTorch tensors:

```{eval-rst}
.. exec_code::
    import ctypes
    from typing import Any

    import torch
    from onnxscript import ir

    # Define utilities to convert PyTorch data types so users do not need to specify manually
    _TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType] = {
        torch.bfloat16: ir.DataType.BFLOAT16,
        torch.bool: ir.DataType.BOOL,
        torch.complex128: ir.DataType.COMPLEX128,
        torch.complex64: ir.DataType.COMPLEX64,
        torch.float16: ir.DataType.FLOAT16,
        torch.float32: ir.DataType.FLOAT,
        torch.float64: ir.DataType.DOUBLE,
        torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
        torch.float8_e4m3fnuz: ir.DataType.FLOAT8E4M3FNUZ,
        torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
        torch.float8_e5m2fnuz: ir.DataType.FLOAT8E5M2FNUZ,
        torch.int16: ir.DataType.INT16,
        torch.int32: ir.DataType.INT32,
        torch.int64: ir.DataType.INT64,
        torch.int8: ir.DataType.INT8,
        torch.uint8: ir.DataType.UINT8,
    }


    def _torch_dtype_to_onnx_dtype(dtype: torch.dtype) -> ir.DataType:
        return _TORCH_DTYPE_TO_ONNX[dtype]

    class TorchTensor(ir.Tensor):
        def __init__(self, tensor: torch.Tensor):
            # Pass the tensor as the raw data to ir.Tensor's constructor
            super().__init__(tensor, dtype=_torch_dtype_to_onnx_dtype(tensor.dtype))

        def __array__(self, dtype: Any = None) -> "np.ndarray":
            # numpy() calls __array__ in ir.Tensor
            if self.dtype == ir.DataType.BFLOAT16:
                return self.raw.view(torch.uint16).__array__(dtype)
            if self.dtype in {
                ir.DataType.FLOAT8E4M3FN,
                ir.DataType.FLOAT8E4M3FNUZ,
                ir.DataType.FLOAT8E5M2,
                ir.DataType.FLOAT8E5M2FNUZ
            }:
                return self.raw.view(torch.uint8).__array__(dtype)
            return self.raw.__array__(dtype)

        def tobytes(self) -> bytes:
            # Implement tobytes to support native PyTorch types so we can use types like bloat16
            # Reading from memory directly is also more efficient because
            # it avoids the copy to NumPy array
            tensor = self.raw.detach().cpu().contiguous()
            return bytes(
                (ctypes.c_ubyte * tensor.element_size() * tensor.numel()).from_address(
                    tensor.data_ptr()
                )
            )

    # Test the implementation
    torch_tensor = torch.tensor([1,2,3], dtype=torch.bfloat16)
    tensor = TorchTensor(torch_tensor)
    print("tensor: ", tensor)
    print("numpy: ", tensor.numpy())
    print("tobytes: ", tensor.tobytes())  # b'\x80?\x00@@@'
    print("nbytes: ", tensor.nbytes)  # 6
```

The `TorchTensor` class above implements `tobytes()` to produce the correct bytes representation for the tensor when it is serialized into an ONNX file / TensorProto. The class also implements the `__array__()` method to return float32 for types NumPy does not support. This way analysis passes can still perform computation on these values.

### String Tensor

Use class:`ir.StringTensor` to create a string tensor.


<!-- TODO(justinchuby): Document make tensor helper -->

### Sparse Tensor

Sparse tensors are not yet supported, but they are on our roadmap.

## Advanced usage

Since `ir.Tensor`
