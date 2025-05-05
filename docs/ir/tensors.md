# Tensor Representation in the IR

The ONNX IR offers the {py:class}`ir.TensorProtocol <onnxscript.ir.TensorProtocol>` interface for using different data structures as backing data for tensors. Besides the traditional {py:class}`onnx.TensorProto`, you can use {py:class}`np.ndarray`, {py:class}`torch.Tensor`, {py:class}`jax.Array`, and virtually anything else to represent tensors in the graph. This allows them to be accessed and serialized via the same `TensorProtocol` interface, without incurring additional copies during initialization.

## The `TensorProtocol`

{py:class}`ir.TensorProtocol <onnxscript.ir.TensorProtocol>` defines a read-only interface for representing tensors. A tensor class implementing the interface has attributes like `name`, `shape`, `dtype`, `size`, `nbytes` and `metadata_props` to describe basic properties of the tensor. Additionally, it should implement two methods {py:meth}`numpy <onnxscript.ir.TensorProtocol.numpy>` and {py:meth}`__array__  <onnxscript.ir.TensorProtocol.__array__>` which will produce equivalent NumPy arrays from the backing data.

:::{note}
When interacting with initializers, constant values and tensor attributes, it is best to assume `TensorProtocol` and only use `isinstance` to check for concrete classes when there is a need.
:::

## Tensor Classes

### ir.TensorProtoTensor

We use the {py:class}`ir.TensorProtoTensor <onnxscript.ir.TensorProtoTensor>` as a wrapper around the proto to implement the `ir.TensorProtocol` interface. You can access `shape`, `dtype` etc. as usual. A copy is incurred only when `numpy()` is called.

:::{note}
Directly initializing an `ir.TensorProtoTensor`, as below, is possible. However, it is usually recommended to use `ir.serde.deserialize_tensor` because it handles all types of `TensorProto`s (`ir.TensorProtoTensor` doesn't handle external tensors, for example). Please refer to [From `TensorProto`s and back](#from-tensorprotos-and-back) for an example.
:::

```{eval-rst}
.. exec_code::

    import onnx
    from onnxscript import ir

    tensor_proto = onnx.helper.make_tensor("tensor", onnx.TensorProto.INT16, (3,), [1, 2, 3])
    tensor = ir.TensorProtoTensor(tensor_proto)
    print("tensor: ", tensor)  # TensorProtoTensor<INT16,[3]>(name='tensor')
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

### String Tensor

Use {py:class}`ir.StringTensor <onnxscript.ir.StringTensor>` to create a string tensor.

<!-- TODO(justinchuby): Document make tensor helper -->

### Sparse Tensor

Sparse tensors are not yet supported, but they are on our roadmap.

## From `TensorProto`s and back

In the following scenario, we show how to go from a `TensorProto` to an `ir.Tensor`, run some computation, then turn it back to an `ir.Tensor` and finally `TensorProto`

```{eval-rst}
.. exec_code::

    from onnxscript import ir
    import onnx
    import numpy as np

    # 1. Create the TensorProto
    proto = onnx.helper.make_tensor(
        "tensor", onnx.TensorProto.FLOAT16, [2, 3], [1, 2, 3, 4, 5, 6]
    )

    # 2. Create an IR Tensor from the Protobuf message
    tensor = ir.serde.deserialize_tensor(proto)
    # Note that we get a TensorProtoTensor that implements the TensorProtocol
    print("tensor:", tensor)  # TensorProtoTensor<FLOAT16,[2,3]>(name='tensor')
    print("tensor.numpy():", tensor.numpy())   # [[1. 2. 3.]
                                               #  [4. 5. 6.]]
    print("tensor.tobytes():", tensor.tobytes())  # b'\x00<\x00@\x00B\x00D\x00E\x00F'

    # 3. Do computation using numpy
    mean = tensor.numpy().mean(axis=0)
    print("mean:", mean)  # array([2.5, 3.5, 4.5], dtype=float16)

    # 4. Create a Tensor from the ndarray. Note that we use ir.Tensor
    tensor_mean = ir.Tensor(mean)
    print("tensor_mean:", tensor_mean)  # Tensor<FLOAT16,[3]>(array([2.5, 3.5, 4.5], dtype=float16), name='')

    # 5. Obtain the TensorProto from ir.Tensor
    mean_tensor_proto: onnx.TensorProto = ir.serde.serialize_tensor(tensor_mean)
    print("mean_tensor_proto:", mean_tensor_proto)
    print(
        "onnx.numpy_helper.to_array(mean_tensor_proto):",
        onnx.numpy_helper.to_array(mean_tensor_proto)
        # array([2.5, 3.5, 4.5], dtype=float16)
    )

    # You can obtain the bytes data as well
    print("tensor_mean.tobytes():", tensor_mean.tobytes())
    print("Bytes same as proto:", mean_tensor_proto.raw_data == tensor_mean.tobytes())

    # Explore other methods defined by TensorProtocol:
    print("\n# Explore other methods defined by TensorProtocol:")
    print("tensor_mean.shape:", tensor_mean.shape)
    print("tensor_mean.dtype:", tensor_mean.dtype)
    print("tensor_mean.name:", tensor_mean.name)
    print("tensor_mean.doc_string:", tensor_mean.doc_string)
    print("tensor_mean.raw:", tensor_mean.raw)
    print("tensor_mean.metadata_props:", tensor_mean.metadata_props)
    print("tensor_mean.size:", tensor_mean.size)
    print("tensor_mean.nbytes:", tensor_mean.nbytes)
    print("tensor_mean.raw:", tensor_mean.raw)
```

## Working with non-native NumPy dtypes: bfloat16, float8, int4

`ir.Tensor.numpy()` produces a NumPy array representation of the tensor's value. When the tensor has dtype `BFLOAT16`, `FLOAT8[...]` or `[U]INT4` which are not supported by NumPy, we use dtypes from the `ml_dtypes` package.

`uint4`/`int4` is always unpacked; **`tobyte()` produces a packed representation** as expected.

Initialization of `ir.Tensor` requires the NumPy array to follow the following typing constraints, or have a `ml_dtypes` dtype.

- `int8` for (unpacked) int4, with the sign bit extended to 8 bits.
- `uint8` for (unpacked) uint4.
- `uint8` for 8-bit data types like float8.
- `uint16` for bfloat16.

The following example shows how to create a `FLOAT8E4M3FN` tensor, transform its values, and create a new tensor to store the transformed values.

```{eval-rst}
.. exec_code::

    from onnxscript import ir
    import numpy as np

    array = np.array([0b1, 0b11], dtype=np.uint8)
    # The array is reinterpreted using the ml_dtypes package
    tensor = ir.Tensor(array, dtype=ir.DataType.FLOAT8E4M3FN)
    print(tensor)  # Tensor<FLOAT8E4M3FN,[2]>(array([0.00195312, 0.00585938], dtype='float8_e4m3fn'), name=None)
    print("tensor.numpy():", tensor.numpy())  # [0.00195312 0.00585938]

    # Compute
    times_100 = tensor.numpy() * np.array(100, dtype=tensor.numpy().dtype)
    print("times_100:", times_100)

    # Create a new tensor out of the new value; dtype must be specified
    new_tensor = ir.Tensor(times_100.view(np.uint8), dtype=ir.DataType.FLOAT8E4M3FN)
    # You can also directly create the tensor from the float8 array without specifying dtype
    # new_tensor = ir.Tensor(times_100)
    print("new_tensor:", new_tensor)  # Tensor<FLOAT8E4M3FN,[2]>(array([0.1875, 0.5625], dtype='float8_e4m3fn'), name=None)
    print("new_tensor == times_100", new_tensor.numpy() == times_100)  # array([ True,  True])
```

## Advanced Usage

### Subclass `ir.Tensor` for More Efficient Access and Broader `dtype` Support

{py:class}`ir.Tensor` internally converts any array compatible objects into NumPy arrays to produce the byte representation in `tobytes()`. This can be inefficient due to the additional conversion. It also limits support for dtypes not supported by NumPy like bfloat16, because the `__array__` method would fail.

To fully support arrays from other frameworks, it is usually a good idea to create specialized classes to handle them. The `TorchTensor` class below demonstrates how you can subclass `ir.Tensor` to handle PyTorch tensors:

```{eval-rst}
.. exec_code::

    import ctypes
    from typing import Any

    import numpy.typing as npt
    import torch

    from onnxscript import ir


    class TorchTensor(ir.Tensor):
        def __init__(
            self, tensor: torch.Tensor, name: str | None = None, doc_string: str | None = None
        ):
            # Pass the tensor as the raw data to ir.Tensor's constructor

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
                torch.uint16: ir.DataType.UINT16,
                torch.uint32: ir.DataType.UINT32,
                torch.uint64: ir.DataType.UINT64,
            }
            super().__init__(
                tensor, dtype=_TORCH_DTYPE_TO_ONNX[tensor.dtype], name=name, doc_string=doc_string
            )

        def numpy(self) -> npt.NDArray:
            self.raw: torch.Tensor
            if self.dtype == ir.DataType.BFLOAT16:
                return self.raw.view(torch.uint16).numpy(force=True).view(self.dtype.numpy())
            if self.dtype in {
                ir.DataType.FLOAT8E4M3FN,
                ir.DataType.FLOAT8E4M3FNUZ,
                ir.DataType.FLOAT8E5M2,
                ir.DataType.FLOAT8E5M2FNUZ,
            }:
                return self.raw.view(torch.uint8).numpy(force=True).view(self.dtype.numpy())

            return self.raw.numpy(force=True)

        def __array__(self, dtype: Any = None, copy: bool | None = None) -> npt.NDArray:
            del copy  # Unused, but needed for the signature
            if dtype is None:
                return self.numpy()
            return self.numpy().__array__(dtype)

        def tobytes(self) -> bytes:
            # Implement tobytes to support native PyTorch types so we can use types like bloat16
            # Reading from memory directly is also more efficient because
            # it avoids copying to a NumPy array
            import torch._subclasses.fake_tensor

            with torch._subclasses.fake_tensor.unset_fake_temporarily():  # pylint: disable=protected-access
                # Disable any fake mode so calling detach() etc. will return a real tensor
                tensor = self.raw.detach().cpu().contiguous()

            if isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor):  # pylint: disable=protected-access
                raise TypeError(
                    f"Cannot take content out from the FakeTensor ('{self.name}'). Please replace the tensor "
                    "with a tensor backed by real data using ONNXProgram.apply_weights() "
                    "or save the model without initializers by setting include_initializers=False."
                )

            return bytes(
                (ctypes.c_ubyte * tensor.element_size() * tensor.numel()).from_address(
                    tensor.data_ptr()
                )
            )

    # Test the implementation
    torch_tensor = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
    tensor = TorchTensor(torch_tensor)
    print("tensor: ", tensor)
    print("numpy: ", tensor.numpy())
    print("tobytes: ", tensor.tobytes())  # b'\x80?\x00@@@'
    print("nbytes: ", tensor.nbytes)  # 6
```

The `TorchTensor` class above implements `tobytes()` to produce the correct bytes representation for the tensor when it is serialized into an ONNX file / TensorProto. The class also implements the `__array__()` method to return the bit representation for types NumPy does not support. This way analysis passes can still perform computation on these values.

### Computation with different Frameworks

Since `ir.Tensor` implements the `__array__` method and `__dlpack__` methods, its content can be shared with computation frameworks without copying. For example:

```{eval-rst}
.. exec_code::

    from onnxscript import ir

    # We can call numpy methods directly on ir.Tensor
    import numpy as np
    print(np.multiply(ir.Tensor(np.array([1, 2])), 42))  # array([42., 84.])

    # We can transfer arrays to different frameworks
    import jax.numpy as jnp
    import jax
    import torch

    # Create ir.Tensor
    jax_array = jnp.array([10., 20.])
    ir_tensor_jax = ir.Tensor(jax_array, dtype=ir.DataType.FLOAT)
    torch_tensor = torch.tensor([30., 40.])
    ir_tensor_torch = ir.Tensor(torch_tensor, dtype=ir.DataType.FLOAT)

    # Use numpy for computation
    print(np.multiply(ir_tensor_jax, ir_tensor_torch))  # array([300., 800.], dtype=float32)

    # Use jax for computation by calling from_dlpack to transfer the tensor data without copying when the device is the same
    jax_array_from_ir = jax.dlpack.from_dlpack(ir_tensor_torch)
    print(jax_array_from_ir + jax_array)  # [40. 60.]

    # Use PyTorch for computation
    torch_tensor_from_ir = torch.from_dlpack(ir_tensor_jax)
    print(torch_tensor_from_ir - torch_tensor)  # tensor([-20., -20.])

    # They can all be serialized into TensorProto
    proto = ir.serde.serialize_tensor(ir_tensor_jax)
    print(type(proto))  # <class 'onnx.onnx_ml_pb2.TensorProto'>
    print(proto)

    # The value is exactly the same as jax_array
    print(ir.serde.deserialize_tensor(proto).numpy())  # [10. 20.]
```

This is particularly useful if you are creating passes on the graph that requires doing computation on concrete values. You are free to use your favorite frameworks to create the passes. The transformed graph that contains newly created `ir.Tensor`s will be compatible with downstream passes even if they leverage other computation frameworks.
