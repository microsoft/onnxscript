# Tensor representation in the IR

The ONNX IR offers the :class:`TensorProtocol` interface for usings different data structures as backing data for tensors. Besides the traditional ``onnx.TensorProto``, you can also use ``np.ndarray``, ``torch.Tensor``, and virtually anything else to represent tensors in the graph. This allows for  They can be accessed and serialized via the same `TensorProtocol` interface.

## The ``TensorProtocol``

:class:`ir.TensorProtocol` defines a read-only interface for representing tensors. A tensor class implementing the interface has attributes like ``name``, ``shape``, ``dtype``, ``size``, ``nbytes`` and ``metadata_props`` to describe basic properties of the tensor. Additionally, it implements two methods ``numpy()`` and ``__array__()`` which will produce equivalent numpy arrays from the backing data.

When interacting with the IR, it is best to assume

## TensorProto

The ONNX spec defines [different ways](https://github.com/onnx/onnx/blob/d6f87121ba256ac6cc4d1da0463c300c278339d2/onnx/onnx.proto#L567-L654) for storing tensor data as a TensorProto protocol buffer message. The IR has corresponding classes for each of these data storage methods.

For tensors whose data are stored in the ``raw_data`` field, we use the :class:`ir.TensorProtoTensor`
