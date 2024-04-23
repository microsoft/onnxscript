# ONNX IR

An in-memory IR that supports the full ONNX spec, designed for graph construction, analysis and transformation.

## Features

- Full ONNX spec support: all valid models representable by ONNX protobuf, and a subset of invalid models (so you can load and fix them).
- Memory friendly representation of initializers and tensors: mmap'ed external tensors; unified interface for ONNX TensorProto, Numpy arrays and PyTorch Tensors etc. No tensor size limitation. Zero copies.
- Straightforward access patterns: Access value information and traverse the graph topology at ease.
- Robust mutation: Create as many iterators as you like on the graph while mutating it.
- Speed: Performant graph manipulation, serialization/deserialization to Protobuf.

## Project structure

- [`_protocols.py`](_protocols.py): Interfaces defined for all entities in the IR.
- [`_core.py`](_core.py): Implementation of the core entities in the IR, including `Model`, `Graph`, `Node`, `Value`, and others.
- [`_enums.py`](_enums.py): Definition of the type enums that correspond to the `DataType` and `AttributeType` in `onnx.proto`.
- [`_name_authority.py`](_name_authority.py): The authority for giving names to entities in the graph, used internally.
- [`_linked_list.py`](_linked_list.py): The data structure as the node container in the graph that supports robust iteration and mutation. Internal.
- [`_metadata.py`](_metadata.py): Metadata store for all entities in the IR.
