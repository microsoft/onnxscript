# ONNXScript

ONNXScript is a subset of Python that can be used to author ONNX functions (as well as ONNX models).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Installation steps

```
pip install onnx onnxruntime
git clone https://github.com/microsoft/onnx-script.git
pip install -e .
```

## Run unit tests

```
pytest onnxscript/test
```

## Design

*onnxscript* implements two main functionalities:

- a converter which translate a python function into ONNX, the converter analyzes the python
  code through the python abstract syntactic tree and converts that tree into an ONNX graph
  equivalent to the function.
- a runtime returning an eager evaluation of this function, this runtime relies on
  *onnxruntime* for every operation described in
  [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

## Example

Let's write a function in file **onnx_fct.py**. The script may contain multiple functions.

```python
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script
from onnxscript.onnx_types import INT64, FLOAT
from onnxscript.opset15 as op


# If the function is a model to export, it must have annotations to specify the type of inputs and outputs.
@script()
def Hardmax(X: FLOAT[], axis=0) -> FLOAT[]:
    # op gives access to ONNX operators for opset 15
    argmax = op.ArgMax(X, axis=axis, keepdims=False)
    # The parser makes the distinction between inputs (unnamed arguments) and attributes (named parameters).
    xshape = op.Shape(X, start=axis)
    # Constant must be declared with operator Constant.
    zero = op.Constant(value_ints=[0])
    depth = op.GatherElements(xshape, zero)
    empty_shape = op.Constant(value_ints=[])
    depth = op.Reshape(depth, empty_shape)
    # Constant Array must be defined with function make_tensor from onnx package.
    values = op.Constant(value=make_tensor('cst01', TensorProto.FLOAT, [2], [0, 1]))
    cast_values = op.CastLike(values, X)
    return op.OneHot(argmax, depth, cast_values, axis=axis)
```

The decorator parses the code of the function and converts it into an intermediate
structure. If it fails, it produces an error message indicating the line where
the error was detected. If it succeeds, the intermediate can be converted into
one ONNX structure of type FunctionProto (a subpart of a model) or ModelProto
(a whole model any runtime can predict with).

- `Hardmax.to_function_proto()` returns a `FunctionProto`,
  annotations are not used as FunctionProto does not requires any information about
  shapes or types.
- `Hardmax.to_model_proto(axis=0)` returns a `ModelProto`,
  annotations are mandatory to specify input and output shapes and types of the model.

**Eager mode**

Eager evaluation mode is mostly use to debug and check intermediate results
are expected.

```python
import numpy as np

v = np.array([[0, 1], [2, 3]], dtype=np.float32)
result = Hardmax(v)
```

More examples can be found in folder [docs/examples](docs/examples).

## Contributing

Every change impacting the converter or the eager evaluation must be unit tested with
class `OnnxScriptTestCase` to ensure both systems do return the same results with the same inputs.
