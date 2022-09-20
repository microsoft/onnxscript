# ONNXScript

ONNXScript is a subset of Python that can be used to author ONNX functions (as well as ONNX models).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Installation steps

```bash
pip install onnx onnxruntime
git clone https://github.com/microsoft/onnx-script.git
pip install -e .
```

A note to use experimental ONNX and ORT packages:

Some of onnx-script functionalities depend on changes in ONNX that are not in official ONNX package yet.
In order to work with those functionalities, one needs to:

```bash
pip uninstall onnx onnxruntime
pip install --pre -f https://onnxruntimepackages.blob.core.windows.net/$web/onnx-function-experiment.html onnx-function-experiment
pip install --pre -f https://onnxruntimepackages.blob.core.windows.net/$web/onnxruntime-function-experiment.html ort-function-experiment-nightly
```

With experimental ONNX, one can write a script function with optional attributes. Examples are in [onnxfns1A.py](https://github.com/microsoft/onnx-script/blob/main/onnxscript/test/models/onnxfns1A.py). To validate that experimental features are enabled:

```bash
pytest onnxscript\test\functions\onnxfns1A_test.py
```

## Run unit tests

```bash
pytest onnxscript/test
```

## Design

*onnxscript* implements two main functionalities:

- a converter which translates a python function into ONNX; the converter analyzes the python
  code using its abstract syntax tree and converts that tree into an ONNX graph
  equivalent to the function.
- a runtime that allows such functions to be executed (in an "eager mode"); this runtime relies on
  *onnxruntime* for executing every operation described in
  [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

The runtime is intended to help understand and debug function-definitions, and performance
is not a goal for this mode.

## Example

Let's write a function in file **onnx_fct.py**. The script may contain multiple functions.

```python
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script
from onnxscript.onnx_types import INT64, FLOAT

# We use ONNX opset 15 to define the function below.
from onnxscript.onnx_opset import opset15 as op

# We use the script decorator to indicate that this is meant to be translated to ONNX.
@script()
def Hardmax(X: FLOAT[...], axis: int = 0) -> FLOAT[...]:
    # The type annotation on X indicates that it is a float tensor of unknown rank.
    # The type annotation on axis indicates that it will be treated as an int attribute in ONNX.

    # Invoke ONNX opset 15 op ArgMax
    # Use unnamed arguments for ONNX input parameters, and named arguments for ONNX
    # attribute parameters.
    argmax = op.ArgMax(X, axis=axis, keepdims=False)

    xshape = op.Shape(X, start=axis)
    # use the Constant operator to create constant tensors
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
representation. If it fails, it produces an error message indicating the line where
the error was detected. If it succeeds, the intermediate representation
can be converted into an ONNX structure of type FunctionProto as shown below.

- `Hardmax.to_function_proto()` returns a `FunctionProto`,

**Eager mode**

Eager evaluation mode is mostly use to debug and check intermediate results
are as expected. The function defined above can be called as below, and this
executes in an eager-evaluation mode.

```python
import numpy as np

v = np.array([[0, 1], [2, 3]], dtype=np.float32)
result = Hardmax(v)
```

More examples can be found in folder [docs/examples](docs/examples).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Microsoft CLA](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

**Unit Test**

Every change impacting the converter or the eager evaluation must be unit tested with
class `OnnxScriptTestCase` to ensure both systems do return the same results with the same inputs.

**Code Style**

We use flake8, black, isort, and mypy to check code format. You can find their configuration in .flake8 and pyproject.toml, and run lint check by:

```bash
./tools/style.sh
```
