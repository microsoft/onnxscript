---
myst:
  substitutions:
    onnxscript: '*ONNX Script*'
---

# ONNX Script

For instructions on how to install **ONNX Script** refer to [ONNX Script Github Repo](https://github.com/microsoft/onnxscript)


## Overview

{{ onnxscript }} enables developers to naturally author ONNX functions and
models using a subset of Python. It is intended to be:

- **Expressive:** enables the authoring of all ONNX functions.
- **Simple and concise:** function code is natural and simple.
- **Debuggable:** allows for eager-mode evaluation that enables
  debugging the code using standard python debuggers.

Note however that {{ onnxscript }} does **not** intend to support the entirety
of the Python language.

{{ onnxscript }} provides a few major capabilities for authoring and debugging
ONNX models and functions:

- A converter which translates a Python {{ onnxscript }} function into an
  ONNX graph, accomplished by traversing the Python Abstract Syntax Tree
  to build an ONNX graph equivalent of the function.
- A runtime shim that allows such functions to be evaluated
  (in an "eager mode"). This functionality currently relies on
  ONNX Runtime for executing ONNX ops
  and there is a Python-only reference runtime for ONNX underway that
  will also be supported.
- A converter that translates ONNX models and functions into {{ onnxscript }}.
  This capability can be used to fully round-trip ONNX Script ↔ ONNX graph.

Note that the runtime is intended to help understand and debug function definitions.
Performance is not a goal here.


## Example

The following toy example illustrates how to use onnxscript.

```python
from onnxscript import script
# We use ONNX opset 15 to define the function below.
from onnxscript import opset15 as op

# We use the script decorator to indicate that the following function is meant
# to be translated to ONNX.
@script()
def MatmulAdd(X, Wt, Bias):
    return op.MatMul(X, Wt) + Bias
```

The decorator parses the code of the function and converts it into an intermediate
representation. If it fails, it produces an error message indicating the error detected.
If it succeeds, the corresponding ONNX representation of the function
(a value of type FunctionProto) can be generated as shown below:

```python
fp = MatmulAdd.to_function_proto()  # returns an onnx.FunctionProto
```

One can similarly generate an ONNX Model. There are a few differences between
ONNX models and ONNX functions. For example, ONNX models must specify the
type of inputs and outputs (unlike ONNX functions).
The following example illustrates how we can generate an ONNX Model:

```python
from onnxscript import script
from onnxscript import opset15 as op
from onnxscript import FLOAT

@script()
def MatmulAddModel(X : FLOAT[64, 128] , Wt: FLOAT[128, 10], Bias: FLOAT[10]) -> FLOAT[64, 10]:
    return op.MatMul(X, Wt) + Bias

model = MatmulAddModel.to_model_proto() # returns an onnx.ModelProto
```

## Eager mode

Eager evaluation mode is mostly use to debug and check intermediate results
are as expected. The function defined earlier can be called as below, and this
executes in an eager-evaluation mode.

```python
import numpy as np

x = np.array([[0, 1], [2, 3]], dtype=np.float32)
wt = np.array([[0, 1], [2, 3]], dtype=np.float32)
bias = np.array([0, 1], dtype=np.float32)
result = MatmulAdd(x, wt, bias)
```

```{toctree}
:maxdepth: 1

Overview <self>
tutorial/index
api/index
ir/index
auto_examples/index
articles/index
```

## License

onnxscript comes with a [MIT](https://github.com/microsoft/onnxscript/blob/main/LICENSE) license.
