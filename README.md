# ONNX Script

[![CI](https://github.com/microsoft/onnxscript/actions/workflows/main.yaml/badge.svg)](https://github.com/microsoft/onnxscript/actions/workflows/main.yaml)
[![Dev Release](https://aiinfra.visualstudio.com/ONNX%20Converters/_apis/build/status%2Fonnxscript-release-dev?branchName=main&label=Dev%20Release)](https://aiinfra.visualstudio.com/ONNX%20Converters/_build/latest?definitionId=1258&branchName=main)
[![PyPI - Version](https://img.shields.io/pypi/v/onnxscript.svg)](https://pypi.org/project/onnxscript)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnxscript.svg)](https://pypi.org/project/onnxscript)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ONNX Script enables developers to naturally author ONNX functions and
models using a subset of Python. ONNX Script is:

* **Expressive:** enables the authoring of all ONNX functions.
* **Simple and concise:** function code is natural and simple.
* **Debuggable:** allows for eager-mode evaluation that provides for a
  more delightful ONNX model debugging experience.

Note however that ONNX Script does **not** intend to support the entirety
of the Python language.

## Design Overview

ONNX Script provides a few major capabilities for authoring and debugging
ONNX models and functions:

* A converter which translates a Python ONNX Script function into an
  ONNX graph, accomplished by traversing the [Python Abstract Syntax Tree][python-ast] to build an ONNX graph equivalent of the function.

* A converter that operates inversely, translating ONNX models and
  functions into ONNX Script. This capability can be used to fully round-trip
  ONNX Script ↔ ONNX graph.

* A runtime shim that allows such functions to be evaluated
  (in an "eager mode"). This functionality currently relies on
  [ONNX Runtime][onnx-runtime] for executing every [ONNX Operator][onnx-ops],
  and there is a Python-only reference runtime for ONNX underway that
  will also be supported.

  Note that the runtime is intended to help understand and debug function definitions. Performance is not a goal here.

## Installing ONNX Script

```bash
pip install --upgrade onnxscript
```

### Install for Development

```bash
git clone https://github.com/microsoft/onnxscript
cd onnxscript
pip install -r requirements-dev.txt
pip install -e .
```

### Run Unit Tests

```bash
pytest .
```

## Example

```python update-readme
import onnx

# We use ONNX opset 15 to define the function below.
from onnxscript import FLOAT, script
from onnxscript import opset15 as op


# We use the script decorator to indicate that
# this is meant to be translated to ONNX.
@script()
def onnx_hardmax(X, axis: int):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""

    # The type annotation on X indicates that it is a float tensor of
    # unknown rank. The type annotation on axis indicates that it will
    # be treated as an int attribute in ONNX.
    #
    # Invoke ONNX opset 15 op ArgMax.
    # Use unnamed arguments for ONNX input parameters, and named
    # arguments for ONNX attribute parameters.
    argmax = op.ArgMax(X, axis=axis, keepdims=False)
    xshape = op.Shape(X, start=axis)
    # use the Constant operator to create constant tensors
    zero = op.Constant(value_ints=[0])
    depth = op.GatherElements(xshape, zero)
    empty_shape = op.Constant(value_ints=[0])
    depth = op.Reshape(depth, empty_shape)
    values = op.Constant(value_ints=[0, 1])
    cast_values = op.CastLike(values, X)
    return op.OneHot(argmax, depth, cast_values, axis=axis)


# We use the script decorator to indicate that
# this is meant to be translated to ONNX.
@script()
def sample_model(X: FLOAT[64, 128], Wt: FLOAT[128, 10], Bias: FLOAT[10]) -> FLOAT[64, 10]:
    matmul = op.MatMul(X, Wt) + Bias
    return onnx_hardmax(matmul, axis=1)


# onnx_model is an in-memory ModelProto
onnx_model = sample_model.to_model_proto()

# Save the ONNX model at a given path
onnx.save(onnx_model, "sample_model.onnx")

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print(f"The model is invalid: {e}")
else:
    print("The model is valid!")
```

The decorator parses the code of the function, converting it into an
intermediate representation. If it fails, it produces an error message
indicating the line where the error was detected. If it succeeds, the
intermediate representation can be converted into an ONNX graph
structure of type `FunctionProto`:

* `Hardmax.to_function_proto()` returns a `FunctionProto`

### Eager Mode Evaluation

Eager mode is mostly used to debug and validate that intermediate results
are as expected. The function defined above can be called as below,
executing in an eager-evaluation mode:

```python
import numpy as np

v = np.array([[0, 1], [2, 3]], dtype=np.float32)
result = Hardmax(v)
```

More examples can be found in the [docs/examples](docs/examples) directory.

## Development Guidelines

Every change impacting the converter or the eager evaluation must be
unit tested with class `OnnxScriptTestCase` to ensure both systems do
return the same results with the same inputs.

### Coding Style

We use `ruff`, `black`, `isort`, and `mypy` etc. to check code formatting and use `lintrunner` to run all linters.
You can install the dependencies and initialize with

```sh
pip install lintrunner lintrunner-adapters
lintrunner init
```

This will install lintrunner on your system and download all the necessary dependencies to run linters locally.
If you want to see what lintrunner init will install, run `lintrunner init --dry-run`.

To lint local changes:

```bash
lintrunner
```

To format files:

```bash
lintrunner f
```

To lint all files:

```bash
lintrunner --all-files
```

Use `--output oneline` to produce a compact list of lint errors, useful when
there are many errors to fix.

See all available options with `lintrunner -h`.

To read more about lintrunner, see [wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).
To update an existing linting rule or create a new one, modify `.lintrunner.toml` or create a
new adapter following examples in https://github.com/justinchuby/lintrunner-adapters.

## Contributing

We're always looking for your help to improve the product (bug fixes, new features, documentation, etc). Currently ONNX Script is under early and heavy development, so we encourage proposing any major changes by [filing an issue](https://github.com/microsoft/onnxscript/issues) to discuss your idea with the team first.

### Report a Security Issue

**Please do not report security vulnerabilities through public GitHub issues.**

Please refer to our guidance on filing [Security Issues](SECURITY.md).

### Licensing Guidelines

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

### Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos is subject to those third-party's policies.

[python-ast]: https://docs.python.org/3/library/ast.html
[onnx-runtime]: https://onnxruntime.ai
[onnx-ops]: https://github.com/onnx/onnx/blob/main/docs/Operators.md
[onnxfns1A.py]: https://github.com/microsoft/onnxscript/blob/main/onnxscript/tests/models/onnxfns1A.py
