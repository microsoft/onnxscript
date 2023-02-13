<table>
<tr>
<td>⚠️</td>
<td>
<strong>NOTE:</strong> ONNX Script is in <strong><em>very early
and active development</em></strong> and the team anticipates
<strong><em>breaking changes</em></strong> as the project evolves.
ONNX Script is <strong><ins>not ready for production</ins></strong>,
but early feedback is welcome.
</td>
<td>⚠️</td>
</tr>
</table>

----

# ONNX Script

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
pip install onnx onnxruntime pytest
git clone https://github.com/microsoft/onnx-script
cd onnx-script
pip install -e .
```

### Using Experimental ONNX and ONNX Runtime Packages

Some ONNX Script features depend on changes to ONNX that are not yet
available in a released ONNX package.

To enable support for these features, experimental dependency packages
must be installed:

```bash
pip uninstall onnx onnxruntime

pip install --pre -f https://onnxruntimepackages.z14.web.core.windows.net/onnx-function-experiment.html onnx-function-experiment

pip install --pre -f https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime-function-experiment.html ort-function-experiment-nightly
```

With experimental ONNX, one can write a script function with optional
attributes. Examples are in [onnxfns1A.py][onnxfns1A.py]. To validate
that experimental features are enabled:

```bash
pytest onnxscript/tests/functions/onnxfns1A_test.py
```

### Run Unit Tests

```bash
pytest onnxscript
```

## Example

```python
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, INT64, FLOAT

# We use ONNX opset 15 to define the function below.
from onnxscript import opset15 as op

# We use the script decorator to indicate that
# this is meant to be translated to ONNX.
@script()
def Hardmax(X: FLOAT[...], axis: int = 0) -> FLOAT[...]:
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
    empty_shape = op.Constant(value_ints=[])
    depth = op.Reshape(depth, empty_shape)
    # Constant Array must be defined with function
    # make_tensor from onnx package.
    values = op.Constant(value=make_tensor(
      'cst01', TensorProto.FLOAT, [2], [0, 1]))
    cast_values = op.CastLike(values, X)
    return op.OneHot(argmax, depth, cast_values, axis=axis)
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
lintrunner -m main
```

To lint all files:

```bash
lintrunner --all-files
```

To format files:

```bash
lintrunner f -m main
```

Use `--output oneline` to produce a compact list of lint errors, useful when
there are many errors to fix.

See all available options with `lintrunner -h`.

To read more about lintrunner, see [wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).
To update an existing linting rule or create a new one, modify `.lintrunner.toml` or create a
new adapter following examples in https://github.com/justinchuby/lintrunner-adapters.

## Contributing

We're always looking for your help to improve the product (bug fixes, new features, documentation, etc). Currently ONNX Script is under early and heavy development, so we encourage proposing any major changes by [filing an issue](https://github.com/microsoft/onnx-script/issues) to discuss your idea with the team first.

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
[onnxfns1A.py]: https://github.com/microsoft/onnx-script/blob/main/onnxscript/tests/models/onnxfns1A.py
