onnxscript: authoring onnx scripts
==================================

ONNXScript is a subset of Python that can be used to author ONNX functions (as well as ONNX models).

.. toctree::
    :maxdepth: 1

    tutorial/index
    open/index
    api/index
    auto_examples/index

*onnxscript* implements two main functionalities:

- a converter which translates a python function into ONNX; the converter analyzes the python
  code using its abstract syntax tree and converts that tree into an ONNX graph
  equivalent to the function.
- a runtime that allows such functions to be executed (in an "eager mode"); this runtime relies on
  *onnxruntime* for executing every operation described in
  `ONNX Operators <https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_).

The runtime is intended to help understand and debug function-definitions, and performance
is not a goal for this mode.

**Example**

The following toy example illustrates how to use onnxscript.

::

    from onnxscript import script
    # We use ONNX opset 15 to define the function below.
    from onnxscript.onnx_opset import opset15 as op

    # We use the script decorator to indicate that the following function is meant
    # to be translated to ONNX.
    @script()
    def MatmulAdd(X, Wt, Bias):
        return op.MatMul(X, Wt) + Bias


The decorator parses the code of the function and converts it into an intermediate
representation. If it fails, it produces an error message indicating the line where
the error was detected. If it succeeds, the corresponding ONNX representation
of the function (a value of type FunctionProto) can be generated as shown below:

::

  fp = MatmulAdd.to_function_proto()  # returns an onnx.FunctionProto


**Eager mode**

Eager evaluation mode is mostly use to debug and check intermediate results
are as expected. The function defined above can be called as below, and this
executes in an eager-evaluation mode.

::

    import numpy as np

    x = np.array([[0, 1], [2, 3]], dtype=np.float32)
    wt = np.array([[0, 1], [2, 3]], dtype=np.float32)
    bias = np.array([0, 1], dtype=np.float32)
    result = MatmulAdd(x, wt, bias)

**License**

onnxscript comes with a `MIT <https://github.com/microsoft/onnx-script/blob/main/LICENSE>`_ license.
