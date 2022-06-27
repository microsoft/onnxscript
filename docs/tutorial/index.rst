Tutorial
========

In this tutorial, we illustrate the features supported by onnxscript using examples.

**Basic Features**

::

    from onnxscript import script
    # We use ONNX opset 15 to define the function below.
    from onnxscript.onnx import opset15 as op

    # We use the script decorator to indicate that the following function is meant
    # to be translated to ONNX.
    @script()
    def Softplus(X):
        return op.Log(op.Exp(X) + 1.0)

In the above example, expressions such as ``op.Log(...)`` and ``op.Exp(...)`` represent
a call to an ONNX operator (and is translated into an ONNX *NodeProto*). Here, ``op``
serves to identify the *opset* containing the called operator. In this example,
we are using the standard ONNX opset version 15 (as identified by the import
statement ``from onnxscript.onnx import opset15 as op``).

Operators such as ``+`` are supported as syntactic shorthand and are mapped to
a corresponding standard ONNX operator (such as ``Add``) in the default opset.

Similarly, constant literals such as ``1.0`` are allowed as syntactic
shorthand (in contexts such as in the above example) and are implicitly promoted
into an ONNX tensor constant.

**Specifying attribute-parameter values**

The example below illustrates how to specify attribute-values in a call.
In this example, we call the ONNX operator ``Shape`` and specify the attribute
values for the attributes ``start`` and ``end``.
::

    @script()
    def FirstDim(X):
        return op.Shape(X, start=0, end=1)

In the translation of a call to an ONNX operator, keyword or named arguments are translated into
attribute parameters, while positional arguments are translated into normal value-parameters.
Thus, ``X`` is treated as a normal value-parameter for this particular call.

**Specifying formal attribute parameters of functions**

The (formal) input parameters of Python functions are treated by the converter as representing
either attribute parameters or input value parameters. The converter uses the type annotation
on the formal input parameters to make this distinction. Thus, in the example below,
`alpha` is treated as an attribute parameter (because of its `float` type annotation).
::

    @script()
    def LeakyRelu(X, alpha: float):
        return op.Where(X < 0.0, alpha * X, X)
    
As illustrated in the above example, when an attribute-parameter is used in a context
requiring a value-parameter, the converter will automatically convert the attribute
into a tensor-value. Specifically, in the sub-expression ``alpha * X``, the attribute
parameter ``alpha`` is used as a value-parameter of the call to the ``Mul`` op (denoted
by the ``*``) and is automatically converted.
