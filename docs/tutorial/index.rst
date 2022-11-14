Tutorial
========

In this tutorial, we illustrate the features supported by onnxscript using examples.

**Basic Features**

The example below shows a definition of ``Softplus`` as an *onnxscript* function.

.. literalinclude:: examples/softplus.py

In the above example, expressions such as ``op.Log(...)`` and ``op.Exp(...)`` represent
a call to an ONNX operator (and is translated into an ONNX *NodeProto*). Here, ``op``
serves to identify the *opset* containing the called operator. In this example,
we are using the standard ONNX opset version 15 (as identified by the import
statement ``from onnxscript.onnx_opset import opset15 as op``).

Operators such as ``+`` are supported as syntactic shorthand and are mapped to
a corresponding standard ONNX operator (such as ``Add``) in the default opset.

Similarly, constant literals such as ``1.0`` are allowed as syntactic
shorthand (in contexts such as in the above example) and are implicitly promoted
into an ONNX tensor constant.

**Omitting optional inputs**

Some of the input arguments of ONNX ops are *optional*: for example, the *min*
and *max* inputs of the ``Clip`` operator. The value ``None`` can be used
to indicate an omitted optional input, as shown below:

.. literalinclude:: examples/omitted_input.py

**Specifying attribute-parameter values**

The example below illustrates how to specify attribute-values in a call.
In this example, we call the ONNX operator ``Shape`` and specify the attribute
values for the attributes ``start`` and ``end``.

.. literalinclude:: examples/firstdim.py

In the translation of a call to an ONNX operator, keyword arguments (aka named arguments)
of Python are translated into attribute parameters (of ONNX), while positional arguments
are translated into normal value-parameters.
Thus, ``X`` is treated as a normal value-parameter (in ONNX) for this particular call, while
``start`` and ``end`` are treated as attribute-parameters.

**Specifying tensor-valued attributes**

Tensor constants can be created using the ONNX utility ``make_tensor`` and these
can be used as attribute values, as shown below:

.. literalinclude:: examples/tensor_attr.py

**Semantics: Script Constants**

Attributes in ONNX are required to be constant values. In *onnxscript*, the
expression specified as an attribute is evaluated at script-time (when the
script decorator is evaluated) in the context in which the script function
is defined. The resulting python value is translated into an ONNX attribute,
as long as it has a valid type. Note that this changes the semantics of the
function. Even if the values of the variables used in the expression are
subsequently modified, this modification has no effect on the attribute-value.
Thus, the example shown above is equivalent to the following:

.. literalinclude:: examples/tensor_attr2.py

This behavior may be different in eager-mode execution.
*TODO*: Need some appropriate warning/error message in such situations.

**Specifying formal attribute parameters of functions**

The (formal) input parameters of Python functions are treated by the converter as representing
either attribute-parameters or input value parameters (of the generated ONNX function).
The converter uses the type annotation on the formal input parameters to make this distinction.
Thus, in the example below, ``alpha`` is treated as an attribute parameter (because of its ``float``
type annotation).

.. literalinclude:: examples/leakyrelu.py

As illustrated in the above example, when an attribute-parameter is used in a context
requiring a value-parameter, the converter will automatically convert the attribute
into a tensor-value. Specifically, in the sub-expression ``alpha * X``, the attribute
parameter ``alpha`` is used as a value-parameter of the call to the ``Mul`` op (denoted
by the ``*``) and is automatically converted.

**Conditional statements**

The function definition below illustrates the use of conditionals.

.. literalinclude:: examples/dropout.py

The use of conditional statements requires that any variable that is *used* in the code
has a *definition* of the same variable along all possible paths to the use.

**Loops**

ONNX implements a loop operator doing a fixed number of iterations
and/or a loop breaking if a condition is not true anymore.
First example below illustrates the use of the most simple case:
a fixed number of iterations.

.. literalinclude:: examples/forloop.py

Second example shows a loop breaking if a condition is not true
any more.

.. literalinclude:: examples/whileloop.py

Third example mixes both types of loops.

.. literalinclude:: examples/forwhileloop.py
