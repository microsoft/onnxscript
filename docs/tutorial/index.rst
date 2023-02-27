.. include:: ../abbreviations.rst

Tutorial
========

In this tutorial, we illustrate the features supported by |onnxscript| using examples.

**Basic Features**

The example below shows a definition of ``Softplus`` as an |onnxscript| function.

.. literalinclude:: examples/softplus.py

In the above example, expressions such as ``op.Log(...)`` and ``op.Exp(...)`` represent
a call to an ONNX operator (and is translated into an ONNX *NodeProto*). Here, ``op``
serves to identify the *opset* containing the called operator. In this example,
we are using the standard ONNX opset version 15 (as identified by the import
statement ``from onnxscript.onnx_opset import opset15 as op``).

Operators such as ``+`` are supported as syntactic shorthand and are mapped to
a corresponding standard ONNX operator (such as ``Add``) in an appropriate opset.
In the above example, the use of `op` indicates opset 15 is to be used.
If the example does not make use of an opset explicitly in this fashion, it
must be specified via the parameter `default_opset` to the `@script()` invocation.

Similarly, constant literals such as ``1.0`` are allowed as syntactic
shorthand (in contexts such as in the above example) and are implicitly promoted
into an ONNX tensor constant.

**Omitting optional inputs**

Some of the input arguments of ONNX ops are *optional*: for example, the *min*
and *max* inputs of the ``Clip`` operator. The value ``None`` can be used
to indicate an omitted optional input, as shown below, or it can be simply
omitted in the case of trailing inputs:

.. literalinclude:: examples/omitted_input.py

**Specifying attribute-parameter values**

The example below illustrates how to specify attribute-values in a call.
In this example, we call the ONNX operator ``Shape`` and specify the attribute
values for the attributes ``start`` and ``end``.

.. literalinclude:: examples/firstdim.py

In the translation of a call to an ONNX operator, the translator makes use of the
opschema specification of the operator to map the actual parameters to appropriate input
parameters and attribute parameters. Since the ONNX specification does not indicate any
ordering for attribute parameters, it is recommended that attribute parameters be specified
using keyword arguments (aka named arguments).

If the translator does not have an opschema for the called op, it uses the following
strategy to map the actual parameters to appropriate input parameters and attribute parameters.
Keyword arguments of Python are translated into attribute parameters (of ONNX), while positional arguments
are translated into normal value-parameters.
Thus, in the above example, ``X`` is treated as a normal value-parameter for this particular call, while
``start`` and ``end`` are treated as attribute-parameters (when an opschema is unavailable).


**Specifying tensor-valued attributes**

Tensor constants can be created using the ONNX utility ``make_tensor`` and these
can be used as attribute values, as shown below:

.. literalinclude:: examples/tensor_attr.py

The code shown above, while verbose, allows the users to explicitly specify what
they want. The converter, as a convenience, allows users to use numeric constants,
as in the example below, which is translated into the same ONNX representation as
the one above.

.. literalinclude:: examples/tensor_attr_short.py

This works for scalar constants. However, if the user wants to
create an 1-dimensional tensor containing a single value, instead of a 0-dimensional
tensor, they need to do so more explicitly (as in the previous example).

**Semantics: Script Constants**

Attributes in ONNX are required to be constant values. In |onnxscript|, the
expression specified as an attribute is evaluated at script-time (when the
script decorator is evaluated) in the context in which the script function
is defined. The resulting python value is translated into an ONNX attribute,
as long as it has a valid type.

This has several significant semantic implications. First, it allows the use
of arbitrary python code in a context where an attribute-value is expected.
However, the python code must be evaluatable using the global context in
which the script-function is defined. For example, computation using
the parameters of the function itself (even if they are attribute-parameters)
is not permitted.

|onnxscript| assumes that such python-code represents constants.
If the values of the variables used in the expression are
subsequently modified, this modification has no effect on the attribute-value
or the ONNX function/model created. This may potentially cause the behavior
of eager-mode execution to be inconsistent with the ONNX construct generated.

Thus, the example shown above is equivalent to the following:

.. literalinclude:: examples/tensor_attr2.py

**Specifying formal attribute parameters of functions**

The (formal) input parameters of Python functions are treated by the converter as representing
either attribute-parameters or input value parameters (of the generated ONNX function).
However, the converter needs to know for each parameter whether it represents an
attribute or input.
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

**Encoding Higher-Order Ops: Scan**

ONNX allows graph-valued attributes. This is the mechanism used to define (quasi)
higher-order ops, such as *If*, *Loop*, *Scan*, and *SequenceMap*.
While we use Python control-flow to encode *If* and *Loop*, |onnxscript|
supports the use of nested Python functions to represent graph-valued attributes,
as shown in the example below:

.. literalinclude:: examples/scanloop.py

In this case, the function-definition of *Sum* is converted into a graph and used
as the attribute-value when invoking the *Scan* op.

Function definitions used as graph-attributes must satisfy some constraints.
They cannot update outer-scope variables, but may reference them.
(Specifically, the functions cannot use *global* or *nonlocal* declarations.)
They are also restricted from using local-variables with the same name
as outer-scope variables (no shadowing).

There is also an interaction between SSA-renaming and the use of outer-scope
variables inside a function-definition. The following code is invalid, since
the function *CumulativeSum* references the global *g*, which is updated
in between the function-definition and function-use. Note that, from an
ONNX perspective, the two assignments to *g* represent two distinct tensors
*g1* and *g2*.

.. literalinclude:: examples/outerscope_redef_error.py
