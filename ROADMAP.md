# onnx-script roadmap

onnx-script is being actively developed. Next section recalls the main milestones
when they are approximatively expected.

- **2022-04-06**: support standard onnx operators, one opset,
  a converter, eager evaluation, unit tests.
- **2022-04-20**: onnxscript decorator to detect as soon as possible
  wrong syntax, discrepancies between converter and eager evaluation
- **2022-05-03**: documentation and tutorial, nested if, multiple opsets, implementation of many onnx operators with onnx primitives
- **2022-05-17**: implementation of FFT functions, unidimensional and multidimensional
- **2022-05-31**: robustness, get feedback from other member of the team and fix bugs
- **2022-06-14**: first version of a library of ONNX functions`1`
- **2022-06-28**: first public release on PyPi

Notes:

- `1`: with a library of ONNX functions, an ONNX model can be defined with a shared ONNX file
  defining many ONNX functions (or `FunctionProto`) and a second file containing the model itself.
  Currently, if a model uses several times a FFT function, a converter can insert many times
  the same combination of nodes implementing FFT function or insert a node FFT refering
  to a function added to the model. Functions could be part a separate file shared by
  multiple models leading to a size reduction.
