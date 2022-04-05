"""
Define an onnx-script library
=============================
Shows how we can export a python (onnx-script) file containing function definitions
into ONNX proto format.

See file mylib.py for the function definitions.
"""

import mylib

from onnxscript import export_onnx_lib

export_onnx_lib(mylib, "mylib.onnxlib")
