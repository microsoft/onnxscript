# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from onnxscript.backend.onnx_export import export2python


def onnx2script(filename, use_operators=False, rename=False):
    """Exports an onnx graph to a script in following onnx-script syntax.
    The result is printed on the standard output.

    :param filename: onnx file to convert
    :param use_operators: converts a numerical operator into op.Add (False) or keep it (True)
    :param rename: to use shorter name
    """
    code = export2python(filename, use_operators=use_operators, rename=rename)
    print(code)


if __name__ == "__main__":
    import fire
    fire.Fire(dict(onnx2script=onnx2script), sys.argv[1:])
