# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import ast
import pprint


class SourceInfo:
    """Information about onnxscript source fragment, used for diagnostic messages."""

    def __init__(self, ast_node: ast.AST, source, code=None):
        from onnxscript.converter import Converter

        assert isinstance(ast_node, ast.AST)
        assert isinstance(source, Converter)
        if hasattr(source, "source"):
            code = source.source
            current_fn = getattr(source, "current_fn", None)
            if current_fn is not None:
                source = getattr(source.current_fn, "name", None)
            else:
                source = None
        if hasattr(ast_node, "lineno"):
            self.ast_obj = ast_node
            self.lineno = ast_node.lineno
        else:
            raise NotImplementedError(
                f"Unable to extract debug information from type {type(ast_node)!r}, "
                f"attributes={pprint.pformat(ast_node.__dict__)}."
            )
        self.source = source
        self.code = None if code is None else code.split("\n")

    def msg(self, text):
        return f"ERROR\n{str(self)}\n    {text}"

    def __str__(self):
        if self.code is None:
            line = ""
        else:
            line = f"    -- line: {self.code[self.lineno - 1]}"
        return f"{self.source}:{int(self.lineno)}{line}"
