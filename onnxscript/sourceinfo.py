# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

"""Source code information used for diagnostic messages."""

from __future__ import annotations

import ast
from typing import Callable, Optional


class SourceInfo:
    """Information about onnxscript source fragment, used for diagnostic messages."""

    def __init__(
        self,
        ast_node: ast.AST,
        code: Optional[str] = None,
        function_name: Optional[str] = None,
    ):
        self.ast_node = ast_node
        self.code = code
        self.function_name = function_name

    @property
    def lineno(self):
        return self.ast_node.lineno

    def msg(self, error_message: str) -> str:
        lineno = self.lineno
        if self.function_name:
            source_loc = f"Function '{self.function_name}', line {lineno}"
        else:
            source_loc = f"Line {lineno}"

        if self.code:
            lines = self.code.split("\n")
            line = lines[lineno - 1]
            marker_prefix = " " * (self.ast_node.col_offset)
            source_line = f"{line}\n{marker_prefix}^\n"
        else:
            source_line = ""

        return f"ERROR: {error_message}\nat: {source_loc}\n{source_line}"

    def __str__(self) -> str:
        raise ValueError("Cannot happen!")


Formatter = Callable[[ast.AST, str], str]


def formatter(source_code: Optional[str]) -> Formatter:
    def format(node: ast.AST, message: str) -> str:
        return SourceInfo(node, source_code).msg(message)

    return format
