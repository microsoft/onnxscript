# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Source code information used for diagnostic messages."""

from __future__ import annotations

import ast
from typing import Callable


class SourceInfo:
    """Information about onnxscript source fragment, used for diagnostic messages."""

    def __init__(
        self,
        ast_node: ast.AST,
        *,
        code: str | None = None,
        function_name: str | None = None,
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


def formatter(source_code: str | None) -> Formatter:
    def format(node: ast.AST, message: str) -> str:
        return SourceInfo(node, code=source_code).msg(message)

    return format
