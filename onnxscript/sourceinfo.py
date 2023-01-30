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
        self.lineno = ast_node.lineno

    def msg(self, text: str) -> str:
        return f"ERROR\n{self}\n    {text}"

    def __str__(self) -> str:
        if self.code:
            lines = self.code.split("\n")
            line = f" ...{lines[self.lineno - 1]}"
        else:
            line = ""
        function_name = self.function_name or ""
        return f"{function_name}:{int(self.lineno)}{line}"


Formatter = Callable[[ast.AST, str], str]


def formatter(source_code: Optional[str]) -> Formatter:
    def format(node: ast.AST, message: str) -> str:
        return SourceInfo(node, source_code).msg(message)

    return format
