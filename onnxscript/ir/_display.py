# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Internal utilities for displaying the intermediate representation of a model.

NOTE: All third-party imports should be scoped and imported only when used to avoid
importing unnecessary dependencies.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

from typing import Any

_LONG_TEXT_LIMIT = 3000


def require_rich() -> Any:
    """Raise an ImportError if rich is not installed."""
    try:
        import rich
    except ImportError:
        return None
    return rich


class PrettyPrintable:
    def display(self, *, page: bool | None = None) -> None:
        """Pretty print the object.

        Args:
            page: Whether to page the output if it is too long.
        """
        rich = require_rich()
        text = str(self)

        if rich is None:
            print(text)
            # Color print this message
            print(
                f"\n\n\u001b[36mTip: Install the rich library with 'pip install rich' to pretty print this {self.__class__.__name__}.\u001b[0m"
            )
            return

        if page is None and len(text) > _LONG_TEXT_LIMIT:
            # By default, page the output if it is too long
            page = True
        if page:
            import rich.console
            import rich.syntax

            console = rich.console.Console()
            syntax = rich.syntax.Syntax(text, "cpp", theme="ansi_light")
            with console.pager(styles=True):
                console.print(syntax)
        else:
            rich.print(text)
