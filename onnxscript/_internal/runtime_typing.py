"""An internal wrapper for the beartype library.

Decorate a function with `@runtime_typing.checked` to enable runtime
type checking. The decorator is a no-op when the `beartype` library is not
installed.
"""

import warnings

__all__ = [
    "checked",
]

try:
    from beartype import beartype as checked
    from beartype import roar as _roar

    # Beartype warns when we import from typing because the types are deprecated
    # in Python 3.9. But there will be a long time until we can move to using
    # the native container types for type annotations (when 3.9 is the lowest
    # supported version). So we silence the warning.
    warnings.filterwarnings(
        "ignore",
        category=_roar.BeartypeDecorHintPep585DeprecationWarning,
    )
except ImportError:

    def checked(func):  # type: ignore[no-redef]
        return func

except Exception as e:  # pylint: disable=broad-exception-caught
    # Warn errors that are not import errors (unexpected).
    warnings.warn(f"{e}", stacklevel=2)

    def checked(func):  # type: ignore[no-redef]
        return func
