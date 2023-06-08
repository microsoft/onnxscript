# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import List, Optional

from onnxscript.diagnostics.infra.sarif import _property_bag


@dataclasses.dataclass
class Message:
    """Encapsulates a message intended to be read by the end user."""

    arguments: Optional[List[str]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "arguments"}
    )
    id: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "id"}
    )
    markdown: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "markdown"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    text: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "text"}
    )


# flake8: noqa
