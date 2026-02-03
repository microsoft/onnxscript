# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Merging metadata_props"""

from __future__ import annotations

from typing import Callable, Iterable

import onnx_ir as ir

# Utilities for merging metadata properties, represented as strings.
# The merging-logic will take care of special cases like missing metadata or
# empty string metadata, and so the functions defined below need not handle
# special cases like empty string. (This does assume that an empty string is
# the same as no metadata, which is a reasonable assumption for most metadata.)

StringMerger = Callable[[str, str], str]


def overwrite(_: str, new: str) -> str:
    return new


def join(separator: str) -> StringMerger:
    """Creates a StringMerger that joins two strings with the given separator.

    Args:
        separator (str): The separator to use when joining the strings.

    Returns:
        StringMerger: A function that joins two strings with the specified separator.
    """

    def merger(first: str, second: str) -> str:
        return f"{first}{separator}{second}"

    return merger


comma_separator_merger = join(", ")


class MetadataMerger:
    """Merges metadata properties using specified merging logic.

    Attributes:
        mergers: A mapping from metadata property keys to their corresponding merging functions.
        default: The default merging function to use when a specific key does not have a defined merger.
           If None, the first value is used. (Specify `overwrite` to always use the second value.)
    """

    def __init__(
        self, mergers: dict[str, StringMerger], default: StringMerger | None = None
    ) -> None:
        self.mergers = mergers
        self.default = default

    def update_dict(self, updated: dict[str, str], updates: dict[str, str]) -> None:
        """Updates the first metadata property dictionary with values from the second.

        Args:
            updated: The metadata dictionary to be updated.
            updates: The updates metadata dictionary.
        """
        for key, new_value in updates.items():
            if new_value == "":
                continue
            if (key in updated) and ((updated_value := updated[key]) != ""):
                merger = self.mergers.get(key, self.default)
                if merger is not None:
                    updated[key] = merger(updated_value, new_value)
            else:
                updated[key] = new_value

    def copy_merged_metadata(
        self, from_nodes: Iterable[ir.Node], to: ir.Node | Iterable[ir.Node]
    ) -> None:
        """Merges metadata from multiple nodes and assigns it to one or more target nodes.

        Args:
            from_nodes: The source nodes from which to merge metadata.
            to: The target node(s) to which the merged metadata will be assigned.
        """
        if isinstance(to, ir.Node):
            updated = to.metadata_props
            for node in from_nodes:
                self.update_dict(updated, node.metadata_props)
        elif len(to) == 1:
            # Handle single node in iterable case
            target_node = next(iter(to))
            updated = target_node.metadata_props
            for node in from_nodes:
                self.update_dict(updated, node.metadata_props)
        else:
            merged_metadata: dict[str, str] = {}
            for node in from_nodes:
                self.update_dict(merged_metadata, node.metadata_props)
            for target_node in to:
                self.update_dict(target_node.metadata_props, merged_metadata)
