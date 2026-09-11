# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-ancestors
# --------------------------------------------------------------------------

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, Sequence, runtime_checkable

import libcst as cst
import libcst.matchers as cstm
import libcst.metadata as cstmeta

__all__ = [
    "format_code",
    "make_name",
    "make_import_alias",
    "make_const_expr",
    "RemoveUnusedImportsTransformer",
    "CstCodeGenerator",
]


def format_code(path: Path | None, code: bytes) -> bytes:
    try:
        import ufmt

        if path is None:
            path = Path(os.curdir)

        return ufmt.ufmt_bytes(
            path,
            code,
            black_config=ufmt.util.make_black_config(path),
            usort_config=ufmt.UsortConfig.find(path),
        )
    except ImportError:
        return code


def make_name(name: str) -> cst.Attribute | cst.Name:
    tokens = name.split(".")
    expr: cst.Name | cst.Attribute = cst.Name(tokens[0])
    for attr in tokens[1:]:
        expr = cst.Attribute(expr, cst.Name(attr))
    return expr


def make_import_alias(name: str, asname: str | None = None) -> cst.ImportAlias:
    return cst.ImportAlias(
        name=make_name(name),
        asname=cst.AsName(cst.Name(asname)) if asname else None,
    )


def make_const_expr(const: str | int | float) -> cst.BaseExpression:
    negate = False
    val: cst.Float | cst.Integer

    if isinstance(const, str):
        return cst.SimpleString('"' + const.replace('"', '\\"') + '"')
    elif isinstance(const, int):
        val = cst.Integer(str(abs(const)))
        negate = const < 0
    elif isinstance(const, float):
        val = cst.Float(str(abs(const)))
        negate = const < 0
    else:
        raise NotImplementedError(repr(const))

    if negate:
        return cst.UnaryOperation(
            operator=cst.Minus(),
            expression=val,
        )

    return val


@dataclass
class ImportAlias:
    name: str
    alias: str | None = None

    def to_cst(self) -> cst.ImportAlias:
        return cst.ImportAlias(
            make_name(self.name), cst.AsName(cst.Name(self.alias)) if self.alias else None
        )


@dataclass
class Import:
    module: ImportAlias

    def to_cst(self) -> cst.Import:
        return cst.Import(names=[self.module.to_cst()])


@dataclass
class ImportFrom:
    module: str
    names: list[ImportAlias]

    def to_cst(self) -> cst.ImportFrom:
        return cst.ImportFrom(
            module=make_name(self.module),
            names=[name.to_cst() for name in self.names],
        )


@runtime_checkable
class ScopeAnalyzer(Protocol):
    def analyze_scopes(self, scopes: set[cstmeta.Scope]):
        pass


class RemoveUnusedImportsTransformer(cst.CSTTransformer, ScopeAnalyzer):
    def __init__(self):
        self.__unused_imports: dict[cst.Import | cst.ImportFrom, set[str]] = defaultdict(set)

    def is_unused_allowed(self, node: cst.Import | cst.ImportFrom, name: str):
        return name == "annotations" and cstm.matches(
            node, cstm.ImportFrom(module=cstm.Name("__future__"))
        )

    def analyze_scopes(self, scopes: set[cstmeta.Scope]):
        for scope in scopes:
            for assignment in scope.assignments:
                if (
                    isinstance(assignment, cstmeta.Assignment)
                    and isinstance(node := assignment.node, (cst.Import, cst.ImportFrom))
                    and len(assignment.references) == 0
                    and not self.is_unused_allowed(node, assignment.name)
                ):
                    self.__unused_imports[node].add(assignment.name)

    def __leave_import_alike(
        self,
        original_node: cst.Import | cst.ImportFrom,
        updated_node: cst.Import | cst.ImportFrom,
    ) -> cst.Import | cst.ImportFrom | cst.RemovalSentinel:
        if original_node not in self.__unused_imports or isinstance(
            updated_node.names, cst.ImportStar
        ):
            return updated_node

        names_to_keep: list[cst.ImportAlias] = []

        for name in updated_node.names:
            if name.asname is not None:
                if not isinstance(name.asname, cst.Name):
                    continue
                name_value = name.asname.name.value
            else:
                name_value = name.name.value
            if name_value not in self.__unused_imports[original_node]:
                names_to_keep.append(name.with_changes(comma=cst.MaybeSentinel.DEFAULT))

        if len(names_to_keep) == 0:
            return cst.RemoveFromParent()

        return updated_node.with_changes(names=names_to_keep)

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import):
        return self.__leave_import_alike(original_node, updated_node)

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom):
        return self.__leave_import_alike(original_node, updated_node)


class CstCodeGenerator:
    def __init__(self):
        self.__imports: Final[list[Import | ImportFrom]] = []

    def add_import(self, module: str, alias: str | None = None):
        if not any(
            isinstance(imp, Import) and imp.module.name == module and imp.module.alias == alias
            for imp in self.__imports
        ):
            self.__imports.append(Import(ImportAlias(module, alias)))

    def add_import_from(self, module: str, name: str, alias: str | None = None):
        for imp in self.__imports:
            if isinstance(imp, ImportFrom) and imp.module == module:
                for existing in imp.names:
                    if existing.name == name and existing.alias == alias:
                        return
                imp.names.append(ImportAlias(name, alias))
                return
        self.__imports.append(ImportFrom(module, [ImportAlias(name, alias)]))

    def make_import_statements(self) -> Sequence[cst.SimpleStatementLine]:
        return [cst.SimpleStatementLine(body=[imp.to_cst()]) for imp in self.__imports]

    def apply_transformers(
        self, module: cst.Module, transformers: Sequence[cst.CSTTransformer]
    ) -> cst.Module:
        for transformer in transformers:
            wrapper = cstmeta.MetadataWrapper(module)
            if isinstance(transformer, ScopeAnalyzer):
                scopes = {
                    scope
                    for scope in wrapper.resolve(cstmeta.ScopeProvider).values()
                    if scope is not None
                }
                transformer.analyze_scopes(scopes)
            module = wrapper.visit(transformer)
        return module
