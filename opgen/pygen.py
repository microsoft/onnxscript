# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=W0613

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from enum import Enum
from textwrap import TextWrapper, dedent
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    Set,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")
TNode = TypeVar("TNode", bound="Node")
TExpr = TypeVar("TExpr", bound="Expr")
NoneType = type(None)


def _assert_instance(instance, expected_type: Union[Type, Tuple[Type, ...]]):
    if not isinstance(instance, expected_type):
        raise TypeError(f"expected: {expected_type!r}; actual: {instance!r}")


__end_of_sequence = StopIteration()


def first_or_none(seq: Iterable[T]) -> Optional[T]:
    return next(iter(seq), None)


def first(seq: Iterable[T]) -> T:
    return next(iter(seq))


def single_or_none(seq: Iterable[T]) -> Optional[T]:
    i = iter(seq)
    value = next(i, __end_of_sequence)
    if value is __end_of_sequence:
        return None
    if next(i, __end_of_sequence) is not __end_of_sequence:
        raise StopIteration("sequence contains more than one element")
    return value


class Role:
    def __init__(self, name: str):
        _assert_instance(name, str)
        self.name = name

    def __str__(self):
        return self.name


class NodePredicate:
    always: NodePredicate

    def __init__(
        self,
        role: Optional[Role] = None,
        type_: Optional[Type[TNode]] = None,
        func: Optional[Callable[[Node], bool]] = None,
    ):
        _assert_instance(role, (Role, NoneType))
        _assert_instance(type_, (type, NoneType))
        self.role = role
        self.type = type_
        self.func = func

    def matches(self, node: Node):
        _assert_instance(node, Node)
        matches = True
        if self.role:
            matches &= node.role is self.role
        if self.type:
            matches &= isinstance(node, self.type)
        if self.func and matches:
            matches &= self.func(node)
        return matches


NodePredicate.always = NodePredicate()


class Node(ABC):
    # pylint: disable=W0212

    def __init__(self):
        self._role: Optional[Role] = None
        self._parent: Optional[Node] = None
        self._prev_sibling: Optional[Node] = None
        self._next_sibling: Optional[Node] = None
        self._first_child: Optional[Node] = None
        self._last_child: Optional[Node] = None
        self.leading_trivia: Optional[str] = None
        self.trailing_trivia: Optional[str] = None

    @property
    def qual_name(self) -> str:
        names = []
        for ancestor in self.get_ancestors(and_self=True):
            names.insert(0, ancestor.name if hasattr(ancestor, "name") else "<unnamed>")
        return ".".join(names)

    @property
    def parent_module(self) -> Optional[Module]:
        return first_or_none(self.get_ancestors_of_type(Module))

    @property
    def parent(self):
        return self._parent

    @property
    def role(self):
        return self._role

    @property
    def prev_sibling(self):
        return self._prev_sibling

    @property
    def next_sibling(self):
        return self._next_sibling

    @property
    def first_child(self):
        return self._first_child

    @property
    def last_child(self):
        return self._last_child

    @property
    def has_children(self):
        return self._first_child is not None

    @property
    def children(self) -> Iterable[Node]:
        current_node = self.first_child
        while current_node is not None:
            # save next then yield to allow removing/replacing nodes while iterating
            next_node = current_node.next_sibling
            yield current_node
            current_node = next_node

    def get_children(self, predicate: NodePredicate) -> Iterable[Node]:
        _assert_instance(predicate, NodePredicate)
        yield from filter(predicate.matches, self.children)

    def get_children_in_role(self, role: Role):
        _assert_instance(role, Role)
        return self.get_children(NodePredicate(role=role))

    def get_children_of_type(self, type_: Type[TNode]) -> Iterable[TNode]:
        _assert_instance(type_, type)
        return self.get_children(NodePredicate(type_=type_))

    def get_ancestors(
        self, predicate: Optional[NodePredicate] = None, and_self=False
    ) -> Iterable[Node]:
        current_node = self if and_self else self.parent
        while current_node:
            # save next then yield to allow removing/replacing nodes while iterating
            next_node = current_node.parent
            if predicate is None or predicate.matches(current_node):
                yield current_node
            current_node = next_node

    def get_ancestors_in_role(self, role: Role, and_self=False):
        _assert_instance(role, Role)
        return self.get_ancestors(NodePredicate(role=role), and_self=and_self)

    def get_ancestors_of_type(self, type_: Type[TNode], and_self=False) -> Iterable[TNode]:
        _assert_instance(type_, type)
        return self.get_ancestors(NodePredicate(type_=type_), and_self=and_self)

    def _set_parent(self, child: Node):
        if child._parent is not None:
            raise ValueError(f"node is already has a parent: {child.parent!r}")
        child._parent = self

    def _get_single_child(self, role: Role) -> Optional[Node]:
        return first_or_none(self.get_children_in_role(role))

    def _set_single_child(self, node: Node, role: Role):
        current_node = self._get_single_child(role)
        if current_node:
            current_node.replace(node)
        else:
            self.append_child(node, role)

    def append_children(self, children: Optional[Union[Node, Iterable[Node]]], role: Role):
        _assert_instance(role, Role)
        if children is None:
            return

        if isinstance(children, Node):
            self.append_child(children, role)
        else:
            for child in children:
                self.append_child(child, role)

    def append_child(self, child: Node, role: Role):
        _assert_instance(role, Role)
        if child is None:
            return
        _assert_instance(child, Node)

        self._set_parent(child)
        child._role = role

        if self._first_child is None:
            self._last_child = child
            self._first_child = child
        else:
            self._last_child._next_sibling = child
            child._prev_sibling = self._last_child
            self._last_child = child

    def insert_child_before(self, next_sibling: Optional[Node], child: Node, role: Role):
        _assert_instance(next_sibling, (Node, type(None)))
        _assert_instance(child, Node)
        _assert_instance(role, Role)

        if next_sibling is None:
            self.append_child(child, role)
            return

        self._set_parent(child)
        child._role = role
        child._next_sibling = next_sibling
        child._prev_sibling = next_sibling._prev_sibling

        if next_sibling._prev_sibling is None:
            self._first_child = child
        else:
            next_sibling._prev_sibling._next_sibling = child

        next_sibling._prev_sibling = child

    def prepend_child(self, child: Node, role: Role):
        _assert_instance(child, Node)
        _assert_instance(role, Role)
        self.insert_child_before(self.first_child, child, role)

    def remove(self):
        if self._prev_sibling is not None:
            self._prev_sibling._next_sibling = self._next_sibling
        else:
            self._parent._first_child = self._next_sibling

        if self._next_sibling is not None:
            self._next_sibling._prev_sibling = self._prev_sibling
        else:
            self._parent._last_child = self._prev_sibling

        self._parent = None
        self._role = None
        self._prev_sibling = None
        self._next_sibling = None

        return self

    def replace(self, new_node: Optional[Node]):
        if new_node is None:
            self.remove()
            return

        if new_node is self:
            return

        if self.parent is None:
            raise ValueError("cannot replace root node")

        _assert_instance(new_node, Node)

        if new_node.parent is not None:
            if self in new_node.ancestors:
                new_node.remove()
            else:
                raise ValueError(f"node is used in another tree: {new_node!r}")

        new_node._parent = self._parent
        new_node._role = self._role
        new_node._prev_sibling = self._prev_sibling
        new_node._next_sibling = self._next_sibling

        if self._prev_sibling is None:
            self._parent._first_child = new_node
        else:
            self._prev_sibling._next_sibling = new_node

        if self._next_sibling is None:
            self._parent._last_child = new_node
        else:
            self._parent._prev_sibling = new_node

        self._parent = None
        self._role = None
        self._prev_sibling = None
        self._next_sibling = None

    @abstractmethod
    def accept(self, visitor: Visitor):
        pass

    def _dispatch_visit(self, dispatch: Callable[[TNode, VisitKind], bool]):
        visitor = dispatch.__self__
        visitor.enter(self)
        if dispatch(self) is True:
            for child in self.children:
                child.accept(dispatch.__self__)
            visitor.leave(self)
            dispatch(self)
        else:
            visitor.leave(self)
        visitor.finish(self)

    def __str__(self):
        buffer = io.StringIO()
        self.accept(PythonWriter(buffer))
        return buffer.getvalue()


class Expr(Node, ABC):
    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_expr)


class ThunkExpr(Expr):
    def __init__(self, code: str):
        super().__init__()
        self.code = code

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_thunk_expr)


class Name(Expr):
    def __init__(self, identifier: str):
        super().__init__()
        self.identifier = identifier

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_name)


class Constant(Expr):
    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_constant)


class ExprList(Expr, Generic[TExpr], ABC):
    class Roles:
        Elements = Role("ExprList.Elements")

    def __init__(self, *elements: TExpr):
        super().__init__()
        self.append_children(elements, ExprList.Roles.Elements)

    @property
    def elements(self) -> Iterable[TExpr]:
        return self.get_children_in_role(ExprList.Roles.Elements)

    def append_element(self, element: TExpr):
        _assert_instance(element, Expr)
        self.append_child(element, ExprList.Roles.Elements)


class BinOp(Expr):
    class Roles:
        Left = Role("BinOp.Left")
        Right = Role("BinOp.Right")

    def __init__(self, left: Expr, op: str, right: Expr):
        super().__init__()
        self.append_child(left, BinOp.Roles.Left)
        self.op = op
        self.append_child(right, BinOp.Roles.Right)

    @property
    def left(self):
        return first(self.get_children_in_role(BinOp.Roles.Left))

    @property
    def right(self):
        return first(self.get_children_in_role(BinOp.Roles.Right))

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_binop)


class Subscript(Expr):
    class Roles:
        Value = Role("Subscript.Value")
        Slice = Role("Subscript.Slice")

    def __init__(self, value: Expr, slice: Expr):
        super().__init__()
        self.append_child(value, Subscript.Roles.Value)
        self.append_child(slice, Subscript.Roles.Slice)

    @property
    def value(self):
        return first(self.get_children_in_role(Subscript.Roles.Value))

    @property
    def slice(self):
        return first(self.get_children_in_role(Subscript.Roles.Slice))

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_subscript)


class Starred(Expr):
    class Roles:
        Expr = Role("Starred.Expr")

    def __init__(self, expr: Expr):
        super().__init__()
        self.append_child(expr, Starred.Roles.Expr)

    @property
    def expr(self):
        return first(self.get_children_in_role(Starred.Roles.Expr))

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_starred)


class Call(Expr):
    class Roles:
        Func = Role("Call.Func")
        Args = Role("Call.Args")

    def __init__(self, func: Expr, *args: Expr):
        super().__init__()
        _assert_instance(func, Expr)
        self.append_child(func, Call.Roles.Func)
        self.append_children(args, Call.Roles.Args)

    @property
    def func(self) -> Expr:
        return first(self.get_children_in_role(Call.Roles.Func))

    @property
    def args(self) -> Iterable[Expr]:
        return self.get_children_in_role(Call.Roles.Args)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_call)


class Lambda(Expr):
    class Roles:
        Args = Role("Lambda.Args")
        Body = Role("Lambda.Body")

    def __init__(self, body: Expr, *args: Arg):
        super().__init__()
        _assert_instance(body, Expr)
        self.append_child(body, Lambda.Roles.Body)
        self.append_children(args, Lambda.Roles.Args)

    @property
    def body(self) -> Expr:
        return first(self.get_children_in_role(Lambda.Roles.Body))

    @property
    def args(self) -> Iterable[Expr]:
        return self.get_children_in_role(Lambda.Roles.Args)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_lambda)


class TupleExpr(ExprList):
    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_tuple_expr)


class ListExpr(ExprList):
    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_list_expr)


class SetExpr(ExprList):
    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_set_expr)


class DictElem(Expr):
    class Roles:
        Key = Role("DictElem.Key")
        Value = Role("DictElem.Value")

    def __init__(self, key: Expr, value: Expr):
        super().__init__()
        _assert_instance(key, Expr)
        _assert_instance(value, Expr)
        self.append_child(key, DictElem.Roles.Key)
        self.append_child(value, DictElem.Roles.Value)

    @property
    def key(self) -> Expr:
        return first(self.get_children_in_role(DictElem.Roles.Key))

    @property
    def value(self) -> Expr:
        return first(self.get_children_in_role(DictElem.Roles.Value))

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_dict_elem)


class DictExpr(ExprList[DictElem]):
    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_dict_expr)


class TypeRef(Expr):
    class Roles:
        TypeArgs = Role("TypeRef.TypeArgs")

    def __init__(
        self,
        module: Optional[str],
        name: str,
        *typeargs: TypeRef,
        default_value: Optional[Constant] = None,
    ):
        super().__init__()
        self.module = module
        self.name = name
        self.default_value = default_value or Constant(None)
        self.imported_by: Optional[ImportBase] = None
        self.append_children(typeargs, TypeRef.Roles.TypeArgs)

    @property
    def typeargs(self) -> Iterable[TypeRef]:
        return self.get_children_in_role(TypeRef.Roles.TypeArgs)

    def append_typearg(self, typearg: TypeRef):
        _assert_instance(typearg, TypeRef)
        self.append_child(typearg, TypeRef.Roles.TypeArgs)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_typeref)

    @staticmethod
    def make_composite_if_multiple(
        composite_type: type[TypeRef], *typeargs: TypeRef
    ) -> TypeRef:
        if len(typeargs) == 0:
            return NoneTypeRef
        elif len(typeargs) == 1:
            return typeargs[0]
        else:
            return composite_type(*typeargs)


class BuiltinTypeRef(TypeRef):
    def __init__(self, name: str, *typeargs: TypeRef, **kwargs):
        super().__init__(None, name, *typeargs, **kwargs)


class NoneTypeRef(BuiltinTypeRef):
    def __init__(self):
        super().__init__("None")


class BoolTypeRef(BuiltinTypeRef):
    def __init__(self):
        super().__init__("bool", default_value=Constant(bool()))  # noqa: UP018


class IntTypeRef(BuiltinTypeRef):
    def __init__(self):
        super().__init__("int", default_value=Constant(int()))  # noqa: UP018


class FloatTypeRef(BuiltinTypeRef):
    def __init__(self):
        super().__init__("float", default_value=Constant(float()))  # noqa: UP018


class ComplexTypeRef(BuiltinTypeRef):
    def __init__(self):
        super().__init__("complex", default_value=Constant(complex()))


class StrTypeRef(BuiltinTypeRef):
    def __init__(self):
        super().__init__("str")


class BytesTypeRef(BuiltinTypeRef):
    def __init__(self):
        super().__init__("bytes")


class EllipsisTypeRef(BuiltinTypeRef):
    def __init__(self):
        super().__init__("...")


class TypingRefs(ABC):
    @abstractmethod
    def __init__(self):
        pass

    class Any(TypeRef):
        def __init__(self):
            super().__init__("typing", "Any")

    class Union(TypeRef):
        def __init__(self, *typeargs: TypeRef):
            super().__init__("typing", "Union", *typeargs)

    class Optional(TypeRef):
        def __init__(self, *typeargs: TypeRef):
            super().__init__("typing", "Optional", *typeargs)

    class Sequence(TypeRef):
        def __init__(self, *typeargs: TypeRef):
            super().__init__("typing", "Sequence", *typeargs)

    class Tuple(TypeRef):
        def __init__(self, *typeargs: TypeRef):
            super().__init__("typing", "Tuple", *typeargs)

    class Mapping(TypeRef):
        def __init__(self, *typeargs: TypeRef):
            super().__init__("typing", "Mapping", *typeargs)

    class List(TypeRef):
        def __init__(self, *typeargs: TypeRef):
            super().__init__("typing", "List", *typeargs)

    class Annotation(TypeRef):
        def __init__(self, *typeargs: TypeRef):
            super().__init__("typing", "Annotation", *typeargs)

    class Callable(TypeRef):
        def __init__(self, *typeargs: TypeRef):
            super().__init__("typing", "Callable", *typeargs)


class Arg(Node):
    class Roles:
        Type = Role("Arg.Type")
        DefaultValue = Role("Arg.DefaultValue")

    def __init__(
        self,
        name: str,
        type: Optional[TypeRef] = None,
        default_value: Optional[Expr] = None,
        is_vararg: bool = False,
        is_kwarg: bool = False,
        doc: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.is_vararg = is_vararg
        self.is_kwarg = is_kwarg
        self.doc = doc
        self.append_child(type, Arg.Roles.Type)
        self.append_child(default_value, Arg.Roles.DefaultValue)

    @property
    def type(self) -> Optional[TypeRef]:
        return first_or_none(self.get_children_in_role(Arg.Roles.Type))

    @property
    def default_value(self) -> Optional[Expr]:
        return first_or_none(self.get_children_in_role(Arg.Roles.DefaultValue))

    @default_value.setter
    def default_value(self, value: Optional[Expr]):
        self._set_single_child(value, Arg.Roles.DefaultValue)

    @property
    def has_default_value(self) -> bool:
        return self.default_value is not None

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_arg)


class Stmt(Node, ABC):
    pass


class BlockStmt(Stmt, ABC):
    pass


class Pass(Stmt):
    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_pass)


class ThunkStmt(Stmt):
    class Roles:
        Thunk = Role("ThunkStmt.Thunk")

    def __init__(self, *thunks: Union[str, Stmt]):
        super().__init__()
        self.thunk: Optional[str] = None
        if len(thunks) == 1 and isinstance(thunks[0], str):
            self.thunk = thunks[0]
        else:
            for thunk in thunks:
                if isinstance(thunk, str):
                    self.append_child(ThunkStmt(thunk), ThunkStmt.Roles.Thunk)
                else:
                    self.append_child(thunk, ThunkStmt.Roles.Thunk)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_thunk_stmt)


class FunctionDef(BlockStmt):
    class Roles:
        Args = Role("FunctionDef.Args")
        ReturnType = Role("FunctionDef.ReturnType")
        Body = Role("FunctionDef.Body")

    def __init__(
        self,
        name: str,
        *args: Arg,
        return_type: Optional[TypeRef] = None,
        body: Union[Stmt, Iterable[Stmt]] = (),
        doc: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.doc = doc
        self.append_children(args, FunctionDef.Roles.Args)
        self.append_children(return_type, FunctionDef.Roles.ReturnType)
        self.append_children(body, FunctionDef.Roles.Body)

    @property
    def args(self) -> Iterable[Arg]:
        return self.get_children_in_role(FunctionDef.Roles.Args)

    def append_arg(self, base: TypeRef):
        _assert_instance(base, TypeRef)
        self.append_child(base, FunctionDef.Roles.Args)

    @property
    def return_type(self) -> Optional[TypeRef]:
        return self._get_single_child(FunctionDef.Roles.ReturnType)

    @return_type.setter
    def return_type(self, return_type: Optional[TypeRef]):
        self._set_single_child(return_type, FunctionDef.Roles.ReturnType)

    @property
    def body(self) -> Iterable[Stmt]:
        return self.get_children_in_role(FunctionDef.Roles.Body)

    def append_body(self, stmt: Stmt):
        _assert_instance(stmt, Stmt)
        self.append_child(stmt, FunctionDef.Roles.Body)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_functiondef)


class ClassDef(BlockStmt):
    class Roles:
        Bases = Role("ClassDef.Bases")
        Body = Role("ClassDef.Body")

    def __init__(self, name: str, *body: Stmt, bases: Union[TypeRef, Iterable[TypeRef]] = ()):
        super().__init__()
        self.name = name
        self.append_children(bases, ClassDef.Roles.Bases)
        self.append_children(body, ClassDef.Roles.Body)

    @property
    def bases(self) -> Iterable[TypeRef]:
        return self.get_children_in_role(ClassDef.Roles.Bases)

    def append_base(self, base: TypeRef):
        _assert_instance(base, TypeRef)
        self.append_child(base, ClassDef.Roles.Bases)

    @property
    def body(self) -> Iterable[Stmt]:
        return self.get_children_in_role(ClassDef.Roles.Body)

    def make_typeref(self) -> TypeRef:
        return TypeRef(self.parent.qual_name if self.parent else None, self.name)

    def append_body(self, stmt: Stmt):
        _assert_instance(stmt, Stmt)
        self.append_child(stmt, ClassDef.Roles.Body)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_classdef)


class Return(Stmt):
    class Roles:
        Expr = Role("Return.Expr")

    def __init__(self, expr: Expr):
        super().__init__()
        self.append_child(expr, Return.Roles.Expr)

    @property
    def expr(self):
        return self._get_single_child(Return.Roles.Expr)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_return)


class Assign(Stmt):
    class Roles:
        Target = Role("Assign.Target")
        Value = Role("Assign.Value")
        Type = Role("Assign.Type")

    def __init__(self, target: Expr, value: Expr, type: Optional[TypeRef] = None):
        super().__init__()
        self.target = target
        self.value = value
        self.type = type

    @property
    def target(self) -> Optional[Expr]:
        return self._get_single_child(Assign.Roles.Target)

    @target.setter
    def target(self, expr: Optional[Expr]):
        self._set_single_child(expr, Assign.Roles.Target)

    @property
    def value(self) -> Optional[Expr]:
        return self._get_single_child(Assign.Roles.Value)

    @value.setter
    def value(self, expr: Optional[Expr]):
        self._set_single_child(expr, Assign.Roles.Value)

    @property
    def type(self) -> Optional[TypeRef]:
        return self._get_single_child(Assign.Roles.Type)

    @type.setter
    def type(self, expr: Optional[TypeRef]):
        self._set_single_child(expr, Assign.Roles.Type)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_assign)


class If(BlockStmt):
    class Roles:
        Condition = Role("If.Condition")
        TrueBody = Role("If.TrueBody")
        FalseBody = Role("If.FalseBody")

    def __init__(
        self,
        condition: Expr,
        true_body: Iterable[Stmt],
        false_body: Optional[Iterable[Stmt]] = None,
    ):
        super().__init__()
        self.condition = condition
        self.append_children(true_body, If.Roles.TrueBody)
        self.append_children(false_body, If.Roles.FalseBody)

    @property
    def condition(self) -> Optional[Expr]:
        return self._get_single_child(If.Roles.Condition)

    @condition.setter
    def condition(self, expr: Optional[Expr]):
        self._set_single_child(expr, If.Roles.Condition)

    @property
    def true_body(self) -> Iterable[Stmt]:
        return self.get_children_in_role(If.Roles.TrueBody)

    @property
    def false_body(self) -> Iterable[Stmt]:
        return self.get_children_in_role(If.Roles.FalseBody)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_if)


class Raise(Node):
    class Roles:
        Expr = Role("Raise.Expr")

    def __init__(self, expr: Expr):
        super().__init__()
        self.append_child(expr, Raise.Roles.Expr)

    @property
    def expr(self):
        return first(self.get_children_in_role(Raise.Roles.Expr))

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_raise)


class Alias(Node):
    def __init__(self, name: str, alias: Optional[str] = None):
        super().__init__()
        self.name = name
        self.alias = alias

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_alias)


class ImportBase(Stmt, ABC):
    class Roles:
        Names = Role("ImportBase.Names")

    def __init__(self, *names: Alias):
        super().__init__()
        self.append_children(names, ImportBase.Roles.Names)

    @property
    def names(self) -> Iterable[Alias]:
        return self.get_children_in_role(ImportBase.Roles.Names)


class Import(ImportBase):
    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_import)


class ImportFrom(ImportBase):
    def __init__(self, module: str, *names: Alias, level: Optional[int] = None):
        super().__init__(*names)
        self.module = module
        self.level = level

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_importfrom)


class Module(Node):
    class Roles:
        Body = Role("Module.Body")

    def __init__(self, *body: Stmt, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.append_children(body, Module.Roles.Body)

    @property
    def body(self) -> Iterable[Stmt]:
        return self.get_children_in_role(Module.Roles.Body)

    def append_body(self, *stmts: Node):
        self.append_children(stmts, Module.Roles.Body)

    def accept(self, visitor: Visitor):
        self._dispatch_visit(visitor.visit_module)


class VisitKind(Enum):
    NONE = 0
    ENTER = 1
    LEAVE = 2


class Visitor:
    def __init__(self):
        self.visit_kind = VisitKind.NONE
        self.node_stack = []

    def enter(self, node: Node):
        self.visit_kind = VisitKind.ENTER
        self.node_stack.append(node)

    def leave(self, node: Node):
        self.visit_kind = VisitKind.LEAVE

    def finish(self, node: Node):
        self.visit_kind = VisitKind.NONE
        self.node_stack.pop()

    def visit_node(self, node: Node) -> Optional[bool]:
        return True

    def visit_expr(self, expr: Expr) -> Optional[bool]:
        return self.visit_node(expr)

    def visit_name(self, name: Name) -> Optional[bool]:
        return self.visit_expr(name)

    def visit_constant(self, constant: Constant) -> Optional[bool]:
        return self.visit_expr(constant)

    def visit_binop(self, binop: BinOp) -> Optional[bool]:
        return self.visit_expr(binop)

    def visit_subscript(self, subscript: Subscript) -> Optional[bool]:
        return self.visit_expr(subscript)

    def visit_starred(self, starred: Starred) -> Optional[bool]:
        return self.visit_expr(starred)

    def visit_call(self, call: Call) -> Optional[bool]:
        return self.visit_expr(call)

    def visit_lambda(self, lambda_: Lambda) -> Optional[bool]:
        return self.visit_expr(lambda_)

    def visit_expr_list(self, expr_list: ExprList) -> Optional[bool]:
        return self.visit_expr(expr_list)

    def visit_thunk_expr(self, thunk: ThunkExpr) -> Optional[bool]:
        return self.visit_expr(thunk)

    def visit_tuple_expr(self, tuple: TupleExpr) -> Optional[bool]:
        return self.visit_expr_list(tuple)

    def visit_list_expr(self, list: ListExpr) -> Optional[bool]:
        return self.visit_expr_list(list)

    def visit_set_expr(self, set: SetExpr) -> Optional[bool]:
        return self.visit_expr_list(set)

    def visit_dict_elem(self, elem: DictElem) -> Optional[bool]:
        return self.visit_expr(elem)

    def visit_dict_expr(self, dict: DictExpr) -> Optional[bool]:
        return self.visit_expr_list(dict)

    def visit_typeref(self, typeref: TypeRef) -> Optional[bool]:
        return self.visit_expr(typeref)

    def visit_arg(self, arg: Arg) -> Optional[bool]:
        return self.visit_node(arg)

    def visit_stmt(self, stmt: Stmt) -> Optional[bool]:
        return self.visit_node(stmt)

    def visit_blockstmt(self, block: BlockStmt) -> Optional[bool]:
        return self.visit_stmt(block)

    def visit_pass(self, pass_: Pass) -> Optional[bool]:
        return self.visit_stmt(pass_)

    def visit_thunk_stmt(self, thunk: ThunkStmt) -> Optional[bool]:
        return self.visit_stmt(thunk)

    def visit_functiondef(self, functiondef: FunctionDef) -> Optional[bool]:
        return self.visit_stmt(functiondef)

    def visit_classdef(self, classdef: ClassDef) -> Optional[bool]:
        return self.visit_stmt(classdef)

    def visit_return(self, return_: Return) -> Optional[bool]:
        return self.visit_stmt(return_)

    def visit_assign(self, assign: Assign) -> Optional[bool]:
        return self.visit_stmt(assign)

    def visit_if(self, if_: If) -> Optional[bool]:
        return self.visit_stmt(if_)

    def visit_raise(self, raise_: Raise) -> Optional[bool]:
        return self.visit_stmt(raise_)

    def visit_alias(self, alias: Alias) -> Optional[bool]:
        return self.visit_node(alias)

    def visit_importbase(self, import_: ImportBase) -> Optional[bool]:
        return self.visit_stmt(import_)

    def visit_import(self, import_: Import) -> Optional[bool]:
        return self.visit_importbase(import_)

    def visit_importfrom(self, importfrom: ImportFrom) -> Optional[bool]:
        return self.visit_importbase(importfrom)

    def visit_module(self, module: Module) -> Optional[bool]:
        return self.visit_node(module)


class FixupVisitor(Visitor, ABC):
    pass


class PopulateEmptyMemberBodies(FixupVisitor):
    def visit_classdef(self, classdef: ClassDef) -> Optional[bool]:
        if self.visit_kind is VisitKind.ENTER and not any(classdef.body):
            classdef.append_child(Pass(), ClassDef.Roles.Body)
        return True

    def visit_functiondef(self, functiondef: FunctionDef) -> Optional[bool]:
        if self.visit_kind is VisitKind.ENTER and not any(functiondef.body):
            functiondef.append_child(Pass(), FunctionDef.Roles.Body)
        return True


class NameCollector(Visitor):
    def __init__(self, predicate: NodePredicate):
        super().__init__()
        _assert_instance(predicate, NodePredicate)
        self._predicate = predicate
        self.names: Set[str] = set()

    def leave(self, node: Node) -> Optional[bool]:
        if self._predicate.matches(node) and hasattr(node, "name"):
            self.names.add(node.name)


class ImportAdjuster(FixupVisitor):
    def __init__(self):
        super().__init__()
        self.naming_conflicts: Set[str] = set()

    def enter(self, node: Node):
        if len(self.node_stack) == 0:
            collector = NameCollector(
                NodePredicate(func=lambda n: isinstance(n, (ClassDef, FunctionDef)))
            )
            node.accept(collector)
            self.naming_conflicts = collector.names
        super().enter(node)

    def leave(self, node: Node):
        super().leave(node)
        if len(self.node_stack) == 0:
            self.naming_conflicts = set()

    def visit_typeref(self, typeref: TypeRef) -> Optional[bool]:
        if self.visit_kind is not VisitKind.ENTER or not typeref.module:
            return True

        module = first_or_none(typeref.get_ancestors_of_type(Module))
        if module is None:
            return True

        def adjust_typeref(import_alias: Optional[str]):
            typeref.module = None
            if import_alias:
                typeref.name = import_alias

        import_from: ImportFrom = None

        # Reuse an existing import if we have one; if so,
        # and the imported name is already specified, return
        # early as there's nothing to import. In that case, also
        # adjust the typeref if the import is aliased due to
        # conflict resolution below from a previous pass.
        for import_ in filter(
            lambda i: i.module == typeref.module, module.get_children_of_type(ImportFrom)
        ):
            import_from = import_
            for imported_name in filter(
                lambda i: i.name in (typeref.name, typeref.name), import_.names
            ):
                adjust_typeref(imported_name.alias)
                return True

        # See if the type name conflicts with other names in the
        # module (class and function names). If so, adjust the
        # name to create an alias on the import. This rewrites
        # conflicts like:
        #   from typing import Optional
        #   def Optional(thing: Optional[str]): ...
        # To:
        #   from typing import Optional as _Optional
        #   def Optional(thing: _Optional[str]): ...
        conflict_alias = typeref.name
        while conflict_alias in self.naming_conflicts:
            conflict_alias = f"_{conflict_alias}"
        if conflict_alias == typeref.name:
            import_alias = Alias(typeref.name)
        else:
            import_alias = Alias(typeref.name, conflict_alias)

        # Expand or create the import
        if import_from is None:
            module.prepend_child(ImportFrom(typeref.module, import_alias), Module.Roles.Body)
        else:
            import_from.append_child(import_alias, ImportBase.Roles.Names)

        adjust_typeref(conflict_alias)
        return True


class NodeWriterOptions:
    def __init__(self, indent="    ", newline="\n", insert_final_newline=True):
        self.indent = indent
        self.newline = newline
        self.insert_final_newline = insert_final_newline


class NodeWriter(Visitor, ABC):
    def __init__(self, stream: TextIO, options: Optional[NodeWriterOptions] = None):
        super().__init__()
        self._stream = stream
        self._options = options or NodeWriterOptions()
        self._indent_level = 0
        self._last_char = ""

    def enter(self, node: Node):
        super().enter(node)
        if node.leading_trivia:
            self.write(node.leading_trivia)

    def finish(self, node: Node):
        super().finish(node)
        if node.trailing_trivia:
            self.write(node.trailing_trivia)
        if self._options.insert_final_newline and len(self.node_stack) == 0:
            self.write("\n")

    def indent(self):
        self._indent_level += 1

    def dedent(self):
        self._indent_level -= 1

    def write_indent(self):
        self._stream.write(self._options.indent * self._indent_level)

    def _raw_write(self, str: str):
        if len(str) > 0:
            if self._options.newline != "\n":
                self._stream.write(str.replace("\n", self._options.newline))
            else:
                self._stream.write(str)
            self._last_char = str[-1]

    def write(self, *texts: str, separator: str = "", allow_empty_text: bool = False):
        for i, text in enumerate(texts):
            if not allow_empty_text and len(text) == 0:
                continue
            if self._last_char == "\n":
                self.write_indent()
            if i > 0:
                self._raw_write(separator)
                if separator == "\n":
                    self.write_indent()
            self._raw_write(text)

    def dispatch_write(
        self,
        separator: Union[str, Callable[[Node], str]],
        nodes: Iterable[Node],
        prefix: str = "",
        suffix: str = "",
    ):
        self.write(prefix)
        for i, node in enumerate(nodes):
            if i > 0:
                if callable(separator):
                    self.write(separator(node))
                else:
                    self.write(separator)
            node.accept(self)
        self.write(suffix)


class PythonWriter(NodeWriter):
    def visit_node(self, node: Node) -> Optional[bool]:
        raise NotImplementedError(f"no visitor for node {node}")

    def visit_module(self, module: Module):
        def sep(node: Node):
            node_is_block = isinstance(node, BlockStmt)
            prev_is_block = isinstance(node.prev_sibling, BlockStmt)
            if prev_is_block or (node_is_block and not prev_is_block):
                return "\n\n\n"
            else:
                return "\n"

        self.dispatch_write(sep, module.body)

    def visit_alias(self, alias: Alias):
        self.write(alias.name)
        if alias.alias:
            self.write(" as ")
            self.write(alias.alias)

    def visit_import(self, import_: Import):
        self.write("import ")
        self.dispatch_write(", ", import_.names)

    def visit_importfrom(self, importfrom: ImportFrom):
        self.write(f"from {importfrom.module} import ")
        self.dispatch_write(", ", importfrom.names)

    def visit_typeref(self, typeref: TypeRef):
        if typeref.module and len(typeref.module) > 0:
            self.write(typeref.module)
            self.write(".")
        self.write(typeref.name)
        if any(typeref.typeargs):
            self.write("[")
            self.dispatch_write(", ", typeref.typeargs)
            self.write("]")

    def visit_arg(self, arg: Arg):
        if arg.is_vararg:
            self.write("*")
        self.write(arg.name)
        if arg.type:
            self.write(": ")
            arg.type.accept(self)
        if arg.default_value:
            self.write(" = ")
            arg.default_value.accept(self)

    def visit_thunk_expr(self, thunk: ThunkExpr):
        self.write(thunk.code)

    def visit_name(self, name: Name):
        self.write(name.identifier)

    def visit_constant(self, constant: Constant):
        self.write(
            repr(constant.value) if isinstance(constant.value, str) else str(constant.value)
        )

    def visit_binop(self, binop: BinOp):
        binop.left.accept(self)
        self.write(f" {binop.op} ")
        binop.right.accept(self)

    def visit_subscript(self, subscript: Subscript):
        subscript.value.accept(self)
        self.write("[")
        subscript.slice.accept(self)
        self.write("]")

    def visit_starred(self, starred: Starred):
        self.write("*")
        starred.expr.accept(self)

    def visit_call(self, call: Call):
        call.func.accept(self)
        self.dispatch_write(", ", call.args, prefix="(", suffix=")")

    def visit_lambda(self, lambda_: Lambda):
        self.write("lambda ")
        self.dispatch_write(", ", lambda_.args)
        self.write(": ")
        lambda_.body.accept(self)

    def visit_tuple_expr(self, tuple: TupleExpr):
        self.dispatch_write(", ", tuple.elements, prefix="(", suffix=",)")

    def visit_list_expr(self, list: ListExpr):
        self.dispatch_write(", ", list.elements, prefix="[", suffix="]")

    def visit_set_expr(self, set: ListExpr):
        self.dispatch_write(", ", set.elements, prefix="{", suffix="}")

    def visit_dict_elem(self, elem: DictElem):
        elem.key.accept(self)
        self.write(": ")
        elem.value.accept(self)

    def visit_dict_expr(self, dict: DictExpr):
        self.dispatch_write(", ", dict.elements, prefix="{", suffix="}")

    def visit_pass(self, pass_: Pass):
        self.write("pass")

    def visit_thunk_stmt(self, thunk: ThunkStmt) -> bool:
        if self.visit_kind == VisitKind.ENTER and thunk.thunk:
            lines = dedent(thunk.thunk).splitlines()
            self.write(*lines, separator="\n", allow_empty_text=True)
            if thunk.next_sibling:
                self.write("\n")
        return True

    def visit_assign(self, assign: Assign):
        assign.target.accept(self)
        if assign.type:
            self.write(": ")
            assign.type.accept(self)
        self.write(" = ")
        assign.value.accept(self)

    def visit_if(self, if_: If):
        self.write("if ")
        if_.condition.accept(self)
        self.write(":\n")
        self.indent()
        self.dispatch_write("\n", if_.true_body)
        self.dedent()
        if first_or_none(if_.false_body) is not None:
            self.write("else:\n")
            self.indent()
            self.dispatch_write("\n", if_.false_body)
            self.dedent()

    def visit_raise(self, raise_: Raise):
        self.write("raise ")
        raise_.expr.accept(self)

    def visit_functiondef(self, functiondef: FunctionDef):
        self.write("def ", functiondef.name, "(")
        self.dispatch_write(", ", functiondef.args)
        self.write(")")
        if functiondef.return_type:
            self.write(" -> ")
            functiondef.return_type.accept(self)
        self.write(":\n")
        self.indent()
        if functiondef.doc:
            self.write('r"""')
            for line in dedent(functiondef.doc).splitlines():
                self.write(line)
                self.write("\n")
            self.write('"""\n\n')
        self.dispatch_write("\n", functiondef.body)
        self.dedent()

    def visit_classdef(self, classdef: ClassDef):
        self.write("class ", classdef.name)
        if any(classdef.bases):
            self.write("(")
            self.dispatch_write(", ", classdef.bases)
            self.write(")")
        self.write(":\n")
        self.indent()
        self.dispatch_write("\n\n", classdef.body)
        self.dedent()

    def visit_return(self, return_: Return):
        self.write("return ")
        return_.expr.accept(self)


class DocCommentBuilder(Visitor):
    def __init__(self, width: int = 80):
        super().__init__()
        self.width = width

    def visit_functiondef(self, functiondef: FunctionDef):
        def wrap(text: str, initial_indent="", subsequent_indent=""):
            return TextWrapper(
                width=self.width,
                initial_indent=initial_indent,
                subsequent_indent=subsequent_indent,
                expand_tabs=False,
                replace_whitespace=False,
                fix_sentence_endings=False,
                break_long_words=False,
                break_on_hyphens=False,
            ).fill(text)

        argsdoc = ""
        for arg in functiondef.args:
            if arg.doc:
                argsdoc += wrap(f"{arg.name}: {arg.doc}", " " * 4, " " * 8) + "\n\n"
        if argsdoc:
            functiondef.doc += "\n\nArgs:\n" + argsdoc

        if functiondef.doc:
            functiondef.doc = functiondef.doc.strip()
