from __future__ import annotations

__all__ = ["assert_onnx_proto_equal"]

import difflib
import typing
from typing import Any, Collection, Sequence

import google.protobuf.message

if typing.TYPE_CHECKING:
    import onnx


def _opset_import_key(opset_import: onnx.OperatorSetIdProto) -> tuple[str, int]:
    return (opset_import.domain, opset_import.version)


def _value_info_key(value_info: onnx.ValueInfoProto) -> str:
    return value_info.name


def _function_key(function: onnx.FunctionProto) -> tuple[str, str, str]:
    return (function.domain, function.name, function.overload)


def _find_duplicates(with_duplicates: Collection[Any]) -> list[Any]:
    """Return a list of duplicated elements in a collection."""
    seen = set()
    duplicates = []
    for x in with_duplicates:
        if x in seen:
            duplicates.append(x)
        seen.add(x)
    return duplicates


def assert_onnx_proto_equal(
    a: google.protobuf.message.Message | Any, b: google.protobuf.message.Message | Any
) -> None:
    """Assert that two ONNX protos are equal.

    Equality is defined as having the same fields with the same values. When
    a field takes the default value, it is considered equal to the field
    not being set.

    Sequential fields with name `opset_import`, `value_info`, and `functions` are
    compared disregarding the order of their elements.

    Args:
        a: The first ONNX proto.
        b: The second ONNX proto.
    """
    assert type(a) == type(b), f"Type not equal: {type(a)} != {type(b)}"

    a_fields = {field.name: value for field, value in a.ListFields()}
    b_fields = {field.name: value for field, value in b.ListFields()}
    all_fields = sorted(set(a_fields.keys()) | set(b_fields.keys()))
    for field in all_fields:
        # Obtain the default value if the field is not set. This way we can compare the two fields.
        a_value = getattr(a, field)
        b_value = getattr(b, field)
        if (
            isinstance(a_value, Sequence)
            and isinstance(b_value, Sequence)
            and not isinstance(a_value, (str, bytes))
            and not isinstance(b_value, (str, bytes))
        ):
            # Check length first
            a_keys: list[Any] = []
            b_keys: list[Any] = []
            if field == "opset_import":
                a_value = sorted(a_value, key=_opset_import_key)
                b_value = sorted(b_value, key=_opset_import_key)
                a_keys = [_opset_import_key(opset_import) for opset_import in a_value]
                b_keys = [_opset_import_key(opset_import) for opset_import in b_value]
            elif field == "value_info":
                a_value = sorted(a_value, key=_value_info_key)
                b_value = sorted(b_value, key=_value_info_key)
                a_keys = [_value_info_key(value_info) for value_info in a_value]
                b_keys = [_value_info_key(value_info) for value_info in b_value]
            elif field == "functions":
                a_value = sorted(a_value, key=_function_key)
                b_value = sorted(b_value, key=_function_key)
                a_keys = [_function_key(functions) for functions in a_value]
                b_keys = [_function_key(functions) for functions in b_value]

            if a_keys != b_keys:
                keys_only_in_a = set(a_keys) - set(b_keys)
                keys_only_in_b = set(b_keys) - set(a_keys)
                error_message = (
                    f"Field {field} not equal: keys_only_in_a={keys_only_in_a}, keys_only_in_b={keys_only_in_b}. "
                    f"Field type: {type(a_value)}. "
                    f"Duplicated a_keys: {_find_duplicates(a_keys)}, duplicated b_keys: {_find_duplicates(b_keys)}"
                )
                raise AssertionError(error_message)
            elif len(a_value) != len(b_value):
                error_message = (
                    f"Field {field} not equal: len(a)={len(a_value)}, len(b)={len(b_value)} "
                    f"Field type: {type(a_value)}"
                )
                raise AssertionError(error_message)
            # Check every element
            for i in range(len(a_value)):
                a_value_i = a_value[i]
                b_value_i = b_value[i]
                if isinstance(
                    a_value_i, google.protobuf.message.Message
                ) and isinstance(b_value_i, google.protobuf.message.Message):
                    try:
                        assert_onnx_proto_equal(a_value_i, b_value_i)
                    except AssertionError as e:
                        error_message = f"Field {field} index {i} in sequence not equal. type(a_value_i): {type(a_value_i)}, type(b_value_i): {type(b_value_i)}, a_value_i: {a_value_i}, b_value_i: {b_value_i}"
                        raise AssertionError(error_message) from e
                elif a_value_i != b_value_i:
                    error_message = f"Field {field} index {i} in sequence not equal. type(a_value_i): {type(a_value_i)}, type(b_value_i): {type(b_value_i)}"
                    for line in difflib.ndiff(
                        str(a_value_i).splitlines(), str(b_value_i).splitlines()
                    ):
                        error_message += "\n" + line
                    raise AssertionError(error_message)
        elif isinstance(a_value, google.protobuf.message.Message) and isinstance(
            b_value, google.protobuf.message.Message
        ):
            assert_onnx_proto_equal(a_value, b_value)
        elif a_value != b_value:
            error_message = (
                f"Field {field} not equal. field_a: {a_value}, field_b: {b_value}"
            )
            raise AssertionError(error_message)
