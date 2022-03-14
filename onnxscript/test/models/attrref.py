# SPDX-License-Identifier: Apache-2.0


def float_attr_ref_test(X, alpha: float):
    return oxs.Foo(X, alpha)


def int_attr_ref_test(X, alpha: int):
    return oxs.Foo(X, alpha)


def str_attr_ref_test(X, alpha: str):
    return oxs.Foo(X, alpha)
