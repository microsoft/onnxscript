from __future__ import annotations

import onnxscript.ir as ir
import onnxscript.rewriter.no_op as no_op
import onnxscript.rewriter.pattern as orp
from onnxscript.rewriter import pattern

op = pattern.onnxop


def transpose_identity(x, perm):
    return op.Transpose(x, perm=perm)


def transpose_identity_check(x: ir.Value, perm: ir.Attr | ir.RefAttr) -> bool:
    if isinstance(perm, ir.RefAttr):
        return False
    if perm.type == ir.AttributeType.INTS:
        if perm.value == list(range(len(perm.value))):
            return True
    return False


def transpose_identity_rewrite(x: ir.Value, perm: ir.Attr | ir.RefAttr):
    return op.Transpose(x, perm=perm)


def transpose_transpose(x, perm1, perm2):
    return op.Transpose(op.Transpose(x, perm1=[1, 0]), perm2=[1, 0])


def transpose_transpose_check(
    x: ir.Value, perm1: ir.Attr | ir.RefAttr, perm2: ir.Attr | ir.RefAttr
) -> bool:
    if isinstance(perm1, ir.RefAttr) or isinstance(perm2, ir.RefAttr):
        return False
    return True


def _apply_transpose(cls, perm: tuple[int, ...], on: list[int | str]) -> list[int | str]:
    assert len(perm) == len(on), "length mismatch"
    res = [None for i in on]
    for i, p in enumerate(perm):
        res[i] = on[p]
    return res


def _apply_transposes(
    cls, perms: list[tuple[int, ...]], on: list[int | str] | None = None
) -> list[int | str]:
    if on is None:
        on = list(range(len(perms[0])))
    for p in perms:
        on = cls.apply_transpose(p, on)
    return on


def transpose_transpose_rewrite(
    x: ir.Value, perm1: ir.Attr | ir.RefAttr, perm2: ir.Attr | ir.RefAttr
):
    first = list(range(len(perm1)))
    last = _apply_transposes([perm1, perm2])
    if first == last:
        return op.Identity(x)
    return op.Transpose(x, perm=last)


transpose_identity_rule = pattern.RewriteRule(
    transpose_identity, transpose_identity_check, transpose_identity_rewrite
)
transpose_transpose_rule = pattern.RewriteRule(
    transpose_transpose, transpose_transpose_check, transpose_transpose_rewrite
)


def llama_p0_rule_set(verbose: int = 0) -> orp.RewriteRuleSet:
    """Returns a set of rules which should be applied
    before anyother one as they usually remove unnecessary computation
    such as the multiplication by 1.

    Args:
        verbose: verbosity
    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            no_op.mul_by_1_rule,
            no_op.add_0_rule,
            no_op.add_0_rule,
            no_op.div_by_1_rule,
            transpose_identity_rule,
            transpose_transpose_rule,
        ]
    )
