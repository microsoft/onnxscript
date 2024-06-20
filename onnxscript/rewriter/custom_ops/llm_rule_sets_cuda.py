# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx

import onnxscript.ir as ir
import onnxscript.rewriter.pattern as orp

op = orp.onnxop


class _CombineBinary(orp.RewriteRuleAsClass):
    @classmethod
    def _same_shape(
        cls, sh1: tuple[int, ...], sh2: tuple[int, ...], broadcast: bool = False
    ) -> bool:
        if broadcast:
            if len(sh1) != len(sh2):
                rk = max(len(sh1), len(sh2))
                sh1 = (1,) * (rk - len(sh1)) + sh1
                sh2 = (1,) * (rk - len(sh2)) + sh2
            allow_one1 = True
            allow_one2 = True
            for a, b in zip(sh1, sh2):
                if a == b:
                    if a != 1:
                        allow_one1 = False
                    if b != 1:
                        allow_one2 = False
                    continue
                if a == 1 and allow_one1:
                    allow_one2 = False
                    continue
                if b == 1 and allow_one2:
                    allow_one1 = False
                    continue
                return False
            return True
        return sh1 == sh2

    @classmethod
    def check(cls, context, x, y, z) -> bool:
        if x.shape is None or y.shape is None or z.shape is None:
            return False
        return cls._same_shape(x.shape, y.shape, broadcast=True) and cls._same_shape(
            y.shape, z.shape, broadcast=True
        )


class CombinedAddAdd1(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Add(x, op.Add(y, z))

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.AddAdd(x, y, z, domain="ai.onnx.contrib")


class CombinedAddAdd2(CombinedAddAdd1):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Add(op.Add(x, y), z)


class CombinedMulMul1(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(x, op.Mul(y, z))

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.MulMul(x, y, z, domain="ai.onnx.contrib")


class CombinedMulMul2(CombinedMulMul1):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(op.Mul(x, y), z)


class CombinedAddMul1(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(op.Add(x, y), z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.AddMul(x, y, z, domain="ai.onnx.contrib")


class CombinedAddMul2(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(x, op.Add(y, z))

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.AddMul(y, z, x, domain="ai.onnx.contrib")


class CombinedMulAdd1(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Add(op.Mul(x, y), z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.MulAdd(x, y, z, domain="ai.onnx.contrib")


class CombinedMulAdd2(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Add(x, op.Mul(y, z))

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.MulAdd(y, z, x, domain="ai.onnx.contrib")


class AddSharedInput1(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Add(x, y), op.Add(x, z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.AddSharedInput(x, y, z, domain="ai.onnx.contrib", outputs=2)


class AddSharedInput2(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Add(y, x), op.Add(x, z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.AddSharedInput(x, y, z, domain="ai.onnx.contrib", outputs=2)


class MulSharedInput1(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(x, y), op.Mul(x, z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.MulSharedInput(x, y, z, domain="ai.onnx.contrib", outputs=2)


class MulSharedInput2(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(y, x), op.Mul(x, z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.MulSharedInput(x, y, z, domain="ai.onnx.contrib", outputs=2)


class CombinedSubMul1(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(op.Sub(x, y), z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.SubMul(x, y, z, negative=0, domain="ai.onnx.contrib")


class CombinedSubMul1c(CombinedSubMul1):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(z, op.Sub(x, y))


class CombinedSubMul2(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(op.Sub(y, x), z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.SubMul(x, y, z, negative=1, domain="ai.onnx.contrib")


class CombinedSubMul2c(CombinedSubMul2):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Mul(z, op.Sub(y, x))


class CombinedMulSub1(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Sub(op.Mul(x, y), z)

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.MulSub(x, y, z, negative=0, domain="ai.onnx.contrib")


class CombinedMulSub1c(CombinedMulSub1):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Sub(op.Mul(y, x), z)


class CombinedMulSub2(_CombineBinary):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Sub(z, op.Mul(x, y))

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.MulSub(x, y, z, negative=1, domain="ai.onnx.contrib")


class CombinedMulSub2c(CombinedMulSub2):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Sub(z, op.Mul(y, x))


class MaskedScatterNDOfShape(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, shape, indices, updates, tensor, masked, zero, reduction):
        cst = op.ConstantOfShape(shape, value=tensor)
        masked_indices = op.Equal(indices, masked)
        masked_updates = op.Where(masked_indices, zero, updates)
        return op.ScatterND(cst, indices, masked_updates, reduction=reduction)

    @classmethod
    def check(cls, context, shape, indices, updates, tensor, masked, zero, reduction) -> bool:
        if reduction.value != "add":
            return False
        if tensor.value.numpy().reshape((1,)).tolist() != [0]:
            return False
        if zero.const_value is None or zero.const_value.numpy().reshape((1,)).tolist() != [0]:
            return False
        if masked.const_value is None or masked.const_value.numpy().size != 1:
            return False
        return True

    @classmethod
    def rewrite(cls, op, shape, indices, updates, tensor, masked, zero, reduction):
        return op.MaskedScatterNDOfShape(
            shape,
            indices,
            updates,
            maskedValue=int(masked.const_value.numpy().reshape((1,))[0]),
            reduction=reduction.value,
            domain="ai.onnx.contrib",
        )


class MulSigmoid(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, x):
        return op.Mul(x, op.Sigmoid(x))

    @classmethod
    def check(cls, context, x) -> bool:
        return True

    @classmethod
    def rewrite(cls, op, x):
        return op.MulSigmoid(x, domain="ai.onnx.contrib")


class NegXPlus1(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, cst, x):
        return op.Sub(cst, x)

    @classmethod
    def check(cls, context, cst, x) -> bool:
        if cst.const_value is None:
            return False
        if cst.shape != (1,):
            return False
        value = float(cst.const_value.numpy().reshape((1,))[0])
        return value == 1

    @classmethod
    def rewrite(cls, op, cst, x):
        return op.NegXplus1(x, domain="ai.onnx.contrib")


class ReplaceZero1(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, x, y):
        return op.Where(op.Cast(x, to=onnx.TensorProto.BOOL), x, y)

    @classmethod
    def check(cls, context, x, y) -> bool:
        return True

    @classmethod
    def rewrite(cls, op, x, y):
        return op.ReplaceZero(x, y, equal=1, domain="ai.onnx.contrib")


class ReplaceZero2(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, x, y):
        return op.Where(op.Cast(x, to=onnx.TensorProto.BOOL), y, x)

    @classmethod
    def check(cls, context, x, y) -> bool:
        return True

    @classmethod
    def rewrite(cls, op, x, y):
        return op.ReplaceZero(x, y, equal=0, domain="ai.onnx.contrib")


class Rotary1(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, x):
        x1, x2 = op.Split(x, axis=-1, outputs=2, num_outputs=2)
        return op.Concat(op.Neg(x2), x1, axis=-1)

    @classmethod
    def check(cls, context, x) -> bool:
        return True

    @classmethod
    def rewrite(cls, op, x):
        return op.Rotary(x, side="right", domain="ai.onnx.contrib")


class Rotary2(Rotary1):
    @classmethod
    def pattern(cls, op, x):
        x1, x2 = op.Split(x, num_outputs=2, axis=-1, outputs=2)
        return op.Concat(x2, op.Neg(x1), axis=-1)

    @classmethod
    def rewrite(cls, op, x):
        return op.Rotary(x, side="left", domain="ai.onnx.contrib")


class Rotary3(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, x, splits):
        x1, x2 = op.Split(x, splits, axis=-1, outputs=2)
        return op.Concat(op.Neg(x2), x1, axis=-1)

    @classmethod
    def check(cls, context, x, splits) -> bool:
        if splits.const_value is None:
            return False
        value = splits.const_value.numpy()
        if value.shape != (2,) or value[0] != value[1]:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, splits):
        return op.Rotary(x, splits, side="right", domain="ai.onnx.contrib")


class Rotary4(Rotary3):
    @classmethod
    def pattern(cls, op, x, splits):
        x1, x2 = op.Split(x, splits, axis=-1, outputs=2)
        return op.Concat(x2, op.Neg(x1), axis=-1)

    @classmethod
    def rewrite(cls, op, x, splits):
        return op.Rotary(x, splits, side="left", domain="ai.onnx.contrib")


class TransposeCast1(orp.RewriteRuleAsClass):
    """Replaces ``Cast + Transpose(. perm=[1, 0])`` by ``TransposeCast2D``."""

    @classmethod
    def pattern(cls, op, x, perm, to):
        return op.Cast(op.Transpose(x, perm=perm), to=to)

    @classmethod
    def check(cls, context, x, perm, to) -> bool:
        if isinstance(perm, ir.RefAttr) or isinstance(to, ir.RefAttr):
            return False
        if perm.value != [1, 0]:
            return False
        if to.value not in {onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT}:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, perm, to):
        if to.value == onnx.TensorProto.FLOAT:
            return op.Transpose2DCastFP32(x, domain="ai.onnx.contrib")
        return op.Transpose2DCastFP16(x, domain="ai.onnx.contrib")


class TransposeCast2(orp.RewriteRuleAsClass):
    """Replaces ``Transpose(. perm=[1, 0]) + Cast`` by ``TransposeCast2D``."""

    @classmethod
    def pattern(cls, op, x, perm, to):
        return op.Transpose(op.Cast(x, to=to), perm=perm)

    @classmethod
    def check(cls, context, x, perm, to) -> bool:
        if isinstance(perm, ir.RefAttr) or isinstance(to, ir.RefAttr):
            return False
        if perm.value != [1, 0]:
            return False
        if to.value not in {onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT}:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, perm, to):
        if to.value == onnx.TensorProto.FLOAT:
            return op.Transpose2DCastFP32(x, domain="ai.onnx.contrib")
        return op.Transpose2DCastFP16(x, domain="ai.onnx.contrib")


class MulAddTranspose(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Transpose(op.MulAdd(x, y, z, domain="ai.onnx.contrib"), perm=[0, 2, 1, 3])

    @classmethod
    def check(cls, context, x, y, z) -> bool:
        return True

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.MulAdd(x, y, z, transposeMiddle=1, domain="ai.onnx.contrib")


class AddMulTranspose(orp.RewriteRuleAsClass):
    @classmethod
    def pattern(cls, op, x, y, z):
        return op.Transpose(op.AddMul(x, y, z, domain="ai.onnx.contrib"), perm=[0, 2, 1, 3])

    @classmethod
    def check(cls, context, x, y, z) -> bool:
        return True

    @classmethod
    def rewrite(cls, op, x, y, z):
        return op.AddMul(x, y, z, transposeMiddle=1, domain="ai.onnx.contrib")


def llm_rule_set_cuda() -> orp.RewriteRuleSet:
    """Returns a set of rules fusing nodes into custom kernels.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            orp.make_rewrite_rule_from_class(AddSharedInput1, True),
            orp.make_rewrite_rule_from_class(AddSharedInput2, True),
            orp.make_rewrite_rule_from_class(MulSharedInput1, True),
            orp.make_rewrite_rule_from_class(MulSharedInput2, True),
            orp.make_rewrite_rule_from_class(AddMulTranspose),
            orp.make_rewrite_rule_from_class(CombinedAddAdd1),
            orp.make_rewrite_rule_from_class(CombinedAddAdd2),
            orp.make_rewrite_rule_from_class(CombinedAddMul1),
            orp.make_rewrite_rule_from_class(CombinedAddMul2),
            orp.make_rewrite_rule_from_class(CombinedMulAdd1),
            orp.make_rewrite_rule_from_class(CombinedMulAdd2),
            orp.make_rewrite_rule_from_class(CombinedMulMul1),
            orp.make_rewrite_rule_from_class(CombinedMulMul2),
            orp.make_rewrite_rule_from_class(CombinedMulSub1),
            orp.make_rewrite_rule_from_class(CombinedMulSub1c),
            orp.make_rewrite_rule_from_class(CombinedMulSub2),
            orp.make_rewrite_rule_from_class(CombinedMulSub2c),
            orp.make_rewrite_rule_from_class(CombinedSubMul1),
            orp.make_rewrite_rule_from_class(CombinedSubMul1c),
            orp.make_rewrite_rule_from_class(CombinedSubMul2),
            orp.make_rewrite_rule_from_class(CombinedSubMul2c),
            orp.make_rewrite_rule_from_class(MaskedScatterNDOfShape),
            orp.make_rewrite_rule_from_class(MulAddTranspose),
            orp.make_rewrite_rule_from_class(MulSigmoid),
            orp.make_rewrite_rule_from_class(NegXPlus1),
            orp.make_rewrite_rule_from_class(ReplaceZero1),
            orp.make_rewrite_rule_from_class(ReplaceZero2),
            orp.make_rewrite_rule_from_class(Rotary1),
            orp.make_rewrite_rule_from_class(Rotary2),
            orp.make_rewrite_rule_from_class(Rotary3),
            orp.make_rewrite_rule_from_class(Rotary4),
            orp.make_rewrite_rule_from_class(TransposeCast1),
            orp.make_rewrite_rule_from_class(TransposeCast2),
        ]
    )
