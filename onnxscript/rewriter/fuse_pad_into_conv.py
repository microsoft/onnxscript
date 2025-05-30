# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fuses Pad nodes into preceding nodes. Supported fusion patterns:
- Pad ∘ Conv          -> Conv
- Pad ∘ ConvInteger   -> ConvInteger
"""

import typing

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import pattern as orp


def fill_pads_with_axes(
    pads: typing.Sequence[int], axes: typing.Sequence[int], rank: int
) -> typing.List[int]:
    new_pads = [0] * 2 * rank
    N = len(axes)
    for start_idx, axis in enumerate(axes):
        new_pads[axis] = pads[start_idx]
        new_pads[axis + rank] = pads[start_idx + N]
    return new_pads


class _FusePadConvBase(orp.RewriteRuleClassBase):
    """Interface for PadConv nodes fusion."""

    def __init__(self, name: str, as_function: bool = False):
        # Remove nodes is set to False to remove unused nodes after the rewrite.
        super().__init__(name=name, remove_nodes=False, as_function=as_function)

    def rewrite(
        self, op: ir.tape.Tape, x: ir.Value, pad: ir.Value, conv: ir.Value
    ) -> ir.Value:
        pnode = pad.producer()
        cnode = conv.producer()

        # Retrieve the padding and axes
        x_rank = len(x.shape)
        pad_pads = pnode.inputs[1].const_value.numpy().tolist()
        if len(pnode.inputs) > 3 and (axes := pnode.inputs[3]) is not None:
            axes = [x if x >= 0 else x_rank + x for x in axes.const_value.numpy()]
        else:
            axes = list(range(x_rank))

        # Fulfill pad_pads in every dimension (filling with zero the other ones)
        pad_pads = fill_pads_with_axes(pad_pads, axes, x_rank)

        # Get only spatial pads
        new_pads = pad_pads[2:x_rank] + pad_pads[x_rank + 2 :]

        # Replace conv pads = new + old
        conv_attr: typing.Mapping[str, ir.Attr] = cnode.attributes.copy()
        if "pads" in conv_attr:
            new_pads = [x + y for x, y in zip(conv_attr["pads"].as_ints(), new_pads)]
        conv_attr["pads"] = ir.convenience.convert_attribute("pads", new_pads)

        return op.op(
            cnode.op_type,
            inputs=(x, *cnode.inputs[1:]),
            attributes=conv_attr,
            domain=cnode.domain,
            name=cnode.name,
        )

    def check(self, context, x: ir.Value, pad: ir.Value, conv: ir.Value) -> orp.MatchResult:
        del context  # Unused
        check_result = orp.MatchResult()
        pnode = pad.producer()
        x_rank = len(x.shape)

        # Pad constraints: attributes
        if (mode := pnode.attributes.get("mode", None)) and mode.as_string() != "constant":
            return check_result.fail(f"{pnode.name} mode must be 'constant'.")

        # Pad constraints: inputs
        if (pads := pnode.inputs[1]).const_value is None:
            return check_result.fail(f"{pads.name} is not a constant/initializer.")
        if len(pnode.inputs) > 2 and (constant_value := pnode.inputs[2]) is not None:
            if constant_value.const_value is None:
                return check_result.fail(
                    f"{constant_value.name} is not a constant/initializer."
                )
            elif constant_value.const_value.numpy().item() != 0:
                return check_result.fail(f"{constant_value.name} must be equal to 0.")
        axes = list(range(x_rank))
        if len(pnode.inputs) > 3 and (axes := pnode.inputs[3]) is not None:
            if axes.const_value is None:
                return check_result.fail(f"{axes.name} is not a constant/initializer.")
            axes_list = [x if x >= 0 else x_rank + x for x in axes.const_value.numpy()]
        else:
            axes_list = list(range(x_rank))

        # Pad constraints: values
        pads_list = fill_pads_with_axes(pads.const_value.numpy(), axes_list, x_rank)
        if np.any(pads_list[:2] + pads_list[x_rank : x_rank + 2]):
            return check_result.fail(f"{pads.name} must be zero in non-spatial dimensions.")

        return check_result


class FusePadConv(_FusePadConvBase):
    """Replaces ``Pad(Conv(x))`` with ``Conv(x)``."""

    def __init__(self, as_function: bool = False):
        super().__init__(name="FusePadConv", as_function=as_function)

    def pattern(self, op: ir.tape.Tape, x: ir.Value) -> ir.Value:
        return op.Conv(
            op.Pad(x, _allow_other_inputs=True, _outputs=["pad"]),
            _allow_other_inputs=True,
            _outputs=["conv"],
        )

    def check(self, context, x: ir.Value, pad: ir.Value, conv: ir.Value) -> orp.MatchResult:
        check_result = super().check(context, x, pad, conv)
        if check_result.reason:
            return check_result

        # Conv constraints: attributes
        cnode = conv.producer()
        if (apad := cnode.attributes.get("auto_pad", None)) and apad.as_string() != "NOTSET":
            return check_result.fail(f"{cnode.name} auto_pad must be 'NOTSET'.")
        return check_result


class FusePadConvInteger(FusePadConv):
    """Replaces ``Pad(ConvInteger(x))`` with ``ConvInteger(x)``."""

    def __init__(self, as_function: bool = False):
        super(FusePadConv, self).__init__(name="FusePadConvInteger", as_function=as_function)

    def pattern(self, op: ir.tape.Tape, x: ir.Value) -> ir.Value:
        return op.ConvInteger(
            op.Pad(x, _allow_other_inputs=True, _outputs=["pad"]),
            _allow_other_inputs=True,
            _outputs=["conv"],
        )


fuse_pad_into_conv = FusePadConv.rule()
fuse_pad_into_conv_integer = FusePadConvInteger.rule()


def fuse_pad_into_conv_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rewrite rules that fuse Pad nodes into preceding:
    - Conv
    - ConvInteger

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            fuse_pad_into_conv,
            fuse_pad_into_conv_integer,
        ]
    )
