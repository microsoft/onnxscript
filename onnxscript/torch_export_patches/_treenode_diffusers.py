try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
except ImportError:
    try:
        import diffusers
    except ImportError:
        diffusers = None
        UNet2DConditionOutput = None
    if diffusers:
        raise

from onnxscript.torch_export_patches import _treenode_transformers

SUPPORTED_DATACLASSES: set[type] = set()


if UNet2DConditionOutput is not None:
    (
        flatten_u_net2_d_condition_output,
        flatten_with_keys_u_net2_d_condition_output,
        unflatten_u_net2_d_condition_output,
    ) = _treenode_transformers.make_serialization_function_for_dataclass(
        UNet2DConditionOutput, SUPPORTED_DATACLASSES
    )
