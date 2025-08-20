import re
from typing import Any, Callable

import torch
from transformers.cache_utils import (
    DynamicCache,
    EncoderDecoderCache,
    HybridCache,
    SlidingWindowCache,
    StaticCache,
)

try:
    from transformers.models.mamba.modeling_mamba import MambaCache
except ImportError:
    from transformers.cache_utils import MambaCache
from transformers.modeling_outputs import BaseModelOutput

from onnxscript.torch_export_patches import _cache_creator

SUPPORTED_DATACLASSES: set[type] = set()


############
# MambaCache
############


def flatten_mamba_cache(
    mamba_cache: MambaCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    assert isinstance(mamba_cache.conv_states, list) and isinstance(
        mamba_cache.ssm_states, list
    ), (
        f"Unexpected types for conv_states and ssm_states {type(mamba_cache.conv_states)}, "
        f"{type(mamba_cache.ssm_states)}"
    )
    flat = [
        ("conv_states", mamba_cache.conv_states),
        ("ssm_states", mamba_cache.ssm_states),
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def unflatten_mamba_cache(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> MambaCache:
    """Restores a :class:`transformers.cache_utils.MambaCache` from python objects."""
    conv_states, ssm_states = values

    class Config:
        def __init__(self):
            if isinstance(conv_states, list):
                self.intermediate_size = conv_states[0].shape[1]
                self.state_size = ssm_states[0].shape[2]
                self.conv_kernel = conv_states[0].shape[2]
                self.num_hidden_layers = len(conv_states)
            else:
                self.intermediate_size = conv_states.shape[2]
                self.state_size = ssm_states.shape[3]
                self.conv_kernel = conv_states.shape[3]
                self.num_hidden_layers = conv_states.shape[0]

    cache = MambaCache(
        Config(),
        max_batch_size=1,
        dtype=values[-1][0].dtype,
        device="cpu" if values[-1][0].get_device() < 0 else "cuda",
    )
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


def flatten_with_keys_mamba_cache(
    cache: MambaCache,
) -> tuple[
    list[tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    values, context = flatten_mamba_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


##############
# DynamicCache
##############


def flatten_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    ca = _cache_creator.CacheKeyValue(dynamic_cache)
    flat = [("key_cache", ca.key_cache), ("value_cache", ca.value_cache)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> tuple[list[tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    values, context = flatten_dynamic_cache(dynamic_cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_dynamic_cache(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> DynamicCache:
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
    return _cache_creator.make_dynamic_cache(list(zip(values[0], values[1])))


#############
# HybridCache
#############


def flatten_hybrid_cache(
    cache: HybridCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.HybridCache` with python objects."""
    ca = _cache_creator.CacheKeyValue(cache)
    flat = [("key_cache", ca.key_cache), ("value_cache", ca.value_cache)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_hybrid_cache(
    cache: HybridCache,
) -> tuple[list[tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.HybridCache` with python objects."""
    values, context = flatten_hybrid_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_hybrid_cache(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> HybridCache:
    """Restores a :class:`transformers.cache_utils.HybridCache` from python objects."""
    return _cache_creator.make_hybrid_cache(list(zip(values[0], values[1])))


#############
# StaticCache
#############


def flatten_static_cache(
    cache: StaticCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.StaticCache` with python objects."""
    ca = _cache_creator.CacheKeyValue(cache)
    assert not ca.key_cache or cache.max_cache_len == ca.key_cache[0].shape[2], (
        f"Serialization doet not work when "
        f"cache.max_cache_len={cache.max_cache_len} != "
        f"cache.key_cache[0].shape[2]={ca.keu_cache[0].shape[2]}"
    )
    flat = [("key_cache", ca.key_cache), ("value_cache", ca.value_cache)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_static_cache(
    cache: StaticCache,
) -> tuple[list[tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.StaticCache` with python objects."""
    values, context = flatten_static_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_static_cache(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> StaticCache:
    """Restores a :class:`transformers.cache_utils.StaticCache` from python objects."""
    return _cache_creator.make_static_cache(
        list(zip(values[0], values[1])), max_cache_len=values[0][0].shape[2]
    )


####################
# SlidingWindowCache
####################


def flatten_sliding_window_cache(
    cache: SlidingWindowCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.SlidingWindowCache`
    with python objects.
    """
    ca = _cache_creator.CacheKeyValue(cache)
    flat = [("key_cache", ca.key_cache), ("value_cache", ca.value_cache)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_sliding_window_cache(
    cache: SlidingWindowCache,
) -> tuple[list[tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.SlidingWindowCache`
    with python objects.
    """
    values, context = flatten_sliding_window_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_sliding_window_cache(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> SlidingWindowCache:
    """Restores a :class:`transformers.cache_utils.SlidingWindowCache` from python objects."""
    key_cache, value_cache = values
    return _cache_creator.make_sliding_window_cache(list(zip(values[0], values[1])))


#####################
# EncoderDecoderCache
#####################


def flatten_encoder_decoder_cache(
    ec_cache: EncoderDecoderCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.EncoderDecoderCache`
    with python objects.
    """
    dictionary = {
        "self_attention_cache": ec_cache.self_attention_cache,
        "cross_attention_cache": ec_cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten(dictionary)


def flatten_with_keys_encoder_decoder_cache(
    ec_cache: EncoderDecoderCache,
) -> tuple[
    list[tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`transformers.cache_utils.EncoderDecoderCache`
    with python objects.
    """
    dictionary = {
        "self_attention_cache": ec_cache.self_attention_cache,
        "cross_attention_cache": ec_cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten_with_keys(dictionary)


def unflatten_encoder_decoder_cache(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> EncoderDecoderCache:
    """Restores a :class:`transformers.cache_utils.EncoderDecoderCache` from python objects."""
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    return EncoderDecoderCache(**dictionary)


#############
# dataclasses
#############


def lower_name_with_(name):
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def make_serialization_function_for_dataclass(
    cls: type, supported_classes: set[type]
) -> tuple[Callable, Callable, Callable]:
    """Automatically creates serialization function for a class decorated with
    ``dataclasses.dataclass``.
    """

    def flatten_cls(obj: cls) -> tuple[list[Any], torch.utils._pytree.Context]:  # type: ignore[valid-type]
        """Serializes a ``%s`` with python objects."""
        return list(obj.values()), list(obj.keys())

    def flatten_with_keys_cls(
        obj: cls,  # type: ignore[valid-type]
    ) -> tuple[list[tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
        """Serializes a ``%s`` with python objects with keys."""
        values, context = list(obj.values()), list(obj.keys())
        return [
            (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
        ], context

    def unflatten_cls(
        values: list[Any], context: torch.utils._pytree.Context, output_type=None
    ) -> cls:  # type: ignore[valid-type]
        """Restores an instance of ``%s`` from python objects."""
        return cls(**dict(zip(context, values)))

    name = lower_name_with_(cls.__name__)
    flatten_cls.__name__ = f"flatten_{name}"
    flatten_with_keys_cls.__name__ = f"flatten_with_keys_{name}"
    unflatten_cls.__name__ = f"unflatten_{name}"
    flatten_cls.__doc__ = flatten_cls.__doc__ % cls.__name__
    flatten_with_keys_cls.__doc__ = flatten_with_keys_cls.__doc__ % cls.__name__
    unflatten_cls.__doc__ = unflatten_cls.__doc__ % cls.__name__
    supported_classes.add(cls)
    return flatten_cls, flatten_with_keys_cls, unflatten_cls


(
    flatten_base_model_output,
    flatten_with_keys_base_model_output,
    unflatten_base_model_output,
) = make_serialization_function_for_dataclass(BaseModelOutput, SUPPORTED_DATACLASSES)
