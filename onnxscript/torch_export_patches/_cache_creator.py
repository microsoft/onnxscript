from typing import Any, Callable

import packaging.version as pv
import torch
import transformers
import transformers.cache_utils

try:
    from transformers.models.mamba.modeling_mamba import MambaCache
except ImportError:
    from transformers.cache_utils import MambaCache


class CacheKeyValue:
    """Cache wrapper for compatibility across transformers versions.

    Starting transformers>=4.54, the cache API has deprecated
    ``cache.key_cache`` and ``cache.value_cache``.
    This class wraps a cache independently from transformers version and enables
    attributes ``key_cache`` and ``value_cache``.

    Example:
        .. code-block:: python

            capi = CacheKeyValue(cache)
            capi.key_cache
            capi.value_cache
    """

    def __init__(self, cache=None):
        if hasattr(cache, "layers"):
            layers = [
                layer
                for layer in cache.layers
                if layer is not None and layer.keys is not None and layer.values is not None
            ]
            self.key_cache = [layer.keys for layer in layers]
            self.value_cache = [layer.values for layer in layers]
            if None in self.key_cache or None in self.value_cache:
                from .helper import string_type

                raise AssertionError(
                    f"issue with key_cache={string_type(self.key_cache)}, "
                    f"or value_cache={string_type(self.value_cache)}, "
                    f"cache.layers={string_type(cache.layers)}"
                )
        elif cache is not None:
            self.key_cache = cache.key_cache
            self.value_cache = cache.value_cache

    def make_dynamic_cache(self):
        """Create a dynamic cache from the wrapped cache.

        Returns:
            transformers.cache_utils.DynamicCache: The dynamic cache created from key-value pairs.
        """
        return make_dynamic_cache(list(zip(self.key_cache, self.value_cache)))


if pv.Version(transformers.__version__) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """Create an instance of transformers.cache_utils.DynamicCache.

        This version is valid for transformers >= 4.50.

        Args:
            key_value_pairs: List of pairs of (key, values).

        Returns:
            transformers.cache_utils.DynamicCache: The created dynamic cache.

        Example:
            .. runpython::
                :showcode:

                import torch
                from onnxscript.torch_export_patches._verbose import string_type
                from onnxscript.torch_export_patches._cache_creator import make_dynamic_cache

                n_layers = 2
                bsize, nheads, slen, dim = 2, 4, 3, 7

                past_key_values = make_dynamic_cache(
                    [
                        (
                            torch.randn(bsize, nheads, slen, dim),
                            torch.randn(bsize, nheads, slen, dim),
                        )
                        for i in range(n_layers)
                    ]
                )
                print(string_type(past_key_values, with_shape=True))
        """
        cache = transformers.cache_utils.DynamicCache(key_value_pairs)
        if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
            # The cache constructor contains the two following lines
            # (in cache_utils.py) which append empty layers when the cache is
            # initialized. We need to remove them.
            # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
            # self.append_new_layers(self.num_hidden_layers - 1)
            cache.layers[:] = cache.layers[-len(key_value_pairs) :]
        assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
            f"Unexpected number of layers in the cache ({len(cache.layers)}), {len(key_value_pairs)} expected."
        )
        return cache

else:

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """Create an instance of transformers.cache_utils.DynamicCache.

        This version is valid for transformers < 4.50.

        Args:
            key_value_pairs: List of pairs of (key, values).

        Returns:
            transformers.cache_utils.DynamicCache: The created dynamic cache.

        Example:
            .. runpython::
                :showcode:

                import torch
                from onnxscript.torch_export_patches._verbose import string_type
                from onnxscript.torch_export_patches._cache_creator import make_dynamic_cache

                n_layers = 2
                bsize, nheads, slen, dim = 2, 4, 3, 7

                past_key_values = make_dynamic_cache(
                    [
                        (
                            torch.randn(bsize, nheads, slen, dim),
                            torch.randn(bsize, nheads, slen, dim),
                        )
                        for i in range(n_layers)
                    ]
                )
                print(string_type(past_key_values, with_shape=True))
        """
        cache = transformers.cache_utils.DynamicCache(len(key_value_pairs))  # type: ignore
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache


def make_static_cache(
    key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    max_cache_len: int | None = None,
) -> transformers.cache_utils.DynamicCache:
    """Create an instance of transformers.cache_utils.StaticCache.

    Args:
        key_value_pairs: List of pairs of (key, values).
        max_cache_len: Max cache length or something inferred from the vector.

    Returns:
        transformers.cache_utils.StaticCache: The created static cache.

    Example:
        .. runpython::
            :showcode:

            import torch
            from onnxscript.torch_export_patches._verbose import string_type
            from onnxscript.torch_export_patches._cache_creator import make_static_cache

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_static_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ],
                max_cache_len=10,
            )
            print(string_type(past_key_values, with_shape=True))
    """

    class Config:
        def __init__(self):
            self.head_dim = key_value_pairs[0][0].shape[-1]
            self.num_attention_heads = key_value_pairs[0][0].shape[1]
            self.num_hidden_layers = len(key_value_pairs)

    assert max_cache_len is not None, (
        f"max_cache_len={max_cache_len} cannot be setup automatically yet from shape {key_value_pairs[0][0].shape}"
    )
    torch._check(
        max_cache_len >= key_value_pairs[0][0].shape[2],
        (
            f"max_cache_len={max_cache_len} cannot be smaller "
            f"shape[2]={key_value_pairs[0][0].shape[2]} in shape "
            f"{key_value_pairs[0][0].shape}"
        ),
    )
    cache = transformers.cache_utils.StaticCache(
        config=Config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        device=key_value_pairs[0][0].device,
        dtype=key_value_pairs[0][0].dtype,
        max_cache_len=max_cache_len,
    )
    ca = CacheKeyValue(cache)
    for i in range(len(key_value_pairs)):
        assert key_value_pairs[i][0].shape == key_value_pairs[i][1].shape, (
            f"Shape mismatch {key_value_pairs[i][0].shape} != {key_value_pairs[i][1].shape}"
        )
        d = key_value_pairs[i][1].shape[2]
        ca.key_cache[i][:, :, :d, :] = key_value_pairs[i][0]
        ca.value_cache[i][:, :, :d, :] = key_value_pairs[i][1]
    if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
        # The cache constructor contains the two following lines
        # (in cache_utils.py) which append empty layers when the cache is
        # initialized. We need to remove them.
        # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
        # self.append_new_layers(self.num_hidden_layers - 1)
        cache.layers[:] = cache.layers[-len(key_value_pairs) :]
    assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
        f"Unexpected number of layers in the cache ({len(cache.layers)}), {len(key_value_pairs)} expected."
    )
    return cache


def make_encoder_decoder_cache(
    self_attention_cache: transformers.cache_utils.DynamicCache,
    cross_attention_cache: transformers.cache_utils.DynamicCache,
) -> transformers.cache_utils.EncoderDecoderCache:
    """Create an EncoderDecoderCache.

    Args:
        self_attention_cache: The self-attention cache.
        cross_attention_cache: The cross-attention cache.

    Returns:
        transformers.cache_utils.EncoderDecoderCache: The created encoder-decoder cache.
    """
    return transformers.cache_utils.EncoderDecoderCache(
        self_attention_cache=self_attention_cache, cross_attention_cache=cross_attention_cache
    )


def make_mamba_cache(key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> MambaCache:
    """Create a MambaCache.

    Args:
        key_value_pairs: List of pairs of (key, values).

    Returns:
        MambaCache: The created Mamba cache.
    """
    dtype = key_value_pairs[0][0].dtype

    class Config:
        def __init__(self):
            self.intermediate_size = key_value_pairs[0][0].shape[1]
            self.conv_kernel = key_value_pairs[0][0].shape[-1]
            self.state_size = key_value_pairs[0][1].shape[-1]
            self.num_hidden_layers = len(key_value_pairs)
            self.dtype = dtype

    cache = MambaCache(
        Config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        device=key_value_pairs[0][0].device,
        dtype=dtype,
    )
    for i in range(len(key_value_pairs)):
        assert cache.conv_states[i].dtype == dtype, (
            f"Type mismatch for cache.conv_states[{i}].dtype={cache.conv_states[i].dtype} != {dtype}"
        )
        assert cache.ssm_states[i].dtype == dtype, (
            f"Type mismatch for cache.ssm_states[{i}].dtype={cache.ssm_states[i].dtype} != {dtype}"
        )
        assert cache.conv_states[i].shape == key_value_pairs[i][0].shape, (
            f"Shape mismatch, expected {cache.conv_states[i].shape}, got {key_value_pairs[i][0].shape}"
        )
        cache.conv_states[i][:, :, :] = key_value_pairs[i][0]
        assert cache.ssm_states[i].shape == key_value_pairs[i][1].shape, (
            f"Shape mismatch, expected {cache.ssm_states[i].shape}, got {key_value_pairs[i][1].shape}"
        )
        cache.ssm_states[i][:, :, :] = key_value_pairs[i][1]
    return cache


def make_sliding_window_cache(
    key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> transformers.cache_utils.SlidingWindowCache:
    """Create a transformers.cache_utils.SlidingWindowCache.

    Args:
        key_value_pairs: List of pairs of (key, values).

    Returns:
        transformers.cache_utils.SlidingWindowCache: The created sliding window cache.
    """

    class Config:
        def __init__(self):
            self.head_dim = key_value_pairs[0][0].shape[-1]
            self.num_attention_heads = key_value_pairs[0][0].shape[1]
            self.num_hidden_layers = len(key_value_pairs)
            self.sliding_window = key_value_pairs[0][0].shape[2]

    cache = transformers.cache_utils.SlidingWindowCache(
        config=Config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        max_cache_len=key_value_pairs[0][0].shape[2],  # same as sliding_window
        device=key_value_pairs[0][0].device,
        dtype=key_value_pairs[0][0].dtype,
    )
    ca = CacheKeyValue(cache)
    for i in range(len(key_value_pairs)):
        assert ca.key_cache[i].shape == key_value_pairs[i][0].shape, (
            f"Shape mismatch, expected {cache.key_cache[i].shape}, got {key_value_pairs[i][0].shape}"
        )
        ca.key_cache[i][:, :, :, :] = key_value_pairs[i][0]
        assert ca.value_cache[i].shape == key_value_pairs[i][1].shape, (
            f"Shape mismatch, expected {cache.value_cache[i].shape}, got {key_value_pairs[i][1].shape}"
        )
        ca.value_cache[i][:, :, :, :] = key_value_pairs[i][1]
    if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
        # The cache constructor contains the two following lines
        # (in cache_utils.py) which append empty layers when the cache is
        # initialized. We need to remove them.
        # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
        # self.append_new_layers(self.num_hidden_layers - 1)
        cache.layers[:] = cache.layers[-len(key_value_pairs) :]
    assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
        f"Unexpected number of layers in the cache ({len(cache.layers)}), {len(key_value_pairs)} expected."
    )
    return cache


def make_hybrid_cache(
    key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    max_cache_len: int | None = None,
    max_batch_size: int | None = None,
    sliding_window: int | None = None,
) -> transformers.cache_utils.HybridCache:
    """Create an instance of transformers.cache_utils.HybridCache.

    This version is valid for transformers < 4.50.

    Args:
        key_value_pairs: List of pairs of (key, values).
        max_cache_len: Maximum cache length.
        max_batch_size: Maximum batch size.
        sliding_window: Sliding window size.

    Returns:
        transformers.cache_utils.HybridCache: The created hybrid cache.

    Example:
        .. runpython::
            :showcode:

            import torch
            from onnxscript.torch_export_patches._verbose import string_type
            from onnxscript.torch_export_patches._cache_creator import make_hybrid_cache

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_hybrid_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ]
            )
            print(string_type(past_key_values, with_shape=True))

    Note:
        This part defines how the shapes are working in one HybridCache:

        .. code-block:: python

                self.max_cache_len = (
                    max_cache_len if max_cache_len is not None else config.max_position_embeddings)

                # Sliding layers can't be larger than the overall max cache len
                self.sliding_window_len = min(config.sliding_window, self.max_cache_len)
                self.max_batch_size = max_batch_size

                self.head_dim = (
                    config.head_dim if hasattr(config, "head_dim")
                    else config.hidden_size // config.num_attention_heads
                )

                self._dtype = dtype
                self.num_key_value_heads = (
                    config.num_attention_heads
                    if getattr(config, "num_key_value_heads", None) is None
                    else config.num_key_value_heads
                )

                # If the attribute does not exist in the config, fallback to a simple StaticCache
                if hasattr(config, "layer_types"):
                    self.is_sliding = [
                        layer_type != "full_attention" for layer_type in config.layer_types]
                else:
                    self.is_sliding = [False] * config.num_hidden_layers

                self.key_cache: list[torch.Tensor] = []
                self.value_cache: list[torch.Tensor] = []
                global_cache_shape = (self.max_batch_size, self.num_key_value_heads,
                                      self.max_cache_len, self.head_dim)
                sliding_cache_shape = (self.max_batch_size, self.num_key_value_heads,
                                       self.sliding_window_len, self.head_dim)
                self.sliding_window = min(config.sliding_window, max_cache_len)
                device = torch.device(device) if device is not None else None
                for i in range(config.num_hidden_layers):
                    layer_device = layer_device_map[i] if layer_device_map is not None else device
                    cache_shape = sliding_cache_shape if self.is_sliding[i] else global_cache_shape
                    new_layer_key_cache = torch.zeros(
                        cache_shape, dtype=self._dtype, device=layer_device)
                    new_layer_value_cache = torch.zeros(
                        cache_shape, dtype=self._dtype, device=layer_device)
                    torch._dynamo.mark_static_address(new_layer_key_cache)
                    torch._dynamo.mark_static_address(new_layer_value_cache)
                    self.key_cache.append(new_layer_key_cache)
                    self.value_cache.append(new_layer_value_cache)
    """
    layer_types = None
    if key_value_pairs:
        assert not max_batch_size and not max_cache_len, (
            "key_value_pairs is not empty, do not specify max_cache_len and max_batch_size"
        )
        max_batch_size = key_value_pairs[0][0].shape[0]
        sets_of_dim = {kv[0].shape[2] for kv in key_value_pairs}
        if len(sets_of_dim) == 1:
            max_cache_len = sets_of_dim.pop()
            sliding_window = max_cache_len
        else:
            assert len(sets_of_dim) == 2, (
                f"Not implemented for more than 2 dimensions {sets_of_dim}"
            )
            max_cache_len = max(sets_of_dim)
            sliding_window = min(sets_of_dim)
            layer_types = [
                "full_attention" if i == max_cache_len else "sliding_attention"
                for i in [kv[0].shape[2] for kv in key_value_pairs]
            ]
    else:
        assert max_batch_size and max_cache_len, (
            "key_value_pairs is empty, max_batch_size and max_cache_len are required"
        )
        if sliding_window is None:
            sliding_window = max_cache_len
    _max_cache_len = max_cache_len
    _sliding_window = sliding_window

    class Config:
        max_cache_len = _max_cache_len
        batch_size = max_batch_size
        num_heads = key_value_pairs[0][0].shape[1] if key_value_pairs else None
        head_dim = key_value_pairs[0][0].shape[-1] if key_value_pairs else None
        num_attention_heads = key_value_pairs[0][1].shape[1] if key_value_pairs else None
        num_hidden_layers = len(key_value_pairs)
        sliding_window = _sliding_window
        num_key_value_heads = key_value_pairs[0][1].shape[1]  # transformers 4.48.3

    if layer_types:
        Config.layer_types = layer_types  # type: ignore[attr-defined]

    cache = transformers.cache_utils.HybridCache(
        config=Config(), max_cache_len=max_cache_len, max_batch_size=max_batch_size
    )
    for i, (key, value) in enumerate(key_value_pairs):
        cache.update(
            key,
            value,
            i,
            cache_kwargs={
                "cache_position": torch.arange(0, key.shape[2], dtype=torch.int64).to(
                    key.device
                )
            },
        )
    if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
        # The cache constructor contains the two following lines
        # (in cache_utils.py) which append empty layers when the cache is
        # initialized. We need to remove them.
        # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
        # self.append_new_layers(self.num_hidden_layers - 1)
        cache.layers[:] = cache.layers[-len(key_value_pairs) :]
    assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
        f"Unexpected number of layers in the cache ({len(cache.layers)}), {len(key_value_pairs)} expected."
    )
    return cache


def flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    use_dict: bool = False,
    change_function: Callable[[torch.Tensor], Any] | None = None,
) -> Any:
    """Returns the object in a different structure similar to what the definition of the dynamic shapes should use.

    Args:
        obj: object from a custom class
        use_dict: closer to the original result but
            :func:`torch.export.export` only considers the values,
            the context gives the dictionary keys but it is not expressed
            in the dynamic shapes, these specifications seems to be different
            for the strict and non strict mode. It also preserves tuple.
        change_function: to modifies the tensor in the structure itself,
            like replace them by a shape

    Returns:
        The serialized object
    """
    if isinstance(obj, torch.Tensor):
        return change_function(obj) if change_function else obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in spec.children_specs:
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = flatten_unflatten_for_dynamic_shapes(
            value, use_dict=use_dict, change_function=change_function
        )
        subtrees.append(value)
        start = end
    if use_dict:
        if spec.type is dict:
            # This a dictionary.
            return dict(zip(spec.context, subtrees))
        if spec.type is tuple:
            return tuple(subtrees)
        if spec.type is list:
            return list(subtrees)
        if spec.context:
            # This is a custom class with attributes.
            # It is returned as a list.
            return list(subtrees)
        raise ValueError(
            f"Unable to interpret spec type {spec.type} (type is {type(spec.type)}, context is {spec.context})."
        )
    # This is a list.
    return subtrees
