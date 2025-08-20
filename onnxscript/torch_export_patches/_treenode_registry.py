import pprint
from typing import Any, Callable

import optree
import torch
import transformers
from diffusers.models.unets import unet_2d_condition
from packaging import version as pv
from transformers import cache_utils, modeling_outputs

from onnxscript.torch_export_patches import _treenode_transformers

try:
    from transformers.models.mamba.modeling_mamba import MambaCache
except ImportError:
    from transformers.cache_utils import MambaCache


PATCH_OF_PATCHES: set[Any] = set()


def register_class_serialization(
    cls,
    f_flatten: Callable,
    f_unflatten: Callable,
    f_flatten_with_keys: Callable,
    f_check: Callable | None = None,
    verbose: int = 0,
) -> bool:
    """Registers a class.

    It can be undone with
    :func:`onnxscript.torch_export_patches._treenode_registry.unregister_class_serialization`.

    Args:
        cls: class to register
        f_flatten: see ``torch.utils._pytree.register_pytree_node``
        f_unflatten: see ``torch.utils._pytree.register_pytree_node``
        f_flatten_with_keys: see ``torch.utils._pytree.register_pytree_node``
        f_check: called to check the registration was successful
        verbose: verbosity

    Returns:
        registered or not
    """
    if cls is not None and cls in torch.utils._pytree.SUPPORTED_NODES:
        if verbose and cls is not None:
            print(f"[register_class_serialization] already registered {cls.__name__}")
        return False

    if verbose:
        print(f"[register_class_serialization] ---------- register {cls.__name__}")
    torch.utils._pytree.register_pytree_node(
        cls,
        f_flatten,
        f_unflatten,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
        flatten_with_keys_fn=f_flatten_with_keys,
    )
    if pv.Version(torch.__version__) < pv.Version("2.7"):
        if verbose:
            print(
                f"[register_class_serialization] ---------- register {cls.__name__} for torch=={torch.__version__}"
            )
        torch.fx._pytree.register_pytree_flatten_spec(cls, lambda x, _: f_flatten(x)[0])

    # check
    if f_check:
        raise NotImplementedError(
            "The check function is not implemented. Please provide a valid check function to verify the registration."
        )
        # inst = f_check()
        # values, spec = torch.utils._pytree.tree_flatten(inst)
        # restored = torch.utils._pytree.tree_unflatten(values, spec)
        # assert string_type(inst, with_shape=True) == string_type(restored, with_shape=True), (
        #     f"Issue with registration of class {cls} "
        #     f"inst={string_type(inst, with_shape=True)}, "
        #     f"restored={string_type(restored, with_shape=True)}"
        # )
    return True


def register_cache_serialization(
    patch_transformers: bool = False, patch_diffusers: bool = True, verbose: int = 0
) -> dict[str, bool]:
    """Registers many classes with register_class_serialization.

    Returns information needed to undo the registration.

    Args:
        patch_transformers: Add serialization function for transformers package.
        patch_diffusers: Add serialization function for diffusers package.
        verbose: Verbosity level.

    Returns:
        Information to unpatch.
    """
    wrong: dict[type, str | None] = {}
    if patch_transformers:
        _wrong_registration = {
            cache_utils.DynamicCache: "4.50",
            modeling_outputs.BaseModelOutput: None,
        }

        wrong |= _wrong_registration
    if patch_diffusers:

        def _make_wrong_registrations() -> dict[type, str | None]:
            res: dict[type, str | None] = {}
            for c in [unet_2d_condition.UNet2DConditionOutput]:
                if c is not None:
                    res[c] = None
            return res

        wrong |= _make_wrong_registrations()

    registration_functions = serialization_functions(
        patch_transformers=patch_transformers, patch_diffusers=patch_diffusers, verbose=verbose
    )

    # DynamicCache serialization is different in transformers and does not
    # play way with torch.export.export.
    # see test test_export_dynamic_cache_cat with NOBYPASS=1
    # :: NOBYBASS=1 python _unittests/ut_torch_export_patches/test_dynamic_class.py -k e_c
    # This is caused by this line:
    # torch.fx._pytree.register_pytree_flatten_spec(
    #           DynamicCache, _flatten_dynamic_cache_for_fx)
    # so we remove it anyway
    # BaseModelOutput serialization is incomplete.
    # It does not include dynamic shapes mapping.
    for cls, version in wrong.items():
        if (
            cls in torch.utils._pytree.SUPPORTED_NODES
            and cls not in PATCH_OF_PATCHES
            # and pv.Version(torch.__version__) < pv.Version("2.7")
            and (
                version is None or pv.Version(transformers.__version__) >= pv.Version(version)
            )
        ):
            assert cls in registration_functions, (
                f"{cls} has no registration functions mapped to it, "
                f"available options are {list(registration_functions)}"
            )
            if verbose:
                print(
                    f"[_fix_registration] {cls.__name__} is unregistered and registered first"
                )
            unregister_class_serialization(cls, verbose=verbose)
            registration_functions[cls](verbose=verbose)  # type: ignore[arg-type, call-arg]
            if verbose:
                print(f"[_fix_registration] {cls.__name__} done.")
            # To avoid doing it multiple times.
            PATCH_OF_PATCHES.add(cls)

    # classes with no registration at all.
    done = {}
    for k, v in registration_functions.items():
        done[k] = v(verbose=verbose)  # type: ignore[arg-type, call-arg]
    return done


def serialization_functions(
    patch_transformers: bool = False, patch_diffusers: bool = False, verbose: int = 0
) -> dict[type, Callable[[int], bool]]:
    """Returns the list of serialization functions."""
    supported_classes: set[type] = set()
    classes: dict[type, Callable[[int], bool]] = {}
    all_functions: dict[type, str | None] = {}

    if patch_transformers:
        from onnxscript.torch_export_patches._treenode_transformers import (
            SUPPORTED_DATACLASSES,
            flatten_dynamic_cache,
            flatten_encoder_decoder_cache,
            flatten_hybrid_cache,
            flatten_mamba_cache,
            flatten_sliding_window_cache,
            flatten_static_cache,
            flatten_with_keys_dynamic_cache,
            flatten_with_keys_encoder_decoder_cache,
            flatten_with_keys_hybrid_cache,
            flatten_with_keys_mamba_cache,
            flatten_with_keys_sliding_window_cache,
            flatten_with_keys_static_cache,
            unflatten_dynamic_cache,
            unflatten_encoder_decoder_cache,
            unflatten_hybrid_cache,
            unflatten_mamba_cache,
            unflatten_sliding_window_cache,
            unflatten_static_cache,
        )
        from onnxscript.torch_export_patches._treenode_transformers import (
            __dict__ as dtr,
        )

        all_functions.update(dtr)
        supported_classes |= SUPPORTED_DATACLASSES

        transformers_classes = {
            cache_utils.DynamicCache: lambda verbose=verbose: register_class_serialization(
                cache_utils.DynamicCache,
                flatten_dynamic_cache,
                unflatten_dynamic_cache,
                flatten_with_keys_dynamic_cache,
                # f_check=make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
                verbose=verbose,
            ),
            cache_utils.HybridCache: lambda verbose=verbose: register_class_serialization(
                cache_utils.HybridCache,
                flatten_hybrid_cache,
                unflatten_hybrid_cache,
                flatten_with_keys_hybrid_cache,
                # f_check=make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
                verbose=verbose,
            ),
            MambaCache: lambda verbose=verbose: register_class_serialization(
                MambaCache,
                flatten_mamba_cache,
                unflatten_mamba_cache,
                flatten_with_keys_mamba_cache,
                verbose=verbose,
            ),
            cache_utils.EncoderDecoderCache: lambda verbose=verbose: register_class_serialization(
                cache_utils.EncoderDecoderCache,
                flatten_encoder_decoder_cache,
                unflatten_encoder_decoder_cache,
                flatten_with_keys_encoder_decoder_cache,
                verbose=verbose,
            ),
            cache_utils.SlidingWindowCache: lambda verbose=verbose: register_class_serialization(
                cache_utils.SlidingWindowCache,
                flatten_sliding_window_cache,
                unflatten_sliding_window_cache,
                flatten_with_keys_sliding_window_cache,
                verbose=verbose,
            ),
            cache_utils.StaticCache: lambda verbose=verbose: register_class_serialization(
                cache_utils.StaticCache,
                flatten_static_cache,
                unflatten_static_cache,
                flatten_with_keys_static_cache,
                verbose=verbose,
            ),
        }
        classes.update(transformers_classes)

    if patch_diffusers:
        from onnxscript.torch_export_patches._treenode_diffusers import SUPPORTED_DATACLASSES
        from onnxscript.torch_export_patches._treenode_diffusers import __dict__ as dfu

        all_functions.update(dfu)
        supported_classes |= SUPPORTED_DATACLASSES

    for cls in supported_classes:
        lname = _treenode_transformers.lower_name_with_(cls.__name__)
        assert f"flatten_{lname}" in all_functions, (
            f"Unable to find function 'flatten_{lname}' in {list(all_functions)}"
        )
        classes[cls] = (
            lambda verbose=verbose,
            _ln=lname,
            cls=cls,
            _al=all_functions: register_class_serialization(
                cls,
                _al[f"flatten_{_ln}"],
                _al[f"unflatten_{_ln}"],
                _al[f"flatten_with_keys_{_ln}"],
                verbose=verbose,
            )
        )
    return classes


def unregister_class_serialization(cls: type, verbose: int = 0):
    """Undo the registration."""
    # torch.utils._pytree._deregister_pytree_flatten_spec(cls)
    if cls in torch.fx._pytree.SUPPORTED_NODES:
        del torch.fx._pytree.SUPPORTED_NODES[cls]
    if cls in torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH:
        del torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH[cls]
    if hasattr(torch.utils._pytree, "_deregister_pytree_node"):
        # torch >= 2.7
        torch.utils._pytree._deregister_pytree_node(cls)
    else:
        if cls in torch.utils._pytree.SUPPORTED_NODES:
            del torch.utils._pytree.SUPPORTED_NODES[cls]
    optree.unregister_pytree_node(cls, namespace="torch")
    if cls in torch.utils._pytree.SUPPORTED_NODES:
        import packaging.version as pv

        if pv.Version(torch.__version__) < pv.Version("2.7.0"):
            del torch.utils._pytree.SUPPORTED_NODES[cls]
    assert cls not in torch.utils._pytree.SUPPORTED_NODES, (
        f"{cls} was not successful unregistered "
        f"from torch.utils._pytree.SUPPORTED_NODES="
        f"{pprint.pformat(list(torch.utils._pytree.SUPPORTED_NODES))}"
    )
    if verbose:
        print(f"[unregister_cache_serialization] unregistered {cls.__name__}")


def unregister_cache_serialization(undo: dict[str, bool], verbose: int = 0):
    """Undo all registrations."""
    cls_ensemble = {
        MambaCache,
        cache_utils.DynamicCache,
        cache_utils.EncoderDecoderCache,
    } | set(undo)
    for cls in cls_ensemble:
        if undo.get(cls.__name__):
            unregister_class_serialization(cls, verbose)
