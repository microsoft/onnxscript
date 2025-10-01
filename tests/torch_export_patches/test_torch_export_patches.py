import unittest
from typing import Any, ClassVar

import numpy as np
import torch
import torch.utils._pytree as py_pytree

from onnxscript.torch_export_patches._core import torch_export_patches


class TestTorchExportPatchesComprehensive(unittest.TestCase):
    """Comprehensive tests for torch_export_patches functionality."""

    def assertEqualArrayAny(
        self, expected: Any, value: Any, atol: float = 0, rtol: float = 0, msg: str = ""
    ):
        if isinstance(expected, (tuple, list, dict)):
            self.assertIsInstance(value, type(expected), msg=msg)
            self.assertEqual(len(expected), len(value), msg=msg)
            if isinstance(expected, dict):
                for k in expected:
                    self.assertIn(k, value, msg=msg)
                    self.assertEqualArrayAny(
                        expected[k], value[k], msg=msg, atol=atol, rtol=rtol
                    )
            else:
                excs = []
                for i, (e, g) in enumerate(zip(expected, value)):
                    try:
                        self.assertEqualArrayAny(e, g, msg=msg, atol=atol, rtol=rtol)
                    except AssertionError as e:
                        excs.append(f"Error at position {i} due to {e}")
                if excs:
                    msg_ = "\n".join(excs)
                    msg = f"{msg}\n{msg_}" if msg else msg_
                    raise AssertionError(f"Found {len(excs)} discrepancies\n{msg}")
        elif expected.__class__.__name__ in (
            "DynamicCache",
            "StaticCache",
            "HybridCache",
            "SlidingWindowCache",
        ):
            atts = {"key_cache", "value_cache"}
            self.assertEqualArrayAny(
                {k: expected.__dict__.get(k, None) for k in atts},
                {k: value.__dict__.get(k, None) for k in atts},
                atol=atol,
                rtol=rtol,
            )
        elif isinstance(expected, (int, float, str)):
            self.assertEqual(expected, value, msg=msg)
        elif hasattr(expected, "shape"):
            self.assertEqual(type(expected), type(value), msg=msg)
            self.assertEqualArray(expected, value, msg=msg, atol=atol, rtol=rtol)
        elif expected is None:
            assert value is None, f"Expected is None but value is of type {type(value)}"
        else:
            raise AssertionError(
                f"Comparison not implemented for types {type(expected)} and {type(value)}"
            )

    def assertEqualArray(
        self,
        expected: Any,
        value: Any,
        atol: float = 0,
        rtol: float = 0,
        msg: str | None = None,
    ):
        if hasattr(expected, "detach") and hasattr(value, "detach"):
            if msg:
                try:
                    self.assertEqual(expected.dtype, value.dtype)
                except AssertionError as e:
                    raise AssertionError(msg) from e
                try:
                    self.assertEqual(expected.shape, value.shape)
                except AssertionError as e:
                    raise AssertionError(msg) from e
            else:
                self.assertEqual(expected.dtype, value.dtype)
                self.assertEqual(expected.shape, value.shape)

            try:
                torch.testing.assert_close(value, expected, atol=atol, rtol=rtol)
            except AssertionError as e:
                expected_max = torch.abs(expected).max()
                expected_value = torch.abs(value).max()
                rows = [
                    f"{msg}\n{e}" if msg else str(e),
                    f"expected max value={expected_max}",
                    f"expected computed value={expected_value}",
                ]
                raise AssertionError("\n".join(rows))
            return

        if hasattr(expected, "detach"):
            expected = expected.detach().cpu().numpy()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if msg:
            try:
                self.assertEqual(expected.dtype, value.dtype)
            except AssertionError as e:
                raise AssertionError(msg) from e
            try:
                self.assertEqual(expected.shape, value.shape)
            except AssertionError as e:
                raise AssertionError(msg) from e
        else:
            self.assertEqual(expected.dtype, value.dtype)
            self.assertEqual(expected.shape, value.shape)

        try:
            np.testing.assert_allclose(desired=expected, actual=value, atol=atol, rtol=rtol)
        except AssertionError as e:
            expected_max = np.abs(expected).max()
            expected_value = np.abs(value).max()
            te = expected.astype(int) if expected.dtype == np.bool_ else expected
            tv = value.astype(int) if value.dtype == np.bool_ else value
            rows = [
                f"{msg}\n{e}" if msg else str(e),
                f"expected max value={expected_max}",
                f"expected computed value={expected_value}\n",
                f"ratio={te / tv}\ndiff={te - tv}",
            ]
            raise AssertionError("\n".join(rows))

    def test_context_manager_basic_functionality(self):
        """Test that torch_export_patches context manager works correctly."""
        with torch_export_patches(verbose=0) as modificator:
            self.assertIsNotNone(modificator)
            self.assertTrue(callable(modificator))

        # Test without patching
        with torch_export_patches(patch=False, verbose=0) as modificator:
            self.assertIsNotNone(modificator)
            self.assertTrue(callable(modificator))

    def test_patch_flags_combinations(self):
        """Test different combinations of patch flags."""
        test_cases = [
            {
                "patch_sympy": True,
                "patch_torch": False,
                "patch_transformers": False,
                "patch_diffusers": False,
            },
            {
                "patch_sympy": False,
                "patch_torch": True,
                "patch_transformers": False,
                "patch_diffusers": False,
            },
            {
                "patch_sympy": False,
                "patch_torch": False,
                "patch_transformers": True,
                "patch_diffusers": False,
            },
            {
                "patch_sympy": False,
                "patch_torch": False,
                "patch_transformers": False,
                "patch_diffusers": True,
            },
            {
                "patch_sympy": True,
                "patch_torch": True,
                "patch_transformers": True,
                "patch_diffusers": True,
            },
        ]

        for config in test_cases:
            with self.subTest(config=config), torch_export_patches(
                **config, verbose=0
            ) as modificator:
                self.assertIsNotNone(modificator)

    def test_static_cache_serialization(self):
        """Test StaticCache serialization and deserialization."""
        try:
            from transformers.cache_utils import StaticCache
        except ImportError:
            self.skipTest("StaticCache not available")

        class Config:
            def __init__(self):
                self.num_attention_heads = 4
                self.num_key_value_heads = 4
                self.hidden_size = 64
                self.head_dim = 16
                self.num_hidden_layers = 3

        config = Config()
        cache = StaticCache(
            config, max_batch_size=2, max_cache_len=5, device="cpu", dtype=torch.float32
        )

        # Add some data
        key_states = torch.randn(2, 4, 5, 16)
        value_states = torch.randn(2, 4, 5, 16)
        cache.update(key_states, value_states, layer_idx=0)

        with torch_export_patches(patch_transformers=True, verbose=0):
            values, spec = py_pytree.tree_flatten(cache)
            restored_cache = py_pytree.tree_unflatten(values, spec)
            # If the registration stops working, this will be hit.
            self.assertNotIsInstance(values[0], StaticCache)
            self.assertIsInstance(restored_cache, StaticCache)
            self.assertEqual(cache.max_cache_len, restored_cache.max_cache_len)
            self.assertEqualArrayAny(cache.key_cache, restored_cache.key_cache)
            self.assertEqualArrayAny(cache.value_cache, restored_cache.value_cache)

    def test_hybrid_cache_serialization(self):
        """Test HybridCache serialization and deserialization."""
        try:
            from transformers.cache_utils import HybridCache
        except ImportError:
            self.skipTest("HybridCache not available")

        class Config:
            def __init__(self):
                self.num_attention_heads = 4
                self.num_key_value_heads = 4
                self.hidden_size = 64
                self.head_dim = 16
                self.sliding_window = 4
                self.num_hidden_layers = 3

        config = Config()
        cache = HybridCache(
            config, max_batch_size=2, max_cache_len=10, device="cpu", dtype=torch.float32
        )

        with torch_export_patches(patch_transformers=True, verbose=0):
            values, spec = py_pytree.tree_flatten(cache)
            restored_cache = py_pytree.tree_unflatten(values, spec)
            self.assertNotIsInstance(values[0], HybridCache)
            self.assertIsInstance(restored_cache, HybridCache)

    def test_sliding_window_cache_serialization(self):
        """Test SlidingWindowCache serialization and deserialization."""
        try:
            from transformers.cache_utils import SlidingWindowCache
        except ImportError:
            self.skipTest("SlidingWindowCache not available")

        class Config:
            def __init__(self):
                self.num_attention_heads = 4
                self.num_key_value_heads = 4
                self.hidden_size = 64
                self.head_dim = 16
                self.sliding_window = 5
                self.num_hidden_layers = 3

        config = Config()
        cache = SlidingWindowCache(
            config, max_batch_size=2, max_cache_len=10, device="cpu", dtype=torch.float32
        )

        with torch_export_patches(patch_transformers=True, verbose=0):
            values, spec = py_pytree.tree_flatten(cache)
            restored_cache = py_pytree.tree_unflatten(values, spec)
            self.assertNotIsInstance(values[0], SlidingWindowCache)
            self.assertIsInstance(restored_cache, SlidingWindowCache)

    def test_encoder_decoder_cache_serialization(self):
        """Test EncoderDecoderCache serialization and deserialization."""
        try:
            from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        except ImportError:
            self.skipTest("EncoderDecoderCache not available")

        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        # Add some data
        key_states = torch.randn(2, 4, 5, 16)
        value_states = torch.randn(2, 4, 5, 16)
        self_attention_cache.update(key_states, value_states, layer_idx=0)
        cross_attention_cache.update(key_states, value_states, layer_idx=0)

        cache = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        with torch_export_patches(patch_transformers=True, verbose=0):
            values, spec = py_pytree.tree_flatten(cache)
            restored_cache = py_pytree.tree_unflatten(values, spec)

            self.assertNotIsInstance(values[0], EncoderDecoderCache)
            self.assertIsInstance(restored_cache, EncoderDecoderCache)
            self.assertIsInstance(restored_cache.self_attention_cache, DynamicCache)
            self.assertIsInstance(restored_cache.cross_attention_cache, DynamicCache)

    def test_torch_vmap_patching(self):
        """Test torch.vmap patching functionality."""

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.vmap(lambda t: t * 2)(x)

        x = torch.randn(3, 4)

        with torch_export_patches(patch_torch=True, verbose=0):
            torch.export.export(Model(), (x,))

    def test_model_with_cache_export(self):
        """Test exporting a model that uses cache objects."""
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            self.skipTest("DynamicCache not available")

        class ModelWithCache(torch.nn.Module):
            def forward(self, x: torch.Tensor, cache: DynamicCache):
                if cache.key_cache and len(cache.key_cache) > 0:
                    cached_key = cache.key_cache[0]
                    return x + cached_key.sum()
                return x

        model = ModelWithCache()
        x = torch.randn(2, 4, 8)
        cache = DynamicCache()
        key_states = torch.randn(2, 4, 8, 16)
        value_states = torch.randn(2, 4, 8, 16)
        cache.update(key_states, value_states, layer_idx=0)

        with torch_export_patches(patch_transformers=True, verbose=0):
            # This should not raise an error
            exported = torch.export.export(model, (x, cache))
            self.assertIsNotNone(exported)

    def test_custom_patches(self):
        """Test applying custom patches."""

        class CustomPatchedClass:
            _PATCHES_: ClassVar[list[str]] = ["test_method"]
            _PATCHED_CLASS_: ClassVar[type] = torch.nn.Linear

            @staticmethod
            def test_method(self):
                return "patched"

        original_method = getattr(torch.nn.Linear, "test_method", None)

        try:
            with torch_export_patches(custom_patches=[CustomPatchedClass], verbose=0):
                linear = torch.nn.Linear(2, 2)
                self.assertTrue(hasattr(linear, "test_method"))
                self.assertEqual(linear.test_method(), "patched")

        finally:
            # Clean up
            if original_method is None:
                if hasattr(torch.nn.Linear, "test_method"):
                    delattr(torch.nn.Linear, "test_method")
            else:
                torch.nn.Linear.test_method = original_method

    def test_verbose_output(self):
        """Test that verbose output is generated correctly."""
        import io
        from contextlib import redirect_stdout

        captured_output = io.StringIO()

        with redirect_stdout(captured_output), torch_export_patches(verbose=2):
            pass

        output = captured_output.getvalue()
        # Should contain some verbose information
        self.assertTrue(len(output) > 0)

    def test_patch_isolation(self):
        """Test that patches are properly isolated and don't leak."""
        original_vmap = torch.vmap

        with torch_export_patches(patch_torch=True, verbose=0):
            # Inside context, vmap should be patched
            self.assertNotEqual(torch.vmap, original_vmap)

        # Outside context, vmap should be restored
        self.assertEqual(torch.vmap, original_vmap)

    def test_error_handling_invalid_config(self):
        """Test error handling for invalid configurations."""
        # Test that rewrite parameter raises an error
        with self.assertRaises(ValueError), torch_export_patches(rewrite=["some_method"]):
            pass

        with self.assertRaises(ValueError), torch_export_patches(dump_rewriting="some_file"):
            pass

    def test_multiple_cache_operations(self):
        """Test multiple cache operations in sequence."""
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            self.skipTest("DynamicCache not available")

        with torch_export_patches(patch_transformers=True, verbose=0):
            cache = DynamicCache()

            # Test multiple updates
            for layer_idx in range(3):
                key_states = torch.randn(2, 4, 5, 16)
                value_states = torch.randn(2, 4, 5, 16)
                cache.update(key_states, value_states, layer_idx=layer_idx)

            # Test serialization after multiple operations
            values, spec = py_pytree.tree_flatten(cache)
            restored_cache = py_pytree.tree_unflatten(values, spec)

            self.assertEqual(len(cache.key_cache), len(restored_cache.key_cache))
            for i in range(len(cache.key_cache)):
                self.assertEqualArray(cache.key_cache[i], restored_cache.key_cache[i])


if __name__ == "__main__":
    unittest.main(verbosity=2)
