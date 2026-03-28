# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the feature flag system."""

from __future__ import annotations

import pytest

from mobius import _flags


class TestDefaultValues:
    """Flags have the expected defaults when no env vars are set."""

    def test_suppress_dedup_warning_default_on(self, monkeypatch):
        monkeypatch.delenv("MOBIUS_SUPPRESS_DEDUP_WARNING", raising=False)
        f = _flags._Flags()
        assert f.suppress_dedup_warning is True


class TestEnvVarOverride:
    """Flags are read from environment variables at construction time."""

    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("MOBIUS_SUPPRESS_DEDUP_WARNING", val)
        f = _flags._Flags()
        assert f.suppress_dedup_warning is True

    @pytest.mark.parametrize("val", ["0", "false", "False", "FALSE", "no", "NO"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("MOBIUS_SUPPRESS_DEDUP_WARNING", val)
        f = _flags._Flags()
        assert f.suppress_dedup_warning is False

    def test_unknown_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MOBIUS_SUPPRESS_DEDUP_WARNING", "maybe")
        f = _flags._Flags()
        assert f.suppress_dedup_warning is True  # default


class TestProgrammaticOverride:
    """Flags can be assigned directly on the singleton."""

    def test_assign_and_restore(self):
        original = _flags.flags.suppress_dedup_warning
        try:
            _flags.flags.suppress_dedup_warning = not original
            assert _flags.flags.suppress_dedup_warning is not original
        finally:
            _flags.flags.suppress_dedup_warning = original


class TestOverrideFlagsContextManager:
    """override_flags() restores original values on exit."""

    def test_single_flag_restored(self):
        original = _flags.flags.suppress_dedup_warning
        with _flags.override_flags(suppress_dedup_warning=not original):
            assert _flags.flags.suppress_dedup_warning is not original
        assert _flags.flags.suppress_dedup_warning is original

    def test_restored_on_exception(self):
        original = _flags.flags.suppress_dedup_warning
        exc = RuntimeError("boom")
        try:
            with _flags.override_flags(suppress_dedup_warning=not original):
                raise exc
        except RuntimeError:
            pass
        assert _flags.flags.suppress_dedup_warning is original

    def test_nested_overrides(self):
        original = _flags.flags.suppress_dedup_warning
        with _flags.override_flags(suppress_dedup_warning=not original):
            assert _flags.flags.suppress_dedup_warning is not original
            with _flags.override_flags(suppress_dedup_warning=original):
                assert _flags.flags.suppress_dedup_warning is original
            assert _flags.flags.suppress_dedup_warning is not original
        assert _flags.flags.suppress_dedup_warning is original

    def test_unknown_flag_raises_value_error(self):
        with (
            pytest.raises(ValueError, match="Unknown flag"),
            _flags.override_flags(nonexistent_flag=True),
        ):
            pass  # pragma: no cover


class TestListFlags:
    """list_flags() returns a plain dict of all current flag values."""

    def test_returns_dict(self):
        result = _flags.list_flags()
        assert isinstance(result, dict)

    def test_contains_suppress_dedup_warning(self):
        assert "suppress_dedup_warning" in _flags.list_flags()

    def test_values_match_singleton(self):
        result = _flags.list_flags()
        assert result["suppress_dedup_warning"] == _flags.flags.suppress_dedup_warning

    def test_returns_snapshot_not_live_view(self):
        """list_flags() returns a copy, not a live reference."""
        snapshot = _flags.list_flags()
        original = _flags.flags.suppress_dedup_warning
        with _flags.override_flags(suppress_dedup_warning=not original):
            assert snapshot["suppress_dedup_warning"] == original
