"""The compatibility helper discovers checkout tests, never unrelated installed tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import sppy


def test_compatibility_runner_uses_top_level_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    """Verify compatibility runner uses top level tests."""
    calls = []
    monkeypatch.setattr(pytest, "main", lambda arguments: calls.append(arguments) or 0)
    expected = Path(__file__).resolve().parent
    monkeypatch.setattr(sppy, "__file__", str(expected.parent / "sppy/__init__.py"))

    # Act
    result = sppy.test("-q", "-k", "metadata")

    # Assert
    assert result == 0
    assert calls == [[str(expected), "-q", "-k", "metadata"]]


def test_installed_runner_reports_missing_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    """Verify installed runner reports missing checkout."""
    monkeypatch.setattr(
        sppy, "__file__", str(tmp_path / "site-packages/sppy/__init__.py")
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="source checkout"):
        sppy.test()
