"""Header-level regressions are compiled by fixtures and reported through pytest."""

import subprocess

import pytest


@pytest.mark.native
@pytest.mark.parametrize(
    "native_executable", ["test_sparse_matrix", "test_sparse_vector"], indirect=True
)
def test_native_contracts(native_executable):
    # Arrange
    command = [str(native_executable)]

    # Act
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr
