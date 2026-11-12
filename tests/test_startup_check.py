"""
Tests for the startup binary check (scripts/check_and_download_binaries.py)

The startup message must describe the binaries that are actually installed,
which can differ from the recommended pin after an update to Latest.
"""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_and_download_binaries.py"


@pytest.fixture(scope="module")
def startup_check():
    spec = importlib.util.spec_from_file_location("check_and_download_binaries", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manager(installed, recommended="b12110"):
    return Mock(LLAMA_CPP_VERSION=recommended, get_installed_version_tag=Mock(return_value=installed))


@pytest.mark.parametrize("installed, expected", [
    ("b12110", "Binaries installed (version b12110, the recommended version)"),
    ("b12500", "Binaries installed (version b12500, newer than recommended b12110)"),
    ("b9544", "Binaries installed (version b9544, older than recommended b12110)"),
    (None, "Binaries installed (version unknown, recommended is b12110)"),
])
def test_describe_installed_binaries(startup_check, installed, expected):
    assert startup_check.describe_installed_binaries(_manager(installed)) == expected
