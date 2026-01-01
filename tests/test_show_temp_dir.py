"""
Helper test to show where integration test files are saved
"""

import pytest
from pathlib import Path


@pytest.mark.integration
def test_show_temp_directory_location(tmp_path_factory):
    """
    Show where pytest creates temporary directories

    Run with: pytest tests/test_show_temp_dir.py -v -s
    """
    temp_dir = tmp_path_factory.mktemp("integration_tests")

    print("\n" + "="*80)
    print("INTEGRATION TEST FILE LOCATIONS")
    print("="*80)
    print(f"\nTest output directory:")
    print(f"  {temp_dir}")
    print(f"\nHuggingFace cache (model downloads):")
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"  {hf_cache}")
    print(f"\nFiles are automatically cleaned up after tests complete.")
    print("="*80 + "\n")

    assert temp_dir.exists()
