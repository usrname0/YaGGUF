"""
Tests for version checking and comparison utilities

These tests ensure version detection works correctly.
This is critical because:
1. llama.cpp releases new versions daily
2. Users need to know when updates are available
3. Version parsing changes could break update detection
"""

import pytest
from unittest.mock import Mock, patch
import subprocess
from gguf_converter.gui_utils import (
    get_current_version,
    check_git_updates_available
)


def test_get_current_version():
    """
    Test that get_current_version returns a valid version string
    """
    version = get_current_version()

    # Should return a string
    assert isinstance(version, str)

    # Should not be empty
    assert len(version) > 0

    # Should be the actual version or "unknown"
    assert version != "" and (version == "unknown" or version.count('.') >= 1)


def test_get_current_version_format():
    """
    Test that version follows semantic versioning format
    """
    version = get_current_version()

    if version != "unknown":
        # Should have at least major.minor
        parts = version.lstrip('v').split('.')
        assert len(parts) >= 2

        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()


@patch('subprocess.run')
def test_check_git_updates_no_tags(mock_run):
    """
    Test handling when no git tags are found
    """
    # Mock git fetch (succeeds)
    # Mock git tag (returns empty)
    mock_run.side_effect = [
        Mock(returncode=0, stdout="", stderr=""),  # git fetch
        Mock(returncode=0, stdout="", stderr="")   # git tag (empty)
    ]

    result = check_git_updates_available()

    assert result["status"] == "unknown"
    assert "Could not fetch" in result["message"] or "No version tags" in result["message"]
    assert result["latest_version"] is None


@patch('subprocess.run')
def test_check_git_updates_up_to_date(mock_run):
    """
    Test when current version matches latest tag
    """
    current = get_current_version()

    # Mock git fetch (succeeds)
    # Mock git tag (returns current version)
    mock_run.side_effect = [
        Mock(returncode=0, stdout="", stderr=""),  # git fetch
        Mock(returncode=0, stdout=f"{current}\n", stderr="")  # git tag
    ]

    result = check_git_updates_available()

    # Should indicate up to date
    assert result["status"] == "up_to_date"
    assert "latest version" in result["message"].lower()


@patch('subprocess.run')
def test_check_git_updates_available(mock_run):
    """
    Test when newer version is available
    """
    # Mock git fetch (succeeds)
    # Mock git tag (returns newer version)
    mock_run.side_effect = [
        Mock(returncode=0, stdout="", stderr=""),  # git fetch
        Mock(returncode=0, stdout="v999.999.999\nv1.0.0\n", stderr="")  # git tag (newer first)
    ]

    result = check_git_updates_available()

    # Should indicate updates available
    assert result["status"] == "updates_available"
    assert "999.999.999" in result["message"]
    assert result["latest_version"] == "v999.999.999"


@patch('subprocess.run')
def test_check_git_updates_git_fetch_fails(mock_run):
    """
    Test handling when git fetch fails
    """
    # Mock git fetch failing
    mock_run.side_effect = subprocess.TimeoutExpired("git", 10)

    result = check_git_updates_available()

    # Should handle gracefully
    assert result["status"] == "unknown"
    assert result["latest_version"] is None


@patch('subprocess.run')
def test_check_git_updates_git_tag_fails(mock_run):
    """
    Test handling when git tag command fails
    """
    # Mock git fetch (succeeds)
    # Mock git tag (fails)
    mock_run.side_effect = [
        Mock(returncode=0, stdout="", stderr=""),  # git fetch
        Mock(returncode=1, stdout="", stderr="error")  # git tag fails
    ]

    result = check_git_updates_available()

    # Should handle gracefully
    assert result["status"] == "unknown"
    assert "Could not fetch" in result["message"]


@patch('subprocess.run')
def test_version_comparison_strips_v_prefix(mock_run):
    """
    Test that version comparison correctly strips 'v' prefix
    """
    # Current version is 1.0.6 (without v)
    # Latest tag is v1.0.6 (with v)
    # These should be considered equal

    mock_run.side_effect = [
        Mock(returncode=0, stdout="", stderr=""),  # git fetch
        Mock(returncode=0, stdout="v1.0.6\n", stderr="")  # git tag
    ]

    # Mock get_current_version to return version without v
    with patch('gguf_converter.gui_utils.get_current_version', return_value='1.0.6'):
        result = check_git_updates_available()

        # Should consider them equal
        assert result["status"] == "up_to_date"


@patch('subprocess.run')
def test_version_tags_sorted_correctly(mock_run):
    """
    Test that latest tag is correctly identified from sorted list
    """
    # Mock git fetch (succeeds)
    # Mock git tag returning multiple versions (sorted)
    mock_run.side_effect = [
        Mock(returncode=0, stdout="", stderr=""),  # git fetch
        Mock(returncode=0, stdout="v2.0.0\nv1.5.0\nv1.0.0\n", stderr="")  # Sorted descending
    ]

    result = check_git_updates_available()

    # Should pick first (latest) tag
    assert result["latest_version"] == "v2.0.0"


def test_check_git_updates_returns_dict():
    """
    Test that check_git_updates_available always returns a dict with expected keys
    """
    result = check_git_updates_available()

    # Should be a dictionary
    assert isinstance(result, dict)

    # Should have required keys
    assert "status" in result
    assert "message" in result
    assert "latest_version" in result

    # Status should be one of expected values
    assert result["status"] in ["up_to_date", "updates_available", "unknown"]

    # Message should be a string
    assert isinstance(result["message"], str)
    assert len(result["message"]) > 0


@patch('subprocess.run')
def test_git_fetch_timeout_handling(mock_run):
    """
    Test that timeouts are handled gracefully
    """
    # Mock git fetch timing out
    mock_run.side_effect = subprocess.TimeoutExpired("git fetch --tags", 10)

    result = check_git_updates_available()

    # Should not crash
    assert result["status"] == "unknown"
    assert result["latest_version"] is None


@patch('subprocess.run')
def test_multiple_version_formats(mock_run):
    """
    Test handling of different version tag formats
    """
    # Test with tags that have different formats
    test_cases = [
        "v1.0.6",      # Standard format
        "1.0.6",       # Without v prefix
        "v1.0.6-rc1",  # Release candidate
        "b7522",       # Build number (llama.cpp style)
    ]

    for tag in test_cases:
        mock_run.side_effect = [
            Mock(returncode=0, stdout="", stderr=""),  # git fetch
            Mock(returncode=0, stdout=f"{tag}\n", stderr="")  # git tag
        ]

        result = check_git_updates_available()

        # Should not crash and should return latest_version
        assert result["latest_version"] == tag


@patch('subprocess.run')
def test_network_error_handling(mock_run):
    """
    Test handling of network errors during git fetch
    """
    # Simulate network error
    mock_run.side_effect = Exception("Network error: Could not resolve host")

    result = check_git_updates_available()

    # Should handle gracefully
    assert result["status"] == "unknown"
    assert result["latest_version"] is None
    assert isinstance(result["message"], str)


def test_version_string_not_empty():
    """
    Test that version is never an empty string
    """
    version = get_current_version()

    # Version should never be empty
    assert version != ""

    # Should be either a valid version or "unknown"
    assert version == "unknown" or len(version.strip()) > 0
