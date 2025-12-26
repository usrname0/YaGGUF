"""
Tests for version checking and comparison utilities

These tests ensure version detection works correctly.
This is critical because:
1. llama.cpp releases new versions daily
2. Users need to know when updates are available
3. Version parsing changes could break update detection
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
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


@patch('urllib.request.urlopen')
def test_check_git_updates_no_releases(mock_urlopen):
    """
    Test handling when no releases are found on GitHub
    """
    # Mock API response with no tag_name
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({}).encode()
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    result = check_git_updates_available()

    assert result["status"] == "unknown"
    assert "No releases" in result["message"]
    assert result["latest_version"] is None


@patch('urllib.request.urlopen')
def test_check_git_updates_up_to_date(mock_urlopen):
    """
    Test when current version matches latest release
    """
    current = get_current_version()

    # Mock API response with current version as latest release
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"tag_name": current}).encode()
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    result = check_git_updates_available()

    # Should indicate up to date
    assert result["status"] == "up_to_date"
    assert "latest version" in result["message"].lower()


@patch('urllib.request.urlopen')
def test_check_git_updates_available(mock_urlopen):
    """
    Test when newer version is available
    """
    # Mock API response with newer version
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"tag_name": "v999.999.999"}).encode()
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    result = check_git_updates_available()

    # Should indicate updates available
    assert result["status"] == "updates_available"
    assert "999.999.999" in result["message"]
    assert result["latest_version"] == "v999.999.999"


@patch('urllib.request.urlopen')
def test_check_git_updates_api_fails(mock_urlopen):
    """
    Test handling when GitHub API request fails
    """
    # Mock API request failing
    mock_urlopen.side_effect = Exception("Network error")

    result = check_git_updates_available()

    # Should handle gracefully
    assert result["status"] == "unknown"
    assert result["latest_version"] is None


@patch('urllib.request.urlopen')
def test_check_git_updates_timeout(mock_urlopen):
    """
    Test handling when GitHub API times out
    """
    # Mock API timeout
    import urllib.error
    mock_urlopen.side_effect = urllib.error.URLError("timeout")

    result = check_git_updates_available()

    # Should handle gracefully
    assert result["status"] == "unknown"
    assert "Could not check" in result["message"]


@patch('urllib.request.urlopen')
def test_version_comparison_strips_v_prefix(mock_urlopen):
    """
    Test that version comparison correctly strips 'v' prefix
    """
    # Current version is 1.0.6 (without v)
    # Latest release tag is v1.0.6 (with v)
    # These should be considered equal

    # Mock API response with v prefix
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"tag_name": "v1.0.6"}).encode()
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    # Mock get_current_version to return version without v
    with patch('gguf_converter.gui_utils.get_current_version', return_value='1.0.6'):
        result = check_git_updates_available()

        # Should consider them equal
        assert result["status"] == "up_to_date"


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


@patch('urllib.request.urlopen')
def test_multiple_version_formats(mock_urlopen):
    """
    Test handling of different version tag formats from GitHub releases
    """
    # Test with tags that have different formats
    test_cases = [
        "v1.0.6",      # Standard format
        "1.0.6",       # Without v prefix
        "v1.0.6-rc1",  # Release candidate
        "b7522",       # Build number (llama.cpp style)
    ]

    for tag in test_cases:
        # Mock API response with this tag format
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"tag_name": tag}).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = check_git_updates_available()

        # Should not crash and should return latest_version
        assert result["latest_version"] == tag


def test_version_string_not_empty():
    """
    Test that version is never an empty string
    """
    version = get_current_version()

    # Version should never be empty
    assert version != ""

    # Should be either a valid version or "unknown"
    assert version == "unknown" or len(version.strip()) > 0
