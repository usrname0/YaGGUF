"""
Tests for path handling utilities
"""

import pytest
from gguf_converter.gui_utils import strip_quotes


def test_strip_quotes_double_quotes():
    """
    Test stripping double quotes from path
    """
    # Windows "Copy as path" adds double quotes
    assert strip_quotes('"C:\\Users\\test\\model"') == 'C:\\Users\\test\\model'
    assert strip_quotes('"C:/Users/test/model"') == 'C:/Users/test/model'


def test_strip_quotes_single_quotes():
    """
    Test stripping single quotes from path
    """
    assert strip_quotes("'C:\\Users\\test\\model'") == 'C:\\Users\\test\\model'
    assert strip_quotes("'/home/user/model'") == '/home/user/model'


def test_strip_quotes_mixed_quotes():
    """
    Test handling of mixed quote styles
    """
    # Should strip outer quotes
    assert strip_quotes('"C:\\Users\\test\\model\'') == 'C:\\Users\\test\\model'
    assert strip_quotes('\'C:\\Users\\test\\model"') == 'C:\\Users\\test\\model'


def test_strip_quotes_no_quotes():
    """
    Test that paths without quotes are unchanged
    """
    assert strip_quotes('C:\\Users\\test\\model') == 'C:\\Users\\test\\model'
    assert strip_quotes('/home/user/model') == '/home/user/model'


def test_strip_quotes_empty_string():
    """
    Test handling of empty string
    """
    assert strip_quotes('') == ''


def test_strip_quotes_only_quotes():
    """
    Test handling of string with only quotes
    """
    assert strip_quotes('""') == ''
    assert strip_quotes("''") == ''
    assert strip_quotes('"\'') == ''


def test_strip_quotes_whitespace():
    """
    Test that whitespace is also stripped
    """
    assert strip_quotes('  "C:\\path"  ') == 'C:\\path'
    assert strip_quotes('\t"C:\\path"\t') == 'C:\\path'
    assert strip_quotes('  C:\\path  ') == 'C:\\path'


def test_strip_quotes_quotes_in_middle():
    """
    Test that quotes in the middle of path are preserved
    """
    # Only outer quotes should be stripped
    path = '"C:\\Users\\test\'s folder\\model"'
    result = strip_quotes(path)
    assert "'" in result  # Inner quote preserved
    assert not result.startswith('"')
    assert not result.endswith('"')


def test_strip_quotes_network_paths():
    """
    Test handling of Windows network paths
    """
    assert strip_quotes('"\\\\server\\share\\model"') == '\\\\server\\share\\model'
    assert strip_quotes('\\\\server\\share\\model') == '\\\\server\\share\\model'


def test_strip_quotes_unix_paths():
    """
    Test handling of Unix paths
    """
    assert strip_quotes('"/home/user/model"') == '/home/user/model'
    assert strip_quotes('"/mnt/data/model"') == '/mnt/data/model'
    assert strip_quotes('~/Documents/model') == '~/Documents/model'


def test_strip_quotes_paths_with_spaces():
    """
    Test paths with spaces (common reason for quoting)
    """
    assert strip_quotes('"C:\\Program Files\\model"') == 'C:\\Program Files\\model'
    assert strip_quotes('"/home/user/My Documents/model"') == '/home/user/My Documents/model'


def test_strip_quotes_relative_paths():
    """
    Test handling of relative paths
    """
    assert strip_quotes('"../models/test"') == '../models/test'
    assert strip_quotes('"./output"') == './output'
    assert strip_quotes('"model"') == 'model'
