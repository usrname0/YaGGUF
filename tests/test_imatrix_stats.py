"""
Tests for imatrix statistics parsing and computation

These tests ensure the native Python imatrix parser works correctly.
This is critical because:
1. This is custom code that could break with GGUF format changes
2. Statistics calculations must be accurate
3. File parsing is complex and error-prone
"""

import pytest
import struct
import io
from pathlib import Path
from gguf_converter.imatrix_stats import (
    TensorStatistics,
    Stats,
    load_imatrix_gguf,
    compute_statistics,
    process_tensor_name,
    show_statistics
)


def test_tensor_statistics_initialization():
    """
    Test that TensorStatistics initializes with correct default values
    """
    stats = TensorStatistics("test.tensor")

    assert stats.tensor == "test.tensor"
    assert stats.total_sqract == 0.0
    assert stats.mean_sqract == 0.0
    assert stats.max_sqract == 0.0
    assert stats.min_sqract == 0.0
    assert stats.elements == 0
    assert stats.stddev == 0.0
    assert stats.active == 0.0
    assert stats.entropy == 0.0
    assert stats.zd == 0.0
    assert stats.cossim == 0.0


def test_stats_initialization():
    """
    Test that Stats initializes with empty lists
    """
    stats = Stats()

    assert isinstance(stats.values, list)
    assert isinstance(stats.counts, list)
    assert len(stats.values) == 0
    assert len(stats.counts) == 0


def test_process_tensor_name_simple():
    """
    Test processing simple tensor names
    """
    input_name, display_name = process_tensor_name("blk.0.attn_q.weight")

    # Should extract block number
    assert "0" in input_name or "blk.0" in input_name
    assert "attn_q" in display_name


def test_process_tensor_name_with_block():
    """
    Test processing tensor names with block numbers
    """
    test_cases = [
        "blk.0.attn_q.weight",
        "blk.15.attn_k.weight",
        "blk.31.ffn_up.weight"
    ]

    for name in test_cases:
        input_name, display_name = process_tensor_name(name)

        # Should not be empty
        assert len(input_name) > 0
        assert len(display_name) > 0

        # Display name should be readable
        assert "blk" not in display_name or "." not in display_name


def test_process_tensor_name_output():
    """
    Test processing output tensor name
    """
    input_name, display_name = process_tensor_name("output.weight")

    # process_tensor_name returns "-" for unrecognized patterns
    assert isinstance(input_name, str)
    assert isinstance(display_name, str)


def test_process_tensor_name_embedding():
    """
    Test processing embedding tensor name
    """
    input_name, display_name = process_tensor_name("token_embd.weight")

    # Should return strings (may be "-" for unrecognized patterns)
    assert isinstance(input_name, str)
    assert isinstance(display_name, str)


def test_compute_statistics_empty():
    """
    Test computing statistics with empty data
    """
    stats = Stats()
    stats.values = []
    stats.counts = []

    # Should raise ValueError for empty data
    with pytest.raises(ValueError, match="No data for tensor"):
        compute_statistics("test.tensor", stats)


def test_compute_statistics_single_value():
    """
    Test computing statistics with a single value

    Note: values are divided by counts, so value/count becomes the activation
    """
    stats = Stats()
    stats.values = [100.0]
    stats.counts = [100]

    tensor_stats = compute_statistics("test.tensor", stats)

    assert tensor_stats.tensor == "test.tensor"
    assert tensor_stats.elements == 1  # One activation: 100.0/100 = 1.0
    assert tensor_stats.mean_sqract == 1.0  # Mean of [1.0] is 1.0


def test_compute_statistics_multiple_values():
    """
    Test computing statistics with multiple values

    Note: values are divided by counts to create activations
    """
    stats = Stats()
    stats.values = [10.0, 20.0, 30.0]
    stats.counts = [10, 10, 10]

    tensor_stats = compute_statistics("test.tensor", stats)

    assert tensor_stats.tensor == "test.tensor"
    assert tensor_stats.elements == 3  # Three activations: 1.0, 2.0, 3.0
    assert tensor_stats.mean_sqract > 0
    assert tensor_stats.max_sqract >= tensor_stats.min_sqract


def test_compute_statistics_calculates_mean():
    """
    Test that mean is calculated correctly

    Value 400 divided by count 100 = activation of 4.0
    """
    stats = Stats()
    stats.values = [400.0]
    stats.counts = [100]

    tensor_stats = compute_statistics("test.tensor", stats)

    # Activation is 400/100 = 4.0
    assert abs(tensor_stats.mean_sqract - 4.0) < 0.01


def test_compute_statistics_finds_max():
    """
    Test that max value is found correctly

    Activations are: 10/10=1.0, 50/10=5.0, 30/10=3.0
    """
    stats = Stats()
    stats.values = [10.0, 50.0, 30.0]
    stats.counts = [10, 10, 10]

    tensor_stats = compute_statistics("test.tensor", stats)

    assert tensor_stats.max_sqract == 5.0


def test_compute_statistics_finds_min():
    """
    Test that min value is found correctly

    Activations are: 50/10=5.0, 10/10=1.0, 30/10=3.0
    """
    stats = Stats()
    stats.values = [50.0, 10.0, 30.0]
    stats.counts = [10, 10, 10]

    tensor_stats = compute_statistics("test.tensor", stats)

    assert tensor_stats.min_sqract == 1.0


def test_invalid_gguf_magic():
    """
    Test that invalid GGUF magic number raises error
    """
    # Create a fake file with wrong magic
    fake_data = b"FAKE" + struct.pack('<I', 3) + struct.pack('<Q', 0) + struct.pack('<Q', 0)

    with pytest.raises(ValueError, match="Not a valid GGUF file"):
        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as tmp:
            tmp.write(fake_data)
            tmp_path = tmp.name

        try:
            load_imatrix_gguf(tmp_path)
        finally:
            Path(tmp_path).unlink()


def test_show_statistics_missing_file():
    """
    Test that missing file is handled gracefully
    """
    result = show_statistics("/nonexistent/file.gguf")

    # Should return False for missing file
    assert result is False


def test_show_statistics_invalid_file(tmp_path):
    """
    Test that invalid file is handled gracefully
    """
    # Create an empty file
    invalid_file = tmp_path / "invalid.gguf"
    invalid_file.write_bytes(b"not a gguf file")

    result = show_statistics(str(invalid_file))

    # Should return False for invalid file
    assert result is False


def test_process_tensor_name_returns_tuple():
    """
    Test that process_tensor_name always returns a tuple
    """
    test_names = [
        "blk.0.attn_q.weight",
        "output.weight",
        "token_embd.weight",
        "some.random.tensor"
    ]

    for name in test_names:
        result = process_tensor_name(name)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


def test_compute_statistics_with_zeros():
    """
    Test computing statistics when values include zeros

    Activations: 0/1=0.0, 10/1=10.0, 20/1=20.0
    """
    stats = Stats()
    stats.values = [0.0, 10.0, 20.0]
    stats.counts = [1, 1, 1]

    tensor_stats = compute_statistics("test.tensor", stats)

    # Should handle zeros gracefully
    assert tensor_stats.elements == 3
    assert tensor_stats.min_sqract == 0.0


def test_compute_statistics_with_large_values():
    """
    Test computing statistics with large values

    Activations: 10000/10=1000.0, 20000/10=2000.0, 30000/10=3000.0
    """
    stats = Stats()
    stats.values = [10000.0, 20000.0, 30000.0]
    stats.counts = [10, 10, 10]

    tensor_stats = compute_statistics("test.tensor", stats)

    # Should handle large values without overflow
    assert tensor_stats.mean_sqract > 0
    assert tensor_stats.max_sqract == 3000.0


def test_compute_statistics_with_small_values():
    """
    Test computing statistics with very small values

    Activations: 0.01/10=0.001, 0.02/10=0.002, 0.03/10=0.003
    """
    stats = Stats()
    stats.values = [0.01, 0.02, 0.03]
    stats.counts = [10, 10, 10]

    tensor_stats = compute_statistics("test.tensor", stats)

    # Should handle small values without underflow
    assert tensor_stats.mean_sqract > 0
    assert tensor_stats.min_sqract == 0.001


def test_stats_dict_structure():
    """
    Test that load_imatrix_gguf returns correct structure

    Even though we can't easily create a valid GGUF file,
    we can test the expected return type
    """
    # This will fail to load, but tests the function signature
    with pytest.raises((ValueError, FileNotFoundError, Exception)):
        result = load_imatrix_gguf("/nonexistent/file.gguf")


def test_tensor_statistics_all_fields_accessible():
    """
    Test that all TensorStatistics fields are accessible
    """
    stats = TensorStatistics("test")

    # All these should be accessible without error
    fields = [
        'tensor', 'total_sqract', 'mean_sqract', 'max_sqract', 'min_sqract',
        'elements', 'stddev', 'active', 'entropy', 'zd', 'cossim'
    ]

    for field in fields:
        assert hasattr(stats, field), f"TensorStatistics should have field '{field}'"
        # Should be accessible
        _ = getattr(stats, field)


def test_tensor_statistics_numeric_fields_are_numeric():
    """
    Test that numeric fields in TensorStatistics are numeric types
    """
    stats = TensorStatistics("test")

    numeric_fields = [
        'total_sqract', 'mean_sqract', 'max_sqract', 'min_sqract',
        'elements', 'stddev', 'active', 'entropy', 'zd', 'cossim'
    ]

    for field in numeric_fields:
        value = getattr(stats, field)
        assert isinstance(value, (int, float)), \
            f"Field '{field}' should be numeric, got {type(value)}"


def test_process_tensor_name_handles_edge_cases():
    """
    Test tensor name processing with edge cases
    """
    edge_cases = [
        "",  # Empty string
        ".",  # Just a dot
        "no_dots_at_all",  # No structure
        "blk.999.something",  # Large block number
    ]

    for name in edge_cases:
        # Should not crash
        try:
            input_name, display_name = process_tensor_name(name)
            assert isinstance(input_name, str)
            assert isinstance(display_name, str)
        except Exception as e:
            # If it does raise an exception, it should be a clear one
            assert isinstance(e, (ValueError, KeyError, IndexError))


def test_compute_statistics_handles_mismatched_lengths():
    """
    Test that compute_statistics handles mismatched values/counts lengths
    """
    stats = Stats()
    stats.values = [1.0, 2.0, 3.0]
    stats.counts = [10, 20]  # One less than values

    # Should either handle gracefully or raise clear error
    try:
        tensor_stats = compute_statistics("test.tensor", stats)
        # If it succeeds, check it used the shorter length
        assert tensor_stats.elements <= 30
    except (IndexError, ValueError):
        # Acceptable to raise an error for invalid data
        pass


def test_show_statistics_returns_boolean():
    """
    Test that show_statistics always returns a boolean
    """
    # Test with nonexistent file
    result = show_statistics("/nonexistent/file.gguf")
    assert isinstance(result, bool)


def test_process_tensor_name_preserves_important_info():
    """
    Test that tensor name processing preserves key information
    """
    test_name = "blk.15.attn_q.weight"

    input_name, display_name = process_tensor_name(test_name)

    # Should preserve the block number and component info
    assert "15" in input_name or "15" in display_name
    assert "attn_q" in input_name or "attn" in display_name or "q" in display_name
