"""
Tests for quantization type validation and rules

These tests ensure quantization validation works correctly.
This is critical because:
1. IQ quants require importance matrices for good quality
2. Some quants are incompatible with certain model architectures
3. New quant types might be added to llama.cpp
"""

import pytest
from gguf_converter.converter import GGUFConverter


def test_all_quantization_types_defined():
    """
    Test that QUANTIZATION_TYPES list exists and is populated
    """
    assert hasattr(GGUFConverter, 'QUANTIZATION_TYPES')
    assert len(GGUFConverter.QUANTIZATION_TYPES) > 0


def test_quantization_types_are_strings():
    """
    Test that all quantization types are strings
    """
    for quant_type in GGUFConverter.QUANTIZATION_TYPES:
        assert isinstance(quant_type, str)
        assert len(quant_type) > 0


def test_standard_quants_present():
    """
    Test that standard quantization types are available
    """
    standard_quants = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

    for quant in standard_quants:
        assert quant in GGUFConverter.QUANTIZATION_TYPES, \
            f"{quant} should be in QUANTIZATION_TYPES"


def test_iq_quants_present():
    """
    Test that IQ (importance quantization) types are available
    """
    iq_quants = [
        "IQ1_S", "IQ1_M",
        "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
        "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
        "IQ4_XS", "IQ4_NL"
    ]

    for quant in iq_quants:
        assert quant in GGUFConverter.QUANTIZATION_TYPES, \
            f"{quant} should be in QUANTIZATION_TYPES"


def test_k_quants_present():
    """
    Test that K-quants are available
    """
    k_quants = [
        "Q2_K", "Q2_K_S",
        "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_K", "Q4_K_S", "Q4_K_M",
        "Q5_K", "Q5_K_S", "Q5_K_M",
        "Q6_K"
    ]

    for quant in k_quants:
        assert quant in GGUFConverter.QUANTIZATION_TYPES, \
            f"{quant} should be in QUANTIZATION_TYPES"


def test_legacy_quants_present():
    """
    Test that legacy quantization types are still available
    """
    legacy_quants = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"]

    for quant in legacy_quants:
        assert quant in GGUFConverter.QUANTIZATION_TYPES, \
            f"{quant} should be in QUANTIZATION_TYPES"


def test_float_quants_present():
    """
    Test that float quantization types are available
    """
    float_quants = ["F16", "BF16", "F32"]

    for quant in float_quants:
        assert quant in GGUFConverter.QUANTIZATION_TYPES, \
            f"{quant} should be in QUANTIZATION_TYPES"


def test_no_duplicate_quantization_types():
    """
    Test that there are no duplicate entries in QUANTIZATION_TYPES
    """
    quant_types = GGUFConverter.QUANTIZATION_TYPES
    assert len(quant_types) == len(set(quant_types)), \
        "QUANTIZATION_TYPES should not contain duplicates"


def test_quantization_types_uppercase():
    """
    Test that quantization type names follow uppercase convention
    """
    for quant_type in GGUFConverter.QUANTIZATION_TYPES:
        # Should be uppercase or contain underscores
        assert quant_type.replace('_', '').replace('BF', '').replace('IQ', '').replace('F', '').replace('Q', '').replace('K', '').replace('M', '').replace('S', '').replace('L', '').replace('X', '').replace('N', '').isdigit() or quant_type.isupper() or 'F' in quant_type or 'BF' in quant_type, \
            f"{quant_type} should follow naming convention"


def test_iq_quants_start_with_iq():
    """
    Test that IQ quants follow naming convention
    """
    iq_quants = [q for q in GGUFConverter.QUANTIZATION_TYPES if q.startswith('IQ')]

    # Should have at least the known IQ quants
    assert len(iq_quants) >= 12, "Should have at least 12 IQ quantization types"

    # All should start with IQ
    for quant in iq_quants:
        assert quant.startswith('IQ'), f"{quant} should start with 'IQ'"


def test_k_quants_contain_k():
    """
    Test that K-quants follow naming convention
    """
    k_quants = [q for q in GGUFConverter.QUANTIZATION_TYPES if '_K' in q]

    # Should have at least the known K-quants
    assert len(k_quants) >= 11, "Should have at least 11 K-quantization types"


def test_quantization_types_no_spaces():
    """
    Test that quantization type names contain no spaces
    """
    for quant_type in GGUFConverter.QUANTIZATION_TYPES:
        assert ' ' not in quant_type, \
            f"{quant_type} should not contain spaces"


def test_imatrix_required_types_subset():
    """
    Test that all IQ quants that require imatrix are in QUANTIZATION_TYPES

    This catches if new IQ quants are added but not to the main list
    """
    # These are the quants that require imatrix (from converter.py)
    imatrix_required = [
        "IQ1_S", "IQ1_M",
        "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
        "IQ3_XXS", "IQ3_XS"
    ]

    for quant in imatrix_required:
        assert quant in GGUFConverter.QUANTIZATION_TYPES, \
            f"Imatrix-required quant {quant} should be in QUANTIZATION_TYPES"


def test_incompatible_quants_are_valid():
    """
    Test that all incompatible quants listed in MODEL_INCOMPATIBILITIES are valid

    This catches typos in the incompatibility registry
    """
    for incompat_type, info in GGUFConverter.MODEL_INCOMPATIBILITIES.items():
        incompatible_quants = info.get("incompatible_quants", [])

        for quant in incompatible_quants:
            assert quant in GGUFConverter.QUANTIZATION_TYPES, \
                f"Incompatible quant {quant} in {incompat_type} should be a valid quantization type"


def test_tied_embeddings_incompatible_quants():
    """
    Test that tied embeddings incompatibility includes all IQ quants
    """
    tied_info = GGUFConverter.MODEL_INCOMPATIBILITIES.get("tied_embeddings", {})
    incompatible = tied_info.get("incompatible_quants", [])

    # All IQ quants should be incompatible with tied embeddings
    iq_quants = [q for q in GGUFConverter.QUANTIZATION_TYPES if q.startswith('IQ')]

    for iq_quant in iq_quants:
        assert iq_quant in incompatible, \
            f"{iq_quant} should be incompatible with tied embeddings"


def test_q2_k_s_incompatible_with_tied_embeddings():
    """
    Test that Q2_K_S is also incompatible with tied embeddings

    This is a specific case noted in the incompatibility registry
    """
    tied_info = GGUFConverter.MODEL_INCOMPATIBILITIES.get("tied_embeddings", {})
    incompatible = tied_info.get("incompatible_quants", [])

    assert "Q2_K_S" in incompatible, \
        "Q2_K_S should be incompatible with tied embeddings (requires output.weight)"


def test_alternatives_are_valid_quants():
    """
    Test that recommended alternatives are valid quantization types
    """
    tied_info = GGUFConverter.MODEL_INCOMPATIBILITIES.get("tied_embeddings", {})
    alternatives = tied_info.get("alternatives", [])

    # Check that alternatives mention valid quant types
    alternatives_text = " ".join(alternatives)

    # These should be mentioned as safe alternatives
    safe_quants = ["Q3_K_M", "Q3_K_S", "Q2_K", "Q4_K_M"]

    for quant in safe_quants:
        assert quant in alternatives_text, \
            f"Alternative {quant} should be mentioned in alternatives"


def test_unquantized_formats_included():
    """
    Test that unquantized formats are included
    """
    unquantized = ["F16", "F32"]

    for fmt in unquantized:
        assert fmt in GGUFConverter.QUANTIZATION_TYPES, \
            f"Unquantized format {fmt} should be available"


def test_bf16_format_included():
    """
    Test that BF16 (bfloat16) format is included

    BF16 is useful for certain training/inference scenarios
    """
    assert "BF16" in GGUFConverter.QUANTIZATION_TYPES, \
        "BF16 format should be available"


def test_quantization_list_order_logical():
    """
    Test that quantization types appear in a logical order

    This is more of a sanity check - the list should be organized
    """
    quant_types = GGUFConverter.QUANTIZATION_TYPES

    # Should have legacy quants first, then K-quants, then IQ quants, then float
    q4_0_idx = quant_types.index("Q4_0")
    q4_k_idx = quant_types.index("Q4_K")
    iq2_xxs_idx = quant_types.index("IQ2_XXS")
    f16_idx = quant_types.index("F16")

    # Basic ordering check - legacy before K-quants before IQ before float
    assert q4_0_idx < q4_k_idx, "Legacy quants should come before K-quants"
    assert q4_k_idx < iq2_xxs_idx, "K-quants should come before IQ quants"
    assert iq2_xxs_idx < f16_idx, "IQ quants should come before float formats"
