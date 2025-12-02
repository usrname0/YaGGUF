"""
Lookup tables for IQ (Importance Quantization) formats

These tables are extracted from llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp
and related files. They contain pre-optimized quantization grids.

IQ quantization uses lookup tables instead of arithmetic quantization,
which provides better compression at the cost of implementation complexity.
"""

import numpy as np

# IQ4_NL: 4-bit non-linear quantization
# Block size: 32 elements
# Format: 16 values (4-bit index) mapped to int8 quantized values
# Source: llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp
KVALUES_IQ4NL = np.array([
    -127, -104, -83, -65,
    -49,  -35, -22, -10,
      1,   13,  25,  38,
     53,   69,  89, 113
], dtype=np.int8)

# Verify it's 16 values (4-bit = 2^4 = 16 possible values)
assert len(KVALUES_IQ4NL) == 16, "IQ4_NL table must have 16 entries"

# Note: These values are asymmetric (not centered at 0) which is intentional
# They're optimized for better quantization quality through empirical testing


def get_iq4nl_table():
    """
    Get the IQ4_NL lookup table

    Returns:
        np.ndarray: 16-element int8 array of quantization values
    """
    return KVALUES_IQ4NL


# IQ3_XXS: 3-bit quantization with lookup grid
# Block size: 256 elements
# Grid: 256 entries, each with 4 int8 values
# Source: llama.cpp gguf-py/gguf/quants.py

# Grid map - the actual int8 values that indices point to
_IQ3XXS_GRID_MAP = np.array([0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e], dtype=np.uint8)

# Grid data in hex format (encodes indices into grid_map)
_IQ3XXS_GRID_HEX = (
    b"0000020004001100130017002000220031004200730075000101030110011201"
    b"2101250130013201410154017001000202020402110220022202310233023702"
    b"5102570275020103070310031203250370031304370444045704730475040105"
    b"0705320552053506640610071407160743076107011003101010121021102310"
    b"3010321034104710501000110211111120112211011203121012121221123012"
    b"7212001302132013301346136613011405145014201524154615711505162217"
    b"4017002002201120132020202220262031204220012103210521102112212121"
    b"3021632167217021002202221122172220222222372240225522012310231423"
    b"7023742335245324032527254125742501270327162745270130103012302130"
    b"2330503065307230003102312031313144314631013203321032253252327232"
    b"1133333330344734723400350635223555351436363663363337603704401740"
    b"3540374053405740744120423742404260426642074345430444514464442545"
    b"4345704505471047124730471250415070500051065126515551145232527252"
    b"0253535310542354275472540255315550562457425724604460466064602161"
    b"6161176264623063366344640565526533660367216703700570077010703270"
    b"5270267140711272457252720073157333736073217441740075027524753076"
)

def _decode_iq3xxs_grid():
    """
    Decode IQ3_XXS grid from hex format to int8 grid values.

    This mimics the init_grid() method from llama.cpp gguf-py implementation.
    The hex data encodes 3-bit indices that map to the grid_map values.

    Returns:
        np.ndarray: (256, 4) array of int8 grid values
    """
    # 8 values in grid_map = 2^3, so 3 bits per element
    bits_per_elem = 3
    elems_per_byte = 8 // bits_per_elem  # 2 elements per byte (with 2 bits unused)

    # Convert hex ASCII to bytes
    grid = np.frombuffer(_IQ3XXS_GRID_HEX, dtype=np.uint8)

    # Decode hexadecimal characters (ASCII '0'-'9', 'a'-'f')
    grid = grid.reshape((-1, 2))
    # Convert ASCII hex to nibbles: '0'-'9' → 0-9, 'a'-'f' → 10-15
    grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
    grid = grid[..., 0] | grid[..., 1]  # Combine nibbles into bytes

    # Unpack 3-bit indices from bytes
    # Each byte contains 2 complete 3-bit values (bits 0-2 and 3-5, bits 6-7 unused)
    grid = grid.reshape((-1, 1)) >> np.array([0, 3], dtype=np.uint8).reshape((1, 2))
    grid = (grid & 0b111).reshape((-1, 1))  # Mask to 3 bits

    # Map indices to actual grid values
    grid_map = _IQ3XXS_GRID_MAP.astype(np.int8).reshape((1, -1))
    grid = np.take_along_axis(grid_map, grid.astype(np.intp), axis=-1)

    # Reshape to final grid shape (256 entries, 4 values each)
    return grid.reshape(256, 4)

# Decode the grid
IQ3XXS_GRID = _decode_iq3xxs_grid()

# Sign lookup table (shared with IQ2_XXS)
# Used to decode sign bits during dequantization
# Source: llama.cpp gguf-py/gguf/quants.py IQ2_XXS.ksigns
KSIGNS_IQ2XS = np.array([
    0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, 0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f,
    0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, 0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f,
    0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, 0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf,
    0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, 0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f,
    0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, 0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf,
    0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, 0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f,
    0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, 0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f,
    0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, 0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff,
    0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, 0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f,
    0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, 0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f,
    0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, 0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf,
    0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, 0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f,
    0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, 0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf,
    0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, 0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f,
    0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, 0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f,
    0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, 0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff,
], dtype=np.uint8)

# Verify grid dimensions
assert IQ3XXS_GRID.shape == (256, 4), f"IQ3_XXS grid should be (256, 4), got {IQ3XXS_GRID.shape}"
assert len(KSIGNS_IQ2XS) == 256, f"ksigns should have 256 entries, got {len(KSIGNS_IQ2XS)}"


# Future tables (to be extracted from llama.cpp)
# TODO: Extract these when implementing additional IQ formats
#
# IQ2XS_GRID = None
# IQ3S_GRID = None
# IQ4XS_GRID = None


def test_tables():
    """Test that lookup tables are loaded correctly"""
    print("IQ Lookup Tables Test")
    print("=" * 60)

    # Test IQ4_NL
    table = get_iq4nl_table()
    print(f"\nIQ4_NL table: {len(table)} entries")
    print(f"  Min value: {table.min()}")
    print(f"  Max value: {table.max()}")
    print(f"  Values: {table.tolist()}")

    # Check properties
    assert table.dtype == np.int8, "Table should be int8"
    assert len(table) == 16, "Should have 16 entries for 4-bit"
    assert table.min() == -127, "Min should be -127"
    assert table.max() == 113, "Max should be 113"
    print("  [OK] IQ4_NL passed")

    # Test IQ3_XXS
    print(f"\nIQ3_XXS grid: {IQ3XXS_GRID.shape} shape")
    print(f"  Min value: {IQ3XXS_GRID.min()}")
    print(f"  Max value: {IQ3XXS_GRID.max()}")
    print(f"  First entry: {IQ3XXS_GRID[0].tolist()}")
    print(f"  Last entry: {IQ3XXS_GRID[-1].tolist()}")

    assert IQ3XXS_GRID.shape == (256, 4), f"Grid should be (256, 4), got {IQ3XXS_GRID.shape}"
    assert IQ3XXS_GRID.dtype == np.int8, "Grid should be int8"
    print("  [OK] IQ3_XXS grid passed")

    # Test ksigns
    print(f"\nKSIGNS_IQ2XS: {len(KSIGNS_IQ2XS)} entries")
    print(f"  First 16: {KSIGNS_IQ2XS[:16].tolist()}")
    assert len(KSIGNS_IQ2XS) == 256, f"ksigns should have 256 entries"
    assert KSIGNS_IQ2XS.dtype == np.uint8, "ksigns should be uint8"
    print("  [OK] ksigns passed")

    print("\n" + "=" * 60)
    print("All table tests passed!")


if __name__ == "__main__":
    test_tables()
