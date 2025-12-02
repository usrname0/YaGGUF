"""
IQ3_S: 3-bit quantization (3.4375 bits per weight)

Block structure (110 bytes per 256 elements):
- d: float16 scale (2 bytes)
- qs: 64 bytes (4-bit indices for 256 elements, 2 indices per byte)
- qh: 8 bytes (high bits for 9-bit grid indices)
- signs: 32 bytes (sign bits, 1 bit per element)
- scales: 4 bytes (4-bit per sub-block scale, 8 sub-blocks)

Based on llama.cpp's IQ3_S implementation (ggml-quants.c)
"""

import numpy as np
import struct
from typing import Optional

# Constants
QK_K = 256  # Elements per block
QK_IQ3S = 256  # Same as QK_K
BLOCK_SIZE = 32  # Sub-block size for quantization
N_SCALE = QK_K // 64  # 4 scales per block (256/64)
BYTES_PER_BLOCK = 110  # 2 + 64 + 8 + 32 + 4

# IQ3_S Grid Table (512 entries)
# Each uint32_t encodes 4 int8 values
# Source: llama.cpp/ggml/src/ggml-common.h lines 1020-1085
_IQ3S_GRID_HEX = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
]

# Convert hex grid to int8 values
# Each uint32 contains 4 bytes that represent int8 values
_IQ3S_GRID = np.zeros((512, 4), dtype=np.int8)
for i, hex_val in enumerate(_IQ3S_GRID_HEX):
    # Extract 4 bytes from uint32 (little-endian)
    _IQ3S_GRID[i, 0] = np.int8((hex_val >> 0) & 0xFF)
    _IQ3S_GRID[i, 1] = np.int8((hex_val >> 8) & 0xFF)
    _IQ3S_GRID[i, 2] = np.int8((hex_val >> 16) & 0xFF)
    _IQ3S_GRID[i, 3] = np.int8((hex_val >> 24) & 0xFF)


def get_iq3s_grid():
    """
    Get the IQ3_S grid table.

    Returns:
        np.ndarray: (512, 4) array of int8 values
    """
    return _IQ3S_GRID


def dequantize_block_iq3_s(block_bytes: bytes, block_idx: int = 0) -> np.ndarray:
    """
    Dequantize a single IQ3_S block (110 bytes -> 256 float32 values).

    Args:
        block_bytes: 110 bytes representing the quantized block
        block_idx: Block index (for debugging)

    Returns:
        np.ndarray: 256 dequantized float32 values
    """
    if len(block_bytes) != BYTES_PER_BLOCK:
        raise ValueError(f"Expected {BYTES_PER_BLOCK} bytes, got {len(block_bytes)}")

    # Parse block structure
    # d (2 bytes) + qs (64 bytes) + qh (8 bytes) + signs (32 bytes) + scales (4 bytes)
    d_bytes = block_bytes[0:2]
    qs = np.frombuffer(block_bytes[2:66], dtype=np.uint8)      # 64 bytes
    qh = np.frombuffer(block_bytes[66:74], dtype=np.uint8)     # 8 bytes
    signs = np.frombuffer(block_bytes[74:106], dtype=np.uint8) # 32 bytes
    scales = np.frombuffer(block_bytes[106:110], dtype=np.uint8) # 4 bytes

    # Decode main scale (float16)
    d = np.frombuffer(d_bytes, dtype=np.float16)[0]

    # Output array
    output = np.zeros(QK_IQ3S, dtype=np.float32)

    # Process 8 sub-blocks of 32 elements each
    grid = get_iq3s_grid()

    for ib in range(8):  # 8 sub-blocks
        # Get scale for this sub-block (4 bits per scale, 2 scales per byte)
        if ib % 2 == 0:
            scale_bits = scales[ib // 2] & 0x0F  # Lower 4 bits
        else:
            scale_bits = (scales[ib // 2] >> 4) & 0x0F  # Upper 4 bits

        # Decode sub-block scale: db = d * (1 + 2*scale_bits)
        db = float(d) * (1 + 2 * scale_bits)

        # Process 8 groups of 4 elements each (32 elements per sub-block)
        for j in range(8):
            # Calculate element index within block
            elem_idx = ib * 32 + j * 4

            # Get quantized index (8 bits = 1 byte per 4-element group)
            qs_byte = qs[ib * 8 + j]

            # Get high bit from qh (1 bit per 4-element group)
            # qh has 8 bytes, each byte stores high bits for 8 groups
            qh_byte_idx = (ib * 8 + j) // 8
            qh_bit_idx = (ib * 8 + j) % 8
            high_bit = (qh[qh_byte_idx] >> qh_bit_idx) & 0x01

            # Combine to form 9-bit grid index
            grid_idx = (high_bit << 8) | qs_byte

            # Clamp grid index to valid range
            if grid_idx >= 512:
                grid_idx = 511

            # Look up 4 values from grid
            grid_vals = grid[grid_idx]

            # Apply signs (1 bit per element)
            sign_byte_idx = elem_idx // 8
            for k in range(4):
                sign_bit_idx = (elem_idx + k) % 8
                sign = 1 if (signs[sign_byte_idx + k // 8] >> sign_bit_idx) & 0x01 else -1

                # Dequantize: value = sign * grid_val * scale
                output[elem_idx + k] = sign * grid_vals[k] * db

    return output


def dequantize_iq3_s(quantized_data: np.ndarray) -> np.ndarray:
    """
    Dequantize IQ3_S data.

    Args:
        quantized_data: Quantized data as uint8 array with shape (..., bytes_per_block)

    Returns:
        np.ndarray: Dequantized float32 array
    """
    # Get shape info
    orig_shape = quantized_data.shape

    # Last dimension should be bytes (multiple of BYTES_PER_BLOCK)
    if orig_shape[-1] % BYTES_PER_BLOCK != 0:
        raise ValueError(f"Last dimension must be multiple of {BYTES_PER_BLOCK}, got {orig_shape[-1]}")

    # Calculate number of blocks
    n_blocks = orig_shape[-1] // BYTES_PER_BLOCK

    # Reshape to (other_dims, n_blocks, BYTES_PER_BLOCK)
    other_dims = orig_shape[:-1]
    data_reshaped = quantized_data.reshape(*other_dims, n_blocks, BYTES_PER_BLOCK)

    # Output shape: (other_dims, n_blocks * QK_IQ3S)
    output_shape = other_dims + (n_blocks * QK_IQ3S,)
    output = np.zeros(output_shape, dtype=np.float32)

    # Dequantize each block
    for block_idx in range(n_blocks):
        # Get block bytes
        block_bytes = data_reshaped[..., block_idx, :].tobytes()

        # Dequantize block
        block_output = dequantize_block_iq3_s(block_bytes, block_idx)

        # Store in output
        output[..., block_idx * QK_IQ3S:(block_idx + 1) * QK_IQ3S] = block_output

    return output


# Lazy-loaded lookup tables
_kmap = None
_kneighbors = None

def _get_lookup_tables():
    """Lazy-load lookup tables (61KB)"""
    global _kmap, _kneighbors
    if _kmap is None:
        from .iq3s_tables_generated import _IQ3S_KMAP, _IQ3S_KNEIGHBORS
        _kmap = _IQ3S_KMAP
        _kneighbors = _IQ3S_KNEIGHBORS
    return _kmap, _kneighbors


def _find_nearest_grid_index(values_4: np.ndarray, importance: Optional[np.ndarray] = None) -> int:
    """
    Find grid index for 4 values using kmap lookup or neighbor search.

    Args:
        values_4: 4 int8 values (each 0-7 range, representing 3-bit)
        importance: Optional 4-element importance weights

    Returns:
        Grid index (0-511)
    """
    kmap, kneighbors = _get_lookup_tables()

    # Create 12-bit index from 4 3-bit values
    u = 0
    for i in range(4):
        u |= (int(values_4[i]) << (3 * i))

    grid_index = kmap[u]

    if grid_index >= 0:
        # On grid - direct lookup
        return grid_index

    # Off grid - search neighbors
    neighbor_offset = -(grid_index + 1)
    num_neighbors = int(kneighbors[neighbor_offset])

    # Search neighbors for best match
    best_dist = float('inf')
    best_idx = 0

    for i in range(num_neighbors):
        neighbor_idx = int(kneighbors[neighbor_offset + 1 + i])
        grid_vals = _IQ3S_GRID[neighbor_idx]

        # Calculate weighted distance
        if importance is not None:
            dist = np.sum(importance * (grid_vals - values_4) ** 2)
        else:
            dist = np.sum((grid_vals - values_4) ** 2)

        if dist < best_dist:
            best_dist = dist
            best_idx = neighbor_idx

    return best_idx


def quantize_block_iq3_s(x: np.ndarray, importance: Optional[np.ndarray] = None) -> bytes:
    """
    Quantize 256 float32 values to IQ3_S block (110 bytes).

    Args:
        x: 256 float32 values
        importance: Optional 256 importance weights

    Returns:
        110 bytes: d(2) + qs(64) + qh(8) + signs(32) + scales(4)
    """
    if len(x) != QK_IQ3S:
        raise ValueError(f"Expected {QK_IQ3S} elements, got {len(x)}")

    if importance is not None and len(importance) != QK_IQ3S:
        raise ValueError(f"Importance must have {QK_IQ3S} elements")

    # Output arrays
    qs = np.zeros(64, dtype=np.uint8)
    qh = np.zeros(8, dtype=np.uint8)
    signs = np.zeros(32, dtype=np.uint8)
    scales_int = np.zeros(4, dtype=np.uint8)

    # Constants
    kMaxQ = 8
    BLOCK_SIZE = 32

    # Temp arrays
    xval = np.zeros(BLOCK_SIZE, dtype=np.float32)
    weight = np.zeros(BLOCK_SIZE, dtype=np.float32)
    L = np.zeros(BLOCK_SIZE, dtype=np.int8)
    Laux = np.zeros(BLOCK_SIZE, dtype=np.int8)
    block_signs = np.zeros(BLOCK_SIZE // 8, dtype=np.uint8)

    # Calculate sigma2 for importance weighting
    sumx2 = np.sum(x * x)
    sigma2 = 2 * sumx2 / QK_IQ3S

    max_scale = 0.0
    scales_float = np.zeros(8, dtype=np.float32)

    # Process 8 sub-blocks of 32 elements
    for ib in range(8):
        xb = x[ib * BLOCK_SIZE:(ib + 1) * BLOCK_SIZE]

        # Setup weights
        if importance is not None:
            imp = importance[ib * BLOCK_SIZE:(ib + 1) * BLOCK_SIZE]
            for i in range(BLOCK_SIZE):
                weight[i] = imp[i] * np.sqrt(sigma2 + xb[i] * xb[i])
        else:
            weight = xb * xb

        # Extract signs
        for k in range(BLOCK_SIZE // 8):
            s = 0
            for i in range(8):
                idx = 8 * k + i
                if xb[idx] >= 0:
                    xval[idx] = xb[idx]
                else:
                    xval[idx] = -xb[idx]
                    s |= (1 << i)
            block_signs[k] = s

        max_val = np.max(xval)
        if max_val == 0:
            scales_float[ib] = 0
            continue

        # Scale optimization: try different scale adjustments
        best = 0.0
        scale = max_val / (2 * kMaxQ - 1)

        for is_offset in range(-9, 10):
            id_scale = (2 * kMaxQ - 1 + is_offset * 0.2) / max_val
            this_scale = 1.0 / id_scale

            # Quantize 4-element groups
            for k in range(BLOCK_SIZE // 4):
                for i in range(4):
                    idx = 4 * k + i
                    l = int(np.round(0.5 * (id_scale * xval[idx] - 1)))
                    Laux[idx] = max(0, min(kMaxQ - 1, l))

            # Calculate quality with weighted least squares
            sumqx = 0.0
            sumq2 = 0.0
            for i in range(BLOCK_SIZE):
                w = weight[i]
                q = 2 * Laux[i] + 1
                sumqx += w * xval[i] * q
                sumq2 += w * q * q

            if sumq2 > 0 and sumqx * sumqx > best * sumq2:
                scale = sumqx / sumq2
                best = scale * sumqx
                L[:] = Laux

        if scale < 0:
            scale = -scale
            block_signs = ~block_signs

        # Encode grid indices
        for k in range(BLOCK_SIZE // 4):
            grid_idx = _find_nearest_grid_index(L[4*k:4*k+4])

            # Store in qs and qh
            qs_idx = ib * 8 + k
            qs[qs_idx] = grid_idx & 0xFF

            # High bit storage (9th bit) - 1 bit per group
            qh_byte = (ib * 8 + k) // 8
            qh_bit = (ib * 8 + k) % 8
            if grid_idx & 0x100:
                qh[qh_byte] |= (1 << qh_bit)

        # Store signs
        for k in range(BLOCK_SIZE // 8):
            signs[ib * 4 + k] = block_signs[k]

        scales_float[ib] = scale
        max_scale = max(max_scale, scale)

    # Encode scales
    if max_scale == 0:
        d = 0.0
    else:
        d = max_scale / 31.0
        # Scale calibration factor from llama.cpp
        d *= 1.033

    # Encode sub-block scales as 4-bit values
    if d > 0:
        id_scale = 1.0 / d
        for ib in range(0, 8, 2):
            l1 = int(np.round(0.5 * (id_scale * scales_float[ib] - 1)))
            l1 = max(0, min(15, l1))
            l2 = int(np.round(0.5 * (id_scale * scales_float[ib + 1] - 1)))
            l2 = max(0, min(15, l2))
            scales_int[ib // 2] = l1 | (l2 << 4)

    # Pack into bytes
    d_bytes = np.float16(d).tobytes()

    return d_bytes + qs.tobytes() + qh.tobytes() + signs.tobytes() + scales_int.tobytes()


def quantize_iq3_s(tensor: np.ndarray, parallel: bool = True, importance: Optional[np.ndarray] = None) -> bytes:
    """
    Quantize a tensor to IQ3_S format.

    Args:
        tensor: Input tensor (float32)
        parallel: Enable parallel processing (not yet implemented for IQ3_S)
        importance: Optional importance weights for each element

    Returns:
        bytes: Quantized data
    """
    # Flatten and ensure contiguous
    tensor_flat = tensor.flatten().astype(np.float32)

    if importance is not None:
        importance_flat = importance.flatten().astype(np.float32)
    else:
        importance_flat = None

    # Must be multiple of QK_IQ3S (256)
    n = len(tensor_flat)
    if n % QK_IQ3S != 0:
        raise ValueError(f"Tensor size must be multiple of {QK_IQ3S}, got {n}")

    n_blocks = n // QK_IQ3S
    output = bytearray()

    # Quantize each block
    for i in range(n_blocks):
        block_data = tensor_flat[i * QK_IQ3S:(i + 1) * QK_IQ3S]

        if importance_flat is not None:
            block_importance = importance_flat[i * QK_IQ3S:(i + 1) * QK_IQ3S]
        else:
            block_importance = None

        block_bytes = quantize_block_iq3_s(block_data, block_importance)
        output.extend(block_bytes)

    return bytes(output)


if __name__ == "__main__":
    # Test grid loading
    grid = get_iq3s_grid()
    print(f"IQ3_S grid shape: {grid.shape}")
    print(f"IQ3_S grid dtype: {grid.dtype}")
    print(f"IQ3_S grid sample values: {grid[0]}")
    print(f"Grid min/max: {grid.min()}/{grid.max()}")
