"""
IQ3_XXS quantization (3.0625 bits per weight)

Block structure:
- Block size: 256 elements (QK_K)
- Block bytes: 98 bytes
  - d: 2 bytes (FP16 scale)
  - qs: 64 bytes (grid indices, each index gives 4 values)
  - scales: 32 bytes (packed sign bits and sub-block scales)

This format uses a large lookup grid (256 entries x 4 values) and complex
sign encoding to achieve high quality at ~3.06 bpw.
"""

import numpy as np
from typing import Optional
from .iq_tables import IQ3XXS_GRID, KSIGNS_IQ2XS

# Constants
QK_K = 256
QK3_XXS = QK_K  # Alias for clarity
BYTES_PER_BLOCK = 98

# Block structure offsets
OFFSET_D = 0
OFFSET_QS = 2
OFFSET_SCALES = 2 + 64  # 2 + QK_K // 4
SIZE_D = 2
SIZE_QS = 64  # QK_K // 4
SIZE_SCALES = 32  # Remaining bytes


def dequantize_iq3_xxs(data: np.ndarray) -> np.ndarray:
    """
    Dequantize IQ3_XXS format to float32.

    Args:
        data: Raw block data (n_blocks * 98 bytes)

    Returns:
        Float32 array of dequantized values (n_blocks * 256 elements)
    """
    # Reshape to blocks
    n_blocks = len(data) // BYTES_PER_BLOCK
    blocks = data[:n_blocks * BYTES_PER_BLOCK].reshape(n_blocks, BYTES_PER_BLOCK)

    # Split block into components
    d_bytes = blocks[:, OFFSET_D:OFFSET_D + SIZE_D]
    qs_bytes = blocks[:, OFFSET_QS:OFFSET_QS + SIZE_QS]
    scale_bytes = blocks[:, OFFSET_SCALES:OFFSET_SCALES + SIZE_SCALES]

    # Decode scale (FP16 → FP32)
    d = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32)
    d = d.reshape(n_blocks, 1)

    # Decode scales (32 bytes → 8 uint32 values)
    scales = np.frombuffer(scale_bytes.tobytes(), dtype=np.uint32).reshape(n_blocks, 8)

    # Compute db (dequantization scale per sub-block)
    # db = d * (0.5 + (scales >> 28)) * 0.5
    # The high 4 bits of each scale uint32 encode a sub-block scale adjustment
    high_bits = (scales >> 28).astype(np.float32)
    db = d.reshape(n_blocks, -1, 1, 1) * (np.float32(0.5) + high_bits.reshape(n_blocks, -1, 1, 1)) * np.float32(0.5)

    # Extract sign bits from scales
    # Each uint32 has 4 x 7-bit indices at positions [0, 7, 14, 21]
    # These indices lookup in ksigns table (128 entries), each entry is 8 sign bits
    signs = scales.reshape(n_blocks, -1, 1) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape(1, 1, 4)
    signs = signs & np.uint32(0x7F)  # Mask to 7 bits
    signs = signs.reshape(n_blocks, -1, 4, 1)

    # Look up in ksigns table (use first 128 entries, table is repeated)
    ksigns = KSIGNS_IQ2XS[:128].reshape(1, 1, 1, 128)
    signs = np.take_along_axis(ksigns, signs, axis=-1)
    signs = signs.reshape(n_blocks, -1, 4, 1)

    # Unpack 8 bits from each ksigns byte
    signs = signs >> np.array([i for i in range(8)], dtype=np.uint8).reshape(1, 1, 1, 8)
    signs = signs & np.uint8(0x01)
    signs = np.where(signs == 0, np.float32(1), np.float32(-1))
    signs = signs.reshape(n_blocks, -1, 4, 8)

    # Grid lookup
    # qs has 64 bytes, each is an index into the grid (256, 4)
    # Reshape to (n_blocks, 64, 1, 1) for broadcasting
    qs = qs_bytes.reshape(n_blocks, -1, 1, 1)
    grid = np.take_along_axis(IQ3XXS_GRID.reshape(1, -1, 4, 1), qs, axis=1)
    grid = grid.reshape(n_blocks, -1, 4, 8)

    # Final dequantization: db * grid * signs
    result = (db * grid * signs).reshape(n_blocks, -1)
    return result.flatten()


def find_nearest_grid_index(values_4, grid, importance=None):
    """
    Find the nearest grid entry for a group of 4 values.

    Args:
        values_4: numpy array of 4 float values
        grid: IQ3XXS_GRID (256, 4) array
        importance: Optional importance weights for the 4 values

    Returns:
        Index (0-255) of the nearest grid entry
    """
    # Compute squared distance to all grid entries
    # Broadcasting: (4,) vs (256, 4) -> (256,)
    errors = (grid - values_4) ** 2

    # Apply importance weighting if provided
    if importance is not None:
        # DEBUG: Verify importance is not all ones
        if not np.allclose(importance, 1.0):
            # Importance weighting: errors for important values are weighted more heavily
            # errors shape: (256, 4), importance shape: (4,)
            # Broadcasting will multiply each column by corresponding importance value
            errors = errors * importance.reshape(1, 4)

    distances = np.sum(errors, axis=1)
    return np.argmin(distances)


def build_ksigns_reverse_lookup():
    """
    Build reverse lookup table for ksigns.
    Maps 8-bit sign patterns to 7-bit indices.

    Returns:
        Dictionary mapping uint8 sign pattern to index (0-127)
    """
    reverse = {}
    for i in range(128):
        pattern = KSIGNS_IQ2XS[i]
        reverse[pattern] = i
    return reverse


# Build reverse lookup once at module load
_KSIGNS_REVERSE = build_ksigns_reverse_lookup()


def find_best_ksigns_index(values_8):
    """
    Find the best ksigns index for an 8-element group.

    Args:
        values_8: numpy array of 8 float values

    Returns:
        ksigns index (0-127)
    """
    # Determine signs from actual values
    # In ksigns: bit i set (1) means negative, bit i clear (0) means positive
    # But in dequant: signs == 0 -> +1, signs == 1 -> -1
    # So: value < 0 -> bit should be 1, value >= 0 -> bit should be 0
    signs = np.where(values_8 < 0, 1, 0).astype(np.uint8)

    # Pack 8 bits into a byte
    sign_pattern = 0
    for i in range(8):
        if signs[i]:
            sign_pattern |= (1 << i)

    # Find this pattern in ksigns table (or closest match)
    if sign_pattern in _KSIGNS_REVERSE:
        return _KSIGNS_REVERSE[sign_pattern]
    else:
        # Find closest match using Hamming distance
        best_idx = 0
        best_dist = 8
        for idx in range(128):
            pattern = KSIGNS_IQ2XS[idx]
            dist = bin(pattern ^ sign_pattern).count('1')
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx


def quantize_block_iq3_xxs(x: np.ndarray, importance: Optional[np.ndarray] = None):
    """
    Quantize a single 256-element block to IQ3_XXS format.

    Layout: 8 sub-blocks of 32 elements each.
    Each sub-block has 4 groups of 8 elements.
    Each group of 8 needs: 2 qs indices (2×4 grid values) + 1 ksigns index (8 signs).

    Args:
        x: numpy array of 256 float32 values
        importance: Optional numpy array of 256 importance weights (default: uniform)

    Returns:
        d: FP16 global scale
        qs: 64 bytes of grid indices
        scales: 32 bytes of packed scale/sign data (8 uint32 values)
    """
    # Use uniform importance if not provided
    if importance is None:
        importance = np.ones_like(x)
    assert len(x) == QK_K, f"Block must have {QK_K} elements"

    # Handle near-zero blocks
    amax = np.max(np.abs(x))
    if amax < 1e-8:
        d = np.float16(0.0)
        qs = np.zeros(64, dtype=np.uint8)
        scales = np.zeros(8, dtype=np.uint32)
        return d, qs, scales

    # Initial global scale estimate
    max_grid_val = 62.0
    d = amax / max_grid_val

    # First pass: quantize with initial scale
    qs_temp = np.zeros(64, dtype=np.uint8)
    scale_bits_temp = np.zeros(8, dtype=np.int32)
    sign_indices_temp = []

    for sub_block in range(8):
        start_idx = sub_block * 32
        sub_values = x[start_idx:start_idx + 32]

        # Compute sub-block scale
        sub_amax = np.max(np.abs(sub_values))
        if sub_amax > 0 and d > 0:
            target_db = sub_amax / max_grid_val
            scale_bits = (2.0 * target_db / d) - 0.5
            scale_bits = int(np.clip(scale_bits, 0.0, 15.0))
        else:
            scale_bits = 0
        scale_bits_temp[sub_block] = scale_bits

        db = d * (0.5 + scale_bits) * 0.5

        # Process 4 groups of 8 elements each
        sub_sign_indices = np.zeros(4, dtype=np.uint8)
        for group in range(4):
            # Each group has 8 elements
            group_start = group * 8
            group_values = sub_values[group_start:group_start + 8]

            # Find ksigns index for this group of 8
            sub_sign_indices[group] = find_best_ksigns_index(group_values)

            # Split this group of 8 into 2 sub-groups of 4 for grid search
            for sub_group in range(2):
                sub_group_values = group_values[sub_group * 4:(sub_group + 1) * 4]

                # Get importance for this sub-group
                val_start_global = start_idx + group * 8 + sub_group * 4
                sub_importance = importance[val_start_global:val_start_global + 4]

                # Normalize and search grid
                if db > 0:
                    normalized = np.abs(sub_group_values) / db
                else:
                    normalized = np.zeros(4, dtype=np.float32)

                grid_idx = find_nearest_grid_index(normalized, IQ3XXS_GRID.astype(np.float32), sub_importance)

                # Store: qs[sub_block * 8 + group * 2 + sub_group]
                qs_idx = sub_block * 8 + group * 2 + sub_group
                qs_temp[qs_idx] = grid_idx

        sign_indices_temp.append(sub_sign_indices)

    # Optimize global scale using least squares
    sumqx = 0.0
    sumq2 = 0.0

    for sub_block in range(8):
        start_idx = sub_block * 32
        sub_values = x[start_idx:start_idx + 32]
        scale_bits = scale_bits_temp[sub_block]
        db_mult = (0.5 + scale_bits) * 0.5

        for group in range(4):
            for sub_group in range(2):
                qs_idx = sub_block * 8 + group * 2 + sub_group
                grid_idx = qs_temp[qs_idx]
                grid_values = IQ3XXS_GRID[grid_idx].astype(np.float32)

                # Get the 4 values this corresponds to
                val_start = group * 8 + sub_group * 4
                group_values = sub_values[val_start:val_start + 4]

                # Reconstruct signs
                signs = np.sign(group_values)
                signs[signs == 0] = 1

                # Q values (what we'll multiply by d)
                q_values = grid_values * signs * db_mult

                # Importance-weighted least squares
                val_start_global = start_idx + group * 8 + sub_group * 4
                sub_importance = importance[val_start_global:val_start_global + 4]

                for i in range(4):
                    # Weight by both value magnitude and importance
                    weight = group_values[i] * group_values[i] * sub_importance[i]
                    sumqx += weight * q_values[i] * group_values[i]
                    sumq2 += weight * q_values[i] * q_values[i]

    # Compute optimized scale
    if sumq2 > 0:
        d = sumqx / sumq2
    d = np.float16(d)

    # Second pass with optimized scale
    qs = np.zeros(64, dtype=np.uint8)
    scales_out = np.zeros(8, dtype=np.uint32)

    for sub_block in range(8):
        start_idx = sub_block * 32
        sub_values = x[start_idx:start_idx + 32]

        # Re-compute sub-block scale with optimized d
        sub_amax = np.max(np.abs(sub_values))
        if sub_amax > 0 and d > 0:
            target_db = sub_amax / max_grid_val
            scale_bits = (2.0 * target_db / float(d)) - 0.5
            scale_bits = int(np.clip(scale_bits, 0.0, 15.0))
        else:
            scale_bits = 0

        db = float(d) * (0.5 + scale_bits) * 0.5

        # Process 4 groups of 8 elements each
        sub_sign_indices = np.zeros(4, dtype=np.uint8)
        for group in range(4):
            group_start = group * 8
            group_values = sub_values[group_start:group_start + 8]

            # Find ksigns index
            sub_sign_indices[group] = find_best_ksigns_index(group_values)

            # Quantize 2 sub-groups of 4
            for sub_group in range(2):
                sub_group_values = group_values[sub_group * 4:(sub_group + 1) * 4]

                # Get importance for this sub-group
                val_start_global = start_idx + group * 8 + sub_group * 4
                sub_importance = importance[val_start_global:val_start_global + 4]

                if db > 0:
                    normalized = np.abs(sub_group_values) / db
                else:
                    normalized = np.zeros(4, dtype=np.float32)

                grid_idx = find_nearest_grid_index(normalized, IQ3XXS_GRID.astype(np.float32), sub_importance)
                qs_idx = sub_block * 8 + group * 2 + sub_group
                qs[qs_idx] = grid_idx

        # Pack into uint32: 4 sign indices (7 bits each) + scale_bits (4 bits)
        packed = np.uint32(0)
        packed |= np.uint32(sub_sign_indices[0] & 0x7F) << 0
        packed |= np.uint32(sub_sign_indices[1] & 0x7F) << 7
        packed |= np.uint32(sub_sign_indices[2] & 0x7F) << 14
        packed |= np.uint32(sub_sign_indices[3] & 0x7F) << 21
        packed |= np.uint32(scale_bits) << 28

        scales_out[sub_block] = packed

    return d, qs, scales_out


def quantize_iq3_xxs(tensor: np.ndarray, parallel: bool = True, importance: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Quantize float32 tensor to IQ3_XXS format.

    Args:
        tensor: Float32 input tensor
        parallel: Whether to use parallel processing (not yet implemented)
        importance: Optional importance weights array (same shape as tensor)

    Returns:
        Quantized data as uint8 array (98 bytes per 256-element block)
    """
    # Flatten and ensure float32
    data = tensor.flatten().astype(np.float32)
    n = len(data)

    # Handle importance weights
    if importance is not None:
        importance_flat = importance.flatten().astype(np.float32)
        if len(importance_flat) != len(data):
            raise ValueError(f"Importance shape mismatch: {len(importance_flat)} vs {len(data)}")
    else:
        importance_flat = None

    # Pad to block size if needed
    remainder = n % QK_K
    if remainder != 0:
        padding = QK_K - remainder
        data = np.pad(data, (0, padding), mode='constant', constant_values=0)
        if importance_flat is not None:
            importance_flat = np.pad(importance_flat, (0, padding), mode='constant', constant_values=1.0)
        n = len(data)

    # Calculate number of blocks
    num_blocks = n // QK_K

    # Allocate output buffer
    output = np.zeros(num_blocks * BYTES_PER_BLOCK, dtype=np.uint8)

    # Process each block
    for i in range(num_blocks):
        block_start = i * QK_K
        block_end = block_start + QK_K
        block_data = data[block_start:block_end]

        # Get importance for this block
        if importance_flat is not None:
            block_importance = importance_flat[block_start:block_end]
        else:
            block_importance = None

        # Quantize block
        d, qs, scales = quantize_block_iq3_xxs(block_data, block_importance)

        # Pack into output buffer
        out_offset = i * BYTES_PER_BLOCK

        # Write scale (FP16, 2 bytes)
        d_bytes = np.array([d], dtype=np.float16).tobytes()
        output[out_offset:out_offset+2] = np.frombuffer(d_bytes, dtype=np.uint8)

        # Write grid indices (64 bytes)
        output[out_offset+2:out_offset+66] = qs

        # Write packed scales (32 bytes = 8 uint32)
        scales_bytes = scales.tobytes()
        output[out_offset+66:out_offset+98] = np.frombuffer(scales_bytes, dtype=np.uint8)

    return output


def test_iq3_xxs():
    """
    Test IQ3_XXS quantization and dequantization with round-trip
    """
    print("IQ3_XXS Quantization Round-Trip Test")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    test_data = np.random.randn(1024).astype(np.float32)

    print(f"Original data: {test_data.shape}, range [{test_data.min():.3f}, {test_data.max():.3f}]")
    print(f"Mean: {test_data.mean():.3f}, Std: {test_data.std():.3f}")

    # Quantize
    print("\nQuantizing...")
    quantized = quantize_iq3_xxs(test_data)

    num_blocks = (len(test_data) + QK_K - 1) // QK_K
    expected_size = num_blocks * BYTES_PER_BLOCK
    print(f"Quantized: {len(quantized)} bytes ({len(quantized) / len(test_data):.3f} bytes per element)")
    print(f"Expected size: {expected_size} bytes")
    print(f"Bits per weight: {(len(quantized) * 8) / len(test_data):.3f}")

    assert len(quantized) == expected_size, f"Size mismatch: {len(quantized)} != {expected_size}"

    # Dequantize
    print("\nDequantizing...")
    dequantized = dequantize_iq3_xxs(quantized)
    dequantized = dequantized[:len(test_data)]  # Trim padding

    print(f"Dequantized: {dequantized.shape}, range [{dequantized.min():.3f}, {dequantized.max():.3f}]")
    print(f"Mean: {dequantized.mean():.3f}, Std: {dequantized.std():.3f}")

    # Compute error metrics
    error = test_data - dequantized
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    correlation = np.corrcoef(test_data.flatten(), dequantized.flatten())[0, 1]

    print(f"\nQuality metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Correlation: {correlation:.6f}")

    # Quality check - 3-bit should have >0.95 correlation
    if correlation > 0.95:
        print(f"\n[OK] IQ3_XXS round-trip test passed! Correlation: {correlation:.4f}")
    else:
        print(f"\n[WARNING] Correlation lower than expected: {correlation:.4f} < 0.95")
        print("This may indicate an issue with the quantization algorithm.")

    return correlation


if __name__ == "__main__":
    test_iq3_xxs()
