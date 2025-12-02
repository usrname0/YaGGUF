"""
IQ4_NL: 4-bit non-linear quantization

Block size: 32 elements → 18 bytes per block
Format:
  - 1x FP16 scale (2 bytes)
  - 16x uint8 quantized pairs (16 bytes)

Total: 18 bytes per 32 elements = 4.5 bits per weight

IQ4_NL uses a pre-optimized lookup table of 16 asymmetric values
instead of uniform quantization. This provides better quality at
the same bit depth compared to Q4_0.
"""

import numpy as np
import struct
import gguf
from typing import Optional

try:
    from .iq_tables import KVALUES_IQ4NL
except ImportError:
    from iq_tables import KVALUES_IQ4NL


# Block configuration
QK4_NL = 32  # Elements per block
BYTES_PER_BLOCK = 18  # 2 (scale) + 16 (quantized data)


def find_nearest_index(value, lookup_table):
    """
    Find the index of the nearest value in the lookup table.

    This is equivalent to llama.cpp's best_index_int8() function.
    Uses binary search for efficiency.

    Args:
        value: The value to quantize
        lookup_table: Array of possible quantized values (sorted)

    Returns:
        Index (0-15) of the nearest value
    """
    # Handle edge cases
    if value <= lookup_table[0]:
        return 0
    if value >= lookup_table[-1]:
        return len(lookup_table) - 1

    # Binary search
    left = 0
    right = len(lookup_table) - 1

    while right - left > 1:
        mid = (left + right) // 2
        if value < lookup_table[mid]:
            right = mid
        else:
            left = mid

    # Return closest
    if abs(value - lookup_table[left]) < abs(lookup_table[right] - value):
        return left
    else:
        return right


def quantize_block_iq4_nl(x, importance: Optional[np.ndarray] = None):
    """
    Quantize a single 32-element block to IQ4_NL format.

    Args:
        x: numpy array of 32 float32 values
        importance: Optional 32-element importance weights for better quantization

    Returns:
        scale: FP16 scale value
        quantized: 16 bytes of packed 4-bit indices
    """
    assert len(x) == QK4_NL, f"Block must have {QK4_NL} elements"

    if importance is not None:
        assert len(importance) == QK4_NL, f"Importance must have {QK4_NL} elements"

    # Find the maximum absolute value
    amax = np.max(np.abs(x))

    # Handle near-zero blocks
    if amax < 1e-8:
        scale = np.float16(0.0)
        quantized = np.zeros(QK4_NL // 2, dtype=np.uint8)
        return scale, quantized

    # Initial scale estimate: map max value to max lookup table value
    # The lookup table ranges from -127 to 113, so use the positive max
    d = amax / 127.0  # Using 127 as it's the largest magnitude in the table

    # Quantize each element by finding nearest in lookup table
    indices = np.zeros(QK4_NL, dtype=np.uint8)

    # Compute initial quantization
    id_val = 1.0 / d if d > 0 else 0.0
    for i in range(QK4_NL):
        scaled_val = id_val * x[i]
        indices[i] = find_nearest_index(scaled_val, KVALUES_IQ4NL)

    # Optimize scale using least squares
    # Minimize || x - d * table[indices] ||^2
    # With importance weighting if provided
    sumqx = 0.0
    sumq2 = 0.0

    for i in range(QK4_NL):
        q = KVALUES_IQ4NL[indices[i]]
        # Weight by squared magnitude, optionally scaled by importance
        if importance is not None:
            weight = importance[i] * x[i] * x[i]
        else:
            weight = x[i] * x[i]
        sumqx += weight * q * x[i]
        sumq2 += weight * q * q

    if sumq2 > 0:
        d = sumqx / sumq2
    else:
        d = 0.0

    # Re-quantize with optimized scale (optional refinement)
    # For simplicity, we'll skip the iterative refinement (ntry loop)
    # and use the weighted least squares scale directly

    scale = np.float16(d)

    # Final quantization pass
    id_val = 1.0 / d if d != 0 else 0.0
    for i in range(QK4_NL):
        scaled_val = id_val * x[i]
        indices[i] = find_nearest_index(scaled_val, KVALUES_IQ4NL)

    # Pack two 4-bit indices into each byte (interleaved order)
    # Lower 4 bits: first half element, Upper 4 bits: second half element
    # llama.cpp unpacks as: y[j] = qs[j] & 0xf, y[j+16] = qs[j] >> 4
    quantized = np.zeros(QK4_NL // 2, dtype=np.uint8)
    for i in range(QK4_NL // 2):
        low = indices[i] & 0xF               # Element i (first half)
        high = indices[i + QK4_NL // 2] & 0xF  # Element i+16 (second half)
        quantized[i] = low | (high << 4)

    return scale, quantized


def quantize_iq4_nl(tensor, importance: Optional[np.ndarray] = None):
    """
    Quantize a tensor to IQ4_NL format.

    Args:
        tensor: numpy array of float32 values (will be reshaped to process rows)
        importance: Optional importance weights (same shape as tensor) for better quantization

    Returns:
        numpy array of uint8 containing quantized data
    """
    # Flatten and ensure it's float32
    data = tensor.flatten().astype(np.float32)
    n = len(data)

    # Handle importance if provided
    if importance is not None:
        importance_flat = importance.flatten().astype(np.float32)
        if len(importance_flat) != n:
            raise ValueError(f"Importance shape {importance.shape} doesn't match tensor shape {tensor.shape}")
    else:
        importance_flat = None

    # Pad to block size if needed
    remainder = n % QK4_NL
    if remainder != 0:
        padding = QK4_NL - remainder
        data = np.pad(data, (0, padding), mode='constant', constant_values=0)
        if importance_flat is not None:
            importance_flat = np.pad(importance_flat, (0, padding), mode='constant', constant_values=1.0)
        n = len(data)

    # Calculate number of blocks
    num_blocks = n // QK4_NL

    # Allocate output buffer
    output = np.zeros(num_blocks * BYTES_PER_BLOCK, dtype=np.uint8)

    # Process each block
    for i in range(num_blocks):
        block_start = i * QK4_NL
        block_end = block_start + QK4_NL
        block_data = data[block_start:block_end]

        # Get importance for this block if available
        if importance_flat is not None:
            block_importance = importance_flat[block_start:block_end]
        else:
            block_importance = None

        # Quantize block
        scale, quantized = quantize_block_iq4_nl(block_data, block_importance)

        # Pack into output
        out_offset = i * BYTES_PER_BLOCK

        # Write scale (FP16, 2 bytes)
        scale_bytes = struct.pack('<e', scale)  # Little-endian FP16
        output[out_offset:out_offset+2] = np.frombuffer(scale_bytes, dtype=np.uint8)

        # Write quantized data (16 bytes)
        output[out_offset+2:out_offset+2+16] = quantized

    return output


def dequantize_block_iq4_nl(scale, quantized):
    """
    Dequantize a single IQ4_NL block.

    Args:
        scale: FP16 scale value
        quantized: 16 bytes of packed 4-bit indices

    Returns:
        numpy array of 32 float32 values
    """
    output = np.zeros(QK4_NL, dtype=np.float32)

    # Unpack and dequantize (interleaved order)
    # Matches llama.cpp: y[j] = d * kvalues[qs[j] & 0xf], y[j+16] = d * kvalues[qs[j] >> 4]
    for i in range(QK4_NL // 2):
        packed = quantized[i]
        idx_low = packed & 0xF
        idx_high = (packed >> 4) & 0xF

        output[i] = float(scale) * KVALUES_IQ4NL[idx_low]            # First half
        output[i + QK4_NL // 2] = float(scale) * KVALUES_IQ4NL[idx_high]  # Second half

    return output


def dequantize_iq4_nl(quantized_data, original_shape):
    """
    Dequantize IQ4_NL data back to float32.

    Args:
        quantized_data: uint8 array of quantized blocks
        original_shape: Original tensor shape (for verification)

    Returns:
        numpy array of float32 values
    """
    num_blocks = len(quantized_data) // BYTES_PER_BLOCK
    output = np.zeros(num_blocks * QK4_NL, dtype=np.float32)

    for i in range(num_blocks):
        in_offset = i * BYTES_PER_BLOCK

        # Read scale (FP16)
        scale_bytes = bytes(quantized_data[in_offset:in_offset+2])
        scale = struct.unpack('<e', scale_bytes)[0]

        # Read quantized data
        quantized = quantized_data[in_offset+2:in_offset+2+16]

        # Dequantize block
        block_output = dequantize_block_iq4_nl(scale, quantized)
        output[i*QK4_NL:(i+1)*QK4_NL] = block_output

    # Trim padding if needed
    original_size = np.prod(original_shape)
    return output[:original_size].reshape(original_shape)


if __name__ == "__main__":
    # Test IQ4_NL quantization
    print("Testing IQ4_NL quantization...")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    test_data = np.random.randn(1024).astype(np.float32)

    print(f"Original data: {test_data.shape}, range [{test_data.min():.3f}, {test_data.max():.3f}]")

    # Quantize
    quantized = quantize_iq4_nl(test_data)
    print(f"Quantized: {len(quantized)} bytes ({len(quantized) / len(test_data):.2f} bytes per element)")

    # Expected size
    num_blocks = (len(test_data) + QK4_NL - 1) // QK4_NL
    expected_size = num_blocks * BYTES_PER_BLOCK
    print(f"Expected size: {expected_size} bytes")
    assert len(quantized) == expected_size, f"Size mismatch: {len(quantized)} != {expected_size}"

    # Dequantize
    dequantized = dequantize_iq4_nl(quantized, test_data.shape)
    print(f"Dequantized: {dequantized.shape}, range [{dequantized.min():.3f}, {dequantized.max():.3f}]")

    # Compute error
    error = test_data - dequantized
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    correlation = np.corrcoef(test_data.flatten(), dequantized.flatten())[0, 1]

    print(f"\nQuality metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Correlation: {correlation:.6f}")

    # Quality should be good (>0.99 correlation for 4-bit)
    assert correlation > 0.98, f"Poor correlation: {correlation}"

    print("\n[OK] IQ4_NL quantization test passed!")
