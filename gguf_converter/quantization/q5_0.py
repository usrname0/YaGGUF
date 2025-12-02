"""
Q5_0 quantization implementation

Q5_0 format:
- Block size: 32 elements
- Each block: 1 FP16 scale + 4 bytes high bits + 16 bytes (32 4-bit values packed)
- Total: 22 bytes per block
- Range: -16 to +15 (5-bit signed, offset by 16)
"""

import numpy as np
import struct


QK5_0 = 32  # Block size for Q5_0


def quantize_q5_0(tensor):
    """
    Quantize a float tensor to Q5_0 format.

    Args:
        tensor: numpy array of floats (any shape)

    Returns:
        bytes: quantized data in Q5_0 format

    Format per block (22 bytes):
        - 2 bytes: FP16 scale factor (d)
        - 4 bytes: uint32 qh (high 5th bits for 32 values)
        - 16 bytes: 32 4-bit quantized values (lower 4 bits, packed 2 per byte)

    Note: 5-bit values are split: lower 4 bits in qs, 5th bit in qh
    """
    original_shape = tensor.shape

    # Reshape to 2D: (batch, elements_per_row)
    if tensor.ndim == 1:
        data = tensor.reshape(1, -1)
    else:
        # Flatten all dims except last into first dim
        data = tensor.reshape(-1, original_shape[-1])

    data = data.astype(np.float32)
    n_rows, n_cols = data.shape

    # Output buffer
    output = bytearray()

    # Process each row
    for row_idx in range(n_rows):
        row = data[row_idx]
        n = len(row)

        # Pad row to multiple of block size
        remainder = n % QK5_0
        if remainder != 0:
            padding = QK5_0 - remainder
            row = np.pad(row, (0, padding), mode='constant', constant_values=0)
            n = len(row)

        # Number of blocks in this row
        nb = n // QK5_0

        # Reshape into blocks
        blocks = row.reshape(nb, QK5_0)

        # Quantize each block in this row
        for block in blocks:
            # Find max absolute value in block
            amax = np.abs(block).max()

            # Calculate scale
            # d = max / -16 (maps range to -16..+15 for 5-bit)
            if amax == 0:
                d = 0.0
                id_scale = 0.0
            else:
                d = amax / 16.0
                id_scale = 1.0 / d

            # Quantize each element to 5-bit (0-31 range, representing -16 to +15)
            # Formula: xi = clamp(x / d + 16.5, 0, 31)
            quantized = (block * id_scale + 16.5).astype(np.int32)
            quantized = np.clip(quantized, 0, 31).astype(np.uint8)

            # Split into lower 4 bits and high 5th bit
            qs = quantized & 0x0F  # Lower 4 bits
            qh_bits = (quantized >> 4) & 0x01  # 5th bit

            # Pack high bits into a uint32 (vectorized)
            qh = int(np.sum(qh_bits.astype(np.uint32) << np.arange(32, dtype=np.uint32)))

            # Convert scale to FP16
            d_fp16 = np.float16(d)

            # Pack into bytes
            # Format: FP16 scale + uint32 qh + 16 bytes (2 4-bit values per byte)
            output.extend(struct.pack('<e', d_fp16))  # '<e' = little-endian FP16
            output.extend(struct.pack('<I', qh))       # '<I' = little-endian uint32

            # Pack pairs of 4-bit values into bytes (vectorized)
            # First value in lower nibble, second in upper nibble
            xi0 = qs[:QK5_0//2]          # First half
            xi1 = qs[QK5_0//2:]          # Second half
            packed_bytes = xi0 | (xi1 << 4)
            output.extend(packed_bytes.tobytes())

    return bytes(output)


def dequantize_q5_0(data, n_elements):
    """
    Dequantize Q5_0 format back to float32.

    Args:
        data: bytes in Q5_0 format
        n_elements: number of elements in original tensor

    Returns:
        numpy array of float32
    """
    nb = (n_elements + QK5_0 - 1) // QK5_0  # Number of blocks
    output = np.zeros(nb * QK5_0, dtype=np.float32)

    offset = 0
    for i in range(nb):
        # Read scale (FP16, 2 bytes)
        d = struct.unpack('<e', data[offset:offset+2])[0]
        offset += 2

        # Read qh (uint32, 4 bytes)
        qh = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4

        # Read 16 bytes of packed 4-bit values (vectorized)
        packed_bytes = np.frombuffer(data[offset:offset + QK5_0//2], dtype=np.uint8)
        offset += QK5_0 // 2

        # Unpack lower and upper nibbles
        xi0 = packed_bytes & 0x0F          # Lower 4 bits
        xi1 = (packed_bytes >> 4) & 0x0F   # Upper 4 bits

        # Dequantize with high bit extraction (semi-vectorized)
        for j in range(QK5_0 // 2):
            # Add high bit from qh to reconstruct 5-bit value
            xh0 = 16 if (qh >> j) & 1 else 0
            xh1 = 16 if (qh >> (j + QK5_0//2)) & 1 else 0

            # Reconstruct 5-bit value and dequantize: value = (q - 16) * d
            output[i * QK5_0 + j] = (int(xi0[j]) + xh0 - 16) * float(d)
            output[i * QK5_0 + j + QK5_0//2] = (int(xi1[j]) + xh1 - 16) * float(d)

    return output[:n_elements]


def test_q5_0():
    """Test Q5_0 quantization/dequantization"""
    print("Testing Q5_0 quantization...")

    # Create test data
    test_data = np.random.randn(128).astype(np.float32)
    print(f"Original shape: {test_data.shape}")
    print(f"Original range: [{test_data.min():.4f}, {test_data.max():.4f}]")

    # Quantize
    quantized = quantize_q5_0(test_data)
    print(f"Quantized size: {len(quantized)} bytes")
    print(f"Expected size: {(128 // QK5_0) * 22} bytes")
    print(f"Compression ratio: {test_data.nbytes / len(quantized):.2f}x")

    # Dequantize
    dequantized = dequantize_q5_0(quantized, len(test_data))
    print(f"Dequantized shape: {dequantized.shape}")
    print(f"Dequantized range: [{dequantized.min():.4f}, {dequantized.max():.4f}]")

    # Calculate error
    error = np.abs(test_data - dequantized).mean()
    max_error = np.abs(test_data - dequantized).max()
    print(f"Mean absolute error: {error:.6f}")
    print(f"Max absolute error: {max_error:.6f}")

    # Check correlation
    correlation = np.corrcoef(test_data, dequantized)[0, 1]
    print(f"Correlation: {correlation:.6f}")

    print("\nTest passed!" if correlation > 0.98 else "\nTest FAILED!")


if __name__ == "__main__":
    test_q5_0()
