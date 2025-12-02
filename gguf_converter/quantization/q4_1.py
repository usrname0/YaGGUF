"""
Q4_1 quantization implementation

Q4_1 format:
- Block size: 32 elements
- Each block: 1 FP16 scale + 1 FP16 min + 16 bytes (32 4-bit values packed)
- Total: 20 bytes per block
- Range: 0 to 15 (4-bit unsigned)
"""

import numpy as np
import struct


QK4_1 = 32  # Block size for Q4_1


def quantize_q4_1(tensor):
    """
    Quantize a float tensor to Q4_1 format.

    Args:
        tensor: numpy array of floats (any shape)

    Returns:
        bytes: quantized data in Q4_1 format

    Format per block (20 bytes):
        - 2 bytes: FP16 scale factor (d)
        - 2 bytes: FP16 minimum value (m)
        - 16 bytes: 32 4-bit quantized values (packed 2 per byte)

    Note: Q4_1 uses min-max quantization to map [min, max] to [0, 15]
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
        remainder = n % QK4_1
        if remainder != 0:
            padding = QK4_1 - remainder
            row = np.pad(row, (0, padding), mode='constant', constant_values=0)
            n = len(row)

        # Number of blocks in this row
        nb = n // QK4_1

        # Reshape into blocks
        blocks = row.reshape(nb, QK4_1)

        # Quantize each block
        for block in blocks:
            # Find min and max
            min_val = block.min()
            max_val = block.max()

            # Calculate scale
            # d = (max - min) / 15 (maps to 0..15 for 4-bit)
            if max_val > min_val:
                d = (max_val - min_val) / 15.0
                id_scale = 1.0 / d
            else:
                d = 0.0
                id_scale = 0.0

            # Quantize each element to 4-bit (0-15 range)
            # Formula: xi = clamp((x - min) / d + 0.5, 0, 15)
            if d > 0:
                quantized = ((block - min_val) * id_scale + 0.5).astype(np.int32)
            else:
                quantized = np.zeros(QK4_1, dtype=np.int32)

            quantized = np.clip(quantized, 0, 15).astype(np.uint8)

            # Convert scale and min to FP16
            d_fp16 = np.float16(d)
            m_fp16 = np.float16(min_val)

            # Pack into bytes
            # Format: FP16 scale + FP16 min + 16 bytes (2 4-bit values per byte)
            output.extend(struct.pack('<e', d_fp16))  # FP16 scale
            output.extend(struct.pack('<e', m_fp16))  # FP16 min

            # Pack pairs of 4-bit values into bytes (vectorized)
            # First value in lower nibble, second in upper nibble
            xi0 = quantized[:QK4_1//2]          # First half
            xi1 = quantized[QK4_1//2:]          # Second half
            packed_bytes = xi0 | (xi1 << 4)
            output.extend(packed_bytes.tobytes())

    return bytes(output)


def dequantize_q4_1(data, n_elements):
    """
    Dequantize Q4_1 format back to float32.

    Args:
        data: bytes in Q4_1 format
        n_elements: number of elements in original tensor

    Returns:
        numpy array of float32
    """
    nb = (n_elements + QK4_1 - 1) // QK4_1  # Number of blocks
    output = np.zeros(nb * QK4_1, dtype=np.float32)

    offset = 0
    for i in range(nb):
        # Read scale (FP16, 2 bytes)
        d = struct.unpack('<e', data[offset:offset+2])[0]
        offset += 2

        # Read min (FP16, 2 bytes)
        m = struct.unpack('<e', data[offset:offset+2])[0]
        offset += 2

        # Read 16 bytes of packed 4-bit values (vectorized)
        packed_bytes = np.frombuffer(data[offset:offset + QK4_1//2], dtype=np.uint8)
        offset += QK4_1 // 2

        # Unpack lower and upper nibbles
        xi0 = packed_bytes & 0x0F          # Lower 4 bits
        xi1 = (packed_bytes >> 4) & 0x0F   # Upper 4 bits

        # Dequantize: value = q * d + m (vectorized)
        output[i * QK4_1:i * QK4_1 + QK4_1//2] = xi0 * float(d) + float(m)
        output[i * QK4_1 + QK4_1//2:(i + 1) * QK4_1] = xi1 * float(d) + float(m)

    return output[:n_elements]


def test_q4_1():
    """Test Q4_1 quantization/dequantization"""
    print("Testing Q4_1 quantization...")

    # Create test data
    test_data = np.random.randn(128).astype(np.float32)
    print(f"Original shape: {test_data.shape}")
    print(f"Original range: [{test_data.min():.4f}, {test_data.max():.4f}]")

    # Quantize
    quantized = quantize_q4_1(test_data)
    print(f"Quantized size: {len(quantized)} bytes")
    print(f"Expected size: {(128 // QK4_1) * 20} bytes")
    print(f"Compression ratio: {test_data.nbytes / len(quantized):.2f}x")

    # Dequantize
    dequantized = dequantize_q4_1(quantized, len(test_data))
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

    print("\nTest passed!" if correlation > 0.97 else "\nTest FAILED!")


if __name__ == "__main__":
    test_q4_1()
