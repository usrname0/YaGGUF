"""
Q8_0 quantization implementation

Q8_0 format:
- Block size: 32 elements
- Each block: 1 FP16 scale + 32 int8 values
- Total: 34 bytes per block
"""

import numpy as np
import struct


QK8_0 = 32  # Block size for Q8_0


def quantize_q8_0(tensor):
    """
    Quantize a float tensor to Q8_0 format.

    Args:
        tensor: numpy array of floats (any shape)

    Returns:
        bytes: quantized data in Q8_0 format

    Format per block (34 bytes):
        - 2 bytes: FP16 scale factor (d)
        - 32 bytes: int8 quantized values (qs)

    Note: Preserves tensor shape by processing last dimension in blocks
    """
    original_shape = tensor.shape

    # Reshape to 2D: (batch, elements_per_row)
    # This preserves the structure needed by GGUF
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
        remainder = n % QK8_0
        if remainder != 0:
            padding = QK8_0 - remainder
            row = np.pad(row, (0, padding), mode='constant', constant_values=0)
            n = len(row)

        # Number of blocks in this row
        nb = n // QK8_0

        # Reshape into blocks
        blocks = row.reshape(nb, QK8_0)

        # Quantize each block in this row
        for block in blocks:
            # Find max absolute value in block
            amax = np.abs(block).max()

            # Calculate scale
            # d = amax / 127 (max value for int8)
            if amax == 0:
                d = 0.0
                id_scale = 0.0
            else:
                d = amax / 127.0
                id_scale = 1.0 / d

            # Quantize each element
            # q = round(x / d) = round(x * id)
            quantized = np.round(block * id_scale).astype(np.int8)

            # Clip to int8 range (should already be in range, but safety first)
            quantized = np.clip(quantized, -128, 127)

            # Convert scale to FP16
            d_fp16 = np.float16(d)

            # Pack into bytes
            # Format: FP16 scale + 32 int8 values
            output.extend(struct.pack('<e', d_fp16))  # '<e' = little-endian FP16
            output.extend(quantized.tobytes())

    return bytes(output)


def dequantize_q8_0(data, n_elements):
    """
    Dequantize Q8_0 format back to float32.

    Args:
        data: bytes in Q8_0 format
        n_elements: number of elements in original tensor

    Returns:
        numpy array of float32
    """
    nb = (n_elements + QK8_0 - 1) // QK8_0  # Number of blocks
    output = np.zeros(nb * QK8_0, dtype=np.float32)

    offset = 0
    for i in range(nb):
        # Read scale (FP16, 2 bytes)
        d = struct.unpack('<e', data[offset:offset+2])[0]
        offset += 2

        # Read quantized values (32 int8 bytes)
        qs = np.frombuffer(data[offset:offset+QK8_0], dtype=np.int8)
        offset += QK8_0

        # Dequantize: value = q * d
        output[i * QK8_0:(i + 1) * QK8_0] = qs.astype(np.float32) * float(d)

    return output[:n_elements]


def test_q8_0():
    """Test Q8_0 quantization/dequantization"""
    print("Testing Q8_0 quantization...")

    # Create test data
    test_data = np.random.randn(128).astype(np.float32)
    print(f"Original shape: {test_data.shape}")
    print(f"Original range: [{test_data.min():.4f}, {test_data.max():.4f}]")

    # Quantize
    quantized = quantize_q8_0(test_data)
    print(f"Quantized size: {len(quantized)} bytes")
    print(f"Compression ratio: {test_data.nbytes / len(quantized):.2f}x")

    # Dequantize
    dequantized = dequantize_q8_0(quantized, len(test_data))
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

    print("\nTest passed!" if correlation > 0.99 else "\nTest FAILED!")


if __name__ == "__main__":
    test_q8_0()
