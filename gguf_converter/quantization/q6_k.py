"""
Q6_K quantization implementation - faithfully recreated from llama.cpp

Q6_K format (from llama.cpp ggml-quants.c):
- Super-block size: 256 elements (QK_K = 256)
- Structure: 16 blocks of 16 elements each
- Each block has an int8 scale factor
- One FP16 super-block scale factor
- 6-bit quantization per element (range: -32 to 31)
- Effective: 6.5625 bits per weight

Block structure (from ggml-common.h):
- uint8_t ql[QK_K/2]: lower 4 bits of quantized values (128 bytes)
- uint8_t qh[QK_K/4]: upper 2 bits of quantized values (64 bytes)
- int8_t scales[QK_K/16]: per-block scales (16 bytes)
- ggml_half d: super-block scale (2 bytes)
- Total: 210 bytes per super-block
"""

import numpy as np
import struct
from .optimizations import make_qx_quants


QK_K = 256  # Super-block size
GROUP_MAX_EPS = 1e-8


def nearest_int(x):
    """Round to nearest integer (matches llama.cpp)"""
    return int(np.round(x))


def quantize_q6_k(tensor, scalar_optimization=False, verbose=False):
    """
    Quantize a float tensor to Q6_K format (llama.cpp implementation)

    Args:
        tensor: numpy array of floats (any shape)
        scalar_optimization: Enable 19-iteration optimization for better quality (20x slower)
        verbose: print progress for debugging

    Returns:
        bytes: quantized data in Q6_K format
    """
    original_shape = tensor.shape

    # Reshape to 2D: (batch, elements_per_row)
    if tensor.ndim == 1:
        data = tensor.reshape(1, -1)
    else:
        data = tensor.reshape(-1, original_shape[-1])

    data = data.astype(np.float32)
    n_rows, n_cols = data.shape

    # Output buffer
    output = bytearray()

    # Process each row
    for row_idx in range(n_rows):
        row = data[row_idx]
        n = len(row)

        # Pad row to multiple of super-block size
        remainder = n % QK_K
        if remainder != 0:
            padding = QK_K - remainder
            row = np.pad(row, (0, padding), mode='constant', constant_values=0)
            n = len(row)

        # Number of super-blocks in this row
        nb = n // QK_K

        # Process each super-block
        for i in range(nb):
            x = row[i * QK_K:(i + 1) * QK_K]

            # Temporary storage for quantized values
            L = np.zeros(QK_K, dtype=np.uint8)
            scales = np.zeros(QK_K // 16, dtype=np.float32)

            # Step 1: Find optimal scale for each 16-element block
            # Use optimized path if scalar_optimization is enabled
            rmse_type = 1 if scalar_optimization else 0
            max_scale = 0
            max_abs_scale = 0

            for ib in range(QK_K // 16):
                block = x[16 * ib:16 * (ib + 1)]
                scale, L_block = make_qx_quants(16, 32, block, rmse_type=rmse_type)
                scales[ib] = scale
                L[16 * ib:16 * (ib + 1)] = L_block

                abs_scale = abs(scale)
                if abs_scale > max_abs_scale:
                    max_abs_scale = abs_scale
                    max_scale = scale

            # Handle all-zero case
            if max_abs_scale < GROUP_MAX_EPS:
                output.extend(bytes(128))  # ql
                output.extend(bytes(64))   # qh
                output.extend(bytes(16))   # scales
                output.extend(struct.pack('<e', np.float16(0.0)))  # d
                continue

            # Step 2: Quantize scales
            iscale = -128.0 / max_scale
            d = np.float16(1.0 / iscale)

            scales_int8 = np.zeros(QK_K // 16, dtype=np.int8)
            for ib in range(QK_K // 16):
                scales_int8[ib] = min(127, nearest_int(iscale * scales[ib]))

            # Step 3: Requantize with final scales
            for j in range(QK_K // 16):
                d_block = float(d) * scales_int8[j]
                if d_block == 0:
                    continue
                # Vectorized quantization for the entire block
                block = x[16 * j:16 * (j + 1)]
                vals = block / d_block
                l = np.round(vals).astype(np.int32)
                l = np.clip(l, -32, 31)
                L[16 * j:16 * (j + 1)] = l + 32

            # Step 4: Pack bits (llama.cpp packing logic)
            ql = np.zeros(QK_K // 2, dtype=np.uint8)
            qh = np.zeros(QK_K // 4, dtype=np.uint8)

            # Process in two 128-element chunks
            ql_idx = 0
            qh_idx = 0
            for j in range(0, QK_K, 128):
                for l in range(32):
                    # Get 4 values with stride 32
                    q1 = L[j + l + 0] & 0xF
                    q2 = L[j + l + 32] & 0xF
                    q3 = L[j + l + 64] & 0xF
                    q4 = L[j + l + 96] & 0xF

                    # Pack lower 4 bits
                    ql[ql_idx + l + 0] = q1 | (q3 << 4)
                    ql[ql_idx + l + 32] = q2 | (q4 << 4)

                    # Pack upper 2 bits
                    qh[qh_idx + l] = (
                        ((L[j + l + 0] >> 4) & 3) |
                        (((L[j + l + 32] >> 4) & 3) << 2) |
                        (((L[j + l + 64] >> 4) & 3) << 4) |
                        (((L[j + l + 96] >> 4) & 3) << 6)
                    )

                ql_idx += 64
                qh_idx += 32

            # Write block
            output.extend(ql.tobytes())
            output.extend(qh.tobytes())
            output.extend(scales_int8.tobytes())
            output.extend(struct.pack('<e', d))

    return bytes(output)


def dequantize_q6_k(data, n_elements):
    """
    Dequantize Q6_K format back to float32 (llama.cpp implementation)

    Args:
        data: bytes in Q6_K format
        n_elements: number of elements in original tensor

    Returns:
        numpy array of float32
    """
    nb = (n_elements + QK_K - 1) // QK_K  # Number of super-blocks
    output = np.zeros(nb * QK_K, dtype=np.float32)

    offset = 0

    for i in range(nb):
        # Read block data
        ql = np.frombuffer(data[offset:offset + 128], dtype=np.uint8)
        offset += 128

        qh = np.frombuffer(data[offset:offset + 64], dtype=np.uint8)
        offset += 64

        sc = np.frombuffer(data[offset:offset + 16], dtype=np.int8)
        offset += 16

        d = struct.unpack('<e', data[offset:offset + 2])[0]
        offset += 2

        # Dequantize (llama.cpp logic)
        y_idx = i * QK_K
        ql_idx = 0
        qh_idx = 0
        sc_idx = 0

        # Process in two 128-element chunks
        for n in range(0, QK_K, 128):
            for l in range(32):
                is_val = l // 16

                # Unpack 6-bit values (cast to int8 before subtracting, like llama.cpp)
                q1 = np.int8((ql[ql_idx + l + 0] & 0xF) | (((qh[qh_idx + l] >> 0) & 3) << 4)) - 32
                q2 = np.int8((ql[ql_idx + l + 32] & 0xF) | (((qh[qh_idx + l] >> 2) & 3) << 4)) - 32
                q3 = np.int8((ql[ql_idx + l + 0] >> 4) | (((qh[qh_idx + l] >> 4) & 3) << 4)) - 32
                q4 = np.int8((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 3) << 4)) - 32

                # Dequantize with scales (note: uses every other scale)
                output[y_idx + l + 0] = d * sc[sc_idx + is_val + 0] * q1
                output[y_idx + l + 32] = d * sc[sc_idx + is_val + 2] * q2
                output[y_idx + l + 64] = d * sc[sc_idx + is_val + 4] * q3
                output[y_idx + l + 96] = d * sc[sc_idx + is_val + 6] * q4

            y_idx += 128
            ql_idx += 64
            qh_idx += 32
            sc_idx += 8

    return output[:n_elements]


def test_q6_k():
    """Test Q6_K quantization/dequantization"""
    print("Testing Q6_K quantization...")

    # Create test data (multiple super-blocks)
    test_data = np.random.randn(512).astype(np.float32)
    print(f"Original shape: {test_data.shape}")
    print(f"Original range: [{test_data.min():.4f}, {test_data.max():.4f}]")

    # Quantize
    quantized = quantize_q6_k(test_data)
    print(f"Quantized size: {len(quantized)} bytes")
    expected_size = (512 // QK_K) * 210
    print(f"Expected size: {expected_size} bytes")
    print(f"Compression ratio: {test_data.nbytes / len(quantized):.2f}x")
    print(f"Bits per weight: {len(quantized) * 8 / len(test_data):.4f}")

    # Dequantize
    dequantized = dequantize_q6_k(quantized, len(test_data))
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
    test_q6_k()
