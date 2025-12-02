"""
Q4_K quantization implementation - simplified version

Q4_K format (from llama.cpp ggml-common.h):
- Super-block size: 256 elements (QK_K = 256)
- Structure: 8 blocks of 32 elements each
- Each block has a 6-bit quantized scale and min
- Affine quantization: x = d*scale*q - dmin*min
- Effective: 4.5 bits per weight

Block structure:
- ggml_half d: super-block scale for quantized scales (2 bytes)
- ggml_half dmin: super-block scale for quantized mins (2 bytes)
- uint8_t scales[12]: scales and mins, quantized with 6 bits (12 bytes)
- uint8_t qs[128]: 4-bit quants (128 bytes)
- Total: 144 bytes per super-block

Dequantization formula: x = d * scale * q - dmin * min
"""

import numpy as np
import struct
from .optimizations import make_qkx3_quants, make_qp_quants


QK_K = 256  # Super-block size
K_SCALE_SIZE = 12
GROUP_MAX_EPS = 1e-8


def nearest_int(x):
    """Round to nearest integer (matches llama.cpp)"""
    return int(np.round(x))


def get_scale_min_k4(j, scales_packed):
    """
    Unpack scale and min from packed 12-byte array (matches llama.cpp)

    Args:
        j: block index (0-7)
        scales_packed: uint8 array of 12 bytes containing packed scales/mins

    Returns:
        tuple: (scale, min) both as uint8
    """
    if j < 4:
        d = scales_packed[j] & 63
        m = scales_packed[j + 4] & 63
    else:
        d = (scales_packed[j + 4] & 0xF) | ((scales_packed[j - 4] >> 6) << 4)
        m = (scales_packed[j + 4] >> 4) | ((scales_packed[j - 0] >> 6) << 4)

    return d, m


def pack_scales_mins_k4(scales_6bit, mins_6bit):
    """
    Pack 8 6-bit scales and 8 6-bit mins into 12 bytes

    Args:
        scales_6bit: uint8 array of 8 scales (values 0-63)
        mins_6bit: uint8 array of 8 mins (values 0-63)

    Returns:
        uint8 array of 12 bytes
    """
    packed = np.zeros(K_SCALE_SIZE, dtype=np.uint8)

    # Pack according to llama.cpp logic
    for j in range(8):
        ls = scales_6bit[j]
        lm = mins_6bit[j]

        if j < 4:
            packed[j] = ls
            packed[j + 4] = lm
        else:
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
            packed[j - 4] |= ((ls >> 4) << 6)
            packed[j - 0] |= ((lm >> 4) << 6)

    return packed


def quantize_q4_k(tensor, scalar_optimization=False, verbose=False):
    """
    Quantize a float tensor to Q4_K format (simplified from llama.cpp)

    Args:
        tensor: numpy array of floats (any shape)
        scalar_optimization: Enable grid search optimization for better quality (~10-15x slower)
        verbose: print progress for debugging

    Returns:
        bytes: quantized data in Q4_K format
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

            # Temporary storage
            L = np.zeros(QK_K, dtype=np.uint8)
            scales = np.zeros(8, dtype=np.float32)  # 8 blocks of 32
            mins = np.zeros(8, dtype=np.float32)

            # Step 1: Find scale and min for each 32-element block
            # Use optimization if enabled
            max_scale = 0.0
            max_min = 0.0

            # Prepare weights for optimization (if enabled)
            sw = np.zeros(8, dtype=np.float32)  # Sum of weights per block
            if scalar_optimization:
                # Compute sigma2 for weighting (matches llama.cpp)
                sum_x2 = np.sum(x * x)
                sigma2 = 2.0 * sum_x2 / QK_K
                av_x = np.sqrt(sigma2)

            Laux = np.zeros(32, dtype=np.uint8)  # Auxiliary array for make_qkx3_quants

            for j in range(8):  # 8 blocks of 32 elements
                block = x[32 * j:32 * (j + 1)]

                if scalar_optimization:
                    # Compute weights for this block
                    weights = av_x + np.abs(block)

                    # Accumulate sum of weights (needed for make_qp_quants)
                    sw[j] = np.sum(weights)

                    # Use grid search optimization (nmax=15 for 4-bit)
                    scale, min_val, L_block = make_qkx3_quants(
                        n=32, nmax=15, x=block, weights=weights,
                        rmin=-0.9, rdelta=0.05, nstep=36, use_mad=False
                    )
                    scales[j] = scale
                    mins[j] = min_val
                    L[32 * j:32 * (j + 1)] = L_block
                else:
                    # Fast path: simple min/max
                    min_val = float(np.min(block))
                    max_val = float(np.max(block))

                    # Calculate scale for affine quantization: q = (x - min) / scale
                    # Range needs to map to 0-15 (4-bit)
                    if max_val > min_val:
                        scale = (max_val - min_val) / 15.0
                    else:
                        scale = 0.0

                    scales[j] = scale
                    mins[j] = -min_val  # Note: mins are stored as negative

                if scales[j] > max_scale:
                    max_scale = scales[j]
                if mins[j] > max_min:
                    max_min = mins[j]

            # Handle all-zero case
            if max_scale < GROUP_MAX_EPS:
                output.extend(struct.pack('<e', np.float16(0.0)))  # d
                output.extend(struct.pack('<e', np.float16(0.0)))  # dmin
                output.extend(bytes(K_SCALE_SIZE))  # scales
                output.extend(bytes(128))  # qs
                continue

            # Step 2: Quantize scales and mins to 6-bit (0-63)
            scales_6bit = np.zeros(8, dtype=np.uint8)
            mins_6bit = np.zeros(8, dtype=np.uint8)

            if scalar_optimization:
                # Use optimized quantization for scales and mins
                d_block, scales_6bit = make_qp_quants(8, 63, scales, sw)
                dmin_block, mins_6bit = make_qp_quants(8, 63, mins, sw)
                d = np.float16(d_block)
                dmin = np.float16(dmin_block)
            else:
                # Fast path: simple quantization
                inv_scale = 63.0 / max_scale if max_scale > 0 else 0.0
                inv_min = 63.0 / max_min if max_min > 0 else 0.0

                for j in range(8):
                    ls = min(63, nearest_int(inv_scale * scales[j]))
                    lm = min(63, nearest_int(inv_min * mins[j]))
                    scales_6bit[j] = ls
                    mins_6bit[j] = lm

                # Step 3: Create FP16 super-block scales
                d = np.float16(max_scale / 63.0)
                dmin = np.float16(max_min / 63.0)

            # Step 4: Pack scales and mins
            scales_packed = pack_scales_mins_k4(scales_6bit, mins_6bit)

            # Step 5: Requantize with final scales and mins
            for j in range(8):
                sc, m = get_scale_min_k4(j, scales_packed)
                d_final = float(d) * sc
                min_final = float(dmin) * m

                if d_final == 0:
                    # Zero scale, set all to 0
                    L[32 * j:32 * (j + 1)] = 0
                    continue

                # Vectorized quantization for the entire block
                block = x[32 * j:32 * (j + 1)]
                vals = (block + min_final) / d_final
                l = np.round(vals).astype(np.int32)
                l = np.clip(l, 0, 15)  # Clip to 4-bit range
                L[32 * j:32 * (j + 1)] = l

            # Step 6: Pack 4-bit values (128 bytes from 256 4-bit values)
            # Process in 64-element chunks with stride packing
            qs = np.zeros(128, dtype=np.uint8)
            q_idx = 0

            for j in range(0, QK_K, 64):
                # Pack 64 elements into 32 bytes
                for l in range(32):
                    # Lower nibble from first 32, upper nibble from second 32
                    qs[q_idx + l] = L[j + l] | (L[j + l + 32] << 4)
                q_idx += 32

            # Write block to output
            output.extend(struct.pack('<e', d))  # 2 bytes
            output.extend(struct.pack('<e', dmin))  # 2 bytes
            output.extend(scales_packed.tobytes())  # 12 bytes
            output.extend(qs.tobytes())  # 128 bytes

    return bytes(output)


def dequantize_q4_k(data, n_elements):
    """
    Dequantize Q4_K format back to float32

    Args:
        data: bytes in Q4_K format
        n_elements: number of elements in original tensor

    Returns:
        numpy array of float32
    """
    nb = (n_elements + QK_K - 1) // QK_K
    output = np.zeros(nb * QK_K, dtype=np.float32)

    offset = 0
    y_idx = 0

    for i in range(nb):
        # Read block structure
        d = struct.unpack('<e', data[offset:offset + 2])[0]
        dmin = struct.unpack('<e', data[offset + 2:offset + 4])[0]
        scales_packed = np.frombuffer(data[offset + 4:offset + 16], dtype=np.uint8)
        qs = np.frombuffer(data[offset + 16:offset + 144], dtype=np.uint8)

        offset += 144

        # Dequantize super-block
        q_idx = 0
        is_idx = 0

        # Process in 64-element chunks
        for j in range(0, QK_K, 64):
            # Get scales for two 32-element blocks
            sc1, m1 = get_scale_min_k4(is_idx + 0, scales_packed)
            d1 = d * sc1
            min1 = dmin * m1

            sc2, m2 = get_scale_min_k4(is_idx + 1, scales_packed)
            d2 = d * sc2
            min2 = dmin * m2

            # Unpack and dequantize first 32 elements (lower nibbles)
            for l in range(32):
                q = qs[q_idx + l] & 0xF
                output[y_idx + l] = d1 * q - min1

            # Unpack and dequantize second 32 elements (upper nibbles)
            for l in range(32):
                q = qs[q_idx + l] >> 4
                output[y_idx + 32 + l] = d2 * q - min2

            q_idx += 32
            y_idx += 64
            is_idx += 2

    return output[:n_elements]


def test_q4_k():
    """Test Q4_K quantization with small tensor"""
    print("Testing Q4_K quantization...")

    # Test 1: Small tensor
    np.random.seed(42)
    test_data = np.random.randn(512).astype(np.float32) * 10.0

    print(f"Input: {len(test_data)} elements, range [{test_data.min():.3f}, {test_data.max():.3f}]")

    # Quantize
    quantized = quantize_q4_k(test_data)
    expected_size = (512 // QK_K) * 144
    print(f"Quantized size: {len(quantized)} bytes (expected {expected_size})")
    print(f"Bits per weight: {len(quantized) * 8 / len(test_data):.4f}")

    # Dequantize
    dequantized = dequantize_q4_k(quantized, len(test_data))

    print(f"Dequantized: {len(dequantized)} elements, range [{dequantized.min():.3f}, {dequantized.max():.3f}]")

    # Check quality
    mse = np.mean((test_data - dequantized) ** 2)
    correlation = np.corrcoef(test_data, dequantized)[0, 1]

    print(f"MSE: {mse:.6f}")
    print(f"Correlation: {correlation:.6f}")

    if correlation > 0.98:
        print("[PASS] Test passed! Correlation > 0.98")
    else:
        print("[FAIL] Test FAILED! Correlation too low")

    # Test 2: Larger tensor for performance
    print("\nPerformance test...")
    import time
    large_data = np.random.randn(2048, 2048).astype(np.float32)
    start = time.time()
    quantized_large = quantize_q4_k(large_data)
    elapsed = time.time() - start

    speed = large_data.size / elapsed / 1000
    print(f"Speed: {speed:.0f}K elements/second")
    print(f"Time: {elapsed:.2f} seconds for {large_data.size / 1e6:.1f}M elements")

    if speed > 100:
        print("[PASS] Performance acceptable (>100K elem/sec)")
    else:
        print("[WARN] Performance slow (<100K elem/sec)")

    # Test 3: Scale/min packing roundtrip
    print("\nScale/min packing test...")
    scales_test = np.array([0, 15, 31, 47, 63, 12, 24, 36], dtype=np.uint8)
    mins_test = np.array([1, 16, 32, 48, 62, 13, 25, 37], dtype=np.uint8)

    packed = pack_scales_mins_k4(scales_test, mins_test)
    print(f"Packed {len(packed)} bytes: {packed}")

    # Unpack and verify
    all_match = True
    for j in range(8):
        sc, m = get_scale_min_k4(j, packed)
        if sc != scales_test[j] or m != mins_test[j]:
            print(f"[FAIL] Mismatch at {j}: expected ({scales_test[j]}, {mins_test[j]}), got ({sc}, {m})")
            all_match = False

    if all_match:
        print("[PASS] Scale/min packing correct")
    else:
        print("[FAIL] Scale/min packing FAILED")


if __name__ == "__main__":
    test_q4_k()
