"""
Q3_K quantization implementation - simplified version

Q3_K format (from llama.cpp ggml-common.h):
- Super-block size: 256 elements (QK_K = 256)
- Structure: 16 blocks of 16 elements each
- Each block has a 6-bit quantized scale (no min)
- Symmetric quantization: x = d * scale * q
- Effective: 3.4375 bits per weight

Block structure:
- uint8_t hmask[32]: quants - high bit (3rd bit) (32 bytes)
- uint8_t qs[64]: quants - low 2 bits (64 bytes)
- uint8_t scales[12]: scales, quantized with 6 bits (12 bytes)
- ggml_half d: super-block scale (2 bytes)
- Total: 110 bytes per super-block

Dequantization formula: x = d * scale * q
where q is 3-bit (-4 to 3)
"""

import numpy as np
import struct
from .optimizations import make_q3_quants


QK_K = 256  # Super-block size
GROUP_MAX_EPS = 1e-8


def nearest_int(x):
    """Round to nearest integer (matches llama.cpp)"""
    return int(np.round(x))


def pack_scales_q3_k(scales_6bit):
    """
    Pack 16 6-bit scales into 12 bytes (matches llama.cpp complex packing)

    Each scale is 6-bit signed (-32 to 31), stored as unsigned (0-63) by adding 32

    Args:
        scales_6bit: int8 array of 16 scales (values -32 to 31)

    Returns:
        uint8 array of 12 bytes
    """
    packed = np.zeros(12, dtype=np.uint8)

    # Convert to unsigned 0-63 range
    scales_unsigned = (scales_6bit + 32).astype(np.uint8)

    # Pack according to llama.cpp logic
    # First 8 scales: lower 4 bits go to bytes 0-7
    for j in range(8):
        packed[j] = scales_unsigned[j] & 0xF

    # Second 8 scales: lower 4 bits go to upper nibbles of bytes 0-7
    for j in range(8, 16):
        packed[j - 8] |= (scales_unsigned[j] & 0xF) << 4

    # Upper 2 bits of all scales go to bytes 8-11
    for j in range(16):
        upper_bits = (scales_unsigned[j] >> 4) & 0x3
        packed[8 + (j % 4)] |= upper_bits << (2 * (j // 4))

    return packed


def unpack_scales_q3_k(scales_packed):
    """
    Unpack 12 bytes into 16 6-bit scales

    Args:
        scales_packed: uint8 array of 12 bytes

    Returns:
        int8 array of 16 scales (values -32 to 31)
    """
    scales = np.zeros(16, dtype=np.int8)

    # Unpack according to llama.cpp logic
    for j in range(16):
        if j < 8:
            lower_4 = int(scales_packed[j] & 0xF)
        else:
            lower_4 = int(scales_packed[j - 8] >> 4)

        upper_2 = int((scales_packed[8 + (j % 4)] >> (2 * (j // 4))) & 0x3)

        scales_unsigned = lower_4 | (upper_2 << 4)
        # Convert unsigned (0-63) to signed (-32 to 31)
        scales[j] = np.int8(scales_unsigned - 32)

    return scales


def quantize_q3_k(tensor, scalar_optimization=False, verbose=False):
    """
    Quantize a float tensor to Q3_K format (simplified from llama.cpp)

    Args:
        tensor: numpy array of floats (any shape)
        scalar_optimization: Enable iterative refinement for better quality (~5x slower)
        verbose: print progress for debugging

    Returns:
        bytes: quantized data in Q3_K format
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
            L = np.zeros(QK_K, dtype=np.int8)
            scales = np.zeros(16, dtype=np.float32)

            # Step 1: Find optimal scale for each 16-element block
            # Use iterative refinement if scalar_optimization is enabled
            max_scale = 0.0
            max_abs_scale = 0.0

            for j in range(16):  # 16 blocks of 16 elements
                block = x[16 * j:16 * (j + 1)]

                # Use optimization function (nmax=4 for Q3_K: range -4 to 3)
                scale, L_block = make_q3_quants(16, 4, block, do_rmse=scalar_optimization)
                scales[j] = scale
                L[16 * j:16 * (j + 1)] = L_block

                abs_scale = abs(scale)
                if abs_scale > max_abs_scale:
                    max_abs_scale = abs_scale
                    max_scale = scale

            # Handle all-zero case
            if max_abs_scale < GROUP_MAX_EPS:
                output.extend(bytes(32))  # hmask
                output.extend(bytes(64))  # qs
                output.extend(bytes(12))  # scales
                output.extend(struct.pack('<e', np.float16(0.0)))  # d
                continue

            # Step 2: Quantize scales to 6-bit (-32 to 31)
            iscale = -32.0 / max_scale
            d = np.float16(1.0 / iscale)

            scales_6bit = np.zeros(16, dtype=np.int8)
            for j in range(16):
                l = nearest_int(iscale * scales[j])
                l = max(-32, min(31, l))
                scales_6bit[j] = l

            # Step 3: Pack scales
            scales_packed = pack_scales_q3_k(scales_6bit)

            # Step 4: Unpack scales for requantization
            scales_unpacked = unpack_scales_q3_k(scales_packed)

            # Step 5: Requantize with final scales
            # Note: L was already filled in Step 1 by make_q3_quants, but we need to
            # requantize with the final packed/unpacked scales for accuracy
            for j in range(16):
                sc = scales_unpacked[j]
                d_final = float(d) * sc

                if abs(d_final) < GROUP_MAX_EPS:
                    L[16 * j:16 * (j + 1)] = 0
                    continue

                # Requantize with final scale (this may differ slightly from Step 1 due to packing)
                block = x[16 * j:16 * (j + 1)]
                vals = block / d_final
                l = np.round(vals).astype(np.int32)
                l = np.clip(l, -4, 3)
                L[16 * j:16 * (j + 1)] = l + 4  # Convert to 0-8 range (0-8, not 0-7)

            # Step 6: Pack 3-bit values (split into low 2 bits and high 1 bit)
            hmask = np.zeros(32, dtype=np.uint8)
            qs = np.zeros(64, dtype=np.uint8)

            # Extract high bit into hmask
            m = 0
            hm = 1
            for j in range(QK_K):
                if L[j] > 3:
                    hmask[m] |= hm
                    L[j] -= 4

                m += 1
                if m == 32:  # QK_K / 8
                    m = 0
                    hm <<= 1

            # Pack low 2 bits into qs (same as Q2_K)
            for j in range(0, QK_K, 128):
                for l in range(32):
                    qs[j // 4 + l] = (
                        L[j + l] |
                        (L[j + l + 32] << 2) |
                        (L[j + l + 64] << 4) |
                        (L[j + l + 96] << 6)
                    )

            # Write block to output
            output.extend(hmask.tobytes())  # 32 bytes
            output.extend(qs.tobytes())  # 64 bytes
            output.extend(scales_packed.tobytes())  # 12 bytes
            output.extend(struct.pack('<e', d))  # 2 bytes

    return bytes(output)


def dequantize_q3_k(data, n_elements):
    """
    Dequantize Q3_K format back to float32

    Args:
        data: bytes in Q3_K format
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
        hmask = np.frombuffer(data[offset:offset + 32], dtype=np.uint8)
        qs = np.frombuffer(data[offset + 32:offset + 96], dtype=np.uint8)
        scales_packed = np.frombuffer(data[offset + 96:offset + 108], dtype=np.uint8)
        d = struct.unpack('<e', data[offset + 108:offset + 110])[0]

        offset += 110

        # Unpack scales
        scales = unpack_scales_q3_k(scales_packed)

        # Dequantize super-block
        is_idx = 0
        m = 1
        q_idx = 0

        # Process in 128-element chunks
        for n in range(0, QK_K, 128):
            shift = 0
            for j in range(4):
                # First block of 16
                dl = d * scales[is_idx]
                is_idx += 1

                for l in range(16):
                    q_low = int((qs[q_idx + l] >> shift) & 3)
                    q_high = 0 if (hmask[l] & m) else -4
                    q = q_low + q_high
                    output[y_idx] = dl * q
                    y_idx += 1

                # Second block of 16
                dl = d * scales[is_idx]
                is_idx += 1

                for l in range(16):
                    q_low = int((qs[q_idx + l + 16] >> shift) & 3)
                    q_high = 0 if (hmask[l + 16] & m) else -4
                    q = q_low + q_high
                    output[y_idx] = dl * q
                    y_idx += 1

                shift += 2
                m <<= 1

            q_idx += 32  # Move to next 32 bytes

    return output[:n_elements]


def test_q3_k():
    """Test Q3_K quantization with small tensor"""
    print("Testing Q3_K quantization...")

    # Test 1: Small tensor
    np.random.seed(42)
    test_data = np.random.randn(512).astype(np.float32) * 10.0

    print(f"Input: {len(test_data)} elements, range [{test_data.min():.3f}, {test_data.max():.3f}]")

    # Quantize
    quantized = quantize_q3_k(test_data)
    expected_size = (512 // QK_K) * 110
    print(f"Quantized size: {len(quantized)} bytes (expected {expected_size})")
    print(f"Bits per weight: {len(quantized) * 8 / len(test_data):.4f}")

    # Dequantize
    dequantized = dequantize_q3_k(quantized, len(test_data))

    print(f"Dequantized: {len(dequantized)} elements, range [{dequantized.min():.3f}, {dequantized.max():.3f}]")

    # Check quality
    mse = np.mean((test_data - dequantized) ** 2)
    correlation = np.corrcoef(test_data, dequantized)[0, 1]

    print(f"MSE: {mse:.6f}")
    print(f"Correlation: {correlation:.6f}")

    if correlation > 0.97:  # Threshold for 3-bit
        print("[PASS] Test passed! Correlation > 0.97")
    else:
        print("[FAIL] Test FAILED! Correlation too low")

    # Test 2: Larger tensor for performance
    print("\nPerformance test...")
    import time
    large_data = np.random.randn(2048, 2048).astype(np.float32)
    start = time.time()
    quantized_large = quantize_q3_k(large_data)
    elapsed = time.time() - start

    speed = large_data.size / elapsed / 1000
    print(f"Speed: {speed:.0f}K elements/second")
    print(f"Time: {elapsed:.2f} seconds for {large_data.size / 1e6:.1f}M elements")

    if speed > 100:
        print("[PASS] Performance acceptable (>100K elem/sec)")
    else:
        print("[WARN] Performance slow (<100K elem/sec)")

    # Test 3: Scale packing roundtrip
    print("\nScale packing test...")
    scales_test = np.array([-32, -20, -10, 0, 5, 10, 15, 20, 25, 28, 30, 31, -15, -5, 10, 20], dtype=np.int8)

    packed = pack_scales_q3_k(scales_test)
    print(f"Packed {len(packed)} bytes")

    unpacked = unpack_scales_q3_k(packed)

    if np.array_equal(scales_test, unpacked):
        print("[PASS] Scale packing correct")
    else:
        print("[FAIL] Scale packing FAILED")
        print(f"Expected: {scales_test}")
        print(f"Got:      {unpacked}")


if __name__ == "__main__":
    test_q3_k()
