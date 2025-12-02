"""
Q2_K quantization implementation - simplified version
"""

import numpy as np
import struct
from .optimizations import make_qkx2_quants, nearest_int

QK_K = 256
Q4_SCALE = 15.0

def quantize_q2_k(tensor: np.ndarray, scalar_optimization=False) -> bytes:
    """
    Quantize a float tensor to Q2_K format (simplified from llama.cpp)
    Args:
        tensor: numpy array of floats (any shape)
    Returns:
        bytes: quantized data in Q2_K format
    """
    original_shape = tensor.shape
    if tensor.ndim == 1:
        data = tensor.reshape(1, -1)
    else:
        data = tensor.reshape(-1, original_shape[-1])
    data = data.astype(np.float32)
    n_rows, n_cols = data.shape

    output = bytearray()

    for row_idx in range(n_rows):
        row = data[row_idx]
        n = len(row)
        remainder = n % QK_K
        if remainder != 0:
            padding = QK_K - remainder
            row = np.pad(row, (0, padding), mode='constant', constant_values=0)
            n = len(row)

        nb = n // QK_K

        for i in range(nb):
            x = row[i * QK_K:(i + 1) * QK_K]
            
            L = np.zeros(QK_K, dtype=np.uint8)
            mins = np.zeros(QK_K // 16, dtype=np.float32)
            scales = np.zeros(QK_K // 16, dtype=np.float32)

            max_scale = 0.0
            max_min = 0.0

            for j in range(QK_K // 16):
                weights = np.abs(x[16 * j:16 * (j + 1)])
                scales[j], mins[j], L[16*j:16*(j+1)] = make_qkx2_quants(16, 3, x[16*j:16*(j+1)], weights, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True)

                if scales[j] > max_scale:
                    max_scale = scales[j]
                if mins[j] > max_min:
                    max_min = mins[j]

            y_scales = np.zeros(QK_K // 16, dtype=np.uint8)
            
            d = np.float16(0.0)
            if max_scale > 0:
                iscale = Q4_SCALE / max_scale
                for j in range(QK_K // 16):
                    l = nearest_int(iscale * scales[j])
                    y_scales[j] = l
                d = np.float16(max_scale / Q4_SCALE)
            
            dmin = np.float16(0.0)
            if max_min > 0:
                iscale = Q4_SCALE / max_min
                for j in range(QK_K // 16):
                    l = nearest_int(iscale * mins[j])
                    y_scales[j] |= l << 4
                dmin = np.float16(max_min / Q4_SCALE)

            for j in range(QK_K // 16):
                d_s = float(d) * (y_scales[j] & 0xF)
                if d_s == 0:
                    L[16*j:16*(j+1)] = 0
                    continue
                dm_s = float(dmin) * (y_scales[j] >> 4)
                for ii in range(16):
                    l = nearest_int((x[16*j + ii] + dm_s) / d_s)
                    L[16*j + ii] = max(0, min(3, l))

            qs = np.zeros(QK_K // 4, dtype=np.uint8)
            for j in range(0, QK_K, 128):
                for l in range(32):
                    qs[j // 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6)

            output.extend(y_scales.tobytes())
            output.extend(qs.tobytes())
            output.extend(struct.pack('<e', d))
            output.extend(struct.pack('<e', dmin))

    return bytes(output)


def dequantize_q2_k(data: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantize Q2_K format back to float32
    """
    output = np.zeros(n_elements, dtype=np.float32)
    
    nb = (n_elements + QK_K - 1) // QK_K
    
    offset = 0
    y_idx = 0

    for i in range(nb):
        scales = np.frombuffer(data[offset:offset+16], dtype=np.uint8)
        qs = np.frombuffer(data[offset+16:offset+16+64], dtype=np.uint8)
        d = struct.unpack('<e', data[offset+80:offset+82])[0]
        dmin = struct.unpack('<e', data[offset+82:offset+84])[0]
        offset += 84

        is_ = 0
        q_offset = 0
        for n in range(0, QK_K, 128):
            shift = 0
            for j in range(4):
                sc = scales[is_]
                is_ += 1
                dl = d * (sc & 0xF)
                ml = dmin * (sc >> 4)
                for l in range(16):
                    if y_idx < n_elements:
                        output[y_idx] = dl * ((qs[q_offset + l] >> shift) & 3) - ml
                        y_idx += 1

                sc = scales[is_]
                is_ += 1
                dl = d * (sc & 0xF)
                ml = dmin * (sc >> 4)
                for l in range(16):
                    if y_idx < n_elements:
                        output[y_idx] = dl * ((qs[q_offset + l + 16] >> shift) & 3) - ml
                        y_idx += 1
                
                shift += 2
            q_offset += 32
    
    return output[:n_elements]

def test_q2_k():
    """Test Q2_K quantization with small tensor"""
    print("Testing Q2_K quantization...")

    # Test 1: Small tensor
    np.random.seed(42)
    test_data = np.random.randn(512).astype(np.float32) * 10.0

    print(f"Input: {len(test_data)} elements, range [{test_data.min():.3f}, {test_data.max():.3f}]")

    # Quantize
    quantized = quantize_q2_k(test_data)
    expected_size = (512 // QK_K) * 84
    print(f"Quantized size: {len(quantized)} bytes (expected {expected_size})")
    print(f"Bits per weight: {len(quantized) * 8 / len(test_data):.4f}")

    # Dequantize
    dequantized = dequantize_q2_k(quantized, len(test_data))

    print(f"Dequantized: {len(dequantized)} elements, range [{dequantized.min():.3f}, {dequantized.max():.3f}]")

    # Check quality
    mse = np.mean((test_data - dequantized) ** 2)
    correlation = np.corrcoef(test_data, dequantized)[0, 1]

    print(f"MSE: {mse:.6f}")
    print(f"Correlation: {correlation:.6f}")

    if correlation > 0.9:
        print("[PASS] Test passed! Correlation > 0.9")
    else:
        print("[FAIL] Test FAILED! Correlation too low")

if __name__ == "__main__":
    test_q2_k()
