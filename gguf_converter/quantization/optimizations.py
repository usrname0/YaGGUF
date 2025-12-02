"""
Quantization optimization functions - matches llama.cpp implementation paths

These functions provide the iterative optimization algorithms used in llama.cpp's
`_impl` functions. Our base quantization uses the fast `_ref` implementations.
When scalar_optimization is enabled, we switch to these slower but higher-quality
algorithms.

Reference: llama.cpp/ggml/src/ggml-quants.c
"""

import numpy as np


GROUP_MAX_EPS = 1e-8


def nearest_int(x):
    """Round to nearest integer (matches llama.cpp)"""
    return int(np.round(x))


def make_qx_quants(n, nmax, x, rmse_type=0, qw=None):
    """
    Symmetric quantization with optional 19-iteration optimization.

    This matches llama.cpp's make_qx_quants() function exactly.
    Used by Q6_K, Q2_K, and potentially other symmetric quantizations.

    Reference: llama.cpp/ggml/src/ggml-quants.c:451-518

    Args:
        n: Number of elements to quantize
        nmax: Maximum quantized value (32 for Q6_K, etc.)
        x: Input float array (numpy array)
        rmse_type: Optimization mode:
            0 = Fast path (no optimization)
            1 = x² weighted least squares + 19-iteration search
            2 = Uniform weighting
            3 = |x| weighting
            (negative values enable early return)
        qw: Optional custom weight array (overrides rmse_type weighting)

    Returns:
        tuple: (scale, L) where:
            scale: Optimal scale factor
            L: Quantized values as uint8 array (range: 0 to 2*nmax-1)
    """
    x = np.asarray(x, dtype=np.float32)
    L = np.zeros(n, dtype=np.int32)

    # Find max absolute value
    max_val = 0.0
    amax = 0.0
    for i in range(n):
        ax = abs(x[i])
        if ax > amax:
            amax = ax
            max_val = x[i]

    # All-zero case
    if amax < GROUP_MAX_EPS:
        return 0.0, np.zeros(n, dtype=np.uint8)

    # Initial scale
    iscale = -nmax / max_val

    # Fast path - no optimization
    if rmse_type == 0:
        for i in range(n):
            l = nearest_int(iscale * x[i])
            l = max(-nmax, min(nmax - 1, l))
            L[i] = l + nmax
        return 1.0 / iscale, L.astype(np.uint8)

    # Optimized path - weighted least squares + 19-iteration search
    return_early = False
    if rmse_type < 0:
        rmse_type = -rmse_type
        return_early = True

    # Initial quantization and compute weighted sums
    sumlx = 0.0
    suml2 = 0.0

    for i in range(n):
        l = nearest_int(iscale * x[i])
        l = max(-nmax, min(nmax - 1, l))
        L[i] = l + nmax

        # Compute weight based on rmse_type
        if qw is not None:
            w = qw[i]
        elif rmse_type == 1:
            w = x[i] * x[i]  # x² weighting (most common)
        elif rmse_type == 2:
            w = 1.0  # Uniform weighting
        elif rmse_type == 3:
            w = abs(x[i])  # |x| weighting
        else:
            w = np.sqrt(abs(x[i]))  # sqrt(|x|) weighting

        sumlx += w * x[i] * (l)
        suml2 += w * (l) * (l)

    # Compute initial scale from weighted least squares
    scale = sumlx / suml2 if suml2 > 0 else 0.0

    # Early return option (for some use cases)
    if return_early:
        if suml2 > 0:
            return 0.5 * (scale + 1.0 / iscale), L.astype(np.uint8)
        else:
            return 1.0 / iscale, L.astype(np.uint8)

    # 19-iteration search: test scale offsets from -9 to +9
    best = scale * sumlx

    for is_offset in range(-9, 10):
        if is_offset == 0:
            continue

        # Perturb the scale
        iscale_test = -(nmax + 0.1 * is_offset) / max_val

        # Re-quantize with this scale
        sumlx_test = 0.0
        suml2_test = 0.0

        for i in range(n):
            l = nearest_int(iscale_test * x[i])
            l = max(-nmax, min(nmax - 1, l))

            # Compute weight (same as before)
            if qw is not None:
                w = qw[i]
            elif rmse_type == 1:
                w = x[i] * x[i]
            elif rmse_type == 2:
                w = 1.0
            elif rmse_type == 3:
                w = abs(x[i])
            else:
                w = np.sqrt(abs(x[i]))

            sumlx_test += w * x[i] * (l)
            suml2_test += w * (l) * (l)

        # Check if this scale is better (using weighted least squares criterion)
        # Matches llama.cpp line 509: if (suml2 > 0 && sumlx*sumlx > best*suml2)
        if suml2_test > 0 and sumlx_test * sumlx_test > best * suml2_test:
            # This scale is better - update L with new quantization
            # Matches llama.cpp lines 510-514
            for i in range(n):
                l = nearest_int(iscale_test * x[i])
                l = max(-nmax, min(nmax - 1, l))
                L[i] = nmax + l  # Note: llama.cpp writes nmax + MAX(...) which is same as nmax + l after the max/min

            scale = sumlx_test / suml2_test
            best = scale * sumlx_test

    return scale, L.astype(np.uint8)


def make_q3_quants(n, nmax, x, do_rmse=False):
    """
    Q3_K-specific quantization with optional iterative refinement.

    This matches llama.cpp's make_q3_quants() function exactly.
    Uses coordinate descent to iteratively improve each quantized value.

    Reference: llama.cpp/ggml/src/ggml-quants.c:520-577

    Args:
        n: Number of elements to quantize
        nmax: Maximum quantized value (4 for Q3_K, giving range -4 to 3)
        x: Input float array (numpy array)
        do_rmse: Enable iterative refinement (up to 5 passes)

    Returns:
        tuple: (scale, L) where:
            scale: Optimal scale factor
            L: Quantized values as int8 array (range: 0 to 2*nmax for storage)
    """
    x = np.asarray(x, dtype=np.float32)
    L = np.zeros(n, dtype=np.int32)

    # Find max absolute value
    max_val = 0.0
    amax = 0.0
    for i in range(n):
        ax = abs(x[i])
        if ax > amax:
            amax = ax
            max_val = x[i]

    # All-zero case
    if amax < GROUP_MAX_EPS:
        return 0.0, np.zeros(n, dtype=np.int8)

    # Initial scale
    iscale = -nmax / max_val

    # Fast path - no optimization
    if not do_rmse:
        for i in range(n):
            l = nearest_int(iscale * x[i])
            l = max(-nmax, min(nmax - 1, l))
            L[i] = l + nmax
        return 1.0 / iscale, L.astype(np.int8)

    # Optimized path - iterative refinement
    # Step 1: Initial quantization (store as -nmax to nmax-1, NOT offset yet)
    sumlx = 0.0
    suml2 = 0.0

    for i in range(n):
        l = nearest_int(iscale * x[i])
        l = max(-nmax, min(nmax - 1, l))
        L[i] = l

        # Compute weighted sums (x² weighting)
        w = x[i] * x[i]
        sumlx += w * x[i] * l
        suml2 += w * l * l

    # Step 2: Iterative refinement (coordinate descent, up to 5 passes)
    for itry in range(5):
        n_changed = 0

        for i in range(n):
            w = x[i] * x[i]

            # Remove current element's contribution
            slx = sumlx - w * x[i] * L[i]

            if slx > 0:
                # Calculate what the element should be without its current contribution
                sl2 = suml2 - w * L[i] * L[i]
                new_l = nearest_int(x[i] * sl2 / slx)
                new_l = max(-nmax, min(nmax - 1, new_l))

                if new_l != L[i]:
                    # Add new element's contribution
                    slx += w * x[i] * new_l
                    sl2 += w * new_l * new_l

                    # Check if this improves the error (weighted least squares criterion)
                    # if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2)
                    if sl2 > 0 and slx * slx * suml2 > sumlx * sumlx * sl2:
                        # Accept the change
                        L[i] = new_l
                        sumlx = slx
                        suml2 = sl2
                        n_changed += 1

        # If no changes in this pass, we've converged
        if n_changed == 0:
            break

    # Step 3: Offset L values for storage (0 to 2*nmax)
    for i in range(n):
        L[i] += nmax

    # Return scale
    scale = sumlx / suml2 if suml2 > 0 else 0.0
    return scale, L.astype(np.int8)


def test_make_qx_quants():
    """Test make_qx_quants with both fast and optimized paths"""
    print("Testing make_qx_quants...")
    print("=" * 70)

    # Test with small block (16 elements, typical for Q6_K)
    print("\nTest 1: Small block (n=16, nmax=32)")
    print("-" * 70)
    np.random.seed(42)
    test_data_small = np.random.randn(16).astype(np.float32)

    print(f"Input: {test_data_small[:8]}...")
    print(f"Range: [{test_data_small.min():.4f}, {test_data_small.max():.4f}]")

    scale_fast, L_fast = make_qx_quants(16, 32, test_data_small, rmse_type=0)
    deq_fast = (L_fast.astype(np.int32) - 32) * scale_fast
    error_fast = np.abs(test_data_small - deq_fast).mean()
    corr_fast = np.corrcoef(test_data_small, deq_fast)[0, 1]

    scale_opt, L_opt = make_qx_quants(16, 32, test_data_small, rmse_type=1)
    deq_opt = (L_opt.astype(np.int32) - 32) * scale_opt
    error_opt = np.abs(test_data_small - deq_opt).mean()
    corr_opt = np.corrcoef(test_data_small, deq_opt)[0, 1]

    print(f"  Fast:      error={error_fast:.6f}, corr={corr_fast:.8f}")
    print(f"  Optimized: error={error_opt:.6f}, corr={corr_opt:.8f}")
    print(f"  Improvement: {(corr_opt - corr_fast) * 100:.6f}%")

    # Test with larger block
    print("\nTest 2: Large block (n=256, nmax=32)")
    print("-" * 70)
    np.random.seed(123)
    test_data_large = np.random.randn(256).astype(np.float32) * 2.0

    print(f"Range: [{test_data_large.min():.4f}, {test_data_large.max():.4f}]")

    scale_fast2, L_fast2 = make_qx_quants(256, 32, test_data_large, rmse_type=0)
    deq_fast2 = (L_fast2.astype(np.int32) - 32) * scale_fast2
    error_fast2 = np.abs(test_data_large - deq_fast2).mean()
    corr_fast2 = np.corrcoef(test_data_large, deq_fast2)[0, 1]

    scale_opt2, L_opt2 = make_qx_quants(256, 32, test_data_large, rmse_type=1)
    deq_opt2 = (L_opt2.astype(np.int32) - 32) * scale_opt2
    error_opt2 = np.abs(test_data_large - deq_opt2).mean()
    corr_opt2 = np.corrcoef(test_data_large, deq_opt2)[0, 1]

    print(f"  Fast:      error={error_fast2:.6f}, corr={corr_fast2:.8f}")
    print(f"  Optimized: error={error_opt2:.6f}, corr={corr_opt2:.8f}")
    print(f"  Improvement: {(corr_opt2 - corr_fast2) * 100:.6f}%")

    # Test with varied distribution
    print("\nTest 3: Varied distribution (n=64, nmax=32)")
    print("-" * 70)
    np.random.seed(456)
    # Create data with outliers (more realistic for weights)
    test_data_varied = np.concatenate([
        np.random.randn(50).astype(np.float32) * 0.5,
        np.random.randn(14).astype(np.float32) * 2.0
    ])
    np.random.shuffle(test_data_varied)

    print(f"Range: [{test_data_varied.min():.4f}, {test_data_varied.max():.4f}]")

    scale_fast3, L_fast3 = make_qx_quants(64, 32, test_data_varied, rmse_type=0)
    deq_fast3 = (L_fast3.astype(np.int32) - 32) * scale_fast3
    error_fast3 = np.abs(test_data_varied - deq_fast3).mean()
    corr_fast3 = np.corrcoef(test_data_varied, deq_fast3)[0, 1]

    scale_opt3, L_opt3 = make_qx_quants(64, 32, test_data_varied, rmse_type=1)
    deq_opt3 = (L_opt3.astype(np.int32) - 32) * scale_opt3
    error_opt3 = np.abs(test_data_varied - deq_opt3).mean()
    corr_opt3 = np.corrcoef(test_data_varied, deq_opt3)[0, 1]

    print(f"  Fast:      error={error_fast3:.6f}, corr={corr_fast3:.8f}")
    print(f"  Optimized: error={error_opt3:.6f}, corr={corr_opt3:.8f}")
    print(f"  Improvement: {(corr_opt3 - corr_fast3) * 100:.6f}%")

    print("\n" + "=" * 70)
    # Pass if optimized is never worse by more than 0.01% or is better in at least one case
    improvements = [corr_opt - corr_fast, corr_opt2 - corr_fast2, corr_opt3 - corr_fast3]
    if any(imp > 0.0001 for imp in improvements):
        print("Test PASSED! Optimization shows improvement in at least one case.")
    elif all(imp > -0.0001 for imp in improvements):
        print("Test PASSED! Optimization is not worse than fast path.")
    else:
        print("Test FAILED! Optimization is significantly worse than fast path.")


def make_qp_quants(n, nmax, x, quant_weights):
    """
    Quantize scales/mins with optimization (used by Q4_K/Q5_K).

    Uses 9-iteration search + iterative refinement to find optimal quantization.

    Reference: llama.cpp/ggml/src/ggml-quants.c:899-968

    Args:
        n: Number of elements to quantize
        nmax: Maximum quantized value (63 for 6-bit scales)
        x: Input float array (scales or mins)
        quant_weights: Weight array for weighted MSE

    Returns:
        tuple: (scale, L) where:
            scale: Optimal scale factor
            L: Quantized values as uint8 array
    """
    x = np.asarray(x, dtype=np.float32)
    quant_weights = np.asarray(quant_weights, dtype=np.float32)
    L = np.zeros(n, dtype=np.uint8)

    # Find max value
    max_val = np.max(x)

    # All-zero case
    if max_val < GROUP_MAX_EPS:
        return 0.0, np.zeros(n, dtype=np.uint8)

    # Initial quantization
    iscale = nmax / max_val

    for i in range(n):
        L[i] = nearest_int(iscale * x[i])

    scale = 1.0 / iscale

    # Compute initial weighted MSE
    best_mse = 0.0
    for i in range(n):
        diff = x[i] - scale * L[i]
        best_mse += quant_weights[i] * diff * diff

    # 9-iteration search: test scale offsets from -4 to +4
    for is_offset in range(-4, 5):
        if is_offset == 0:
            continue

        iscale_test = (0.1 * is_offset + nmax) / max_val
        scale_test = 1.0 / iscale_test

        # Compute MSE for this scale
        mse = 0.0
        for i in range(n):
            l = nearest_int(iscale_test * x[i])
            l = min(nmax, l)
            diff = x[i] - scale_test * l
            mse += quant_weights[i] * diff * diff

        if mse < best_mse:
            best_mse = mse
            iscale = iscale_test

    # Requantize with best scale
    sumlx = 0.0
    suml2 = 0.0

    for i in range(n):
        l = nearest_int(iscale * x[i])
        l = min(nmax, l)
        L[i] = l
        sumlx += quant_weights[i] * x[i] * l
        suml2 += quant_weights[i] * l * l

    # Iterative refinement (up to 5 passes)
    for itry in range(5):
        n_changed = 0

        for i in range(n):
            w = quant_weights[i]

            # Remove current element's contribution
            slx = sumlx - w * x[i] * L[i]
            sl2 = suml2 - w * L[i] * L[i]

            if slx > 0 and sl2 > 0:
                # Calculate what the element should be
                new_l = nearest_int(x[i] * sl2 / slx)
                new_l = min(nmax, new_l)

                if new_l != L[i]:
                    # Add new element's contribution
                    slx += w * x[i] * new_l
                    sl2 += w * new_l * new_l

                    # Check if this improves the error
                    if slx * slx * suml2 > sumlx * sumlx * sl2:
                        L[i] = new_l
                        sumlx = slx
                        suml2 = sl2
                        n_changed += 1

        # If no changes in this pass, we've converged
        if n_changed == 0:
            break

    # Return final scale
    scale = sumlx / suml2 if suml2 > 0 else 0.0
    return scale, L


def make_qkx3_quants(n, nmax, x, weights, rmin=-0.9, rdelta=0.05, nstep=36, use_mad=False):
    """
    Affine quantization with grid search optimization (used by Q4_K/Q5_K).

    Tests multiple scale/min combinations to minimize weighted error.

    Reference: llama.cpp/ggml/src/ggml-quants.c:816-897

    Args:
        n: Number of elements to quantize
        nmax: Maximum quantized value (15 for Q4_K, 31 for Q5_K)
        x: Input float array
        weights: Weight array for weighted error (None = use x²)
        rmin: Minimum offset for grid search (default: -0.9)
        rdelta: Step size for grid search (default: 0.05)
        nstep: Number of search steps (default: 36)
        use_mad: Use MAD instead of MSE (default: False = use MSE)

    Returns:
        tuple: (scale, min_val, L) where:
            scale: Optimal scale factor
            min_val: Optimal min value (note: returned as -min for compatibility)
            L: Quantized values as uint8 array
    """
    x = np.asarray(x, dtype=np.float32)
    L = np.zeros(n, dtype=np.uint8)
    Laux = np.zeros(n, dtype=np.uint8)

    # Find min and max
    min_val = float(np.min(x))
    max_val = float(np.max(x))

    # Compute weighted sum
    sum_w = 0.0
    sum_x = 0.0
    for i in range(n):
        w = weights[i] if weights is not None else x[i] * x[i]
        sum_w += w
        sum_x += w * x[i]

    # Ensure min is at most 0
    if min_val > 0:
        min_val = 0

    # All same value case
    if max_val <= min_val:
        return 0.0, -min_val, np.zeros(n, dtype=np.uint8)

    # Initial quantization
    iscale = nmax / (max_val - min_val)
    scale = 1.0 / iscale

    best_mad = 0.0
    for i in range(n):
        l = nearest_int(iscale * (x[i] - min_val))
        l = max(0, min(nmax, l))
        L[i] = l
        diff = scale * L[i] + min_val - x[i]
        diff = abs(diff) if use_mad else diff * diff
        w = weights[i] if weights is not None else x[i] * x[i]
        best_mad += w * diff

    # Grid search over scale values (if nstep > 0)
    if nstep < 1:
        return scale, -min_val, L.astype(np.uint8)

    for is_step in range(nstep + 1):
        # Test this scale offset
        iscale_test = (rmin + rdelta * is_step + nmax) / (max_val - min_val)

        # Quantize with this scale
        sum_l = 0.0
        sum_l2 = 0.0
        sum_xl = 0.0

        for i in range(n):
            l = nearest_int(iscale_test * (x[i] - min_val))
            l = max(0, min(nmax, l))
            Laux[i] = l
            w = weights[i] if weights is not None else x[i] * x[i]
            sum_l += w * l
            sum_l2 += w * l * l
            sum_xl += w * l * x[i]

        # Solve for optimal scale and min using weighted least squares
        D = sum_w * sum_l2 - sum_l * sum_l

        if D > 0:
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D

            # Constrain min to be at most 0
            if this_min > 0:
                this_min = 0
                this_scale = sum_xl / sum_l2

            # Compute error with this scale/min
            mad = 0.0
            for i in range(n):
                diff = this_scale * Laux[i] + this_min - x[i]
                diff = abs(diff) if use_mad else diff * diff
                w = weights[i] if weights is not None else x[i] * x[i]
                mad += w * diff

            # Keep if better
            if mad < best_mad:
                for i in range(n):
                    L[i] = Laux[i]
                best_mad = mad
                scale = this_scale
                min_val = this_min

    return scale, -min_val, L.astype(np.uint8)


def make_qkx2_quants(n, nmax, x, weights, rmin, rdelta, nstep, use_mad):
    """
    Affine quantization with grid search optimization (used by Q2_K).
    Reference: llama.cpp/ggml/src/ggml-quants.c:608-685
    """
    x = np.asarray(x, dtype=np.float32)
    L = np.zeros(n, dtype=np.uint8)
    Laux = np.zeros(n, dtype=np.uint8)

    min_val = np.min(x)
    max_val = np.max(x)

    sum_w = np.sum(weights)
    sum_x = np.sum(weights * x)

    if min_val > 0:
        min_val = 0

    if max_val == min_val:
        return 0.0, -min_val, np.zeros(n, dtype=np.uint8)

    iscale = nmax / (max_val - min_val)
    scale = 1.0 / iscale

    best_error = 0.0
    for i in range(n):
        l = nearest_int(iscale * (x[i] - min_val))
        L[i] = max(0, min(nmax, l))
        diff = scale * L[i] + min_val - x[i]
        diff = abs(diff) if use_mad else diff * diff
        best_error += weights[i] * diff

    if nstep < 1:
        return scale, -min_val, L

    for is_step in range(nstep + 1):
        iscale_test = (rmin + rdelta * is_step + nmax) / (max_val - min_val)
        sum_l = 0.0
        sum_l2 = 0.0
        sum_xl = 0.0
        for i in range(n):
            l = nearest_int(iscale_test * (x[i] - min_val))
            l = max(0, min(nmax, l))
            Laux[i] = l
            w = weights[i]
            sum_l += w * l
            sum_l2 += w * l * l
            sum_xl += w * l * x[i]

        D = sum_w * sum_l2 - sum_l * sum_l
        if D > 0:
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
            if this_min > 0:
                this_min = 0
                this_scale = sum_xl / sum_l2 if sum_l2 > 0 else 0.0

            cur_error = 0.0
            for i in range(n):
                diff = this_scale * Laux[i] + this_min - x[i]
                diff = abs(diff) if use_mad else diff * diff
                cur_error += weights[i] * diff

            if cur_error < best_error:
                L[:] = Laux[:]
                best_error = cur_error
                scale = this_scale
                min_val = this_min

    return scale, -min_val, L


def test_make_q3_quants():
    """Test make_q3_quants with both fast and optimized paths"""
    print("\n\nTesting make_q3_quants...")
    print("=" * 70)

    # Test with 16-element block (typical for Q3_K)
    print("\nTest: Q3_K block (n=16, nmax=4)")
    print("-" * 70)
    np.random.seed(42)
    test_data = np.random.randn(16).astype(np.float32)

    print(f"Input: {test_data[:8]}...")
    print(f"Range: [{test_data.min():.4f}, {test_data.max():.4f}]")

    # Fast path
    scale_fast, L_fast = make_q3_quants(16, 4, test_data, do_rmse=False)
    deq_fast = (L_fast.astype(np.int32) - 4) * scale_fast
    error_fast = np.abs(test_data - deq_fast).mean()
    corr_fast = np.corrcoef(test_data, deq_fast)[0, 1]

    print(f"\nFast path (do_rmse=False):")
    print(f"  Scale: {scale_fast:.6f}")
    print(f"  L: {L_fast[:8]}...")
    print(f"  Error: {error_fast:.6f}, Correlation: {corr_fast:.8f}")

    # Optimized path
    scale_opt, L_opt = make_q3_quants(16, 4, test_data, do_rmse=True)
    deq_opt = (L_opt.astype(np.int32) - 4) * scale_opt
    error_opt = np.abs(test_data - deq_opt).mean()
    corr_opt = np.corrcoef(test_data, deq_opt)[0, 1]

    print(f"\nOptimized path (do_rmse=True):")
    print(f"  Scale: {scale_opt:.6f}")
    print(f"  L: {L_opt[:8]}...")
    print(f"  Error: {error_opt:.6f}, Correlation: {corr_opt:.8f}")

    print(f"\nImprovement: {(corr_opt - corr_fast) * 100:.6f}%")

    # Test with larger block
    print("\n" + "-" * 70)
    print("Test: Larger block (n=64, nmax=4)")
    print("-" * 70)
    np.random.seed(123)
    test_data2 = np.random.randn(64).astype(np.float32) * 2.0

    scale_fast2, L_fast2 = make_q3_quants(64, 4, test_data2, do_rmse=False)
    deq_fast2 = (L_fast2.astype(np.int32) - 4) * scale_fast2
    corr_fast2 = np.corrcoef(test_data2, deq_fast2)[0, 1]

    scale_opt2, L_opt2 = make_q3_quants(64, 4, test_data2, do_rmse=True)
    deq_opt2 = (L_opt2.astype(np.int32) - 4) * scale_opt2
    corr_opt2 = np.corrcoef(test_data2, deq_opt2)[0, 1]

    print(f"Fast:      Correlation: {corr_fast2:.8f}")
    print(f"Optimized: Correlation: {corr_opt2:.8f}")
    print(f"Improvement: {(corr_opt2 - corr_fast2) * 100:.6f}%")

    print("\n" + "=" * 70)
    if corr_opt >= corr_fast - 0.0001 and corr_opt2 >= corr_fast2 - 0.0001:
        print("Test PASSED! Optimization is not worse than fast path.")
    else:
        print("Test FAILED! Optimization is worse than fast path.")


if __name__ == "__main__":
    test_make_qx_quants()
    test_make_q3_quants()
