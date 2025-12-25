"""
Python implementation of llama-imatrix --show-statistics
Reads GGUF imatrix files and displays statistics exactly as llama.cpp does.
"""

import struct
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
import math


class TensorStatistics:
    """Statistics for a single tensor"""
    def __init__(self, name: str):
        self.tensor = name
        self.total_sqract = 0.0
        self.mean_sqract = 0.0
        self.max_sqract = 0.0
        self.min_sqract = 0.0
        self.elements = 0
        self.stddev = 0.0
        self.active = 0.0
        self.entropy = 0.0
        self.zd = 0.0
        self.cossim = 0.0


class Stats:
    """Raw statistics data from imatrix file"""
    def __init__(self):
        self.values = []
        self.counts = []


def load_imatrix_gguf(file_path: str) -> Dict[str, Stats]:
    """
    Load imatrix data from GGUF format file

    Args:
        file_path: Path to .gguf imatrix file

    Returns:
        Dictionary mapping tensor names to Stats objects
    """
    stats_dict = {}

    with open(file_path, 'rb') as f:
        # Read GGUF header
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not a valid GGUF file: {file_path}")

        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        # Read key-value metadata (we skip this for now as we only need tensors)
        for _ in range(n_kv):
            # Read key
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')

            # Read value type
            value_type = struct.unpack('<I', f.read(4))[0]

            # Skip value based on type
            if value_type == 8:  # String
                str_len = struct.unpack('<Q', f.read(8))[0]
                f.read(str_len)
            elif value_type == 4:  # uint32
                f.read(4)
            elif value_type == 9:  # Array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                if arr_type == 8:  # String array
                    for _ in range(arr_len):
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        f.read(str_len)
                else:
                    # Skip other array types
                    type_sizes = {4: 4, 5: 8, 6: 4, 7: 8}
                    if arr_type in type_sizes:
                        f.read(type_sizes[arr_type] * arr_len)
            # Add more type handlers as needed

        # Read tensor info
        tensor_infos = []
        for _ in range(n_tensors):
            # Read tensor name
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')

            # Read dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))

            # Read tensor type
            tensor_type = struct.unpack('<I', f.read(4))[0]

            # Read offset
            offset = struct.unpack('<Q', f.read(8))[0]

            tensor_infos.append({
                'name': name,
                'dims': dims,
                'type': tensor_type,
                'offset': offset
            })

        # Align to next multiple of 32
        pos = f.tell()
        alignment = 32
        if pos % alignment != 0:
            f.seek(alignment - (pos % alignment), 1)

        data_start = f.tell()

        # Read tensor data
        for info in tensor_infos:
            name = info['name']
            dims = info['dims']
            offset = info['offset']

            # Calculate number of elements
            n_elements = 1
            for d in dims:
                n_elements *= d

            # Seek to tensor data
            f.seek(data_start + offset)

            # Read float32 data
            data = struct.unpack(f'<{n_elements}f', f.read(n_elements * 4))

            # Parse tensor name to determine if it's values or counts
            if name.endswith('.in_sum2'):
                base_name = name[:-8]  # Remove '.in_sum2'
                if base_name not in stats_dict:
                    stats_dict[base_name] = Stats()
                stats_dict[base_name].values = list(data)
            elif name.endswith('.counts'):
                base_name = name[:-7]  # Remove '.counts'
                if base_name not in stats_dict:
                    stats_dict[base_name] = Stats()
                stats_dict[base_name].counts = [int(round(x)) for x in data]

    return stats_dict


def compute_statistics(name: str, stats: Stats) -> TensorStatistics:
    """
    Compute statistics for a tensor (exact replica of C++ implementation)

    Args:
        name: Tensor name
        stats: Raw statistics data

    Returns:
        TensorStatistics object with computed values
    """
    if not stats.values or not stats.counts:
        raise ValueError(f"No data for tensor {name}")

    if len(stats.values) % len(stats.counts) != 0:
        raise ValueError(f"Activation size mismatch for {name}")

    n_mat = len(stats.counts)
    row_size = len(stats.values) // n_mat

    # Normalize activations by counts
    activations = []
    for i in range(n_mat):
        for j in range(row_size):
            activations.append(stats.values[i * row_size + j] / stats.counts[i])

    # Calculate basic statistics
    act_total = sum(activations)
    act_max = max(activations)
    act_min = min(activations)
    act_mean = act_total / len(activations)

    act_sqr_total = sum(a * a for a in activations)
    act_var = (act_sqr_total / len(activations)) - (act_mean * act_mean)
    act_dev = math.sqrt(max(0.0, act_var))

    # Calculate active ratio (threshold from C++ code)
    threshold = 1e-5
    inactive_count = sum(1 for a in activations if abs(a) <= threshold)
    active_ratio = 1.0 - (inactive_count / len(activations))

    # Calculate entropy
    entropy = 0.0
    if act_total > 0:
        for act in activations:
            p = act / act_total
            if p > 0:
                entropy -= p * math.log2(p)

    # Calculate z-score distribution
    z_score = 0
    if act_dev > 0.0:
        for act in activations:
            p = (act - act_mean) / act_dev
            if p > 1:
                z_score += 1

    # Create TensorStatistics object
    ts = TensorStatistics(name)
    ts.total_sqract = act_total
    ts.mean_sqract = act_mean
    ts.max_sqract = act_max
    ts.min_sqract = act_min
    ts.elements = len(activations)
    ts.stddev = act_dev
    ts.active = active_ratio
    ts.entropy = entropy
    ts.zd = z_score / ts.elements if ts.elements > 0 else 0.0

    return ts


def compute_cossim(tstats: List[TensorStatistics], stats_dict: Dict[str, Stats]):
    """
    Compute cosine similarity between consecutive block layers

    Args:
        tstats: List of TensorStatistics objects
        stats_dict: Dictionary of raw statistics data
    """
    pattern = re.compile(r'blk\.(\d+)\.')

    for ts in tstats:
        match = pattern.search(ts.tensor)
        if match:
            blk = int(match.group(1))
            # Create name for previous block
            prev_name = ts.tensor.replace(f'blk.{blk}.', f'blk.{blk-1}.')

            # Find previous tensor in list
            prev_ts = None
            for t in tstats:
                if t.tensor == prev_name:
                    prev_ts = t
                    break

            if prev_ts and prev_name in stats_dict and ts.tensor in stats_dict:
                # Calculate cosine similarity using raw values
                curr_values = stats_dict[ts.tensor].values
                prev_values = stats_dict[prev_name].values

                if len(curr_values) == len(prev_values):
                    # Dot product
                    dp = sum(c * p for c, p in zip(curr_values, prev_values))

                    # Magnitudes
                    curr_mag = math.sqrt(sum(c * c for c in curr_values))
                    prev_mag = math.sqrt(sum(p * p for p in prev_values))

                    # Cosine similarity
                    if curr_mag > 0 and prev_mag > 0:
                        ts.cossim = dp / (curr_mag * prev_mag)
        else:
            ts.cossim = 0.0


def process_tensor_name(input_name: str) -> Tuple[str, str]:
    """
    Extract layer and tensor type from tensor name

    Args:
        input_name: Full tensor name

    Returns:
        Tuple of (layer, tensor_type)
    """
    parts = input_name.split('.')

    layer = '-'
    tensor = input_name

    # Find layer number
    for i, part in enumerate(parts):
        if part == 'blk' and i + 1 < len(parts):
            layer = parts[i + 1]
            break

    # Find tensor type (word before 'weight')
    for i, part in enumerate(parts):
        if part == 'weight' and i > 0:
            tensor = parts[i - 1]
            break

    return layer, tensor


def show_statistics(imatrix_path: str) -> bool:
    """
    Display statistics for an imatrix file (exact replica of C++ output)

    Args:
        imatrix_path: Path to imatrix file

    Returns:
        True on success, False on error
    """
    try:
        # Load imatrix data
        stats_dict = load_imatrix_gguf(imatrix_path)

        if not stats_dict:
            print(f"\nError: {imatrix_path} is not a valid imatrix file\n")
            return False

        # Compute statistics for each tensor
        tstats = []
        for name, stats in stats_dict.items():
            try:
                ts = compute_statistics(name, stats)
                tstats.append(ts)
            except Exception as e:
                print(f"Warning: Could not compute statistics for {name}: {e}")
                continue

        if not tstats:
            print(f"Error: cannot compute statistics for {imatrix_path}\n")
            return False

        # Compute cosine similarity
        compute_cossim(tstats, stats_dict)

        # Sort tensors (by tensor type name, then by total)
        def tensor_comparer(ts):
            layer, name = process_tensor_name(ts.tensor)
            return (name, -ts.total_sqract)

        tstats.sort(key=tensor_comparer)

        # Calculate weighted averages per layer
        weighted_stats = {}

        # Print header
        print(f"\nComputing statistics for {imatrix_path} ({len(tstats)} tensors)")
        print(f"\n{'Layer':>5}  {'Tensor':<20}  {'Σ(Act²)':>14}  {'Min':>10}  {'Max':>14}  {'μ':>10}  {'σ':>10}  {'% Active':>9}  {'N':>10}  {'Entropy':>12}  {'E (norm)':>7}  {'ZD':>11}  {'CosSim':>8}")
        print("=" * 190)

        # Print tensor statistics
        for ts in tstats:
            layer, name = process_tensor_name(ts.tensor)

            # Parse layer number for weighted stats
            try:
                blk = int(layer)
            except (ValueError, TypeError):
                blk = -1

            # Calculate normalized entropy
            norm_entropy = 0.0
            if ts.elements > 1:
                norm_entropy = 100.0 * (ts.entropy / math.log2(ts.elements))

            print(f"{layer:>5}  {name:<20}  {ts.total_sqract:>14.2f}  {ts.min_sqract:>10.4f}  {ts.max_sqract:>14.4f}  {ts.mean_sqract:>10.2f}  {ts.stddev:>10.2f}  {ts.active * 100:>8.2f}%  {ts.elements:>10}  {ts.entropy:>12.4f}  {norm_entropy:>6.2f}%  {ts.zd * 100:>10.2f}%  {ts.cossim:>8.4f}")

            # Accumulate weighted stats
            weighted_bias = ts.elements * ts.total_sqract
            weighted_zd = ts.elements * ts.zd
            weighted_cossim = ts.elements * ts.cossim

            if blk not in weighted_stats:
                weighted_stats[blk] = {
                    'weighted_bias': 0.0,
                    'weighted_zd': 0.0,
                    'weighted_cossim': 0.0,
                    'total_elements': 0
                }

            weighted_stats[blk]['weighted_bias'] += weighted_bias
            weighted_stats[blk]['weighted_zd'] += weighted_zd
            weighted_stats[blk]['weighted_cossim'] += weighted_cossim
            weighted_stats[blk]['total_elements'] += ts.elements

        # Print weighted averages per layer
        layers = [k for k in weighted_stats.keys() if k >= 0]
        if layers:
            print(f"\nComputing weighted average statistics per layer ({len(layers)} layers)")
            print(f"\n{'Layer':>5}  {'μΣ(Act²)':>14}  {'μZD':>11}  {'μCosSim':>8}")
            print("=" * 50)

            for blk in sorted(layers):
                stats = weighted_stats[blk]
                if stats['total_elements'] > 0:
                    bias = stats['weighted_bias'] / stats['total_elements']
                    zd = stats['weighted_zd'] / stats['total_elements']
                    cossim = stats['weighted_cossim'] / stats['total_elements']

                    print(f"{blk:>5}  {bias:>14.2f}  {zd * 100:>10.4f}%  {cossim:>8.4f}")

        print()
        return True

    except Exception as e:
        print(f"\nError: Failed to show statistics: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python imatrix_stats.py <imatrix.gguf>")
        sys.exit(1)

    imatrix_path = sys.argv[1]

    if not Path(imatrix_path).exists():
        print(f"Error: File not found: {imatrix_path}")
        sys.exit(1)

    success = show_statistics(imatrix_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
