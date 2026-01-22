"""
VRAM Calculator for GGUF Models

Calculates estimated VRAM usage and recommended GPU layers (-ngl) based on
available VRAM, model architecture, and desired headroom.
"""

import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    vendor: str  # "nvidia" or "amd"


@dataclass
class GGUFModelInfo:
    """Information extracted from a GGUF model file."""
    file_path: Path
    file_size_mb: float
    num_layers: Optional[int]
    architecture: Optional[str]
    context_length: Optional[int]
    embedding_length: Optional[int]
    head_count: Optional[int]
    head_count_kv: Optional[int]
    vocab_size: Optional[int]
    quantization_version: Optional[int]
    file_type: Optional[str]


@dataclass
class VRAMCalculation:
    """Results of VRAM calculation."""
    recommended_layers: int
    total_layers: int
    mb_per_layer: float
    available_vram_mb: float
    headroom_mb: float
    context_overhead_mb: float
    estimated_usage_mb: float
    fits_entirely: bool
    error: Optional[str] = None


def get_system_ram_mb() -> tuple[int, int, int]:
    """
    Get system RAM usage (Total, Used, Available) in MB.
    
    Returns:
        tuple: (total_mb, used_mb, available_mb)
    """
    import platform
    
    total_mb = 0
    available_mb = 0
    
    try:
        if platform.system() == "Windows":
            import ctypes
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            
            total_mb = int(mem.ullTotalPhys / (1024 * 1024))
            available_mb = int(mem.ullAvailPhys / (1024 * 1024))
            
        elif platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                mem_info = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        # values are in kB
                        try:
                            mem_info[key] = int(parts[1])
                        except ValueError:
                            pass

                if 'MemTotal' in mem_info:
                    total_mb = int(mem_info['MemTotal'] / 1024)

                if 'MemAvailable' in mem_info:
                    available_mb = int(mem_info['MemAvailable'] / 1024)
                elif 'MemFree' in mem_info:
                    # Fallback if MemAvailable not present
                    available_mb = int(mem_info['MemFree'] / 1024)

        elif platform.system() == "Darwin":  # macOS
            # Get total RAM using sysctl
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip())
                total_mb = int(total_bytes / (1024 * 1024))

            # Get memory pressure/available using vm_stat
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse vm_stat output for page size and free/inactive pages
                page_size = 4096  # Default page size
                free_pages = 0
                inactive_pages = 0

                for line in result.stdout.split('\n'):
                    if 'page size of' in line:
                        match = re.search(r'(\d+) bytes', line)
                        if match:
                            page_size = int(match.group(1))
                    elif line.startswith('Pages free:'):
                        match = re.search(r'(\d+)', line)
                        if match:
                            free_pages = int(match.group(1))
                    elif line.startswith('Pages inactive:'):
                        match = re.search(r'(\d+)', line)
                        if match:
                            inactive_pages = int(match.group(1))

                # Available = free + inactive (inactive can be reclaimed)
                available_bytes = (free_pages + inactive_pages) * page_size
                available_mb = int(available_bytes / (1024 * 1024))

    except Exception:
        pass
        
    # Ensure reasonable values
    if total_mb <= 0:
        total_mb = 16 * 1024 # Assumed default if detection fails
        available_mb = 8 * 1024
        
    used_mb = max(0, total_mb - available_mb)
    
    return total_mb, used_mb, available_mb


def get_nvidia_gpus() -> List[GPUInfo]:
    """
    Query NVIDIA GPU VRAM using nvidia-smi.

    Returns:
        List of GPUInfo objects for detected NVIDIA GPUs.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
    except FileNotFoundError:
        return []
    except subprocess.CalledProcessError:
        return []
    except subprocess.TimeoutExpired:
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mb=int(float(parts[2])),
                    used_mb=int(float(parts[3])),
                    free_mb=int(float(parts[4])),
                    vendor="nvidia"
                ))
            except (ValueError, IndexError):
                continue
    return gpus


def get_amd_gpus() -> List[GPUInfo]:
    """
    Query AMD GPU VRAM using rocm-smi.

    Returns:
        List of GPUInfo objects for detected AMD GPUs.
    """
    try:
        # Try rocm-smi first (Linux ROCm)
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Try alternative: rocm-smi with different flags
        try:
            result = subprocess.run(
                ["rocm-smi", "-a", "--csv"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []

    gpus = []
    lines = result.stdout.strip().split("\n")

    # Parse CSV output - format varies by rocm-smi version
    # Common format: device,GPU Memory Used,GPU Memory Total (or similar)
    if len(lines) < 2:
        return gpus

    header = lines[0].lower()

    # Try to find memory columns
    for i, line in enumerate(lines[1:]):
        if not line.strip():
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue

        try:
            # Extract GPU index from first column (usually "cardN" or just index)
            idx_str = parts[0]
            idx_match = re.search(r'\d+', idx_str)
            idx = int(idx_match.group()) if idx_match else i

            # Try to find memory values - rocm-smi output varies
            total_mb = 0
            used_mb = 0

            # Look for memory columns by header or position
            for j, val in enumerate(parts[1:], 1):
                col_name = ""
                if j < len(header.split(",")):
                    col_name = header.split(",")[j].lower()

                # Parse memory values (may be in bytes, MB, or have units)
                val_clean = val.strip()
                if not val_clean:
                    continue

                # Convert to MB if needed
                mb_val = _parse_memory_value(val_clean)
                if mb_val is None:
                    continue

                if "total" in col_name or (j == 2 and total_mb == 0):
                    total_mb = mb_val
                elif "used" in col_name or (j == 1 and used_mb == 0):
                    used_mb = mb_val

            if total_mb > 0:
                gpus.append(GPUInfo(
                    index=idx,
                    name=f"AMD GPU {idx}",
                    total_mb=total_mb,
                    used_mb=used_mb,
                    free_mb=total_mb - used_mb,
                    vendor="amd"
                ))
        except (ValueError, IndexError):
            continue

    # If CSV parsing failed, try plain text parsing
    if not gpus:
        gpus = _parse_rocm_smi_text(result.stdout)

    return gpus


def _parse_memory_value(val: str) -> Optional[int]:
    """
    Parse a memory value string to MB.

    Args:
        val: Memory value string (e.g., "8192", "8192MB", "8GB", "8589934592")

    Returns:
        Value in MB, or None if parsing fails.
    """
    val = val.strip().upper()

    # Remove any non-numeric suffixes except unit indicators
    match = re.match(r'^([\d.]+)\s*(GB|MB|KB|B)?', val)
    if not match:
        return None

    try:
        num = float(match.group(1))
        unit = match.group(2) or ""

        if unit == "GB":
            return int(num * 1024)
        elif unit == "MB":
            return int(num)
        elif unit == "KB":
            return int(num / 1024)
        elif unit == "B" or (not unit and num > 1000000):
            # Likely bytes if no unit and large number
            return int(num / (1024 * 1024))
        else:
            # Assume MB for reasonable values, bytes for large values
            if num > 100000:
                return int(num / (1024 * 1024))
            return int(num)
    except ValueError:
        return None


def _parse_rocm_smi_text(output: str) -> List[GPUInfo]:
    """
    Parse rocm-smi plain text output as fallback.

    Args:
        output: Raw rocm-smi output text.

    Returns:
        List of GPUInfo objects.
    """
    gpus = []
    current_gpu = None
    total_mb = 0
    used_mb = 0

    for line in output.split("\n"):
        line = line.strip()

        # Look for GPU identifier
        gpu_match = re.search(r'GPU\[(\d+)\]|card(\d+)', line, re.IGNORECASE)
        if gpu_match:
            # Save previous GPU if exists
            if current_gpu is not None and total_mb > 0:
                gpus.append(GPUInfo(
                    index=current_gpu,
                    name=f"AMD GPU {current_gpu}",
                    total_mb=total_mb,
                    used_mb=used_mb,
                    free_mb=total_mb - used_mb,
                    vendor="amd"
                ))
            current_gpu = int(gpu_match.group(1) or gpu_match.group(2))
            total_mb = 0
            used_mb = 0

        # Look for memory values
        if "vram total" in line.lower():
            mem_match = re.search(r'([\d.]+)\s*(GB|MB|B)?', line)
            if mem_match:
                total_mb = _parse_memory_value(mem_match.group(0)) or 0

        if "vram used" in line.lower():
            mem_match = re.search(r'([\d.]+)\s*(GB|MB|B)?', line)
            if mem_match:
                used_mb = _parse_memory_value(mem_match.group(0)) or 0

    # Don't forget last GPU
    if current_gpu is not None and total_mb > 0:
        gpus.append(GPUInfo(
            index=current_gpu,
            name=f"AMD GPU {current_gpu}",
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=total_mb - used_mb,
            vendor="amd"
        ))

    return gpus


def get_all_gpus() -> List[GPUInfo]:
    """
    Detect all available GPUs (NVIDIA and AMD).

    Returns:
        List of GPUInfo objects for all detected GPUs.
    """
    gpus = []
    gpus.extend(get_nvidia_gpus())
    gpus.extend(get_amd_gpus())
    return gpus


def get_split_total_size(filepath: Path) -> int:
    """
    Calculate total size of all shards for a split GGUF file.

    Args:
        filepath: Path to any shard (e.g., model-00001-of-00026.gguf)

    Returns:
        Total size in bytes across all shards, or single file size if not split.
    """
    filename = filepath.name

    # Match pattern like: name-00001-of-00026.gguf
    match = re.search(r'-(\d+)-of-(\d+)\.gguf$', filename, re.IGNORECASE)

    if not match:
        return filepath.stat().st_size

    total_shards = int(match.group(2))
    prefix = filename[:match.start()]

    total_size = 0
    for i in range(1, total_shards + 1):
        shard_name = f"{prefix}-{i:05d}-of-{total_shards:05d}.gguf"
        shard_path = filepath.parent / shard_name
        if shard_path.exists():
            total_size += shard_path.stat().st_size
        else:
            # If shard doesn't exist, estimate based on first shard
            total_size += filepath.stat().st_size

    return total_size


def get_gguf_model_info(filepath: Path) -> GGUFModelInfo:
    """
    Read model information from a GGUF file.

    Extracts architecture details including layer count, context length,
    and other relevant parameters for VRAM calculation.

    Uses a minimal metadata reader that skips tensor info for faster loading.

    Args:
        filepath: Path to GGUF model file.

    Returns:
        GGUFModelInfo with extracted model details.
    """
    file_size_bytes = get_split_total_size(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Initialize with defaults
    info = GGUFModelInfo(
        file_path=filepath,
        file_size_mb=file_size_mb,
        num_layers=None,
        architecture=None,
        context_length=None,
        embedding_length=None,
        head_count=None,
        head_count_kv=None,
        vocab_size=None,
        quantization_version=None,
        file_type=None
    )

    # Try to read metadata from GGUF using minimal reader (skips tensor info)
    try:
        metadata = _read_gguf_metadata_only(filepath)

        # Get architecture first
        arch = metadata.get("general.architecture")
        if arch is not None:
            if isinstance(arch, bytes):
                arch = arch.decode('utf-8')
            info.architecture = arch

        # Map of metadata keys to extract (suffix -> attribute name)
        key_mappings = {
            "block_count": "num_layers",
            "context_length": "context_length",
            "embedding_length": "embedding_length",
            "attention.head_count": "head_count",
            "attention.head_count_kv": "head_count_kv",
            "vocab_size": "vocab_size",
        }

        # Search for architecture-specific or general keys
        for key, value in metadata.items():
            for key_suffix, attr_name in key_mappings.items():
                if key.endswith(f".{key_suffix}") or key == f"general.{key_suffix}":
                    if value is not None and isinstance(value, (int, float)):
                        setattr(info, attr_name, int(value))
                    break

            # Get file type (quantization type)
            if key == "general.file_type":
                if value is not None:
                    info.file_type = _get_file_type_name(int(value))
                    info.quantization_version = int(value)

        # Fallback for vocab_size: use tokenizer.ggml.tokens length if not found
        if info.vocab_size is None:
            tokens = metadata.get("tokenizer.ggml.tokens")
            if tokens is not None and isinstance(tokens, list):
                info.vocab_size = len(tokens)

    except Exception:
        pass  # Failed to read metadata

    # Fallback: estimate layers from file size if not found
    if info.num_layers is None:
        info.num_layers = _estimate_layers_from_size(file_size_mb)

    return info


def _read_gguf_metadata_only(filepath: Path) -> Dict[str, Any]:
    """
    Read only the metadata key-value pairs from a GGUF file.

    This is an optimized reader that skips tensor info entirely,
    providing significant speedup for large models with many tensors.

    Args:
        filepath: Path to GGUF model file.

    Returns:
        Dictionary of metadata key-value pairs.
    """
    import struct

    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

    # GGUF value types
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12

    metadata: Dict[str, Any] = {}

    with open(filepath, 'rb') as f:
        # Read and validate magic
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError('Invalid GGUF magic')

        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f'Unsupported GGUF version: {version}')

        # Read counts
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]

        def read_string() -> str:
            """Read a GGUF string (length-prefixed)."""
            length = struct.unpack('<Q', f.read(8))[0]
            return f.read(length).decode('utf-8')

        def read_value(value_type: int) -> Any:
            """Read a value of the given GGUF type."""
            if value_type == GGUF_TYPE_UINT8:
                return struct.unpack('<B', f.read(1))[0]
            elif value_type == GGUF_TYPE_INT8:
                return struct.unpack('<b', f.read(1))[0]
            elif value_type == GGUF_TYPE_UINT16:
                return struct.unpack('<H', f.read(2))[0]
            elif value_type == GGUF_TYPE_INT16:
                return struct.unpack('<h', f.read(2))[0]
            elif value_type == GGUF_TYPE_UINT32:
                return struct.unpack('<I', f.read(4))[0]
            elif value_type == GGUF_TYPE_INT32:
                return struct.unpack('<i', f.read(4))[0]
            elif value_type == GGUF_TYPE_FLOAT32:
                return struct.unpack('<f', f.read(4))[0]
            elif value_type == GGUF_TYPE_BOOL:
                return struct.unpack('<B', f.read(1))[0] != 0
            elif value_type == GGUF_TYPE_STRING:
                return read_string()
            elif value_type == GGUF_TYPE_UINT64:
                return struct.unpack('<Q', f.read(8))[0]
            elif value_type == GGUF_TYPE_INT64:
                return struct.unpack('<q', f.read(8))[0]
            elif value_type == GGUF_TYPE_FLOAT64:
                return struct.unpack('<d', f.read(8))[0]
            elif value_type == GGUF_TYPE_ARRAY:
                # Read array type and length
                array_type = struct.unpack('<I', f.read(4))[0]
                array_len = struct.unpack('<Q', f.read(8))[0]
                # Read array elements
                return [read_value(array_type) for _ in range(array_len)]
            else:
                raise ValueError(f'Unknown GGUF value type: {value_type}')

        # Read all key-value pairs (this is what we need)
        for _ in range(kv_count):
            key = read_string()
            value_type = struct.unpack('<I', f.read(4))[0]
            value = read_value(value_type)
            metadata[key] = value

        # Stop here - don't read tensor info (this is the optimization)

    return metadata


def _get_file_type_name(file_type: int) -> str:
    """
    Convert GGUF file type enum to human-readable name.

    Args:
        file_type: GGUF file type integer.

    Returns:
        Human-readable quantization type name.
    """
    # Common GGUF file types
    # Based on llama.h enum llama_ftype
    file_types = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        # 4-6 removed/legacy
        7: "Q8_0",
        8: "Q5_0",
        9: "Q5_1",
        10: "Q2_K",
        11: "Q3_K_S",
        12: "Q3_K_M",
        13: "Q3_K_L",
        14: "Q4_K_S",
        15: "Q4_K_M",
        16: "Q5_K_S",
        17: "Q5_K_M",
        18: "Q6_K",
        19: "IQ2_XXS",
        20: "IQ2_XS",
        21: "Q2_K_S",
        22: "IQ3_XS",
        23: "IQ3_XXS",
        24: "IQ1_S",
        25: "IQ4_NL",
        26: "IQ3_S",
        27: "IQ3_M",
        28: "IQ2_S",
        29: "IQ2_M",
        30: "IQ4_XS",
        31: "IQ1_M",
        32: "BF16",
        36: "TQ1_0",
        37: "TQ2_0",
        38: "MXFP4_MOE",
    }
    return file_types.get(file_type, f"Unknown ({file_type})")


def _estimate_layers_from_size(size_mb: float) -> int:
    """
    Estimate layer count based on file size.

    This is a rough heuristic for common model sizes assuming
    mid-range quantization (Q4_K_M or similar).

    Args:
        size_mb: Model file size in MB.

    Returns:
        Estimated number of layers.
    """
    if size_mb < 1500:       # ~1B models
        return 22
    elif size_mb < 3000:     # ~1.5-3B models
        return 24
    elif size_mb < 5000:     # ~3-7B models
        return 28
    elif size_mb < 8000:     # ~7-8B models
        return 32
    elif size_mb < 12000:    # ~13B models
        return 40
    elif size_mb < 20000:    # ~20B models
        return 48
    elif size_mb < 30000:    # ~27-34B models
        return 60
    elif size_mb < 50000:    # ~34-45B models
        return 64
    elif size_mb < 80000:    # ~70B models
        return 80
    elif size_mb < 150000:   # ~100-120B models
        return 96
    else:                    # Larger models
        return 128


def calculate_vram(
    model_info: GGUFModelInfo,
    available_vram_mb: float,
    headroom_mb: float = 2048,
    context_size: int = 4096,
    kv_cache_quant: str = "f16"
) -> VRAMCalculation:
    """
    Calculate recommended GPU layers and VRAM usage.

    Args:
        model_info: Model information from get_gguf_model_info().
        available_vram_mb: Total available VRAM in MB.
        headroom_mb: VRAM to reserve for other applications.
        context_size: Context window size for KV cache calculation.
        kv_cache_quant: KV cache quantization type ("f16", "q8_0", or "q4_0").

    Returns:
        VRAMCalculation with detailed results.
    """
    num_layers = model_info.num_layers or 32
    model_size_mb = model_info.file_size_mb

    # Calculate approximate MB per layer
    # Model weights are distributed across layers plus embeddings
    # Embeddings typically account for ~5-10% of model size
    embedding_overhead = 0.08  # 8% for embeddings and other non-layer weights
    layer_weights_mb = model_size_mb * (1 - embedding_overhead)
    mb_per_layer = layer_weights_mb / num_layers

    # Estimate KV cache size
    # KV cache depends on context size, head dimensions, and number of KV heads
    context_overhead_mb = _estimate_context_overhead(
        model_info,
        context_size,
        kv_cache_quant
    )

    # Calculate usable VRAM
    usable_vram = available_vram_mb - headroom_mb - context_overhead_mb

    if usable_vram <= 0:
        return VRAMCalculation(
            recommended_layers=0,
            total_layers=num_layers,
            mb_per_layer=mb_per_layer,
            available_vram_mb=available_vram_mb,
            headroom_mb=headroom_mb,
            context_overhead_mb=context_overhead_mb,
            estimated_usage_mb=0,
            fits_entirely=False,
            error="Not enough VRAM after headroom and context allocation"
        )

    # Calculate how many layers fit
    calculated_layers = int(usable_vram / mb_per_layer)
    recommended = min(calculated_layers, num_layers)

    # Estimate actual VRAM usage
    estimated_usage = (recommended * mb_per_layer) + context_overhead_mb

    return VRAMCalculation(
        recommended_layers=recommended,
        total_layers=num_layers,
        mb_per_layer=mb_per_layer,
        available_vram_mb=available_vram_mb,
        headroom_mb=headroom_mb,
        context_overhead_mb=context_overhead_mb,
        estimated_usage_mb=estimated_usage,
        fits_entirely=calculated_layers >= num_layers,
        error=None
    )


def _estimate_context_overhead(
    model_info: GGUFModelInfo,
    context_size: int,
    kv_cache_quant: str = "f16"
) -> float:
    """
    Estimate VRAM needed for KV cache and context.

    Args:
        model_info: Model information.
        context_size: Context window size.
        kv_cache_quant: KV cache quantization type ("f16", "q8_0", or "q4_0").

    Returns:
        Estimated context overhead in MB.
    """
    # KV cache size formula:
    # 2 * num_layers * context_size * num_kv_heads * head_dim * bytes_per_element
    # Bytes per element depends on quantization:
    # - F16: 2 bytes
    # - Q8_0: 1 byte
    # - Q4_0: 0.5 bytes

    bytes_per_element = {
        "f16": 2.0,
        "q8_0": 1.0,
        "q4_0": 0.5
    }.get(kv_cache_quant.lower(), 2.0)

    num_layers = model_info.num_layers or 32
    embedding_length = model_info.embedding_length or 4096
    head_count = model_info.head_count or 32
    head_count_kv = model_info.head_count_kv or head_count

    # Head dimension
    head_dim = embedding_length // head_count if head_count > 0 else 128

    # KV cache: 2 (K and V) * layers * context * kv_heads * head_dim * bytes_per_element
    kv_cache_bytes = 2 * num_layers * context_size * head_count_kv * head_dim * bytes_per_element
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)

    # Add some overhead for activations during inference (~10% of KV cache)
    activation_overhead = kv_cache_mb * 0.1

    return kv_cache_mb + activation_overhead


def format_size(size_mb: float) -> str:
    """
    Format size in MB to human-readable string.

    Args:
        size_mb: Size in megabytes.

    Returns:
        Formatted string (e.g., "8.5 GB" or "512 MB").
    """
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb:.0f} MB"
