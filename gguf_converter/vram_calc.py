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
    model_size_mb: float
    available_vram_mb: float
    headroom_mb: float
    context_overhead_mb: float
    usable_vram_mb: float
    estimated_usage_mb: float
    fits_entirely: bool
    offload_percentage: float
    error: Optional[str] = None


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

    # Try to read metadata from GGUF
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(filepath))

        # Get architecture first (needed for architecture-specific keys)
        arch = None
        for field in reader.fields.values():
            if field.name == "general.architecture":
                arch = _extract_field_value(field)
                if isinstance(arch, bytes):
                    arch = arch.decode('utf-8')
                info.architecture = arch
                break

        # Map of metadata keys to extract
        # Keys can be architecture-specific (e.g., "llama.block_count")
        # or general (e.g., "general.context_length")
        key_mappings = {
            "block_count": "num_layers",
            "context_length": "context_length",
            "embedding_length": "embedding_length",
            "attention.head_count": "head_count",
            "attention.head_count_kv": "head_count_kv",
            "vocab_size": "vocab_size",
        }

        for field in reader.fields.values():
            name = field.name

            # Check for architecture-specific keys
            for key_suffix, attr_name in key_mappings.items():
                # Try both "{arch}.{key}" and "general.{key}" patterns
                if name.endswith(f".{key_suffix}") or name == f"general.{key_suffix}":
                    value = _extract_field_value(field)
                    if value is not None and isinstance(value, (int, float)):
                        setattr(info, attr_name, int(value))
                    break

            # Get file type (quantization type)
            if name == "general.file_type":
                value = _extract_field_value(field)
                if value is not None:
                    info.file_type = _get_file_type_name(int(value))
                    info.quantization_version = int(value)

    except ImportError:
        pass  # gguf library not installed
    except Exception:
        pass  # Failed to read metadata

    # Fallback: estimate layers from file size if not found
    if info.num_layers is None:
        info.num_layers = _estimate_layers_from_size(file_size_mb)

    return info


def _extract_field_value(field: Any) -> Any:
    """
    Extract the actual value from a GGUF field.

    Args:
        field: GGUF field object.

    Returns:
        Extracted value, or None if extraction fails.
    """
    try:
        # The field structure has 'parts' where the value is typically in the last part
        if hasattr(field, 'parts') and len(field.parts) > 0:
            val_part = field.parts[-1]

            # Handle numpy arrays
            if hasattr(val_part, 'tolist'):
                val = val_part.tolist()
                if isinstance(val, list):
                    if len(val) == 1:
                        return val[0]
                    elif len(val) > 0:
                        # For strings stored as byte arrays
                        if all(isinstance(x, int) and 0 <= x < 256 for x in val[:10]):
                            try:
                                return bytes(val).rstrip(b'\x00')
                            except (ValueError, TypeError):
                                pass
                        return val[0]
                return val

            return val_part
    except Exception:
        pass

    return None


def _get_file_type_name(file_type: int) -> str:
    """
    Convert GGUF file type enum to human-readable name.

    Args:
        file_type: GGUF file type integer.

    Returns:
        Human-readable quantization type name.
    """
    # Common GGUF file types
    file_types = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1",
        8: "Q8_0",
        9: "Q8_1",
        10: "Q2_K",
        11: "Q3_K_S",
        12: "Q3_K_M",
        13: "Q3_K_L",
        14: "Q4_K_S",
        15: "Q4_K_M",
        16: "Q5_K_S",
        17: "Q5_K_M",
        18: "Q6_K",
        19: "Q8_K",
        20: "IQ2_XXS",
        21: "IQ2_XS",
        22: "IQ3_XXS",
        23: "IQ1_S",
        24: "IQ4_NL",
        25: "IQ3_S",
        26: "IQ2_S",
        27: "IQ4_XS",
        28: "IQ1_M",
        29: "BF16",
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
    batch_size: int = 512
) -> VRAMCalculation:
    """
    Calculate recommended GPU layers and VRAM usage.

    Args:
        model_info: Model information from get_gguf_model_info().
        available_vram_mb: Total available VRAM in MB.
        headroom_mb: VRAM to reserve for other applications.
        context_size: Context window size for KV cache calculation.
        batch_size: Batch size for inference.

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
        batch_size
    )

    # Calculate usable VRAM
    usable_vram = available_vram_mb - headroom_mb - context_overhead_mb

    if usable_vram <= 0:
        return VRAMCalculation(
            recommended_layers=0,
            total_layers=num_layers,
            mb_per_layer=mb_per_layer,
            model_size_mb=model_size_mb,
            available_vram_mb=available_vram_mb,
            headroom_mb=headroom_mb,
            context_overhead_mb=context_overhead_mb,
            usable_vram_mb=usable_vram,
            estimated_usage_mb=0,
            fits_entirely=False,
            offload_percentage=0,
            error="Not enough VRAM after headroom and context allocation"
        )

    # Calculate how many layers fit
    calculated_layers = int(usable_vram / mb_per_layer)
    recommended = min(calculated_layers, num_layers)

    # Estimate actual VRAM usage
    estimated_usage = (recommended * mb_per_layer) + context_overhead_mb

    # Calculate offload percentage
    offload_pct = (recommended / num_layers) * 100 if num_layers > 0 else 0

    return VRAMCalculation(
        recommended_layers=recommended,
        total_layers=num_layers,
        mb_per_layer=mb_per_layer,
        model_size_mb=model_size_mb,
        available_vram_mb=available_vram_mb,
        headroom_mb=headroom_mb,
        context_overhead_mb=context_overhead_mb,
        usable_vram_mb=usable_vram,
        estimated_usage_mb=estimated_usage,
        fits_entirely=calculated_layers >= num_layers,
        offload_percentage=offload_pct,
        error=None
    )


def _estimate_context_overhead(
    model_info: GGUFModelInfo,
    context_size: int,
    batch_size: int
) -> float:
    """
    Estimate VRAM needed for KV cache and context.

    Args:
        model_info: Model information.
        context_size: Context window size.
        batch_size: Batch size.

    Returns:
        Estimated context overhead in MB.
    """
    # KV cache size formula:
    # 2 * num_layers * context_size * num_kv_heads * head_dim * bytes_per_element
    # For FP16 KV cache: bytes_per_element = 2

    num_layers = model_info.num_layers or 32
    embedding_length = model_info.embedding_length or 4096
    head_count = model_info.head_count or 32
    head_count_kv = model_info.head_count_kv or head_count

    # Head dimension
    head_dim = embedding_length // head_count if head_count > 0 else 128

    # KV cache: 2 (K and V) * layers * context * kv_heads * head_dim * 2 bytes (FP16)
    kv_cache_bytes = 2 * num_layers * context_size * head_count_kv * head_dim * 2
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)

    # Add some overhead for activations during inference (~10% of KV cache)
    activation_overhead = kv_cache_mb * 0.1

    # Batch size overhead (minimal for inference)
    batch_overhead = (batch_size * embedding_length * 2) / (1024 * 1024)

    return kv_cache_mb + activation_overhead + batch_overhead


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
