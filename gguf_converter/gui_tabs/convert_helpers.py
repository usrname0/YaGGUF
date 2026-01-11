"""
Helper functions for the Convert & Quantize tab
"""

import re
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
from colorama import Style
from ..theme import THEME as theme


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing problematic characters.

    Removes:
    - Leading/trailing whitespace
    - Invalid Windows filename characters: < > : " / \\ | ? *
    - Control characters
    - Collapses multiple spaces to single space

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename safe for all platforms
    """
    if not filename:
        return filename

    # Strip leading/trailing whitespace
    clean = filename.strip()

    # Remove invalid filename characters (Windows is most restrictive)
    # < > : " / \ | ? * and control characters
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', clean)

    # Collapse multiple spaces to single space
    clean = re.sub(r'\s+', ' ', clean)

    # Strip again in case we created leading/trailing spaces
    clean = clean.strip()

    return clean


def detect_dtype_from_safetensors(safetensors_file: Path) -> Optional[str]:
    """
    Read dtype directly from safetensors file header (source of truth).

    This reads only the first few KB of the file to extract the dtype from
    the safetensors metadata, avoiding the need to load the entire model.

    Args:
        safetensors_file: Path to a .safetensors file

    Returns:
        Dtype string like "BF16", "F16", "F32", "F8_E4M3", or None if unable to detect
    """
    import struct

    try:
        with open(safetensors_file, 'rb') as f:
            # Read first 8 bytes to get header length (uint64 little-endian)
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return None

            header_len = struct.unpack('<Q', header_len_bytes)[0]

            # Safety check: header shouldn't be unreasonably large
            if header_len > 10_000_000:  # 10MB max
                return None

            # Read the JSON header
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes.decode('utf-8'))

            # Find the first actual tensor (skip __metadata__)
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                # Found a tensor, return its dtype
                dtype_raw = info.get('dtype', '').upper()

                # Map safetensors dtypes to our format
                dtype_map = {
                    "BF16": "BF16",
                    "F16": "F16",
                    "F32": "F32",
                    "F64": "F64",
                    "F8_E4M3": "FP8",
                    "F8_E5M2": "FP8",
                }

                return dtype_map.get(dtype_raw, dtype_raw)

    except Exception:
        pass

    return None


def detect_source_dtype(model_path: Path) -> Optional[str]:
    """
    Detect the data type (precision) of source model.

    Uses a two-tier approach:
    1. Check config.json (fast, covers most cases)
    2. Fallback to safetensors header (source of truth)

    Args:
        model_path: Path to model directory

    Returns:
        Dtype string like "BF16", "F16", "F32", "FP8", "Pre-Quantized (GPTQ)", etc.
    """
    config_json = model_path / "config.json"
    dtype_from_config = None

    # Level 1: Try config.json first
    if config_json.exists():
        try:
            with open(config_json, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

                # Check for pre-quantized models (GPTQ, AWQ, etc.)
                if "quantization_config" in config_data:
                    quant_method = config_data["quantization_config"].get("quant_method", "unknown")
                    return f"Pre-Quantized ({quant_method.upper()})"

                # Check multiple possible locations for dtype
                # 1. Top-level dtype or torch_dtype (most models)
                dtype_value = config_data.get("dtype", config_data.get("torch_dtype", ""))

                # 2. For multimodal models, check text_config.dtype
                if not dtype_value and "text_config" in config_data:
                    dtype_value = config_data["text_config"].get("dtype",
                                  config_data["text_config"].get("torch_dtype", ""))

                # 3. For vision-only models, check vision_config.dtype
                if not dtype_value and "vision_config" in config_data:
                    dtype_value = config_data["vision_config"].get("dtype",
                                  config_data["vision_config"].get("torch_dtype", ""))

                dtype_value = dtype_value.lower() if dtype_value else ""

                # Map torch dtypes to our format
                dtype_map = {
                    "bfloat16": "BF16",
                    "float16": "F16",
                    "float32": "F32",
                    "float64": "F64",
                }

                if dtype_value in dtype_map:
                    dtype_from_config = dtype_map[dtype_value]
        except Exception:
            pass

    # If we got a dtype from config, return it
    if dtype_from_config:
        return dtype_from_config

    # Level 2: Fallback to safetensors header (source of truth)
    # Find any safetensors file in the directory
    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        # Check for model-*.safetensors pattern
        safetensors_files = list(model_path.glob("model*.safetensors"))

    if safetensors_files:
        # Use the first safetensors file to detect dtype
        return detect_dtype_from_safetensors(safetensors_files[0])

    return None


def validate_calibration_file(config: Dict[str, Any]) -> Optional[Path]:
    """
    Validate and return the calibration file path.

    Builds the path from config, validates it exists, and shows errors if not found.

    Args:
        config: Configuration dictionary

    Returns:
        Path to calibration file if valid, None if not found
    """
    cal_dir = config.get("imatrix_calibration_dir", "")
    cal_file = config.get("imatrix_calibration_file", "wiki.train.raw")

    if cal_dir:
        calibration_file_path = Path(cal_dir) / cal_file
    else:
        # Use default calibration_data directory (one level up from gguf_converter module)
        default_cal_dir = Path(__file__).parent.parent.parent / "calibration_data"
        calibration_file_path = default_cal_dir / cal_file

    # Validate calibration file exists
    if not calibration_file_path.exists():
        # Print detailed error to terminal
        print(f"\n{theme['error']}Calibration file not found:{Style.RESET_ALL}")
        print(f"{theme['error']}  Looking for: {calibration_file_path}{Style.RESET_ALL}")
        print(f"{theme['error']}  Config has: {cal_file}{Style.RESET_ALL}")
        if cal_dir:
            print(f"{theme['error']}  In directory: {cal_dir}{Style.RESET_ALL}")
        print(f"{theme['warning']}  Tip: Go to Imatrix Settings tab and reselect your calibration file{Style.RESET_ALL}\n")

        # Show detailed error in GUI
        st.error(f"**Calibration file not found:** `{cal_file}`\n\n"
               f"Expected path: `{calibration_file_path}`\n\n"
               f"**Fix:** Go to the **Imatrix Settings** tab and select a valid calibration file, "
               f"use an existing imatrix, or turn off imatrix generation.")
        return None

    return calibration_file_path


def detect_intermediate_gguf_files(model_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Detect intermediate GGUF files in model directory.

    Looks for files matching patterns:
    - Single: {anything}_F16.gguf, {anything}_F32.gguf, {anything}_BF16.gguf
    - Split: {anything}_F16-00001-of-00003.gguf, etc.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary mapping format+type to file info:
        {
            'F16_single': {
                'format': 'F16',
                'type': 'single',
                'files': [Path objects sorted],
                'primary_file': Path (first file or single file),
                'shard_count': int (1 for single, N for split),
                'total_size_gb': float
            },
            'F16_split': {
                'format': 'F16',
                'type': 'split',
                ...
            },
            ...
        }
    """
    intermediates = {}

    # Pattern for intermediate files
    single_pattern = re.compile(r'^(.+)_(F16|F32|BF16)\.gguf$')
    split_pattern = re.compile(r'^(.+)_(F16|F32|BF16)-(\d+)-of-(\d+)\.gguf$')

    # Scan directory for GGUF files
    if not model_path.exists() or not model_path.is_dir():
        return {}

    for gguf_file in model_path.glob("*.gguf"):
        # Skip mmproj files (vision model projectors) - they're not intermediates
        if gguf_file.name.startswith("mmproj-"):
            continue

        # Try split pattern first (more specific)
        split_match = split_pattern.match(gguf_file.name)
        if split_match:
            format_type = split_match.group(2)
            shard_num = int(split_match.group(3))
            total_shards = int(split_match.group(4))

            # Use format+type as key to allow both single and split of same format
            key = f"{format_type}_split"

            if key not in intermediates:
                intermediates[key] = {
                    'format': format_type,
                    'type': 'split',
                    'files': [],
                    'shard_numbers': [],
                    'total_expected': total_shards
                }

            intermediates[key]['files'].append(gguf_file)
            intermediates[key]['shard_numbers'].append(shard_num)
        else:
            # Try single pattern
            single_match = single_pattern.match(gguf_file.name)
            if single_match:
                format_type = single_match.group(2)

                # Use format+type as key to allow both single and split of same format
                key = f"{format_type}_single"

                # Add single file (multiple single files of same format will overwrite, which is expected)
                if key not in intermediates:
                    intermediates[key] = {
                        'format': format_type,
                        'type': 'single',
                        'files': [gguf_file]
                    }

    # Validate and finalize each format
    validated = {}
    for key, info in intermediates.items():
        if info['type'] == 'split':
            # Sort files by shard number
            sorted_files = sorted(info['files'], key=lambda p:
                int(split_pattern.match(p.name).group(3)))

            # Validate complete set
            expected = set(range(1, info['total_expected'] + 1))
            found = set(info['shard_numbers'])

            if expected == found:
                # Complete set
                validated[key] = {
                    'format': info['format'],
                    'type': 'split',
                    'files': sorted_files,
                    'primary_file': sorted_files[0],
                    'shard_count': len(sorted_files),
                    'total_size_gb': sum(f.stat().st_size for f in sorted_files) / (1024**3)
                }
        else:
            # Single file
            file = info['files'][0]
            validated[key] = {
                'format': info['format'],
                'type': 'single',
                'files': [file],
                'primary_file': file,
                'shard_count': 1,
                'total_size_gb': file.stat().st_size / (1024**3)
            }

    return validated
