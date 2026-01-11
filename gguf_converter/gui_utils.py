"""
Utility functions for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import json
import subprocess
import platform
from typing import Dict, Optional, Tuple, Any, Callable, List
from colorama import Style
from .theme import THEME as theme

# Optional tkinter import for native file dialogs
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None
    filedialog = None

# Export TKINTER_AVAILABLE for use in other modules
__all__ = ['TKINTER_AVAILABLE', 'browse_folder', 'open_folder', 'strip_quotes',
           'save_config', 'load_config', 'make_config_saver', 'path_input_columns',
           'extract_repo_id_from_url', 'get_platform_path', 'CONFIG_FILE', 'HF_TOKEN_PATH']


# Config file location
CONFIG_FILE = Path.home() / ".gguf_converter_config.json"

# HuggingFace token location (managed by huggingface_hub)
try:
    from huggingface_hub.constants import HF_TOKEN_PATH
except ImportError:
    # Fallback for older versions or if constant is moved
    HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        # Sidebar settings
        "verbose": True,
        "use_imatrix": True,
        "num_threads": None,  # None = auto-detect
        "ignore_imatrix_warnings": False,  # Allow IQ quants without imatrix

        # Imatrix mode (on Convert & Quantize tab)
        "imatrix_mode": "generate",  # "generate", "generate_custom", or "reuse"
        "imatrix_generate_name": "",  # Custom filename for generated imatrix (empty = auto)
        "imatrix_reuse_path": "",  # Filename of imatrix to reuse from output directory

        # Imatrix Settings tab
        "imatrix_ctx_size": 512,
        "imatrix_chunks": 150,  # 100-200 recommended, 0 = all chunks
        "imatrix_collect_output_weight": False,
        "imatrix_calibration_file": "wiki.train.raw",  # Selected calibration file from the directory
        "imatrix_calibration_dir": "",  # Directory to scan for calibration files (empty = use default)
        "imatrix_from_chunk": 0,  # Skip first N chunks
        "imatrix_no_ppl": False,  # Disable perplexity
        "imatrix_parse_special": False,  # Parse special tokens
        "imatrix_output_frequency": 10,  # Save interval
        "imatrix_num_gpu_layers": 0,  # GPU layers (0 = CPU only)
        "imatrix_stats_model": "",  # Model for statistics utility
        "imatrix_stats_path": "",  # Imatrix file for statistics
        "max_preview_lines": 1000,  # Maximum lines to show in calibration preview
        "preview_height": 700,  # Preview area height in pixels

        # Convert & Quantize tab
        "model_path": "",
        "output_dir": "",
        "intermediate_type": "F16",
        "pure_quantization": False,
        "file_mode": "Single files",
        "max_shard_size_gb": 2.0,
        "output_tensor_type": "Same as quant type (default)",
        "token_embedding_type": "Same as quant type (default)",

        # File handling options
        "overwrite_intermediates": True,  # Overwrite F32/F16/BF16 by default
        "overwrite_quants": True,  # Overwrite quantized formats by default

        # Quantization types - all stored in other_quants dict
        "other_quants": {
            "Q4_K_M": True,  # Default to Q4_K_M
        },

        # Saved states for unquantized format checkboxes (before they get disabled as intermediate)
        "unquantized_saved_states": {},

        # Download tab
        "repo_id": "",
        "download_dir": "",

        # Merge tab (deprecated - now in Split/Merge tab)
        "merge_input_dir": "",
        "merge_output_dir": "",

        # Split/Merge tab
        "split_merge_input_dir": "",
        "split_merge_selected_file": None,
        "split_merge_output_dir": "",
        "split_merge_operation_mode": "Split",
        "split_merge_max_shard_size_gb": 2.0,
        "split_merge_copy_aux_files": True,

        # Custom intermediate (used when selecting existing intermediate file)
        "custom_intermediate_mode": None,
        "custom_intermediate_format": None,
        "custom_intermediate_path": None,
        "custom_intermediate_file_type": None,
        "custom_intermediate_saved_intermediate_type": None,

        # Imatrix statistics output
        "imatrix_stats_output_dir": "",

        # Custom binaries
        "use_custom_binaries": False,
        "custom_binaries_folder": "",

        # Custom conversion script
        "use_custom_conversion_script": False,
        "custom_llama_cpp_repo": ""
    }


def load_config() -> Dict[str, Any]:
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)

            config = get_default_config()
            config.update(saved_config)
            return config
        except Exception as e:
            print(f"{theme['warning']}Warning: Could not load config: {e}{Style.RESET_ALL}", flush=True)
            return get_default_config()
    return get_default_config()


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"{theme['warning']}Warning: Could not save config: {e}{Style.RESET_ALL}", flush=True)


def make_config_saver(config: Dict[str, Any], config_key: str, session_key: str) -> Callable[[], None]:
    """
    Factory function to create config save callbacks for Streamlit widgets

    Args:
        config: Config dictionary to update
        config_key: Key in config dict to update
        session_key: Key in st.session_state to read from

    Returns:
        Callback function that saves the value from session state to config

    Example:
        st.checkbox("Enable feature",
                    key="my_feature_checkbox",
                    on_change=make_config_saver(config, "my_feature", "my_feature_checkbox"))
    """
    def save() -> None:
        if session_key in st.session_state:
            config[config_key] = st.session_state[session_key]
            save_config(config)
    return save


def reset_config() -> Dict[str, Any]:
    """Reset configuration to defaults"""
    config = get_default_config()
    save_config(config)
    return config


def strip_quotes(path_str: str | None) -> str:
    """
    Strip surrounding quotes from a path string (Windows "Copy as path" adds them)

    Args:
        path_str: Path string that may have quotes

    Returns:
        Path string without surrounding quotes
    """
    if not path_str:
        return ""
    return path_str.strip().strip('"').strip("'")


def get_platform_path(windows_path: str, unix_path: str) -> str:
    """
    Return platform-appropriate path string for placeholders

    Args:
        windows_path: Path string with Windows-style separators (backslashes)
        unix_path: Path string with Unix-style separators (forward slashes)

    Returns:
        The appropriate path string for the current platform

    Example:
        >>> placeholder = get_platform_path("C:\\Models\\output", "/home/user/Models/output")
    """
    return windows_path if platform.system() == "Windows" else unix_path


def open_folder(folder_path: str) -> None:
    """
    Open folder in file explorer (platform-specific)

    Args:
        folder_path: Path to folder to open
    """
    path = Path(strip_quotes(folder_path))

    if path.is_file():
        path = path.parent

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    system = platform.system()

    # Note: We don't use check=True because some file explorers (notably Windows explorer)
    # can return non-zero exit codes even when they successfully open the folder
    if system == "Windows":
        subprocess.run(["explorer", str(path.resolve())])
    elif system == "Darwin":  # macOS
        subprocess.run(["open", str(path.resolve())])
    else:  # Linux and others
        subprocess.run(["xdg-open", str(path.resolve())])


def get_current_version() -> str:
    """
    Get current version from __init__.py

    Returns:
        str: Version string or "unknown" if not found
    """
    try:
        from gguf_converter import __version__
        return __version__
    except Exception:
        return "unknown"


def check_git_updates_available() -> Dict[str, Any]:
    """
    Check if updates are available from GitHub Releases

    Supports authentication via GITHUB_TOKEN environment variable for private repos

    Returns:
        dict: Status info with keys 'status', 'message', 'latest_version'
    """
    try:
        import urllib.request
        import os

        # Get current version
        current_version = get_current_version()

        # Check GitHub Releases API for latest release
        url = "https://api.github.com/repos/usrname0/YaGGUF/releases/latest"
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/vnd.github.v3+json')

        # Add authentication if token is available
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            req.add_header('Authorization', f'token {github_token}')

        with urllib.request.urlopen(req, timeout=10) as response:
            release_data = json.loads(response.read().decode())
            latest_tag = release_data.get('tag_name', '')

            if latest_tag:
                # Compare versions (remove 'v' prefix if present)
                current = current_version.lstrip('v')
                latest = latest_tag.lstrip('v')

                if latest != current:
                    return {
                        "status": "updates_available",
                        "message": f"New version available: **{latest_tag}** (current: {current_version})",
                        "latest_version": latest_tag
                    }
                else:
                    return {
                        "status": "up_to_date",
                        "message": "You are on the latest version of YaGGUF",
                        "latest_version": current_version
                    }
            else:
                return {
                    "status": "unknown",
                    "message": "No releases found",
                    "latest_version": None
                }
    except Exception:
        return {
            "status": "unknown",
            "message": "Could not check for updates",
            "latest_version": None
        }


def path_input_columns() -> Tuple[List[Any], bool]:
    """
    Create column layout for path inputs with conditional browse button.

    Returns:
        tuple: (columns, has_browse_column)
            - columns: list of Streamlit column objects
            - has_browse_column: bool indicating if browse column exists

    Example:
        cols, has_browse = path_input_columns()
        with cols[0]:
            path = st.text_input("Path", ...)
        if has_browse:
            with cols[1]:
                if st.button("Browse", ...):
                    # browse logic
        with cols[-1]:  # Last column is always check button
            if st.button("Check Folder", ...):
                # check logic
    """
    if TKINTER_AVAILABLE:
        return st.columns([4, 1, 1]), True
    else:
        return st.columns([5, 1]), False


def browse_folder(initial_dir: Optional[str] = None) -> Optional[str]:
    """
    Open a native folder picker dialog

    Args:
        initial_dir: Initial directory to open (optional)

    Returns:
        str: Selected folder path or None if cancelled
    """
    if not TKINTER_AVAILABLE:
        st.error(
            "Folder browser requires tkinter.\n\n"
            "**On Linux**, install it with:\n"
            "- Ubuntu/Debian: `sudo apt install python3-tk`\n"
            "- Fedora/RHEL: `sudo dnf install python3-tkinter`\n"
            "- Arch: `sudo pacman -S tk`\n\n"
            "Then restart the GUI."
        )
        return None

    try:
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)

        # Open folder picker
        folder_path = filedialog.askdirectory(
            parent=root,
            initialdir=initial_dir,
            title="Select Folder"
        )

        root.destroy()
        # Convert to OS-native path separators (tkinter returns forward slashes on all platforms)
        if folder_path:
            return str(Path(folder_path))
        return None
    except Exception as e:
        print(f"{theme['error']}Error opening folder picker: {e}{Style.RESET_ALL}")
        return None


def get_binary_version(converter: Any) -> Dict[str, Any]:
    """
    Get version of llama.cpp binaries

    Args:
        converter: GGUFConverter instance

    Returns:
        dict: Binary version info
    """
    try:
        # Use llama_cpp_manager method to get version info
        version_info = converter.llama_cpp_manager.get_installed_version_info()

        if version_info['full_version']:
            return {
                "status": "ok",
                "version": version_info['full_version'],
                "message": "Binaries are installed"
            }

        # Check if binaries exist
        bin_dir = converter.llama_cpp_manager.bin_dir
        if bin_dir.exists() and any(bin_dir.iterdir()):
            return {
                "status": "ok",
                "version": "unknown",
                "message": "Binaries installed (version check unavailable)"
            }
        else:
            return {
                "status": "missing",
                "version": None,
                "message": "Binaries not installed. Run a conversion or use the update button to download them."
            }

    except Exception as e:
        return {
            "status": "error",
            "version": None,
            "message": f"Error checking binaries: {e}"
        }


def get_binary_version_from_path(binary_path: Optional[Path]) -> Optional[str]:
    """
    Get version from a specific binary path by running it with --version

    Args:
        binary_path: Path to the binary (e.g., llama-quantize or llama-imatrix)

    Returns:
        str: Version string or None if unable to get version
    """
    if binary_path is None:
        return None

    try:
        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            # Parse version from output (may be in stdout or stderr)
            output = result.stderr if result.stderr else result.stdout
            if output:
                # Extract just the version line
                for line in output.split('\n'):
                    if line.startswith('version:'):
                        return line.strip()
                # If no version line found, return first non-empty line
                first_line = output.strip().split('\n')[0]
                if first_line:
                    return first_line
        return None
    except Exception:
        return None


def display_binary_version_status(converter: Any) -> None:
    """
    Display binary version information with status message

    Shows version code block and status message comparing installed vs expected version

    Args:
        converter: GGUFConverter instance

    Returns:
        None (displays directly using Streamlit)
    """
    binary_info = get_binary_version(converter)
    expected_version = converter.llama_cpp_manager.LLAMA_CPP_VERSION

    if binary_info["status"] == "ok" and binary_info["version"]:
        st.code(binary_info["version"], language=None)

        # Show version match status
        expected_numeric = expected_version.lstrip('b')
        if expected_numeric not in binary_info["version"]:
            # Show mismatch warning
            st.warning(f"Expected binary version: **{expected_version}**")
        else:
            # Show positive confirmation when versions match
            st.info("You are on the latest YaGGUF-tested llama.cpp binary")
    elif binary_info["status"] == "missing":
        st.info(f"Expected binary version: **{expected_version}**")


def get_conversion_scripts_info(converter: Any) -> Dict[str, Any]:
    """
    Get version info for conversion scripts

    Args:
        converter: GGUFConverter instance

    Returns:
        dict: Conversion scripts info with status and message
    """
    from pathlib import Path

    try:
        # Use llama_cpp_manager method to get version info
        version_info = converter.llama_cpp_manager.get_installed_conversion_scripts_version_info()

        if version_info['full_version']:
            return {
                "status": "ok",
                "version": version_info['full_version'],
                "message": "Conversion scripts repository found"
            }

        # Check if directory exists
        project_root = Path(__file__).parent.parent
        llama_cpp_dir = project_root / "llama.cpp"

        if llama_cpp_dir.exists():
            return {
                "status": "ok",
                "version": "unknown",
                "message": "Conversion scripts installed (version check unavailable)"
            }
        else:
            return {
                "status": "missing",
                "version": None,
                "message": "Conversion scripts repository not found"
            }

    except Exception as e:
        return {
            "status": "error",
            "version": None,
            "message": f"Error checking conversion scripts: {e}"
        }


def display_conversion_scripts_version_status(converter: Any) -> None:
    """
    Display conversion scripts version information with status message

    Shows version code block and status message comparing installed vs expected version

    Args:
        converter: GGUFConverter instance

    Returns:
        None (displays directly using Streamlit)
    """
    scripts_info = get_conversion_scripts_info(converter)
    expected_version = converter.llama_cpp_manager.LLAMA_CPP_VERSION

    if scripts_info["status"] == "ok" and scripts_info["version"]:
        st.code(f"version: {scripts_info['version']}", language=None)

        # Show version match status
        expected_numeric = expected_version.lstrip('b')
        if expected_numeric in scripts_info["version"]:
            st.info("You are on the latest YaGGUF-tested conversion scripts")
        else:
            st.warning(f"Expected conversion scripts version: **{expected_version}**")
    elif scripts_info["status"] == "missing":
        st.info(f"Expected conversion scripts version: **{expected_version}**")


def extract_repo_id_from_url(url_or_repo_id: str) -> Optional[str]:
    """
    Extract repository ID from HuggingFace URL or return the repo ID if already in correct format

    Handles various HuggingFace URL formats:
    - https://huggingface.co/username/model-name
    - https://huggingface.co/username/model-name/tree/main
    - hf.co/username/model-name
    - username/model-name (already in correct format)

    Args:
        url_or_repo_id: HuggingFace URL or repository ID

    Returns:
        str: Extracted repository ID (username/model-name) or None if invalid

    Examples:
        >>> extract_repo_id_from_url("https://huggingface.co/meta-llama/Llama-3.2-3B")
        'meta-llama/Llama-3.2-3B'
        >>> extract_repo_id_from_url("meta-llama/Llama-3.2-3B")
        'meta-llama/Llama-3.2-3B'
    """
    import re

    if not url_or_repo_id:
        return None

    url_or_repo_id = url_or_repo_id.strip()

    # Check if it's a URL
    if "huggingface.co/" in url_or_repo_id or "hf.co/" in url_or_repo_id:
        # Match patterns like https://huggingface.co/username/model-name or hf.co/username/model-name
        match = re.search(r'(?:huggingface\.co|hf\.co)/([^/]+/[^/]+)', url_or_repo_id)
        if match:
            return match.group(1)
        return None

    # If not a URL, check if it's already in username/model-name format
    if re.match(r'^[^/]+/[^/]+$', url_or_repo_id):
        return url_or_repo_id

    return None


def run_and_stream_command(command: List[str]) -> None:
    """
    Runs a command and streams its output to a Streamlit container.

    Args:
        command (list): The command and its arguments.
    """
    st.info(f"Running command: `{' '.join(command)}`")
    output_container = st.empty()
    output_container.code("Starting process...", language='bash')

    full_output = ""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        for line in iter(process.stdout.readline, ''):
            full_output += line
            output_container.code(full_output, language='bash')

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            st.toast("Command completed successfully!")
        else:
            full_output += f"\n--- Command failed with exit code {return_code} ---"
            output_container.code(full_output, language='bash')
            st.toast(f"Command failed with exit code {return_code}")

    except FileNotFoundError:
        full_output = f"Error: Command not found: `{command[0]}`. Make sure it is in your PATH."
        output_container.code(full_output, language='bash')
        st.toast("Command not found.")
    except Exception as e:
        full_output += f"\n--- An error occurred ---\n{str(e)}"
        output_container.code(full_output, language='bash')
        st.toast(f"An error occurred: {e}")


def detect_all_model_files(model_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Detect all GGUF and safetensors files in a directory (both single and split).

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary mapping file identifier to file info:
        {
            'filename_single': {
                'type': 'single',
                'extension': 'gguf' or 'safetensors',
                'files': [Path object],
                'primary_file': Path,
                'shard_count': 1,
                'total_size_gb': float,
                'display_name': str  # Formatted for dropdown
            },
            'filename_split': {
                'type': 'split',
                'extension': 'gguf' or 'safetensors',
                'files': [Path objects sorted],
                'primary_file': Path,
                'shard_count': int,
                'total_size_gb': float,
                'display_name': str  # Formatted for dropdown
            },
            ...
        }
    """
    import re
    from collections import defaultdict

    if not model_path.exists() or not model_path.is_dir():
        return {}

    detected_files = {}

    # Detect GGUF files
    for extension in ['gguf', 'safetensors']:
        # Pattern for split files: {base}-00001-of-00003.{ext}
        split_pattern = re.compile(rf'^(.+)-(\d+)-of-(\d+)\.{extension}$')

        # Track split file groups
        split_groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'files': [],
            'shard_numbers': [],
            'total_expected': 0
        })

        all_files = list(model_path.glob(f"*.{extension}"))

        for file_path in all_files:
            # Try split pattern first
            split_match = split_pattern.match(file_path.name)
            if split_match:
                base_name = split_match.group(1)
                shard_num = int(split_match.group(2))
                total_shards = int(split_match.group(3))

                split_groups[base_name]['files'].append(file_path)
                split_groups[base_name]['shard_numbers'].append(shard_num)
                if split_groups[base_name]['total_expected'] == 0:
                    split_groups[base_name]['total_expected'] = total_shards
            else:
                # Single file
                key = f"{file_path.stem}_{extension}_single"
                file_size_gb = file_path.stat().st_size / (1024**3)

                # For safetensors, show full filename; for GGUF, show stem only
                if extension == 'safetensors':
                    display_name = f"{file_path.name} (single file, {file_size_gb:.2f} GB)"
                else:
                    display_name = f"{file_path.stem} (single file, {file_size_gb:.2f} GB)"

                detected_files[key] = {
                    'type': 'single',
                    'extension': extension,
                    'files': [file_path],
                    'primary_file': file_path,
                    'shard_count': 1,
                    'total_size_gb': file_size_gb,
                    'display_name': display_name
                }

        # Process split file groups
        for base_name, group_info in split_groups.items():
            # Sort files by shard number
            sorted_files = sorted(group_info['files'],
                                key=lambda p: int(split_pattern.match(p.name).group(2)))  # type: ignore[arg-type, union-attr]

            # Check if complete
            expected = set(range(1, group_info['total_expected'] + 1))
            found = set(group_info['shard_numbers'])

            if expected == found:
                # Complete set
                key = f"{base_name}_{extension}_split"
                total_size_gb = sum(f.stat().st_size for f in sorted_files) / (1024**3)

                # For safetensors, show full filename; for GGUF, show base name only
                if extension == 'safetensors':
                    display_name = f"{base_name}.{extension} ({len(sorted_files)} shards, {total_size_gb:.2f} GB)"
                else:
                    display_name = f"{base_name} ({len(sorted_files)} shards, {total_size_gb:.2f} GB)"

                detected_files[key] = {
                    'type': 'split',
                    'extension': extension,
                    'files': sorted_files,
                    'primary_file': sorted_files[0],
                    'shard_count': len(sorted_files),
                    'total_size_gb': total_size_gb,
                    'display_name': display_name
                }

    return detected_files
