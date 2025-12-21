"""
Utility functions for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import json
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog


# Config file location
CONFIG_FILE = Path.home() / ".gguf_converter_config.json"


def get_default_config():
    """Get default configuration"""
    return {
        # Sidebar settings
        "verbose": True,
        "use_imatrix": True,
        "nthreads": None,  # None = auto-detect
        "ignore_incompatibilities": False,  # Allow incompatible quantizations (advanced users only)

        # Imatrix mode (on Convert & Quantize tab)
        "imatrix_mode": "generate",  # "generate", "generate_custom", or "reuse"
        "imatrix_generate_name": "",  # Custom filename for generated imatrix (empty = auto)
        "imatrix_reuse_path": "",  # Filename of imatrix to reuse from output directory

        # Imatrix Settings tab
        "imatrix_ctx_size": 512,
        "imatrix_chunks": 150,  # 100-200 recommended, 0 = all chunks
        "imatrix_collect_output_weight": False,
        "imatrix_calibration_file": "_default.txt",  # Selected calibration file from the directory
        "imatrix_calibration_dir": "",  # Directory to scan for calibration files (empty = use default)
        "imatrix_from_chunk": 0,  # Skip first N chunks
        "imatrix_no_ppl": False,  # Disable perplexity
        "imatrix_parse_special": False,  # Parse special tokens
        "imatrix_output_frequency": 10,  # Save interval
        "imatrix_ngl": 0,  # GPU layers (0 = CPU only)
        "imatrix_stats_model": "",  # Model for statistics utility
        "imatrix_stats_path": "",  # Imatrix file for statistics

        # Convert & Quantize tab
        "model_path": "",
        "output_dir": "",
        "intermediate_type": "F16",
        "allow_requantize": False,
        "leave_output_tensor": False,
        "pure_quantization": False,
        "keep_split": False,

        # Quantization types - all stored in other_quants dict
        "other_quants": {
            "Q4_K_M": True,  # Default to Q4_K_M
        },

        # Saved states for unquantized format checkboxes (before they get disabled as intermediate)
        "unquantized_saved_states": {},

        # Saved states for quantizations disabled due to incompatibility
        "incompatible_saved_states": {},

        # Download tab
        "repo_id": "",
        "download_dir": "",

        # Custom binaries
        "use_custom_binaries": False,
        "custom_binaries_folder": ""
    }


def load_config():
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)

            config = get_default_config()
            config.update(saved_config)
            return config
        except Exception as e:
            print(f"Warning: Could not load config: {e}", flush=True)
            return get_default_config()
    return get_default_config()


def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}", flush=True)


def make_config_saver(config, config_key, session_key):
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
    def save():
        config[config_key] = st.session_state[session_key]
        save_config(config)
    return save


def reset_config():
    """Reset configuration to defaults"""
    config = get_default_config()
    save_config(config)
    return config


def strip_quotes(path_str):
    """
    Strip surrounding quotes from a path string (Windows "Copy as path" adds them)

    Args:
        path_str: Path string that may have quotes

    Returns:
        Path string without surrounding quotes
    """
    if not path_str:
        return path_str
    return path_str.strip().strip('"').strip("'")


def open_folder(folder_path):
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


def get_current_version():
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


def check_git_updates_available():
    """
    Check if git updates are available from remote

    Returns:
        dict: Status info with keys 'status', 'message', 'behind_count'
    """
    try:
        # Fetch remote refs without updating local branches
        result = subprocess.run(
            ["git", "fetch", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Check how many commits behind
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            behind_count = int(result.stdout.strip()) if result.stdout.strip() else 0
            if behind_count > 0:
                return {
                    "status": "updates_available",
                    "message": f"Updates available ({behind_count} commit{'s' if behind_count > 1 else ''} behind)",
                    "behind_count": behind_count
                }
            else:
                return {
                    "status": "up_to_date",
                    "message": "You are on the latest version of YaGUFF",
                    "behind_count": 0
                }
        else:
            return {
                "status": "unknown",
                "message": "Use 'Check for Updates' to pull the latest version from GitHub",
                "behind_count": 0
            }
    except Exception:
        return {
            "status": "unknown",
            "message": "Use 'Check for Updates' to pull the latest version from GitHub",
            "behind_count": 0
        }


def browse_folder(initial_dir=None):
    """
    Open a native folder picker dialog

    Args:
        initial_dir: Initial directory to open (optional)

    Returns:
        str: Selected folder path or None if cancelled
    """
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
        return folder_path if folder_path else None
    except Exception as e:
        print(f"Error opening folder picker: {e}")
        return None


def get_binary_version(converter):
    """
    Get version of llama.cpp binaries

    Args:
        converter: GGUFConverter instance

    Returns:
        dict: Binary version info
    """
    try:
        # Try to get version from llama-cli
        try:
            cli_path = converter.binary_manager.get_binary_path('llama-cli')
            result = subprocess.run(
                [str(cli_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse version from output (llama-cli outputs to stderr)
                output = result.stderr if result.stderr else result.stdout
                if output:
                    # Extract just the version line
                    for line in output.split('\n'):
                        if line.startswith('version:'):
                            version_line = line.strip()
                            return {
                                "status": "ok",
                                "version": version_line,
                                "message": "Binaries are installed"
                            }
                    # If no version line found, return full output
                    version_line = output.strip().split('\n')[0]
                    return {
                        "status": "ok",
                        "version": version_line,
                        "message": "Binaries are installed"
                    }
        except Exception:
            pass

        # Check if binaries exist
        bin_dir = converter.binary_manager.bin_dir
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


def get_binary_version_from_path(binary_path):
    """
    Get version from a specific binary path by running it with --version

    Args:
        binary_path: Path to the binary (e.g., llama-quantize or llama-imatrix)

    Returns:
        str: Version string or None if unable to get version
    """
    try:
        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5
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


def display_binary_version_status(converter):
    """
    Display binary version information with status message

    Shows version code block and status message comparing installed vs expected version

    Args:
        converter: GGUFConverter instance

    Returns:
        None (displays directly using Streamlit)
    """
    binary_info = get_binary_version(converter)
    expected_version = converter.binary_manager.LLAMA_CPP_VERSION

    if binary_info["status"] == "ok" and binary_info["version"]:
        st.code(binary_info["version"], language=None)

        # Show version match status
        expected_numeric = expected_version.lstrip('b')
        if expected_numeric not in binary_info["version"]:
            # Show mismatch warning
            st.warning(f"Expected binary version: **{expected_version}**")
        else:
            # Show positive confirmation when versions match
            st.info("You are on the latest YaGUFF-tested llama.cpp binary")
    elif binary_info["status"] == "missing":
        st.info(f"Expected binary version: **{expected_version}**")


def run_and_stream_command(command):
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
