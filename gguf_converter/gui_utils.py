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

            # Merge with defaults to handle new settings
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

    # If it's a file, get its parent directory
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
    Get current version from VERSION file

    Returns:
        str: Version string or "unknown" if file not found
    """
    try:
        version_file = Path(__file__).parent.parent / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip()
        return "unknown"
    except Exception:
        return "unknown"


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
