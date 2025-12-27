"""
Utility functions for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import json
import subprocess
import platform
from typing import Dict, Optional, Tuple, Any, Callable, List

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
           'save_config', 'load_config', 'make_config_saver', 'path_input_columns']


# Config file location
CONFIG_FILE = Path.home() / ".gguf_converter_config.json"


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        # Sidebar settings
        "verbose": True,
        "use_imatrix": True,
        "nthreads": None,  # None = auto-detect
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
        "imatrix_ngl": 0,  # GPU layers (0 = CPU only)
        "imatrix_stats_model": "",  # Model for statistics utility
        "imatrix_stats_path": "",  # Imatrix file for statistics
        "max_preview_lines": 1000,  # Maximum lines to show in calibration preview
        "preview_height": 700,  # Preview area height in pixels

        # Convert & Quantize tab
        "model_path": "",
        "output_dir": "",
        "intermediate_type": "F16",
        "allow_requantize": False,
        "leave_output_tensor": False,
        "pure_quantization": False,
        "keep_split": False,
        "output_tensor_type": "Same as quantization type",
        "token_embedding_type": "Same as quantization type",

        # Quantization types - all stored in other_quants dict
        "other_quants": {
            "Q4_K_M": True,  # Default to Q4_K_M
        },

        # Saved states for unquantized format checkboxes (before they get disabled as intermediate)
        "unquantized_saved_states": {},

        # Download tab
        "repo_id": "",
        "download_dir": "",

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
            print(f"Warning: Could not load config: {e}", flush=True)
            return get_default_config()
    return get_default_config()


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}", flush=True)


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
        import json
        import urllib.request
        import os

        # Get current version
        current_version = get_current_version()

        # Check GitHub Releases API for latest release
        url = "https://api.github.com/repos/usrname0/YaGUFF/releases/latest"
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
                        "message": "You are on the latest version of YaGUFF",
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
        return folder_path if folder_path else None
    except Exception as e:
        print(f"Error opening folder picker: {e}")
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
        except (FileNotFoundError, subprocess.SubprocessError, OSError, PermissionError):
            # llama-cli not found or not executable - that's fine, we'll check bin_dir existence next
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


def get_conversion_scripts_info(converter: Any) -> Dict[str, Any]:
    """
    Get version info for conversion scripts

    Args:
        converter: GGUFConverter instance

    Returns:
        dict: Conversion scripts info with status and message
    """
    from pathlib import Path
    import subprocess

    project_root = Path(__file__).parent.parent
    llama_cpp_dir = project_root / "llama.cpp"

    if not llama_cpp_dir.exists() or not (llama_cpp_dir / ".git").exists():
        return {
            "status": "missing",
            "version": None,
            "message": "Conversion scripts repository not found"
        }

    try:
        # Get current commit info
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h - %s (%cr)"],
            cwd=llama_cpp_dir,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            return {
                "status": "ok",
                "version": result.stdout.strip(),
                "message": "Conversion scripts repository found"
            }
        else:
            return {
                "status": "error",
                "version": None,
                "message": "Could not read conversion scripts version"
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
    from pathlib import Path
    import subprocess

    scripts_info = get_conversion_scripts_info(converter)
    expected_version = converter.binary_manager.LLAMA_CPP_VERSION

    if scripts_info["status"] == "ok":
        # Check and display current tag/version
        try:
            project_root = Path(__file__).parent.parent
            llama_cpp_dir = project_root / "llama.cpp"

            result = subprocess.run(
                ["git", "describe", "--tags", "--always"],
                cwd=llama_cpp_dir,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                current_version = result.stdout.strip()
                expected_numeric = expected_version.lstrip('b')

                # Show version tag in code block
                st.code(f"version: {current_version}", language=None)

                # Show status message
                if expected_numeric in current_version:
                    st.info("You are on the latest YaGUFF-tested conversion scripts")
                else:
                    st.warning(f"Expected conversion scripts version: **{expected_version}**")
            else:
                st.info(f"Expected conversion scripts version: **{expected_version}**")
        except Exception:
            st.info(f"Expected conversion scripts version: **{expected_version}**")
    elif scripts_info["status"] == "missing":
        st.info(f"Expected conversion scripts version: **{expected_version}**")


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
