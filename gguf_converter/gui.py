"""
Streamlit GUI for GGUF Converter
"""

import streamlit as st
from pathlib import Path
import sys
import json
import webbrowser
import subprocess
import platform

# Handle both direct execution and module import
try:
    from .converter import GGUFConverter
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gguf_converter.converter import GGUFConverter


# Config file location
CONFIG_FILE = Path.home() / ".gguf_converter_config.json"


def get_default_config():
    """Get default configuration"""
    return {
        # Sidebar settings
        "verbose": False,
        "use_imatrix": True,
        "nthreads": None,  # None = auto-detect

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
        "imatrix_stats_model": "",  # Model for statistics utility
        "imatrix_stats_path": "",  # Imatrix file for statistics

        # Convert & Quantize tab
        "hf_repo": "",
        "model_path": "",
        "output_dir": "",
        "intermediate_type": "F16",

        # Quantization types - all stored in other_quants dict
        "other_quants": {
            "Q4_K_M": True,  # Default to Q4_K_M
        },

        # Saved states for full precision checkboxes (before they get disabled as intermediate)
        "full_precision_saved_states": {},

        # Download tab
        "repo_id": "",
        "download_dir": ""
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


def check_git_status():
    """
    Check if repository is clean (no uncommitted changes)

    Returns:
        tuple: (is_clean: bool, status_message: str)
    """
    try:
        # Check if we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode != 0:
            return False, "Not a git repository"

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if result.stdout.strip():
            return False, "Local modifications detected"

        return True, "Clean working tree"
    except Exception as e:
        return False, f"Error checking git status: {e}"


def check_for_updates():
    """
    Check if updates are available from remote repository

    Returns:
        tuple: (update_available: bool, remote_version: str, message: str)
    """
    try:
        repo_path = Path(__file__).parent.parent

        # Fetch from remote
        result = subprocess.run(
            ["git", "fetch", "origin"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=10
        )

        if result.returncode != 0:
            return False, None, f"Failed to fetch updates: {result.stderr}"

        # Get remote VERSION file content
        result = subprocess.run(
            ["git", "show", "origin/main:VERSION"],
            capture_output=True,
            text=True,
            cwd=repo_path
        )

        if result.returncode != 0:
            return False, None, "Could not read remote VERSION file"

        remote_version = result.stdout.strip()
        current_version = get_current_version()

        if remote_version != current_version:
            return True, remote_version, f"Update available: {current_version} → {remote_version}"
        else:
            return False, remote_version, "You're up to date!"

    except subprocess.TimeoutExpired:
        return False, None, "Update check timed out"
    except Exception as e:
        return False, None, f"Error checking for updates: {e}"


def perform_update():
    """
    Perform git pull to update the repository

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        repo_path = Path(__file__).parent.parent

        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=30
        )

        if result.returncode != 0:
            return False, f"Update failed: {result.stderr}"

        return True, "Update successful! Please restart the application."

    except subprocess.TimeoutExpired:
        return False, "Update timed out"
    except Exception as e:
        return False, f"Error during update: {e}"


def get_binary_version():
    """
    Get version of llama.cpp binaries

    Returns:
        dict: Binary version info
    """
    try:
        from .converter import GGUFConverter
        converter = GGUFConverter()

        # Try to get version from llama-cli
        try:
            cli_path = converter.binary_manager.get_cli_path()
            result = subprocess.run(
                [str(cli_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse version from output
                version_line = result.stdout.strip().split('\n')[0] if result.stdout else "unknown"
                return {
                    "status": "ok",
                    "version": version_line,
                    "message": "Binaries are installed"
                }
        except Exception:
            pass

        # Check if binaries exist
        bin_dir = converter.binary_manager.bin_dir
        if bin_dir.exists():
            return {
                "status": "ok",
                "version": "unknown",
                "message": "Binaries installed (version check unavailable)"
            }
        else:
            return {
                "status": "missing",
                "version": None,
                "message": "Binaries not installed - run a conversion to download"
            }

    except Exception as e:
        return {
            "status": "error",
            "version": None,
            "message": f"Error checking binaries: {e}"
        }


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="GGUF Converter",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Reduce top padding
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Yet Another GGUF Converter")
    st.markdown("*Because there are simultaneously too many and not enough GGUF converters*")

    # Initialize converter
    if 'converter' not in st.session_state:
        st.session_state.converter = GGUFConverter()

    # Load config on first run
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

    # Track reset count to force widget refresh
    if 'reset_count' not in st.session_state:
        st.session_state.reset_count = 0

    converter = st.session_state.converter
    config = st.session_state.config

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        st.markdown("---")

        # Auto-save callback for verbose
        def save_verbose():
            config["verbose"] = st.session_state.verbose_checkbox
            save_config(config)

        verbose = st.checkbox(
            "Verbose output",
            value=config.get("verbose", False),
            help="Show detailed command output in the terminal for debugging and monitoring progress",
            key="verbose_checkbox",
            on_change=save_verbose
        )

        st.markdown("---")
        st.markdown("**Performance:**")
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        default_threads = max(1, max_workers - 1)  # Leave one core free for system

        # Auto-save callback for nthreads
        def save_nthreads():
            config["nthreads"] = int(st.session_state.nthreads_input)
            save_config(config)

        nthreads = st.number_input(
            "Thread count",
            min_value=1,
            max_value=max_workers,
            value=int(config.get("nthreads") or default_threads),
            step=1,
            help=f"Number of threads for llama.cpp (CPU cores: {max_workers}, default: {default_threads} to keep system responsive)",
            key="nthreads_input",
            on_change=save_nthreads
        )

        # Reset settings button (removed Save button)
        st.markdown("---")
        if st.button("Reset to defaults", use_container_width=True, help="Reset all settings to default values"):
            st.session_state.config = reset_config()
            st.session_state.reset_count += 1  # Increment to force widget refresh
            st.rerun()

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Convert & Quantize", "Imatrix Settings", "Imatrix Statistics", "HuggingFace Downloader", "Update", "Info"])

    with tab1:
        st.header("Convert and Quantize Model")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")

            # HuggingFace Repo ID with Check Repo button
            col_hf, col_hf_btn = st.columns([5, 1])
            with col_hf:
                hf_repo = st.text_input(
                    "HuggingFace Repo ID (optional)",
                    value=config.get("hf_repo", ""),
                    placeholder="username/model-name",
                    help="Optional: Download model from HuggingFace. If provided, model will be downloaded to Model Path below."
                )
            with col_hf_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input + help icon
                hf_repo_populated = bool(hf_repo and hf_repo.strip())
                if st.button(
                    "Check Repo",
                    key="check_repo_btn",
                    use_container_width=True,
                    disabled=not hf_repo_populated,
                    help="Open HuggingFace repo in browser" if hf_repo_populated else "Enter a repo ID first"
                ):
                    if hf_repo_populated:
                        url = f"https://huggingface.co/{hf_repo.strip()}"
                        webbrowser.open(url)
                        st.toast(f"Opened {url}")

            # Model path with Open folder button
            col_model, col_model_btn = st.columns([5, 1])
            with col_model:
                model_path = st.text_input(
                    "Model path",
                    value=config.get("model_path", ""),
                    placeholder="E:/Models/my-model",
                    help="Local model directory. If HuggingFace repo provided above, model will be downloaded here first."
                )
            with col_model_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input + help icon
                model_path_clean = strip_quotes(model_path)
                model_path_exists = bool(model_path_clean and Path(model_path_clean).exists())
                if st.button(
                    "Check Folder",
                    key="open_model_folder_btn",
                    use_container_width=True,
                    disabled=not model_path_exists,
                    help="Open model folder in file explorer" if model_path_exists else "Path doesn't exist yet"
                ):
                    if model_path_exists:
                        try:
                            open_folder(model_path_clean)
                            st.toast("Opened folder")
                        except Exception as e:
                            st.toast(f"Could not open folder: {e}")

            # Output directory with Open folder button
            col_output, col_output_btn = st.columns([5, 1])
            with col_output:
                output_dir = st.text_input(
                    "Output directory",
                    value=config.get("output_dir", ""),
                    placeholder="E:/Models/converted",
                    help="Where to save the converted files"
                )
            with col_output_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input + help icon
                output_dir_clean = strip_quotes(output_dir)
                output_dir_exists = bool(output_dir_clean and Path(output_dir_clean).exists())
                if st.button(
                    "Check Folder",
                    key="open_output_folder_btn",
                    use_container_width=True,
                    disabled=not output_dir_exists,
                    help="Open output folder in file explorer" if output_dir_exists else "Path doesn't exist yet"
                ):
                    if output_dir_exists:
                        try:
                            open_folder(output_dir_clean)
                            st.toast("Opened folder")
                        except Exception as e:
                            st.toast(f"Could not open folder: {e}")

            st.subheader("Conversion Options")
            # Determine index for intermediate format radio button
            saved_intermediate = config.get("intermediate_type", "F16").upper()
            if saved_intermediate == "F32":
                intermediate_index = 0
            elif saved_intermediate == "BF16":
                intermediate_index = 2
            else:  # F16 or default
                intermediate_index = 1

            # Track previous intermediate format to restore checkbox state
            if 'previous_intermediate' not in st.session_state:
                st.session_state.previous_intermediate = saved_intermediate

            intermediate_type = st.radio(
                "Intermediate format",
                ["F32", "F16", "BF16"],
                index=intermediate_index,
                help="Format used before quantization (F32=highest quality, F16=balanced, BF16=brain float)",
                horizontal=True
            )

            # Ensure the current intermediate format's state is saved before disabling it
            if "full_precision_saved_states" not in config:
                config["full_precision_saved_states"] = {}
            if "other_quants" not in config:
                config["other_quants"] = {}

            # On first load or if not yet saved, save the current intermediate's state
            if intermediate_type not in config["full_precision_saved_states"]:
                current_state = config["other_quants"].get(intermediate_type, False)
                config["full_precision_saved_states"][intermediate_type] = current_state
                # Also ensure it's checked in other_quants
                config["other_quants"][intermediate_type] = True
                save_config(config)

            # If intermediate format changed, update checkbox states
            if intermediate_type != st.session_state.previous_intermediate:
                prev_format = st.session_state.previous_intermediate

                # Ensure config dicts exist
                if "other_quants" not in config:
                    config["other_quants"] = {}
                if "full_precision_saved_states" not in config:
                    config["full_precision_saved_states"] = {}

                # Save all non-intermediate checkbox states from current render
                for qtype in ["F32", "F16", "BF16"]:
                    checkbox_key = f"full_{qtype}_{prev_format}"
                    if checkbox_key in st.session_state:
                        # Save this checkbox's state, but skip the old intermediate since it was locked
                        if qtype != prev_format:
                            config["other_quants"][qtype] = st.session_state[checkbox_key]

                # Restore the previous intermediate's saved state (if it was saved before being disabled)
                if prev_format in config["full_precision_saved_states"]:
                    config["other_quants"][prev_format] = config["full_precision_saved_states"][prev_format]
                    # Remove from saved states since it's no longer disabled
                    del config["full_precision_saved_states"][prev_format]
                else:
                    # If no saved state exists, default to unchecked
                    if prev_format not in config["other_quants"]:
                        config["other_quants"][prev_format] = False

                # Save the new intermediate's current state before disabling it
                current_new_state = config["other_quants"].get(intermediate_type, False)
                config["full_precision_saved_states"][intermediate_type] = current_new_state

                # Force the new intermediate to be checked
                config["other_quants"][intermediate_type] = True

                st.session_state.previous_intermediate = intermediate_type
                config["intermediate_type"] = intermediate_type
                save_config(config)
                # Rerun to reflect the changes in the UI
                st.rerun()

            st.subheader("Importance Matrix (imatrix)")

            # Callback to save imatrix checkbox state
            def save_use_imatrix():
                config["use_imatrix"] = st.session_state[f"use_imatrix_checkbox_{st.session_state.reset_count}"]
                save_config(config)

            use_imatrix = st.checkbox(
                "Use importance matrix",
                value=config.get("use_imatrix", False),
                help="Use importance matrix for better low-bit quantization (IQ2, IQ3).  \nRequired for best IQ2/IQ3 quality.",
                key=f"use_imatrix_checkbox_{st.session_state.reset_count}",
                on_change=save_use_imatrix
            )

            # Show imatrix options when enabled
            imatrix_mode = None
            imatrix_generate_name = None
            imatrix_reuse_path = None

            if use_imatrix:
                # Scan output directory for .imatrix files
                imatrix_files = []
                if output_dir_clean and Path(output_dir_clean).exists():
                    imatrix_files = sorted([f.name for f in Path(output_dir_clean).glob("*.imatrix")])

                # Radio button for Generate vs Generate (custom name) vs Reuse
                saved_mode = config.get("imatrix_mode", "generate")
                if saved_mode == "generate":
                    mode_index = 0
                elif saved_mode == "generate_custom":
                    mode_index = 1
                else:  # reuse
                    mode_index = 2

                # Callback to save imatrix mode state
                def save_imatrix_mode():
                    mode = st.session_state[f"imatrix_mode_radio_{st.session_state.reset_count}"]
                    if mode == "Generate":
                        config["imatrix_mode"] = "generate"
                    elif mode == "Generate (custom name)":
                        config["imatrix_mode"] = "generate_custom"
                    else:  # Reuse existing
                        config["imatrix_mode"] = "reuse"
                    save_config(config)

                imatrix_mode = st.radio(
                    "Imatrix mode",
                    ["Generate", "Generate (custom name)", "Reuse existing"],
                    index=mode_index,
                    help="Generate with default name, custom name, or reuse an existing imatrix file",
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"imatrix_mode_radio_{st.session_state.reset_count}",
                    on_change=save_imatrix_mode
                )

                # Show appropriate field based on mode
                if imatrix_mode == "Generate (custom name)":
                    # Text field for custom imatrix filename with Set to default button
                    col_imatrix_name, col_imatrix_default = st.columns([5, 1])
                    with col_imatrix_name:
                        imatrix_generate_name = st.text_input(
                            "Imatrix filename",
                            value=config.get("imatrix_generate_name", ""),
                            placeholder="model.imatrix",
                            help="Filename for the generated imatrix file (saved in output directory). Leave empty to use default naming (model_name.imatrix)."
                        )
                    with col_imatrix_default:
                        st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with text input
                        if st.button("Set to default", key="set_default_imatrix_name", use_container_width=True):
                            # If we have model path info, populate with the calculated default
                            if model_path_clean:
                                default_name = f"{Path(model_path_clean).name}.imatrix"
                                config["imatrix_generate_name"] = default_name
                            else:
                                # Otherwise just clear to empty
                                config["imatrix_generate_name"] = ""
                            save_config(config)
                            st.rerun()
                elif imatrix_mode == "Generate":
                    # Use default name
                    imatrix_generate_name = ""
                else:  # Reuse existing
                    # Dropdown with Update File List button for reuse
                    col_imatrix_select, col_imatrix_update = st.columns([5, 1])
                    with col_imatrix_select:
                        if imatrix_files:
                            imatrix_reuse_path = st.selectbox(
                                "Select imatrix file",
                                options=imatrix_files,
                                index=0,
                                help="Choose an existing imatrix file from the output directory"
                            )
                        else:
                            st.warning("No .imatrix files found in output directory")
                            imatrix_reuse_path = None
                    with col_imatrix_update:
                        st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with selectbox
                        if st.button("Update File List", key="update_imatrix_reuse_list", use_container_width=True):
                            st.rerun()

        with col2:
            st.subheader("Quantization Types")
            st.markdown("Select one or more quantization types:")

            # Full Precision Outputs
            st.markdown("**Full Precision Outputs:**")
            full_cols = st.columns(3)
            full_quants = {
                "F32": "32-bit float (full precision)",
                "F16": "16-bit float (half precision)",
                "BF16": "16-bit bfloat (brain float)",
            }
            full_checkboxes = {}
            for idx, (qtype, tooltip) in enumerate(full_quants.items()):
                with full_cols[idx]:
                    # Highlight the intermediate format (radio returns uppercase already)
                    is_intermediate = qtype == intermediate_type
                    label = f"{qtype} (intermediate)" if is_intermediate else qtype

                    # If this is the intermediate format, force checked and disabled
                    # Otherwise, use saved value
                    if is_intermediate:
                        checkbox_value = True
                        checkbox_disabled = True
                    else:
                        checkbox_value = config.get("other_quants", {}).get(qtype, False)
                        checkbox_disabled = False

                    full_checkboxes[qtype] = st.checkbox(
                        label,
                        value=checkbox_value,
                        help=tooltip,
                        key=f"full_{qtype}_{intermediate_type}",
                        disabled=checkbox_disabled
                    )

            # Legacy Quants
            st.markdown("**Legacy Quants:**")
            trad_cols = st.columns(3)
            trad_quants = {
                "Q8_0": "8-bit (highest quality)",
                "Q5_1": "5-bit improved",
                "Q5_0": "5-bit",
                "Q4_1": "4-bit improved",
                "Q4_0": "4-bit",
            }
            trad_checkboxes = {}
            for idx, (qtype, tooltip) in enumerate(trad_quants.items()):
                with trad_cols[idx % 3]:
                    trad_checkboxes[qtype] = st.checkbox(
                        qtype,
                        value=config.get("other_quants", {}).get(qtype, qtype == "Q8_0" if qtype == "Q8_0" else False),
                        help=tooltip,
                        key=f"trad_{qtype}"
                    )

            # K Quants
            st.markdown("**K Quants (Recommended):**")
            k_quants = {
                "Q6_K": "6-bit K (very high quality)",
                "Q5_K_M": "5-bit K medium",
                "Q5_K_S": "5-bit K small",
                "Q4_K_M": "4-bit K medium (best balance)",
                "Q4_K_S": "4-bit K small",
                "Q3_K_L": "3-bit K large",
                "Q3_K_M": "3-bit K medium",
                "Q3_K_S": "3-bit K small",
                "Q2_K_S": "2-bit K small",
                "Q2_K": "2-bit K",
            }
            k_checkboxes = {}
            k_cols = st.columns(3)
            for idx, (qtype, tooltip) in enumerate(k_quants.items()):
                with k_cols[idx % 3]:
                    default_val = config.get("other_quants", {}).get(qtype, qtype == "Q4_K_M")
                    k_checkboxes[qtype] = st.checkbox(
                        qtype,
                        value=default_val,
                        help=tooltip,
                        key=f"k_{qtype}"
                    )

            # I Quants
            st.markdown("**I Quants (Importance Matrix Recommended):**")
            i_quants = {
                "IQ4_NL": "4-bit IQ non-linear",
                "IQ4_XS": "4-bit IQ extra-small",
                "IQ3_M": "3-bit IQ medium",
                "IQ3_S": "3.4-bit IQ small",
                "IQ3_XS": "3-bit IQ extra-small",
                "IQ3_XXS": "3-bit IQ extra-extra-small",
                "IQ2_M": "2-bit IQ medium",
                "IQ2_S": "2-bit IQ small",
                "IQ2_XS": "2-bit IQ extra-small",
                "IQ2_XXS": "2-bit IQ extra-extra-small",
                "IQ1_M": "1-bit IQ medium",
                "IQ1_S": "1-bit IQ small",
            }
            i_checkboxes = {}
            i_cols = st.columns(3)
            for idx, (qtype, tooltip) in enumerate(i_quants.items()):
                with i_cols[idx % 3]:
                    i_checkboxes[qtype] = st.checkbox(
                        qtype,
                        value=config.get("other_quants", {}).get(qtype, False),
                        help=tooltip,
                        key=f"i_{qtype}"
                    )

        # Collect selected quantization types
        selected_quants = []

        # Always include the intermediate format (it will be created anyway)
        intermediate_upper = intermediate_type.upper()
        selected_quants.append(intermediate_upper)

        # Add other selected types (skip if it's the same as intermediate to avoid duplicates)
        for qtype, checked in full_checkboxes.items():
            if checked and qtype != intermediate_upper:
                selected_quants.append(qtype)
        for qtype, checked in trad_checkboxes.items():
            if checked:
                selected_quants.append(qtype)
        for qtype, checked in k_checkboxes.items():
            if checked:
                selected_quants.append(qtype)
        for qtype, checked in i_checkboxes.items():
            if checked:
                selected_quants.append(qtype)

        # Convert button
        st.markdown("---")
        if st.button("Start Conversion", type="primary", use_container_width=True):
            # Strip quotes from paths
            model_path_clean = strip_quotes(model_path)
            output_dir_clean = strip_quotes(output_dir)
            hf_repo_clean = strip_quotes(hf_repo)

            if not model_path_clean:
                st.error("Please provide a model path")
            elif not output_dir_clean:
                st.error("Please provide an output directory")
            elif not selected_quants:
                st.error("Please select at least one quantization type")
            else:
                # Update config with current values
                config["verbose"] = verbose
                config["nthreads"] = nthreads
                config["use_imatrix"] = use_imatrix
                config["hf_repo"] = hf_repo_clean
                config["model_path"] = model_path_clean
                config["output_dir"] = output_dir_clean
                config["intermediate_type"] = intermediate_type

                # Save all quantization selections in other_quants
                all_quant_selections = {}
                all_quant_selections.update(full_checkboxes)
                all_quant_selections.update(trad_checkboxes)
                all_quant_selections.update(k_checkboxes)
                all_quant_selections.update(i_checkboxes)
                config["other_quants"] = all_quant_selections

                save_config(config)

                try:
                    # Download from HuggingFace if repo is provided
                    actual_model_path = model_path_clean
                    if hf_repo_clean and hf_repo_clean.strip():
                        with st.spinner(f"Downloading {hf_repo_clean} from HuggingFace..."):
                            download_path = converter.download_model(
                                repo_id=hf_repo_clean.strip(),
                                output_dir=Path(model_path_clean)
                            )
                            actual_model_path = str(download_path)
                            st.info(f"Downloaded to: {actual_model_path}")

                    # Determine imatrix parameters based on mode
                    generate_imatrix_flag = False
                    imatrix_path_to_use = None
                    calibration_file_path = None
                    imatrix_output_filename = None

                    if use_imatrix:
                        # Save imatrix settings to config
                        if imatrix_mode == "Generate":
                            config["imatrix_mode"] = "generate"
                        elif imatrix_mode == "Generate (custom name)":
                            config["imatrix_mode"] = "generate_custom"
                        else:
                            config["imatrix_mode"] = "reuse"

                        if imatrix_mode in ["Generate", "Generate (custom name)"]:
                            generate_imatrix_flag = True
                            config["imatrix_generate_name"] = imatrix_generate_name
                            imatrix_output_filename = imatrix_generate_name if imatrix_generate_name else None

                            # Build calibration file path for generation
                            cal_dir = config.get("imatrix_calibration_dir", "")
                            cal_file = config.get("imatrix_calibration_file", "_default.txt")

                            if cal_dir:
                                calibration_file_path = Path(cal_dir) / cal_file
                            else:
                                # Use default calibration_data directory (one level up from gguf_converter module)
                                default_cal_dir = Path(__file__).parent.parent / "calibration_data"
                                calibration_file_path = default_cal_dir / cal_file
                        else:  # Reuse existing
                            generate_imatrix_flag = False
                            if imatrix_reuse_path:
                                imatrix_path_to_use = Path(output_dir_clean) / imatrix_reuse_path
                                config["imatrix_reuse_path"] = imatrix_reuse_path

                        save_config(config)

                    with st.spinner("Converting and quantizing... This may take a while."):
                        output_files = converter.convert_and_quantize(
                            model_path=actual_model_path,
                            output_dir=output_dir_clean,
                            quantization_types=selected_quants,
                            intermediate_type=intermediate_type,
                            verbose=verbose,
                            generate_imatrix=generate_imatrix_flag,
                            imatrix_path=imatrix_path_to_use,
                            nthreads=nthreads,
                            imatrix_ctx_size=int(config.get("imatrix_ctx_size", 512)),
                            imatrix_chunks=int(config.get("imatrix_chunks", 100)) if config.get("imatrix_chunks", 100) > 0 else None,
                            imatrix_collect_output=config.get("imatrix_collect_output_weight", False),
                            imatrix_calibration_file=calibration_file_path,
                            imatrix_output_name=imatrix_output_filename
                        )

                    st.success(f"Successfully created {len(output_files)} files!")

                    # If imatrix was used, save paths for statistics tab
                    if use_imatrix:
                        # Determine which imatrix file was used
                        actual_imatrix_path = None
                        if imatrix_mode in ["Generate", "Generate (custom name)"]:
                            # Use the generated imatrix file
                            if imatrix_output_filename:
                                actual_imatrix_path = Path(output_dir_clean) / imatrix_output_filename
                            else:
                                model_name = Path(actual_model_path).name
                                actual_imatrix_path = Path(output_dir_clean) / f"{model_name}.imatrix"
                        else:  # Reuse existing
                            actual_imatrix_path = imatrix_path_to_use

                        # Save paths for statistics tab
                        intermediate_path = Path(output_dir_clean) / f"{Path(actual_model_path).name}_{intermediate_type.upper()}.gguf"

                        if actual_imatrix_path and actual_imatrix_path.exists():
                            config["imatrix_stats_path"] = str(actual_imatrix_path)
                        if intermediate_path.exists():
                            config["imatrix_stats_model"] = str(intermediate_path)
                        save_config(config)

                    st.subheader("Created Files")
                    for file_path in output_files:
                        file_size = file_path.stat().st_size / (1024**3)  # GB
                        st.write(f"`{file_path.name}` ({file_size:.2f} GB)")
                        st.code(str(file_path), language=None)

                except Exception as e:
                    # Show in Streamlit UI
                    st.error(f"Error: {e}")
                    if verbose:
                        st.exception(e)

                    # ALSO print to terminal so user sees it
                    print(f"\nError: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    with tab2:
        st.header("Importance Matrix Settings")
        st.markdown("Configure how importance matrices are generated for low-bit quantization")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Calibration Data")

            # Get the default calibration_data directory (one level up from gguf_converter module)
            default_calibration_dir = Path(__file__).parent.parent / "calibration_data"

            # Get the configured directory or use default
            saved_cal_dir = config.get("imatrix_calibration_dir", "")
            if saved_cal_dir:
                calibration_data_dir = Path(saved_cal_dir)
            else:
                calibration_data_dir = default_calibration_dir

            # Directory input field with Check Folder button
            col_dir, col_dir_btn = st.columns([5, 1])
            with col_dir:
                calibration_dir_input = st.text_input(
                    "Calibration files directory",
                    value=str(calibration_data_dir.resolve()),  # Show absolute path
                    placeholder=str(default_calibration_dir.resolve()),
                    help="Full path to directory containing calibration .txt files",
                    key=f"imatrix_cal_dir_input_{st.session_state.reset_count}",
                    on_change=lambda: None  # Trigger to update when user changes the path
                )
            with col_dir_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input + help icon
                cal_dir_exists_check = calibration_data_dir.exists() and calibration_data_dir.is_dir()
                if st.button(
                    "Check Folder",
                    key="check_cal_dir_btn",
                    use_container_width=True,
                    disabled=not cal_dir_exists_check,
                    help="Open calibration directory in file explorer" if cal_dir_exists_check else "Directory doesn't exist"
                ):
                    if cal_dir_exists_check:
                        try:
                            open_folder(str(calibration_data_dir))
                            st.toast("Opened folder")
                        except Exception as e:
                            st.toast(f"Could not open folder: {e}")

            # Update directory path from input (strip quotes)
            calibration_dir_input_clean = strip_quotes(calibration_dir_input)
            if calibration_dir_input_clean != str(calibration_data_dir.resolve()):
                config["imatrix_calibration_dir"] = calibration_dir_input_clean
                save_config(config)
                calibration_data_dir = Path(calibration_dir_input_clean)

            # Scan directory for .txt files
            calibration_files = []
            if calibration_data_dir.exists() and calibration_data_dir.is_dir():
                calibration_files = sorted([f.name for f in calibration_data_dir.glob("*.txt")])

            if not calibration_files:
                st.warning(f"No .txt files found in: {calibration_data_dir}")
                calibration_files = ["(no files found)"]

            # Determine default selection
            saved_calibration = config.get("imatrix_calibration_file", "_default.txt")
            default_index = 0
            if saved_calibration in calibration_files:
                default_index = calibration_files.index(saved_calibration)

            # Calibration file selection with Update Files button
            col_cal, col_cal_btn = st.columns([5, 1])
            with col_cal:
                # Auto-save callback for calibration file
                def save_calibration_file():
                    config["imatrix_calibration_file"] = st.session_state[f"imatrix_cal_selection_{st.session_state.reset_count}"]
                    save_config(config)

                calibration_selection = st.selectbox(
                    "Calibration file",
                    options=calibration_files,
                    index=default_index,
                    help="Select a calibration file from the directory above",
                    key=f"imatrix_cal_selection_{st.session_state.reset_count}",
                    on_change=save_calibration_file
                )
            with col_cal_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with selectbox + help icon
                if st.button(
                    "Update File List",
                    key="update_cal_files_btn",
                    use_container_width=True,
                    help="Rescan directory for calibration files"
                ):
                    # Trigger a rerun to rescan the directory
                    st.toast("Updated calibration file list")
                    st.rerun()

            st.subheader("Processing Settings")

            # Auto-save callback for chunks
            def save_chunks():
                config["imatrix_chunks"] = int(st.session_state[f"imatrix_chunks_input_{st.session_state.reset_count}"])
                save_config(config)

            imatrix_chunks_input = st.number_input(
                "Chunks to process",
                min_value=0,
                max_value=10000,
                value=int(config.get("imatrix_chunks", 100)),
                step=10,
                help="Number of chunks to process (0 = all). 100-200 recommended for good coverage.",
                key=f"imatrix_chunks_input_{st.session_state.reset_count}",
                on_change=save_chunks
            )

            # Auto-save callback for ctx size
            def save_ctx():
                config["imatrix_ctx_size"] = int(st.session_state[f"imatrix_ctx_input_{st.session_state.reset_count}"])
                save_config(config)

            imatrix_ctx_input = st.number_input(
                "Context size",
                min_value=128,
                max_value=8192,
                value=int(config.get("imatrix_ctx_size", 512)),
                step=128,
                help="Context window size. Larger = more context but more memory.",
                key=f"imatrix_ctx_input_{st.session_state.reset_count}",
                on_change=save_ctx
            )

            # Auto-save callback for from chunk
            def save_from_chunk():
                config["imatrix_from_chunk"] = int(st.session_state[f"imatrix_from_chunk_input_{st.session_state.reset_count}"])
                save_config(config)

            imatrix_from_chunk_input = st.number_input(
                "Skip first N chunks",
                min_value=0,
                max_value=10000,
                value=int(config.get("imatrix_from_chunk", 0)),
                step=1,
                help="Skip the first N chunks (useful for resuming interrupted runs)",
                key=f"imatrix_from_chunk_input_{st.session_state.reset_count}",
                on_change=save_from_chunk
            )

            # Auto-save callback for output frequency
            def save_output_freq():
                config["imatrix_output_frequency"] = int(st.session_state[f"imatrix_output_freq_input_{st.session_state.reset_count}"])
                save_config(config)

            imatrix_output_freq_input = st.number_input(
                "Output frequency (chunks)",
                min_value=1,
                max_value=1000,
                value=int(config.get("imatrix_output_frequency", 10)),
                step=1,
                help="Save interval in chunks (default: 10)",
                key=f"imatrix_output_freq_input_{st.session_state.reset_count}",
                on_change=save_output_freq
            )

            # Auto-save callback for no ppl
            def save_no_ppl():
                config["imatrix_no_ppl"] = st.session_state[f"imatrix_no_ppl_input_{st.session_state.reset_count}"]
                save_config(config)

            imatrix_no_ppl_input = st.checkbox(
                "Disable perplexity calculation",
                value=config.get("imatrix_no_ppl", False),
                help="Skip PPL calculation to speed up processing",
                key=f"imatrix_no_ppl_input_{st.session_state.reset_count}",
                on_change=save_no_ppl
            )

            # Auto-save callback for parse special
            def save_parse_special():
                config["imatrix_parse_special"] = st.session_state[f"imatrix_parse_special_input_{st.session_state.reset_count}"]
                save_config(config)

            imatrix_parse_special_input = st.checkbox(
                "Parse special tokens",
                value=config.get("imatrix_parse_special", False),
                help="Enable parsing of special tokens (e.g., <|im_start|>)",
                key=f"imatrix_parse_special_input_{st.session_state.reset_count}",
                on_change=save_parse_special
            )

            # Auto-save callback for collect output
            def save_collect_output():
                config["imatrix_collect_output_weight"] = st.session_state[f"imatrix_collect_output_input_{st.session_state.reset_count}"]
                save_config(config)

            imatrix_collect_output_input = st.checkbox(
                "Collect output.weight tensor",
                value=config.get("imatrix_collect_output_weight", False),
                help="Collect importance matrix for output.weight tensor",
                key=f"imatrix_collect_output_input_{st.session_state.reset_count}",
                on_change=save_collect_output
            )

            # Reset button
            st.markdown("---")
            if st.button("Reset to Defaults", use_container_width=True, key="reset_imatrix_settings_btn"):
                # Reset imatrix settings to defaults
                defaults = get_default_config()
                config["imatrix_calibration_file"] = defaults["imatrix_calibration_file"]
                config["imatrix_calibration_dir"] = defaults["imatrix_calibration_dir"]
                config["imatrix_chunks"] = defaults["imatrix_chunks"]
                config["imatrix_ctx_size"] = defaults["imatrix_ctx_size"]
                config["imatrix_from_chunk"] = defaults["imatrix_from_chunk"]
                config["imatrix_no_ppl"] = defaults["imatrix_no_ppl"]
                config["imatrix_parse_special"] = defaults["imatrix_parse_special"]
                config["imatrix_collect_output_weight"] = defaults["imatrix_collect_output_weight"]
                config["imatrix_output_frequency"] = defaults["imatrix_output_frequency"]
                save_config(config)
                st.session_state.reset_count += 1
                st.session_state.imatrix_just_reset = True
                st.rerun()

            # Show success message if we just reset
            if st.session_state.get('imatrix_just_reset', False):
                st.success("Reset to defaults! And Saved!")
                st.session_state.imatrix_just_reset = False

        with col2:
            st.subheader("Calibration Preview")

            # Get the calibration file path
            preview_cal_dir = config.get("imatrix_calibration_dir", "")
            if preview_cal_dir:
                preview_calibration_path = Path(preview_cal_dir) / calibration_selection
            else:
                default_cal_dir = Path(__file__).parent.parent / "calibration_data"
                preview_calibration_path = default_cal_dir / calibration_selection

            # Try to read and preview the calibration file
            if preview_calibration_path.exists() and calibration_selection != "(no files found)":
                try:
                    with open(preview_calibration_path, 'r', encoding='utf-8') as f:
                        full_content = f.read()

                    # Preview mode selection
                    preview_mode = st.radio(
                        "Preview mode",
                        ["Full file", "Processed data"],
                        index=0,
                        help="Full file shows entire calibration file. Processed data shows what will actually be used based on your settings.",
                        horizontal=True,
                        key=f"preview_mode_{st.session_state.reset_count}"
                    )

                    # Split into lines for chunk calculation
                    lines = full_content.split('\n')
                    total_lines = len(lines)

                    if preview_mode == "Full file":
                        # Show full file info
                        total_chars = len(full_content)
                        total_words = len(full_content.split())

                        info_msg = f"""**Full File Overview:**
- **Lines**: {total_lines:,} total lines
- **Content**: {total_chars:,} characters, ~{total_words:,} words
- **File**: {calibration_selection}"""

                        st.markdown(info_msg)

                        # Show entire file with word wrap
                        st.text_area(
                            "Content",
                            value=full_content,
                            height=400,
                            disabled=True,
                            label_visibility="collapsed"
                        )

                    else:  # "Processed data"
                        # Calculate what will be processed based on settings
                        from_chunk = int(imatrix_from_chunk_input)
                        chunks_to_process = int(imatrix_chunks_input)

                        # Estimate lines per chunk (llama-imatrix uses context size to determine chunks)
                        ctx_size = int(imatrix_ctx_input)
                        # Rough estimate: each chunk processes ~ctx_size tokens, ~1 token per word, ~5 words per line
                        estimated_lines_per_chunk = max(1, ctx_size // 5)

                        if chunks_to_process > 0:
                            start_line = from_chunk * estimated_lines_per_chunk
                            end_line = start_line + (chunks_to_process * estimated_lines_per_chunk)
                            processed_lines = lines[start_line:end_line]
                            processed_content = '\n'.join(processed_lines)

                            # Calculate totals
                            total_chars = len(processed_content)
                            total_words = len(processed_content.split())
                            actual_end_line = min(end_line, total_lines)

                            # Calculate actual chunks we're getting
                            actual_chunks_shown = len(processed_lines) // estimated_lines_per_chunk
                            max_chunks_available = (total_lines - start_line) // estimated_lines_per_chunk

                            # Check if we're showing the whole file
                            if len(processed_lines) >= total_lines - (from_chunk * estimated_lines_per_chunk):
                                coverage_note = f"**Note**: Requested {chunks_to_process} chunks but only {max_chunks_available} available - showing all remaining data"
                            else:
                                coverage_note = ""

                            info_msg = f"""**Processed Data Overview:**
- **Lines**: {start_line+1} to {actual_end_line} ({len(processed_lines)} of {total_lines} total)
- **Chunks**: Showing ~{actual_chunks_shown} chunks of {chunks_to_process} requested (at {estimated_lines_per_chunk} lines/chunk, {ctx_size} ctx size)
- **Content**: {total_chars:,} characters, ~{total_words:,} words
- **Skipped**: First {from_chunk} chunks ({from_chunk * estimated_lines_per_chunk} lines)

{coverage_note}"""
                        else:
                            # Process all chunks
                            if from_chunk > 0:
                                start_line = from_chunk * estimated_lines_per_chunk
                                processed_lines = lines[start_line:]
                                processed_content = '\n'.join(processed_lines)
                                total_chars = len(processed_content)
                                total_words = len(processed_content.split())
                                estimated_chunks = len(processed_lines) // estimated_lines_per_chunk

                                info_msg = f"""**Processed Data Overview:**
- **Lines**: {start_line+1} to {total_lines} ({len(processed_lines)} of {total_lines} total)
- **Chunks**: ~{estimated_chunks} remaining chunks (at {estimated_lines_per_chunk} lines/chunk)
- **Content**: {total_chars:,} characters, ~{total_words:,} words
- **Skipped**: First {from_chunk} chunks ({from_chunk * estimated_lines_per_chunk} lines)"""
                            else:
                                processed_content = full_content
                                total_chars = len(processed_content)
                                total_words = len(processed_content.split())
                                estimated_chunks = total_lines // estimated_lines_per_chunk

                                info_msg = f"""**Processed Data Overview:**
- **Lines**: All {total_lines} lines
- **Chunks**: ~{estimated_chunks} total chunks (at {estimated_lines_per_chunk} lines/chunk, {ctx_size} ctx size)
- **Content**: {total_chars:,} characters, ~{total_words:,} words"""

                        st.markdown(info_msg)

                        # Show processed data with word wrap
                        st.text_area(
                            "Content",
                            value=processed_content,
                            height=400,
                            disabled=True,
                            label_visibility="collapsed"
                        )

                except Exception as e:
                    st.warning(f"Could not preview calibration file: {e}")
            else:
                st.info("Select a calibration file to preview")

    with tab3:
        st.header("Imatrix Statistics")
        st.markdown("Analyze existing importance matrix files to view statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Settings")

            # Output directory to analyze with Check Output Directory button
            col_stats_dir, col_stats_dir_btn = st.columns([5, 1])
            with col_stats_dir:
                stats_output_dir = st.text_input(
                    "Output directory to analyze",
                    value=config.get("output_dir", ""),
                    placeholder="E:/Models/output",
                    help="Directory containing imatrix and GGUF files to analyze (uses output directory from Convert & Quantize tab)"
                )
            with col_stats_dir_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input + help icon
                stats_dir_clean = strip_quotes(stats_output_dir)
                stats_dir_exists = bool(stats_dir_clean and Path(stats_dir_clean).exists())
                if st.button(
                    "Check Output Directory",
                    key="check_imatrix_output_dir_btn",
                    use_container_width=True,
                    disabled=not stats_dir_exists,
                    help="Open output folder in file explorer" if stats_dir_exists else "Path doesn't exist yet"
                ):
                    if stats_dir_exists:
                        try:
                            open_folder(stats_dir_clean)
                            st.toast("Opened folder")
                        except Exception as e:
                            st.toast(f"Could not open folder: {e}")

            # Scan directory for imatrix files
            imatrix_files = []
            if stats_dir_clean and Path(stats_dir_clean).exists() and Path(stats_dir_clean).is_dir():
                imatrix_files = sorted([str(f) for f in Path(stats_dir_clean).glob("*.imatrix")])

            if not imatrix_files:
                imatrix_files = ["(no .imatrix files found)"]

            # Determine default selection for imatrix file
            saved_imatrix = config.get("imatrix_stats_path", "")
            imatrix_default_index = 0
            if saved_imatrix in imatrix_files:
                imatrix_default_index = imatrix_files.index(saved_imatrix)

            # Imatrix file dropdown with Update File List button
            col_imatrix, col_imatrix_btn = st.columns([5, 1])
            with col_imatrix:
                imatrix_stats_path = st.selectbox(
                    "Imatrix file to analyze",
                    options=imatrix_files,
                    index=imatrix_default_index,
                    help="Select an imatrix file from the directory above",
                    key=f"imatrix_stats_path_{st.session_state.reset_count}"
                )
            with col_imatrix_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with selectbox + help icon
                if st.button(
                    "Update File List",
                    key="update_imatrix_files_btn",
                    use_container_width=True,
                    help="Rescan directory for imatrix files"
                ):
                    st.toast("Updated imatrix file list")
                    st.rerun()

            # Scan directory for GGUF model files
            gguf_files = []
            if stats_dir_clean and Path(stats_dir_clean).exists() and Path(stats_dir_clean).is_dir():
                gguf_files = sorted([str(f) for f in Path(stats_dir_clean).glob("*.gguf")])

            if not gguf_files:
                gguf_files = ["(no .gguf files found)"]

            # Determine default selection for model file
            saved_model = config.get("imatrix_stats_model", "")
            model_default_index = 0
            if saved_model in gguf_files:
                model_default_index = gguf_files.index(saved_model)

            # Model path dropdown with Update File List button
            col_model, col_model_btn = st.columns([5, 1])
            with col_model:
                imatrix_stats_model = st.selectbox(
                    "Model path for statistics",
                    options=gguf_files,
                    index=model_default_index,
                    help="Select a GGUF model file from the directory above (required by llama-imatrix for showing statistics)",
                    key=f"imatrix_stats_model_{st.session_state.reset_count}"
                )
            with col_model_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with selectbox + help icon
                if st.button(
                    "Update File List",
                    key="update_model_files_btn",
                    use_container_width=True,
                    help="Rescan directory for GGUF files"
                ):
                    st.toast("Updated model file list")
                    st.rerun()

            if st.button("Show Statistics", use_container_width=True, key="show_stats_btn"):
                # Strip quotes from paths
                imatrix_stats_path_clean = strip_quotes(imatrix_stats_path)
                imatrix_stats_model_clean = strip_quotes(imatrix_stats_model)

                if not imatrix_stats_path_clean:
                    st.error("Please provide an imatrix file path")
                elif not imatrix_stats_model_clean:
                    st.error("Please provide a model path")
                else:
                    try:
                        with st.spinner("Reading imatrix statistics..."):
                            stats = converter.show_imatrix_statistics(
                                imatrix_stats_path_clean,
                                imatrix_stats_model_clean,
                                verbose=verbose
                            )
                        st.session_state.imatrix_stats_result = stats
                        st.session_state.imatrix_stats_error = None
                    except Exception as e:
                        st.session_state.imatrix_stats_result = None
                        st.session_state.imatrix_stats_error = str(e)

            # Show info/error messages below the button
            if 'imatrix_stats_error' in st.session_state and st.session_state.imatrix_stats_error:
                st.error(f"Error: {st.session_state.imatrix_stats_error}")
                if verbose:
                    st.exception(st.session_state.imatrix_stats_error)
            elif 'imatrix_stats_result' in st.session_state and st.session_state.imatrix_stats_result:
                st.success("Statistics generated!")
            elif 'imatrix_stats_result' not in st.session_state or not st.session_state.imatrix_stats_result:
                st.info("Click 'Show Statistics' to analyze an imatrix file")

        with col2:
            st.subheader("Statistics Output")

            # Display results if available
            if 'imatrix_stats_result' in st.session_state and st.session_state.imatrix_stats_result:
                # Add horizontal scroll for wide statistics output
                st.markdown("""
                    <style>
                    .stats-output pre {
                        overflow-x: auto;
                        white-space: pre;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown('<div class="stats-output">', unsafe_allow_html=True)
                st.code(st.session_state.imatrix_stats_result, language=None)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.header("HuggingFace Downloader")
        st.markdown("Download a model without converting")
        st.markdown("[Browse models on HuggingFace](https://huggingface.co/models)")

        # Repository ID with Check Repo button
        col_repo, col_repo_btn = st.columns([5, 1])
        with col_repo:
            repo_id = st.text_input(
                "Repository ID",
                value=config.get("repo_id", ""),
                placeholder="username/model-name"
            )
        with col_repo_btn:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input
            repo_id_populated = bool(repo_id and repo_id.strip())
            if st.button(
                "Check Repo",
                key="check_repo_download_btn",
                use_container_width=True,
                disabled=not repo_id_populated,
                help="Open HuggingFace repo in browser" if repo_id_populated else "Enter a repo ID first"
            ):
                if repo_id_populated:
                    url = f"https://huggingface.co/{repo_id.strip()}"
                    webbrowser.open(url)
                    st.toast(f"Opened {url}")

        # Download directory with Check Folder button
        col_download, col_download_btn = st.columns([5, 1])
        with col_download:
            download_dir = st.text_input(
                "Download directory",
                value=config.get("download_dir", ""),
                placeholder="E:/Models/downloads"
            )
        with col_download_btn:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input
            download_dir_check = strip_quotes(download_dir)
            download_dir_exists = bool(download_dir_check and Path(download_dir_check).exists())
            if st.button(
                "Check Folder",
                key="check_download_folder_btn",
                use_container_width=True,
                disabled=not download_dir_exists,
                help="Open download folder in file explorer" if download_dir_exists else "Path doesn't exist yet"
            ):
                if download_dir_exists:
                    try:
                        open_folder(download_dir_check)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

        if st.button("Download", use_container_width=True):
            # Strip quotes from paths
            repo_id_clean = strip_quotes(repo_id)
            download_dir_clean = strip_quotes(download_dir)

            if not repo_id_clean:
                st.error("Please provide a repository ID")
            elif not download_dir_clean:
                st.error("Please provide a download directory")
            else:
                # Save current settings before downloading
                config["repo_id"] = repo_id_clean
                config["download_dir"] = download_dir_clean
                save_config(config)

                try:
                    with st.spinner(f"Downloading {repo_id_clean}..."):
                        model_path = converter.download_model(repo_id_clean, download_dir_clean)

                    st.success(f"Downloaded to: {model_path}")
                    st.code(str(model_path), language=None)

                except Exception as e:
                    # Show in Streamlit UI
                    st.error(f"Error: {e}")

                    # ALSO print to terminal so user sees it
                    print(f"\nError: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    with tab5:
        st.header("Update")
        st.markdown("Check for updates and manage your installation")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Current Version")

            current_version = get_current_version()
            st.info(f"**Version:** {current_version}")

            # Git status
            st.markdown("---")
            st.subheader("Repository Status")

            is_clean, status_msg = check_git_status()
            if is_clean:
                st.success(f"✓ {status_msg}")
            else:
                st.warning(f"⚠ {status_msg}")

            # Check for updates button
            st.markdown("---")
            if st.button("Check for Updates", use_container_width=True, type="primary"):
                with st.spinner("Checking for updates..."):
                    update_available, remote_version, message = check_for_updates()

                # Store results in session state
                st.session_state.update_check_result = {
                    "available": update_available,
                    "remote_version": remote_version,
                    "message": message
                }
                st.rerun()

            # Show update check results
            if 'update_check_result' in st.session_state:
                result = st.session_state.update_check_result
                if result["available"]:
                    st.success(f"✓ {result['message']}")

                    # Update button (only if repo is clean)
                    if is_clean:
                        if st.button("Update Now", use_container_width=True, type="primary"):
                            with st.spinner("Updating..."):
                                success, update_msg = perform_update()

                            if success:
                                st.success(update_msg)
                                st.info("Please restart the application to use the new version.")
                                # Clear the update check result
                                del st.session_state.update_check_result
                            else:
                                st.error(update_msg)
                    else:
                        st.warning("Cannot update: Local modifications detected. Commit or stash your changes first.")
                else:
                    if result["remote_version"]:
                        st.info(result["message"])
                    else:
                        st.error(result["message"])

        with col2:
            st.subheader("Binary Information")

            binary_info = get_binary_version()

            if binary_info["status"] == "ok":
                st.success(f"✓ {binary_info['message']}")
                if binary_info["version"]:
                    st.code(binary_info["version"], language=None)
            elif binary_info["status"] == "missing":
                st.warning(f"⚠ {binary_info['message']}")
            else:
                st.error(f"✗ {binary_info['message']}")

            st.markdown("---")
            st.subheader("About Updates")
            st.markdown("""
            **Update Process:**
            1. Click "Check for Updates" to see if a new version is available
            2. If available and your repository is clean, click "Update Now"
            3. Restart the application after updating

            **Important Notes:**
            - Updates require a clean git repository (no local modifications)
            - If you have local changes, commit or stash them first
            - Advanced users can update manually with `git pull`
            - Version numbers only change on stable releases
            - llama.cpp binaries are auto-downloaded when needed

            **Troubleshooting:**
            - If update fails, try running `git pull origin main` manually
            - Check that you have git installed and in your PATH
            - Make sure you have internet connection
            """)

    with tab6:
        st.header("About")
        st.markdown(f"""
        ### Yet Another GGUF Converter

        A user-friendly GGUF converter that shields you from llama.cpp complexity.
        No manual compilation or terminal commands required!

        **Features:**
        - **Convert & Quantize** - HuggingFace models to GGUF with multiple quantization formats at once
        - **Importance Matrix** - Generate or reuse imatrix files for better low-bit quantization (IQ2, IQ3)
        - **Imatrix Statistics** - Analyze importance matrix files to view statistics
        - **HuggingFace Downloader** - Download models without converting
        - **Auto-downloads binaries** - Pre-compiled llama.cpp binaries (no compilation needed!)
        - **Cross-platform** - Windows, Mac, Linux support
        - **Persistent settings** - Automatically saves your preferences
        - **All quantization types** - Full support for llama.cpp quantization types

        **Tabs:**
        1. **Convert & Quantize** - Main conversion interface with imatrix options
        2. **Imatrix Settings** - Configure calibration data and processing settings
        3. **Imatrix Statistics** - Analyze existing imatrix files
        4. **HuggingFace Downloader** - Download models from HuggingFace
        5. **Update** - Check for updates and manage installation
        6. **Info** - This tab

        **Settings:**
        - Your settings are automatically saved as you change them
        - Settings are stored in: `{CONFIG_FILE}`
        - Use "Reset to defaults" in the sidebar to restore default settings

        **Quantization Types (via llama.cpp):**

        | Type | Size | Quality | Category | Notes |
        |------|------|---------|----------|-------|
        | **F32** | Largest | Original | Full Precision | Full 32-bit precision |
        | **F16** | Large | Near-original | Full Precision | Half precision (default intermediate) |
        | **BF16** | Large | Near-original | Full Precision | Brain float 16-bit |
        | **Q8_0** | Very Large | Excellent | Legacy | Near-original quality |
        | Q5_1, Q5_0 | Medium | Good | Legacy | Legacy 5-bit |
        | Q4_1, Q4_0 | Small | Fair | Legacy | Legacy 4-bit |
        | **Q6_K** | Large | Very High | K-Quant | Near-F16 quality |
        | **Q5_K_M** | Medium | Better | K-Quant | Higher quality |
        | Q5_K_S | Medium | Better | K-Quant | 5-bit K small |
        | **Q4_K_M** | Small | Good | K-Quant | **Recommended** - best balance |
        | Q4_K_S | Small | Good | K-Quant | 4-bit K small |
        | Q3_K_L | Very Small | Fair | K-Quant | 3-bit K large |
        | Q3_K_M | Very Small | Fair | K-Quant | 3-bit K medium |
        | Q3_K_S | Very Small | Fair | K-Quant | 3-bit K small |
        | Q2_K | Tiny | Minimal | K-Quant | 2-bit K |
        | Q2_K_S | Tiny | Minimal | K-Quant | 2-bit K small |
        | **IQ4_NL** | Small | Good | I-Quant | 4-bit non-linear (use imatrix) |
        | IQ4_XS | Small | Good | I-Quant | 4-bit extra-small (use imatrix) |
        | IQ3_M | Very Small | Fair | I-Quant | 3-bit medium (use imatrix) |
        | IQ3_S | Very Small | Fair+ | I-Quant | 3.4-bit (use imatrix) |
        | IQ3_XS | Very Small | Fair | I-Quant | 3-bit extra-small (use imatrix) |
        | IQ3_XXS | Very Small | Fair | I-Quant | 3-bit extra-extra-small (use imatrix) |
        | IQ2_M | Tiny | Minimal | I-Quant | 2-bit medium (use imatrix) |
        | IQ2_S | Tiny | Minimal | I-Quant | 2-bit small (use imatrix) |
        | IQ2_XS | Tiny | Minimal | I-Quant | 2-bit extra-small (use imatrix) |
        | IQ2_XXS | Tiny | Minimal | I-Quant | 2-bit extra-extra-small (use imatrix) |
        | IQ1_M | Extreme | Poor | I-Quant | 1-bit medium (use imatrix) |
        | IQ1_S | Extreme | Poor | I-Quant | 1-bit small (use imatrix) |

        **Quick Guide:**
        - Just starting? Use **Q4_K_M**
        - Want better quality? Use **Q5_K_M** or **Q6_K**
        - Need original quality? Use **Q8_0** or **F16**
        - Want smallest size? Use IQ3_M or IQ2_M with importance matrix

        Quantization is powered by llama.cpp - battle-tested and fully compatible!

        **Requirements:**
        - Python 3.8+
        - huggingface-hub (for downloading models)
        - streamlit (for GUI)
        - llama.cpp binaries (auto-downloaded)

        **Command Line Usage:**
        ```bash
        # Convert and quantize
        python -m gguf_converter /path/to/model output/ -q Q4_K_M Q5_K_M

        # Download from HuggingFace
        python -m gguf_converter username/model-name output/ -q Q4_K_M

        # List available types
        python -m gguf_converter --list-types
        ```
        """)


if __name__ == "__main__":
    main()
