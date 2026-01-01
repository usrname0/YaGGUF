"""
llama.cpp tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, get_binary_version,
    get_binary_version_from_path,
    TKINTER_AVAILABLE, get_platform_path
)

if TYPE_CHECKING:
    from ..converter import GGUFConverter


def render_llama_cpp_tab(converter: "GGUFConverter", config: Dict[str, Any]) -> None:
    """Render the llama.cpp tab"""
    # Handle pending binaries folder update
    if 'pending_binaries_folder' in st.session_state:
        st.session_state.custom_binaries_folder_input_update = st.session_state.pending_binaries_folder
        del st.session_state.pending_binaries_folder

    # Handle pending repo folder update
    if 'pending_repo_folder' in st.session_state:
        st.session_state.custom_llama_cpp_repo_input = st.session_state.pending_repo_folder
        del st.session_state.pending_repo_folder

    st.header("llama.cpp Custom Install")

    st.markdown("""
    YaGGUF automatically downloads CPU binaries (good for most cases) and a repo for model conversion scripts.
    Those can be managed on the "Update" tab.  
    You can also opt to use your own specific [llama.cpp](https://github.com/ggml-org/llama.cpp) resources below.  
    """)

    # Local/Custom Binary Settings Section
    col_bin1, col_bin2 = st.columns(2)
    with col_bin1:
        st.subheader("Local/Custom Binary Settings")

        # Auto-save callback for use_custom_binaries
        def save_use_custom_binaries():
            config["use_custom_binaries"] = st.session_state.use_custom_binaries_checkbox_update
            save_config(config)
            # Clear version cache when switching
            if "custom_binary_versions" in st.session_state:
                del st.session_state.custom_binary_versions
            # Reinitialize converter with new settings
            custom_binaries = config.get("custom_binaries_folder", "") if config.get("use_custom_binaries", False) else None
            custom_repo = config.get("custom_llama_cpp_repo", "") if config.get("use_custom_conversion_script", False) else None
            from ..converter import GGUFConverter
            st.session_state.converter = GGUFConverter(
                custom_binaries_folder=custom_binaries,
                custom_llama_cpp_repo=custom_repo
            )

        use_custom_binaries = st.checkbox(
            "Use local/custom binaries",
            value=config.get("use_custom_binaries", False),
            help="Use local or system llama.cpp binaries instead of YaGGUF's auto-downloaded ones. Leave path blank to use system PATH.",
            key="use_custom_binaries_checkbox_update",
            on_change=save_use_custom_binaries
        )

        st.markdown("**Binaries Folder Path** (leave blank for system PATH):")

        # Single folder path with Select Folder and Open Folder buttons
        if TKINTER_AVAILABLE:
            col_folder, col_browse, col_check = st.columns([4, 1, 1])
        else:
            col_folder, col_check = st.columns([5, 1])
            col_browse = None  # Not used when tkinter unavailable

        with col_folder:
            def save_binaries_folder():
                new_folder = st.session_state.custom_binaries_folder_input_update
                if new_folder != config.get("custom_binaries_folder", ""):
                    config["custom_binaries_folder"] = new_folder
                    save_config(config)
                    # Clear version cache when folder changes
                    if "custom_binary_versions" in st.session_state:
                        del st.session_state.custom_binary_versions
                    # Reinitialize converter with new folder
                    from ..converter import GGUFConverter
                    st.session_state.converter = GGUFConverter(custom_binaries_folder=new_folder)

            # Only set value if key not in session state (prevents warning)
            binaries_folder_kwargs = {
                "label": "llama.cpp binaries folder",
                "placeholder": get_platform_path(
                    "C:\\path\\to\\llama.cpp\\bin or leave blank for PATH",
                    "/path/to/llama.cpp/bin or leave blank for PATH"
                ),
                "help": "Path to folder containing llama-quantize and llama-imatrix. Leave blank to use system PATH.",
                "key": "custom_binaries_folder_input_update",
                "label_visibility": "collapsed",
                "disabled": not use_custom_binaries,
                "on_change": save_binaries_folder
            }
            if "custom_binaries_folder_input_update" not in st.session_state:
                binaries_folder_kwargs["value"] = config.get("custom_binaries_folder", "")
            binaries_folder = st.text_input(**binaries_folder_kwargs)  # type: ignore[arg-type]

        if TKINTER_AVAILABLE:
            with col_browse:  # type: ignore[union-attr]
                if st.button(
                    "Select Folder",
                    key="browse_binaries_folder_btn",
                    use_container_width=True,
                    help="Select binaries folder",
                    disabled=not use_custom_binaries
                ):
                    binaries_folder_clean = strip_quotes(binaries_folder)
                    initial_dir = binaries_folder_clean if binaries_folder_clean and Path(binaries_folder_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["custom_binaries_folder"] = selected_folder
                        save_config(config)
                        # Clear version cache when folder changes
                        if "custom_binary_versions" in st.session_state:
                            del st.session_state.custom_binary_versions
                        # Set pending flag - will be applied before widget creation on next run
                        st.session_state.pending_binaries_folder = selected_folder
                        # Reinitialize converter
                        from ..converter import GGUFConverter
                        st.session_state.converter = GGUFConverter(custom_binaries_folder=selected_folder)
                        st.rerun()

        with col_check:
            binaries_folder_clean = strip_quotes(binaries_folder)
            binaries_folder_exists = bool(binaries_folder_clean and Path(binaries_folder_clean).exists())
            if st.button(
                "Open Folder",
                key="check_binaries_folder_btn",
                use_container_width=True,
                disabled=not use_custom_binaries or not binaries_folder_exists,
                help="Open folder in file explorer" if binaries_folder_exists else "Path doesn't exist yet"
            ):
                if binaries_folder_exists:
                    try:
                        open_folder(binaries_folder_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

        # Show help text below folder path
        st.markdown("""
        Enable local/custom binaries if you want to:
        - Use a custom-compiled llama.cpp with GPU support (CUDA/ROCm/Metal/Vulkan)
        - Use a specific llama.cpp version

        **Note:** Custom binaries with GPU support are required for GPU offloading in imatrix generation (see Imatrix Settings tab).
        Gains will depend on hardware. Quantization only uses the CPU regardless of setup.
        """)

    with col_bin2:
        col_header, col_refresh = st.columns([4, 1])
        with col_header:
            st.subheader("Binary Detection")
        with col_refresh:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with header
            if st.button("Refresh", key="refresh_custom_binary_detection", use_container_width=True, disabled=not config.get("use_custom_binaries", False)):
                # Clear cached version info
                if "custom_binary_versions" in st.session_state:
                    del st.session_state.custom_binary_versions
                st.rerun()

        # If custom binaries enabled, show system/custom binary info
        if config.get("use_custom_binaries", False):
            custom_folder = config.get("custom_binaries_folder", "")

            # Check which binaries are found
            quantize_found = False
            imatrix_found = False
            quantize_path = None
            imatrix_path = None

            try:
                quantize_path = st.session_state.converter.llama_cpp_manager.get_quantize_path()
                quantize_found = quantize_path.exists()
            except (RuntimeError, OSError, FileNotFoundError):
                # Binary not found - quantize_found stays False, UI will show appropriate status
                pass

            try:
                imatrix_path = st.session_state.converter.llama_cpp_manager.get_imatrix_path()
                imatrix_found = imatrix_path.exists()
            except (RuntimeError, OSError, FileNotFoundError):
                # Binary not found - imatrix_found stays False, UI will show appropriate status
                pass

            if custom_folder:
                st.info(f"Looking in: `{custom_folder}`")
            else:
                st.info("Looking in system PATH")

            # Initialize version cache if needed
            if "custom_binary_versions" not in st.session_state:
                st.session_state.custom_binary_versions = {}

            # Cache key based on path
            cache_key = f"{custom_folder}_{imatrix_found}"

            # Show found binaries
            if quantize_found and imatrix_found:
                st.success("All required binaries found")

                # Get version from cache or fetch it
                if cache_key not in st.session_state.custom_binary_versions:
                    with st.spinner("Detecting version..."):
                        system_version = get_binary_version_from_path(imatrix_path)
                    # Only cache successful detections (not None)
                    if system_version:
                        st.session_state.custom_binary_versions[cache_key] = system_version
                else:
                    system_version = st.session_state.custom_binary_versions[cache_key]

                if system_version:
                    st.code(system_version, language=None)
                else:
                    st.info("Version: Unable to detect (click Refresh to retry)")

                # Show binary paths
                st.markdown(f"- `llama-quantize`: {quantize_path}")
                st.markdown(f"- `llama-imatrix`: {imatrix_path}")
            elif quantize_found or imatrix_found:
                st.warning("Some binaries missing")
                if quantize_found:
                    st.markdown(f"- `llama-quantize`: {quantize_path}")
                else:
                    st.markdown("- `llama-quantize`: Not found")

                if imatrix_found:
                    st.markdown(f"- `llama-imatrix`: {imatrix_path}")

                    # Get version from cache or fetch it
                    if cache_key not in st.session_state.custom_binary_versions:
                        with st.spinner("Detecting version..."):
                            imatrix_version = get_binary_version_from_path(imatrix_path)
                        # Only cache successful detections (not None)
                        if imatrix_version:
                            st.session_state.custom_binary_versions[cache_key] = imatrix_version
                    else:
                        imatrix_version = st.session_state.custom_binary_versions[cache_key]

                    if imatrix_version:
                        st.code(imatrix_version, language=None)
                    else:
                        st.info("Version: Unable to detect (click Refresh to retry)")
                else:
                    st.markdown("- `llama-imatrix`: Not found")
            else:
                st.error("No binaries found")
                st.markdown("- `llama-quantize`: Not found")
                st.markdown("- `llama-imatrix`: Not found")
        else:
            st.info("Disabled - Using YaGGUF's llama.cpp binaries")

    st.markdown("---")

    # Conversion Script Settings Section
    col_script1, col_script2 = st.columns(2)
    with col_script1:
        st.subheader("Local/Custom Conversion Script Settings")

        # Auto-save callback for use_custom_conversion_script
        def save_use_custom_conversion_script():
            config["use_custom_conversion_script"] = st.session_state.use_custom_conversion_script_checkbox
            save_config(config)
            # Reinitialize converter with new settings
            custom_binaries = config.get("custom_binaries_folder", "") if config.get("use_custom_binaries", False) else None
            custom_repo = config.get("custom_llama_cpp_repo", "") if config.get("use_custom_conversion_script", False) else None
            from ..converter import GGUFConverter
            st.session_state.converter = GGUFConverter(
                custom_binaries_folder=custom_binaries,
                custom_llama_cpp_repo=custom_repo
            )

        use_custom_conversion_script = st.checkbox(
            "Use custom llama.cpp repository",
            value=config.get("use_custom_conversion_script", False),
            help="Use a custom llama.cpp repository for model conversion. Leave path blank to use YaGGUF's repository.",
            key="use_custom_conversion_script_checkbox",
            on_change=save_use_custom_conversion_script
        )

        st.markdown("**Repository Path** (blank defaults back to YaGGUF's auto-cloned repo):")

        # Repository path input with Select Folder and Open Folder buttons
        if TKINTER_AVAILABLE:
            col_repo, col_repo_browse, col_repo_check = st.columns([4, 1, 1])
        else:
            col_repo, col_repo_check = st.columns([5, 1])
            col_repo_browse = None  # Not used when tkinter unavailable

        with col_repo:
            def save_repo_path():
                new_path = st.session_state.custom_llama_cpp_repo_input
                if new_path != config.get("custom_llama_cpp_repo", ""):
                    config["custom_llama_cpp_repo"] = new_path
                    save_config(config)
                    # Reinitialize converter with new repo
                    custom_binaries = config.get("custom_binaries_folder", "") if config.get("use_custom_binaries", False) else None
                    from ..converter import GGUFConverter
                    st.session_state.converter = GGUFConverter(
                        custom_binaries_folder=custom_binaries,
                        custom_llama_cpp_repo=new_path
                    )

            # Only set value if key not in session state (prevents warning)
            repo_path_kwargs = {
                "label": "llama.cpp repository path",
                "placeholder": get_platform_path(
                    "C:\\path\\to\\llama.cpp or leave blank for YaGGUF's repo",
                    "/path/to/llama.cpp or leave blank for YaGGUF's repo"
                ),
                "help": "Path to llama.cpp repository containing convert_hf_to_gguf.py. Leave blank to use YaGGUF's auto-cloned repository.",
                "key": "custom_llama_cpp_repo_input",
                "label_visibility": "collapsed",
                "disabled": not use_custom_conversion_script,
                "on_change": save_repo_path
            }
            if "custom_llama_cpp_repo_input" not in st.session_state:
                repo_path_kwargs["value"] = config.get("custom_llama_cpp_repo", "")
            repo_path = st.text_input(**repo_path_kwargs)  # type: ignore[arg-type]

        if TKINTER_AVAILABLE:
            with col_repo_browse:  # type: ignore[union-attr]
                if st.button(
                    "Select Folder",
                    key="browse_repo_folder_btn",
                    use_container_width=True,
                    help="Select llama.cpp repository folder",
                    disabled=not use_custom_conversion_script
                ):
                    repo_path_clean = strip_quotes(repo_path)
                    initial_dir = repo_path_clean if repo_path_clean and Path(repo_path_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["custom_llama_cpp_repo"] = selected_folder
                        save_config(config)
                        # Set pending flag - will be applied before widget creation on next run
                        st.session_state.pending_repo_folder = selected_folder
                        # Reinitialize converter
                        custom_binaries = config.get("custom_binaries_folder", "") if config.get("use_custom_binaries", False) else None
                        from ..converter import GGUFConverter
                        st.session_state.converter = GGUFConverter(
                            custom_binaries_folder=custom_binaries,
                            custom_llama_cpp_repo=selected_folder
                        )
                        st.rerun()

        with col_repo_check:
            repo_path_clean = strip_quotes(repo_path)
            repo_path_exists = bool(repo_path_clean and Path(repo_path_clean).exists())
            if st.button(
                "Open Folder",
                key="check_repo_folder_btn",
                use_container_width=True,
                disabled=not use_custom_conversion_script or not repo_path_exists,
                help="Open folder in file explorer" if repo_path_exists else "Path doesn't exist yet"
            ):
                if repo_path_exists:
                    try:
                        open_folder(repo_path_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

        # Show help text below folder path
        st.markdown("""
        Enable custom repository if you want to:
        - Use a modified or forked llama.cpp with custom conversion scripts
        - Use a specific llama.cpp version for conversion
        """)

    with col_script2:
        st.subheader("Conversion Script Detection")

        # If custom repo enabled, check for script
        if config.get("use_custom_conversion_script", False):
            custom_repo = config.get("custom_llama_cpp_repo", "")

            if custom_repo:
                st.info(f"Looking in: `{custom_repo}`")

                # Check if script exists
                script_path = Path(custom_repo) / "convert_hf_to_gguf.py"
                script_found = script_path.exists()

                if script_found:
                    st.success("Conversion script found")

                    # Try to get version info from git
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["git", "describe", "--tags", "--always"],
                            cwd=custom_repo,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            version = result.stdout.strip()
                            st.code(f"version: {version}", language=None)
                        else:
                            st.info("Version: Unable to detect (not a git repository)")
                    except Exception:
                        st.info("Version: Unable to detect")

                    st.markdown(f"- `convert_hf_to_gguf.py`: {script_path}")
                else:
                    st.error("Conversion script not found")
                    st.markdown("- `convert_hf_to_gguf.py`: Not found")
                    st.warning(f"Expected location: `{Path(custom_repo) / 'convert_hf_to_gguf.py'}`")
            else:
                # No custom path specified, using YaGGUF's repo
                st.info("Using YaGGUF's auto-cloned llama.cpp repository")

                # Check if script exists
                project_root = Path(__file__).parent.parent.parent
                script_path = project_root / "llama.cpp" / "convert_hf_to_gguf.py"

                if script_path.exists():
                    st.success("Conversion script found")

                    # Try to get version info from git
                    try:
                        import subprocess
                        llama_cpp_dir = project_root / "llama.cpp"
                        result = subprocess.run(
                            ["git", "describe", "--tags", "--always"],
                            cwd=llama_cpp_dir,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            version = result.stdout.strip()
                            st.code(f"version: {version}", language=None)
                        else:
                            st.info("Version: Unable to detect (not a git repository)")
                    except Exception:
                        st.info("Version: Unable to detect")

                    st.markdown(f"- `convert_hf_to_gguf.py`: {script_path}")
                else:
                    st.warning("Conversion script not found in YaGGUF's llama.cpp repository")
        else:
            st.info("Disabled - Using YaGGUF's auto-cloned llama.cpp repository")

    st.markdown("---")

    # YaGGUF Binary Information (Reference)
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.subheader("YaGGUF Binary Information (for comparison)")

    with col_info2:
        binary_info = get_binary_version(st.session_state.converter)
        if binary_info["status"] == "ok" and binary_info.get("version"):
            st.code(binary_info["version"], language=None)
        else:
            st.info("Not installed")
