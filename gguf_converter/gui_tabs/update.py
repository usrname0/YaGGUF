"""
Update tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import subprocess
import platform
import io
import os
import re
import sys
from contextlib import redirect_stdout
from typing import Dict, Any, TYPE_CHECKING
from colorama import Style
from ..theme import THEME as theme


def strip_ansi_codes(text: str) -> str:
    """
    Strip ANSI color/style escape codes from text for clean GUI display.

    Args:
        text: String potentially containing ANSI escape sequences

    Returns:
        Clean string with all ANSI codes removed
    """
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

from ..gui_utils import (
    get_current_version, check_git_updates_available,
    run_and_stream_command, get_binary_version,
    display_binary_version_status, get_conversion_scripts_info,
    display_conversion_scripts_version_status,
    strip_quotes, open_folder, browse_folder, save_config,
    get_binary_version_from_path, TKINTER_AVAILABLE, get_platform_path
)

if TYPE_CHECKING:
    from ..converter import GGUFConverter


class TeeOutput(io.TextIOBase):
    """
    Write to both StringIO (for GUI capture) and stdout (for terminal output)
    """
    def __init__(self, stringio_obj: io.StringIO) -> None:
        self.stringio = stringio_obj
        # Use sys.__stdout__ to bypass Streamlit's stdout capture and write
        # directly to the real terminal. Falls back to sys.stdout if unavailable.
        self.terminal = sys.__stdout__ or sys.stdout

    def write(self, message: str) -> int:
        self.stringio.write(message)
        self.terminal.write(message)
        return len(message)

    def flush(self) -> None:
        self.stringio.flush()
        self.terminal.flush()


def render_update_tab(converter: "GGUFConverter", config: Dict[str, Any]) -> None:
    """Render the Update tab"""
    st.header("Update")

    # Initialize update message in session state if not present
    if 'yagguf_update_message' not in st.session_state:
        st.session_state.yagguf_update_message = None

    # Handle pending folder updates from browse dialogs (must run before widgets are created)
    if 'pending_binaries_folder' in st.session_state:
        st.session_state.custom_binaries_folder_input_update = st.session_state.pending_binaries_folder
        del st.session_state.pending_binaries_folder
    if 'pending_repo_folder' in st.session_state:
        st.session_state.custom_llama_cpp_repo_input = st.session_state.pending_repo_folder
        del st.session_state.pending_repo_folder

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Update YaGGUF")
        st.markdown("Update to the latest version of YaGGUF from GitHub.")
        st.markdown("The GUI will close and restart automatically. All updates will run in the terminal.")
        st.markdown("[View YaGGUF on GitHub](https://github.com/usrname0/YaGGUF)")
        if st.button("Update YaGGUF & Restart"):
            # Check what version is available
            update_status = check_git_updates_available()

            if update_status["status"] == "updates_available" and update_status.get("latest_version"):
                # Get the current Streamlit port from the running URL
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(st.context.url)
                    port = str(parsed_url.port) if parsed_url.port else "8501"
                except Exception:
                    # Fallback to default port if we can't detect it
                    port = "8501"

                # Determine platform-specific restart script
                is_windows = platform.system() == "Windows"
                if is_windows:
                    restart_script = Path(__file__).parent.parent.parent / "scripts" / "update_yagguf_and_restart.bat"
                else:
                    restart_script = Path(__file__).parent.parent.parent / "scripts" / "update_yagguf_and_restart.sh"

                if restart_script.exists():
                    # Store the update message in session state and force a rerun to display it
                    update_msg = f"Updating to {update_status['latest_version']}... See terminal for progress. Please wait."
                    st.session_state.yagguf_update_message = update_msg
                    st.session_state.pending_update = {
                        'script': str(restart_script),
                        'port': port,
                        'version': update_status["latest_version"],
                        'is_windows': is_windows
                    }
                    st.rerun()
                else:
                    st.error(f"Restart script not found: {restart_script}")
            elif update_status["status"] == "up_to_date":
                # Already on latest version
                st.toast("Already on the latest version")
            else:
                # Error checking for updates
                st.toast("Could not check for updates")

    with col2:
        st.subheader("YaGGUF Version Information")
        current_version = get_current_version()
        st.code(f"version: {current_version}", language=None)

        # Display update message if present, otherwise show version comparison
        if st.session_state.yagguf_update_message:
            st.info(st.session_state.yagguf_update_message)

            # If there's a pending update, execute it after displaying the message
            if 'pending_update' in st.session_state:
                pending = st.session_state.pending_update

                # Give Streamlit time to send the message to browser before exiting
                import time
                time.sleep(1.0)

                # Start the update script with port and version parameters
                if pending['is_windows']:
                    subprocess.Popen(
                        ["cmd", "/c", pending['script'], pending['port'], pending['version']],
                        cwd=str(Path(pending['script']).parent)
                    )
                else:
                    subprocess.Popen(
                        ["/bin/bash", pending['script'], pending['port'], pending['version']],
                        cwd=str(Path(pending['script']).parent)
                    )

                # Clear the pending update
                del st.session_state.pending_update

                # Exit Streamlit - this releases the port
                os._exit(0)
        else:
            # Check if updates are available
            update_status = check_git_updates_available()
            if update_status["status"] == "updates_available":
                st.warning(update_status["message"])
            elif update_status["status"] == "up_to_date":
                st.info(update_status["message"])
            else:
                st.info(update_status["message"])

    st.markdown("---")

    # Update YaGGUF Binaries Section
    col_bin1, col_bin2 = st.columns(2)
    with col_bin1:
        st.subheader("Update: Llama.cpp Binaries")
        st.markdown("Update the `llama.cpp` binaries. Choose **Recommended** for the tested version bundled with YaGGUF, or **Latest** for the newest llama.cpp release.")
        st.markdown("[View llama.cpp on GitHub](https://github.com/ggml-org/llama.cpp)")

        # GPU backend selector (platform-specific options)
        available_backends = st.session_state.converter.llama_cpp_manager.get_available_gpu_backends()
        if len(available_backends) > 1:
            backend_labels = [label for label, _ in available_backends]
            backend_values = [value for _, value in available_backends]
            bin_info = st.session_state.converter.llama_cpp_manager._read_bin_info()
            installed_backend = bin_info.get("gpu_backend")
            saved_backend = installed_backend if installed_backend and installed_backend in backend_values else backend_values[0][1]
            saved_index = backend_values.index(saved_backend)

            st.radio(
                "GPU Backend",
                options=backend_labels,
                index=saved_index,
                key="gpu_backend_selector",
                help="Select the GPU backend for llama.cpp. Takes effect when downloading binaries. CPU works everywhere; CUDA/Vulkan require compatible hardware."
            )
            selected_backend = backend_values[backend_labels.index(st.session_state.get("gpu_backend_selector", backend_labels[saved_index]))]
        else:
            selected_backend = available_backends[0][1]

        if st.button("Update Binaries - Recommended Version"):
            output_container = st.empty()
            output_container.code("Starting binary update (Recommended version)...\nThis may take a moment.", language='bash')

            f = io.StringIO()
            tee = TeeOutput(f)
            with redirect_stdout(tee):  # type: ignore[arg-type]
                try:
                    st.session_state.converter.llama_cpp_manager.update_binaries(gpu_backend=selected_backend)
                    st.toast("Binaries updated successfully!")
                    success = True
                except Exception as e:
                    print(f"\n{theme['error']}--- An error occurred ---{Style.RESET_ALL}\n{str(e)}")
                    st.toast(f"An error occurred during binary update: {e}")
                    success = False

            output = f.getvalue()
            output_container.code(strip_ansi_codes(output), language='bash')

            if success:
                st.rerun()

        if st.button("Update Binaries - Latest Version"):
            output_container = st.empty()
            output_container.code("Fetching latest llama.cpp version...\nThis may take a moment.", language='bash')

            f = io.StringIO()
            tee = TeeOutput(f)
            with redirect_stdout(tee):  # type: ignore[arg-type]
                try:
                    latest_version = st.session_state.converter.llama_cpp_manager.get_latest_version()
                    print(f"{theme['info']}Latest version: {latest_version}{Style.RESET_ALL}")
                    st.session_state.converter.llama_cpp_manager.update_binaries(version=latest_version, gpu_backend=selected_backend)
                    st.toast("Binaries updated successfully!")
                    success = True
                except Exception as e:
                    print(f"\n{theme['error']}--- An error occurred ---{Style.RESET_ALL}\n{str(e)}")
                    st.toast(f"An error occurred during binary update: {e}")
                    success = False

            output = f.getvalue()
            output_container.code(strip_ansi_codes(output), language='bash')

            if success:
                st.rerun()

    with col_bin2:
        st.subheader("Binary Information")
        binary_info = get_binary_version(st.session_state.converter)
        if binary_info["status"] == "ok":
            st.success(binary_info['message'])
        elif binary_info["status"] == "missing":
            st.warning(binary_info['message'])
        else:
            st.error(binary_info['message'])

        display_binary_version_status(st.session_state.converter)

    # Custom binaries row — checkbox left, path right (same row = guaranteed alignment)
    col_cbin_check, col_cbin_path = st.columns(2)
    with col_cbin_check:
        st.markdown("**Custom Binaries** — use your own compiled llama.cpp with GPU support (CUDA/ROCm/Metal/Vulkan) for imatrix generation.")

        def save_use_custom_binaries():
            config["use_custom_binaries"] = st.session_state.use_custom_binaries_checkbox_update
            save_config(config)
            if "custom_binary_versions" in st.session_state:
                del st.session_state.custom_binary_versions
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

    with col_cbin_path:
        if use_custom_binaries:
            if TKINTER_AVAILABLE:
                col_folder, col_browse, col_open = st.columns([4, 1, 1])
            else:
                col_folder, col_open = st.columns([5, 1])
                col_browse = None

            with col_folder:
                def save_binaries_folder():
                    new_folder = st.session_state.custom_binaries_folder_input_update
                    if new_folder != config.get("custom_binaries_folder", ""):
                        config["custom_binaries_folder"] = new_folder
                        save_config(config)
                        if "custom_binary_versions" in st.session_state:
                            del st.session_state.custom_binary_versions
                        from ..converter import GGUFConverter
                        st.session_state.converter = GGUFConverter(custom_binaries_folder=new_folder)

                binaries_folder_kwargs = {
                    "label": "llama.cpp binaries folder",
                    "placeholder": get_platform_path(
                        "C:\\path\\to\\llama.cpp\\bin or leave blank for PATH",
                        "/path/to/llama.cpp/bin or leave blank for PATH"
                    ),
                    "help": "Path to folder containing llama-quantize and llama-imatrix. Leave blank to use system PATH.",
                    "key": "custom_binaries_folder_input_update",
                    "label_visibility": "collapsed",
                    "on_change": save_binaries_folder
                }
                if "custom_binaries_folder_input_update" not in st.session_state:
                    binaries_folder_kwargs["value"] = config.get("custom_binaries_folder", "")
                binaries_folder = st.text_input(**binaries_folder_kwargs)  # type: ignore[arg-type]

            if TKINTER_AVAILABLE:
                with col_browse:  # type: ignore[union-attr]
                    if st.button("Select", key="browse_binaries_folder_btn", use_container_width=True, help="Select binaries folder"):
                        binaries_folder_clean = strip_quotes(binaries_folder)
                        initial_dir = binaries_folder_clean if binaries_folder_clean and Path(binaries_folder_clean).exists() else None
                        selected_folder = browse_folder(initial_dir)
                        if selected_folder:
                            config["custom_binaries_folder"] = selected_folder
                            save_config(config)
                            if "custom_binary_versions" in st.session_state:
                                del st.session_state.custom_binary_versions
                            st.session_state.pending_binaries_folder = selected_folder
                            from ..converter import GGUFConverter
                            st.session_state.converter = GGUFConverter(custom_binaries_folder=selected_folder)
                            st.rerun()

            with col_open:
                binaries_folder_clean = strip_quotes(binaries_folder)
                binaries_folder_exists = bool(binaries_folder_clean and Path(binaries_folder_clean).exists())
                if st.button(
                    "Open",
                    key="check_binaries_folder_btn",
                    use_container_width=True,
                    disabled=not binaries_folder_exists,
                    help="Open folder in file explorer" if binaries_folder_exists else "Path doesn't exist yet"
                ):
                    try:
                        open_folder(binaries_folder_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

            # Binary detection status
            custom_folder = config.get("custom_binaries_folder", "")
            quantize_found = False
            imatrix_found = False
            quantize_path = None
            imatrix_path = None
            try:
                quantize_path = st.session_state.converter.llama_cpp_manager.get_quantize_path()
                quantize_found = quantize_path.exists()
            except (RuntimeError, OSError, FileNotFoundError):
                pass
            try:
                imatrix_path = st.session_state.converter.llama_cpp_manager.get_imatrix_path()
                imatrix_found = imatrix_path.exists()
            except (RuntimeError, OSError, FileNotFoundError):
                pass

            if quantize_found and imatrix_found:
                st.success("All required binaries found")
                if "custom_binary_versions" not in st.session_state:
                    st.session_state.custom_binary_versions = {}
                cache_key = f"{custom_folder}_{imatrix_found}"
                if cache_key not in st.session_state.custom_binary_versions:
                    with st.spinner("Detecting version..."):
                        system_version = get_binary_version_from_path(imatrix_path)
                    if system_version:
                        st.session_state.custom_binary_versions[cache_key] = system_version
                else:
                    system_version = st.session_state.custom_binary_versions[cache_key]
                if system_version:
                    st.code(system_version, language=None)
            elif quantize_found or imatrix_found:
                st.warning("Some binaries missing")
            else:
                st.error("No binaries found")

    st.markdown("---")

    # Update Conversion Scripts Section
    col_scripts1, col_scripts2 = st.columns(2)
    with col_scripts1:
        st.subheader("Update: Llama.cpp Conversion Scripts")
        st.markdown("Update the `llama.cpp` repository that contains the `convert_hf_to_gguf.py` script. Choose **Recommended** for the tested version matching YaGGUF binaries, or **Latest** for the newest conversion scripts.")
        st.markdown("[View llama.cpp on GitHub](https://github.com/ggml-org/llama.cpp)")

        if st.button("Update Conversion Scripts - Recommended Version"):
            output_container = st.empty()
            output_container.code("Updating conversion scripts (Recommended version)...\nThis may take a moment.", language='bash')

            f = io.StringIO()
            tee = TeeOutput(f)
            with redirect_stdout(tee):  # type: ignore[arg-type]
                try:
                    result = st.session_state.converter.llama_cpp_manager.update_conversion_scripts()
                    if result['status'] in ['success', 'already_updated']:
                        st.toast("Conversion scripts updated successfully!")
                        success = True
                    else:
                        print(f"\n{theme['error']}--- An error occurred ---{Style.RESET_ALL}\n{result['message']}")
                        st.toast(f"An error occurred: {result['message']}")
                        success = False
                except Exception as e:
                    print(f"\n{theme['error']}--- An error occurred ---{Style.RESET_ALL}\n{str(e)}")
                    st.toast(f"An error occurred during update: {e}")
                    success = False

            output = f.getvalue()
            output_container.code(strip_ansi_codes(output), language='bash')

            if success:
                st.rerun()

        if st.button("Update Conversion Scripts - Latest Version"):
            output_container = st.empty()
            output_container.code("Fetching latest llama.cpp version...\nThis may take a moment.", language='bash')

            f = io.StringIO()
            tee = TeeOutput(f)
            with redirect_stdout(tee):  # type: ignore[arg-type]
                try:
                    latest_version = st.session_state.converter.llama_cpp_manager.get_latest_version()
                    print(f"{theme['info']}Latest version: {latest_version}{Style.RESET_ALL}")
                    result = st.session_state.converter.llama_cpp_manager.update_conversion_scripts(version=latest_version)
                    if result['status'] in ['success', 'already_updated']:
                        st.toast("Conversion scripts updated successfully!")
                        success = True
                    else:
                        print(f"\n{theme['error']}--- An error occurred ---{Style.RESET_ALL}\n{result['message']}")
                        st.toast(f"An error occurred: {result['message']}")
                        success = False
                except Exception as e:
                    print(f"\n{theme['error']}--- An error occurred ---{Style.RESET_ALL}\n{str(e)}")
                    st.toast(f"An error occurred during update: {e}")
                    success = False

            output = f.getvalue()
            output_container.code(strip_ansi_codes(output), language='bash')

            if success:
                st.rerun()

    with col_scripts2:
        st.subheader("Conversion Scripts Information")
        scripts_info = get_conversion_scripts_info(st.session_state.converter)

        if scripts_info["status"] == "ok":
            st.success(scripts_info['message'])
        elif scripts_info["status"] == "missing":
            st.warning(scripts_info['message'])
        else:
            st.error(scripts_info['message'])

        display_conversion_scripts_version_status(st.session_state.converter)

    # Custom conversion scripts row — checkbox left, path right (same row = guaranteed alignment)
    col_cscr_check, col_cscr_path = st.columns(2)
    with col_cscr_check:
        st.markdown("**Custom Conversion Scripts** — use a custom llama.cpp repository for `convert_hf_to_gguf.py` (modified/forked or specific version).")

        def save_use_custom_conversion_script():
            config["use_custom_conversion_script"] = st.session_state.use_custom_conversion_script_checkbox
            save_config(config)
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

    with col_cscr_path:
        if use_custom_conversion_script:
            if TKINTER_AVAILABLE:
                col_repo, col_repo_browse, col_repo_open = st.columns([4, 1, 1])
            else:
                col_repo, col_repo_open = st.columns([5, 1])
                col_repo_browse = None

            with col_repo:
                def save_repo_path():
                    new_path = st.session_state.custom_llama_cpp_repo_input
                    if new_path != config.get("custom_llama_cpp_repo", ""):
                        config["custom_llama_cpp_repo"] = new_path
                        save_config(config)
                        custom_binaries = config.get("custom_binaries_folder", "") if config.get("use_custom_binaries", False) else None
                        from ..converter import GGUFConverter
                        st.session_state.converter = GGUFConverter(
                            custom_binaries_folder=custom_binaries,
                            custom_llama_cpp_repo=new_path
                        )

                repo_path_kwargs = {
                    "label": "llama.cpp repository path",
                    "placeholder": get_platform_path(
                        "C:\\path\\to\\llama.cpp or leave blank for YaGGUF's repo",
                        "/path/to/llama.cpp or leave blank for YaGGUF's repo"
                    ),
                    "help": "Path to llama.cpp repository containing convert_hf_to_gguf.py.",
                    "key": "custom_llama_cpp_repo_input",
                    "label_visibility": "collapsed",
                    "on_change": save_repo_path
                }
                if "custom_llama_cpp_repo_input" not in st.session_state:
                    repo_path_kwargs["value"] = config.get("custom_llama_cpp_repo", "")
                repo_path = st.text_input(**repo_path_kwargs)  # type: ignore[arg-type]

            if TKINTER_AVAILABLE:
                with col_repo_browse:  # type: ignore[union-attr]
                    if st.button("Select", key="browse_repo_folder_btn", use_container_width=True, help="Select llama.cpp repository folder"):
                        repo_path_clean = strip_quotes(repo_path)
                        initial_dir = repo_path_clean if repo_path_clean and Path(repo_path_clean).exists() else None
                        selected_folder = browse_folder(initial_dir)
                        if selected_folder:
                            config["custom_llama_cpp_repo"] = selected_folder
                            save_config(config)
                            st.session_state.pending_repo_folder = selected_folder
                            custom_binaries = config.get("custom_binaries_folder", "") if config.get("use_custom_binaries", False) else None
                            from ..converter import GGUFConverter
                            st.session_state.converter = GGUFConverter(
                                custom_binaries_folder=custom_binaries,
                                custom_llama_cpp_repo=selected_folder
                            )
                            st.rerun()

            with col_repo_open:
                repo_path_clean = strip_quotes(repo_path)
                repo_path_exists = bool(repo_path_clean and Path(repo_path_clean).exists())
                if st.button(
                    "Open",
                    key="check_repo_folder_btn",
                    use_container_width=True,
                    disabled=not repo_path_exists,
                    help="Open folder in file explorer" if repo_path_exists else "Path doesn't exist yet"
                ):
                    try:
                        open_folder(repo_path_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

            # Conversion script detection
            custom_repo = config.get("custom_llama_cpp_repo", "")
            if custom_repo:
                script_path = Path(custom_repo) / "convert_hf_to_gguf.py"
                if script_path.exists():
                    st.success("Conversion script found")
                else:
                    st.error(f"Script not found at `{script_path}`")

    st.markdown("---")

    st.subheader("Update Dependencies")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("Update PyTorch and Python dependencies from `requirements.txt`.")
        st.markdown("The GUI will close and restart automatically. All updates will run in the terminal.")

        if st.button("Update Dependencies & Restart"):
            # Get the current Streamlit port from the running URL
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(st.context.url)
                port = str(parsed_url.port) if parsed_url.port else "8501"
            except Exception:
                # Fallback to default port if we can't detect it
                port = "8501"

            # Determine platform-specific restart script
            is_windows = platform.system() == "Windows"
            if is_windows:
                restart_script = Path(__file__).parent.parent.parent / "scripts" / "update_dependencies_and_restart.bat"
            else:
                restart_script = Path(__file__).parent.parent.parent / "scripts" / "update_dependencies_and_restart.sh"

            if restart_script.exists():
                st.info("See terminal for update progress... Please wait.")

                # Give Streamlit time to send the message to browser before exiting
                import time
                time.sleep(0.5)

                # Start the update script with port parameter - output will show in original terminal
                if is_windows:
                    subprocess.Popen(
                        ["cmd", "/c", str(restart_script), port],
                        cwd=str(restart_script.parent)
                    )
                else:
                    subprocess.Popen(
                        ["/bin/bash", str(restart_script), port],
                        cwd=str(restart_script.parent)
                    )

                # Exit Streamlit - this releases the port
                os._exit(0)
            else:
                st.error(f"Restart script not found: {restart_script}")

    with col4:
        try:
            req_path = Path(__file__).parent.parent.parent / "requirements.txt"
            if req_path.exists():
                st.markdown("`requirements.txt`")
                st.code(req_path.read_text(), language='text')
            else:
                st.warning("`requirements.txt` not found.")
        except Exception as e:
            st.error(f"Error reading requirements.txt: {e}")
