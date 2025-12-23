"""
llama.cpp tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, get_binary_version,
    display_binary_version_status, get_binary_version_from_path
)


def render_llama_cpp_tab(converter, config):
    """Render the llama.cpp tab"""
    st.header("llama.cpp Binaries")

    st.markdown("""
    By default, YaGUFF automatically downloads pre-compiled llama.cpp binaries that use the CPU (good for most cases).
    You can update binaries from the **Update** tab.
    """)

    # YaGUFF Binary Information (Reference)
    st.subheader("YaGUFF Binary Information")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("[View llama.cpp on GitHub](https://github.com/ggml-org/llama.cpp)")

    with col_info2:
        binary_info = get_binary_version(st.session_state.converter)
        if binary_info["status"] == "ok":
            st.success(binary_info['message'])
        elif binary_info["status"] == "missing":
            st.warning(binary_info['message'])
        else:
            st.error(binary_info['message'])

        display_binary_version_status(st.session_state.converter)

    st.markdown("---")

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
            if config.get("use_custom_binaries", False):
                custom_folder = config.get("custom_binaries_folder", "")
                from ..converter import GGUFConverter
                st.session_state.converter = GGUFConverter(custom_binaries_folder=custom_folder)
            else:
                from ..converter import GGUFConverter
                st.session_state.converter = GGUFConverter()

        use_custom_binaries = st.checkbox(
            "Use local/custom binaries",
            value=config.get("use_custom_binaries", False),
            help="Use local or system llama.cpp binaries instead of YaGUFF's auto-downloaded ones. Leave path blank to use system PATH.",
            key="use_custom_binaries_checkbox_update",
            on_change=save_use_custom_binaries
        )

        st.markdown("**Binaries Folder Path** (leave blank for system PATH):")

        # Single folder path with Browse and Check Folder buttons
        col_folder, col_browse, col_check = st.columns([4, 1, 1])
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

            binaries_folder = st.text_input(
                "llama.cpp binaries folder",
                value=config.get("custom_binaries_folder", ""),
                placeholder="D:/llama.cpp/build/bin or leave blank for PATH",
                help="Path to folder containing llama-quantize and llama-imatrix. Leave blank to use system PATH.",
                key="custom_binaries_folder_input_update",
                label_visibility="collapsed",
                disabled=not use_custom_binaries,
                on_change=save_binaries_folder
            )
        with col_browse:
            if st.button(
                "Browse",
                key="browse_binaries_folder_btn",
                use_container_width=True,
                help="Browse for binaries folder",
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
                    # Reinitialize converter
                    from ..converter import GGUFConverter
                    st.session_state.converter = GGUFConverter(custom_binaries_folder=selected_folder)
                    st.rerun()
        with col_check:
            binaries_folder_clean = strip_quotes(binaries_folder)
            binaries_folder_exists = bool(binaries_folder_clean and Path(binaries_folder_clean).exists())
            if st.button(
                "Check Folder",
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
        - Use binaries from your system PATH
        - Use a specific llama.cpp version

        **Note:** Custom binaries with GPU support are required for GPU offloading in imatrix generation (see Imatrix Settings tab).
        """)

    with col_bin2:
        col_header, col_refresh = st.columns([4, 1])
        with col_header:
            st.subheader("Local/Custom Binary Detection")
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
                quantize_path = st.session_state.converter.binary_manager.get_quantize_path()
                quantize_found = quantize_path.exists()
            except (RuntimeError, OSError, FileNotFoundError):
                # Binary not found - quantize_found stays False, UI will show appropriate status
                pass

            try:
                imatrix_path = st.session_state.converter.binary_manager.get_imatrix_path()
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
                    system_version = get_binary_version_from_path(imatrix_path)
                    st.session_state.custom_binary_versions[cache_key] = system_version
                else:
                    system_version = st.session_state.custom_binary_versions[cache_key]

                if system_version:
                    st.code(system_version, language=None)
                else:
                    st.info("Version: Unable to detect")

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
                        imatrix_version = get_binary_version_from_path(imatrix_path)
                        st.session_state.custom_binary_versions[cache_key] = imatrix_version
                    else:
                        imatrix_version = st.session_state.custom_binary_versions[cache_key]

                    if imatrix_version:
                        st.code(imatrix_version, language=None)
                else:
                    st.markdown("- `llama-imatrix`: Not found")
            else:
                st.error("No binaries found")
                st.markdown("- `llama-quantize`: Not found")
                st.markdown("- `llama-imatrix`: Not found")
        else:
            st.info("Disabled - Using YaGUFF binaries")
