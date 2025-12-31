"""
Streamlit GUI for GGUF Converter
"""

import streamlit as st
import sys
from pathlib import Path
import multiprocessing

# Handle both direct execution and module import
try:
    from .converter import GGUFConverter
    from .gui_utils import (
        load_config, save_config, reset_config, get_default_config
    )
    from .gui_tabs import (
        render_convert_tab,
        render_imatrix_settings_tab,
        render_imatrix_stats_tab,
        render_downloader_tab,
        render_llama_cpp_tab,
        render_info_tab,
        render_update_tab
    )
except ImportError:
    # Add parent directory for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gguf_converter.converter import GGUFConverter
    from gguf_converter.gui_utils import (
        load_config, save_config, reset_config, get_default_config
    )
    from gguf_converter.gui_tabs import (
        render_convert_tab,
        render_imatrix_settings_tab,
        render_imatrix_stats_tab,
        render_downloader_tab,
        render_llama_cpp_tab,
        render_info_tab,
        render_update_tab
    )


def main() -> None:
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

    st.title("YaGGUF - Yet Another GGUF Converter")

    # Initialize converter with custom settings if specified
    if 'converter' not in st.session_state:
        config = st.session_state.get('config', load_config())
        custom_binaries = config.get("custom_binaries_folder", "") if config.get("use_custom_binaries", False) else None
        custom_repo = config.get("custom_llama_cpp_repo", "") if config.get("use_custom_conversion_script", False) else None
        st.session_state.converter = GGUFConverter(
            custom_binaries_folder=custom_binaries,
            custom_llama_cpp_repo=custom_repo
        )

    # Load config on first run
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

    if 'reset_count' not in st.session_state:
        st.session_state.reset_count = 0

    # Handle model path from downloader (before widget creation)
    if 'pending_model_path' in st.session_state:
        st.session_state.model_path_input = st.session_state.pending_model_path
        del st.session_state.pending_model_path

    # Handle reset to defaults (before widget creation)
    if 'pending_reset_defaults' in st.session_state:
        defaults = get_default_config()
        st.session_state.verbose_checkbox = defaults["verbose"]
        del st.session_state.pending_reset_defaults

    converter = st.session_state.converter
    config = st.session_state.config

    with st.sidebar:
        st.header("Settings")
        st.markdown("[YaGGUF (GitHub)](https://github.com/usrname0/YaGGUF)")
        st.markdown("---")

        def save_verbose():
            config["verbose"] = st.session_state.verbose_checkbox
            save_config(config)

        # Only set value if not already in session state (prevents warning)
        verbose_kwargs = {
            "label": "Verbose output",
            "help": "Show detailed command output in the terminal for debugging and monitoring progress",
            "key": "verbose_checkbox",
            "on_change": save_verbose
        }
        if "verbose_checkbox" not in st.session_state:
            verbose_kwargs["value"] = config.get("verbose", False)
        verbose = st.checkbox(**verbose_kwargs)  # type: ignore[arg-type]
        st.markdown("---")
        st.markdown("**Performance:**")
        max_workers = multiprocessing.cpu_count()
        default_threads = max(1, max_workers - 1)

        def save_num_threads():
            config["num_threads"] = int(st.session_state.num_threads_input)
            save_config(config)

        num_threads = st.number_input(
            "Thread count",
            min_value=1,
            max_value=max_workers,
            value=int(config.get("num_threads") or default_threads),
            step=1,
            help=f"Number of threads for llama.cpp (logical processors: {max_workers}, default: {default_threads} to keep system responsive)",
            key="num_threads_input",
            on_change=save_num_threads
        )

        st.markdown("---")
        if st.button("Reset to defaults", use_container_width=True, help="Reset all settings to default values"):
            st.session_state.config = reset_config()
            st.session_state.reset_count += 1
            if "download_just_completed" in st.session_state:
                st.session_state.download_just_completed = False
            st.session_state.model_path_input = ""
            if "pending_model_path" in st.session_state:
                del st.session_state.pending_model_path
            keys_to_delete = [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(('trad_', 'k_', 'i_'))]
            for key in keys_to_delete:
                del st.session_state[key]
            if "iq_checkbox_states" in st.session_state:
                st.session_state.iq_checkbox_states = {}
            st.session_state.pending_reset_defaults = True
            st.rerun()

    # Main content - tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Convert & Quantize",
        "Imatrix Settings",
        "Imatrix Statistics",
        "HuggingFace Downloader",
        "Info",
        "llama.cpp",
        "Update"
    ])

    with tab1:
        render_convert_tab(converter, config, verbose, num_threads)

    with tab2:
        render_imatrix_settings_tab(converter, config)

    with tab3:
        render_imatrix_stats_tab(converter, config)

    with tab4:
        render_downloader_tab(converter, config)

    with tab5:
        render_info_tab(converter, config)

    with tab6:
        render_llama_cpp_tab(converter, config)

    with tab7:
        render_update_tab(converter, config)


if __name__ == "__main__":
    main()
