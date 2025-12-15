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
        render_info_tab,
        render_update_tab
    )
except ImportError:
    # Add parent directory to path for direct execution
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
        render_info_tab,
        render_update_tab
    )


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

    st.title("YaGUFF - Yet Another GGUF Converter")
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

    # Handle setting model path from downloader (must happen before widget creation)
    if 'pending_model_path' in st.session_state:
        st.session_state.model_path_input = st.session_state.pending_model_path
        del st.session_state.pending_model_path

    # Handle reset to defaults (must happen before widget creation)
    if 'pending_reset_defaults' in st.session_state:
        defaults = get_default_config()
        st.session_state.verbose_checkbox = defaults["verbose"]
        st.session_state.incompatibility_warnings_checkbox = not defaults["ignore_incompatibilities"]
        del st.session_state.pending_reset_defaults

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

        # Only set value if not already in session state (prevents warning)
        verbose_kwargs = {
            "label": "Verbose output",
            "help": "Show detailed command output in the terminal for debugging and monitoring progress",
            "key": "verbose_checkbox",
            "on_change": save_verbose
        }
        if "verbose_checkbox" not in st.session_state:
            verbose_kwargs["value"] = config.get("verbose", False)
        verbose = st.checkbox(**verbose_kwargs)

        # Auto-save callback for incompatibility warnings (inverted logic)
        def save_incompatibility_warnings():
            # Checkbox checked = warnings ON, so ignore_incompatibilities = False
            config["ignore_incompatibilities"] = not st.session_state.incompatibility_warnings_checkbox
            save_config(config)

        # Only set value if not already in session state (prevents warning)
        incomp_kwargs = {
            "label": "Incompatibility warnings",
            "help": "Detect and prevent incompatible quantizations (e.g., IQ quants on Qwen models). Uncheck to override warnings (advanced users only).",
            "key": "incompatibility_warnings_checkbox",
            "on_change": save_incompatibility_warnings
        }
        if "incompatibility_warnings_checkbox" not in st.session_state:
            incomp_kwargs["value"] = not config.get("ignore_incompatibilities", False)  # Inverted: default to checked (warnings ON)
        incompatibility_warnings_enabled = st.checkbox(**incomp_kwargs)

        # For internal use, invert back to ignore_incompatibilities
        ignore_incompatibilities = not incompatibility_warnings_enabled

        st.markdown("---")
        st.markdown("**Performance:**")
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
            # Clear download success message
            if "download_just_completed" in st.session_state:
                st.session_state.download_just_completed = False
            # Clear model path widget state - set to empty instead of deleting
            st.session_state.model_path_input = ""
            # Also clear pending model path flag if it exists
            if "pending_model_path" in st.session_state:
                del st.session_state.pending_model_path
            # Clear all quantization checkbox states
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith(('trad_', 'k_', 'i_'))]
            for key in keys_to_delete:
                del st.session_state[key]
            # Clear IQ checkbox state tracking
            if "iq_checkbox_states" in st.session_state:
                st.session_state.iq_checkbox_states = {}
            # Set pending flag to update verbose and incompatibility warnings on next run
            st.session_state.pending_reset_defaults = True
            st.rerun()

    # Main content - tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Convert & Quantize",
        "Imatrix Settings",
        "Imatrix Statistics",
        "HuggingFace Downloader",
        "Info",
        "Update"
    ])

    with tab1:
        render_convert_tab(converter, config, verbose, nthreads, ignore_incompatibilities)

    with tab2:
        render_imatrix_settings_tab(converter, config)

    with tab3:
        render_imatrix_stats_tab(converter, config)

    with tab4:
        render_downloader_tab(converter, config)

    with tab5:
        render_info_tab(converter, config)

    with tab6:
        render_update_tab(converter, config)


if __name__ == "__main__":
    main()
