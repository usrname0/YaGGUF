"""
Imatrix Statistics tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, TKINTER_AVAILABLE
)


def render_imatrix_stats_tab(converter: Any, config: Dict[str, Any]) -> None:
    """Render the Imatrix Statistics tab"""
    st.header("Imatrix Statistics")

    # Get verbose setting from config
    verbose = config.get("verbose", True)
    st.markdown("Analyze existing importance matrix files to view statistics")

    st.subheader("Settings")

    # Output directory to analyze with Select Folder and Open Folder buttons
    if TKINTER_AVAILABLE:
        col_stats_dir, col_stats_dir_browse, col_stats_dir_check = st.columns([4, 1, 1])
    else:
        col_stats_dir, col_stats_dir_check = st.columns([5, 1])
        col_stats_dir_browse = None  # Not used when tkinter unavailable

    with col_stats_dir:
        stats_output_dir = st.text_input(
            "Output directory to analyze",
            value=config.get("output_dir", ""),
            placeholder="~/Models/output",
            help="Directory containing imatrix and GGUF files to analyze (uses output directory from Convert & Quantize tab)"
        )

    if TKINTER_AVAILABLE:
        with col_stats_dir_browse:  # type: ignore[union-attr]
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input + help icon
            if st.button(
                "Select Folder",
                key="browse_imatrix_output_dir_btn",
                use_container_width=True,
                help="Select output directory"
            ):
                stats_dir_clean = strip_quotes(stats_output_dir)
                initial_dir = stats_dir_clean if stats_dir_clean and Path(stats_dir_clean).exists() else None
                selected_folder = browse_folder(initial_dir)
                if selected_folder:
                    config["output_dir"] = selected_folder
                    save_config(config)
                    st.rerun()

    with col_stats_dir_check:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input + help icon
        stats_dir_clean = strip_quotes(stats_output_dir)
        stats_dir_exists = bool(stats_dir_clean and Path(stats_dir_clean).exists())
        if st.button(
            "Open Folder",
            key="check_imatrix_output_dir_btn",
            use_container_width=True,
            disabled=not stats_dir_exists,
            help="Open folder in file explorer" if stats_dir_exists else "Path doesn't exist yet"
        ):
            if stats_dir_exists:
                try:
                    open_folder(stats_dir_clean)
                    st.toast("Opened folder")
                except Exception as e:
                    st.toast(f"Could not open folder: {e}")

    # Strip quotes from path for later use
    stats_dir_clean = strip_quotes(stats_output_dir)

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
            "Refresh File List",
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

    if st.button("Show Statistics", use_container_width=True, key="show_stats_btn"):
        # Strip quotes from paths
        imatrix_stats_path_clean = strip_quotes(imatrix_stats_path)

        if not imatrix_stats_path_clean:
            st.error("Please provide an imatrix file path")
        else:
            try:
                with st.spinner("Reading imatrix statistics..."):
                    stats = converter.show_imatrix_statistics(
                        imatrix_stats_path_clean,
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
    elif 'imatrix_stats_result' in st.session_state and st.session_state.imatrix_stats_result:
        st.success("Statistics generated!")

    # Display results if available
    if 'imatrix_stats_result' in st.session_state and st.session_state.imatrix_stats_result:
        st.markdown("---")
        st.subheader("Statistics Output")

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
