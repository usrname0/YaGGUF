"""
Imatrix Settings tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import itertools

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, make_config_saver, get_default_config, TKINTER_AVAILABLE
)


def render_imatrix_settings_tab(converter, config):
    """Render the Imatrix Settings tab"""
    st.header("Importance Matrix Settings")
    st.markdown("Configure how importance matrices are generated for low-bit quantization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calibration Data")
        st.markdown("Select calibration files for importance matrix generation")

        # Get the default calibration_data directory (one level up from gguf_converter module)
        default_calibration_dir = Path(__file__).parent.parent.parent / "calibration_data"

        # Get the configured directory or use default
        saved_cal_dir = config.get("imatrix_calibration_dir", "")
        if saved_cal_dir:
            calibration_data_dir = Path(saved_cal_dir)
        else:
            calibration_data_dir = default_calibration_dir

        # Directory input field with Browse and Check Folder buttons
        if TKINTER_AVAILABLE:
            col_dir, col_dir_browse, col_dir_check = st.columns([4, 1, 1])
        else:
            col_dir, col_dir_check = st.columns([5, 1])
            col_dir_browse = None  # Not used when tkinter unavailable

        with col_dir:
            calibration_dir_input = st.text_input(
                "Calibration files directory",
                value=str(calibration_data_dir.resolve()),  # Show absolute path
                placeholder=str(default_calibration_dir.resolve()),
                help="Full path to directory containing calibration .txt files",
                key=f"imatrix_cal_dir_input_{st.session_state.reset_count}",
                on_change=lambda: None  # Trigger to update when user changes the path
            )

        if TKINTER_AVAILABLE:
            with col_dir_browse:  # type: ignore[union-attr]
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Browse",
                    key="browse_cal_dir_btn",
                    use_container_width=True,
                    help="Browse for calibration directory"
                ):
                    initial_dir = str(calibration_data_dir) if calibration_data_dir.exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["imatrix_calibration_dir"] = selected_folder
                        save_config(config)
                        st.rerun()

        with col_dir_check:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            cal_dir_exists = calibration_data_dir.exists() and calibration_data_dir.is_dir()
            if st.button(
                "Check Folder",
                key="check_cal_dir_btn",
                use_container_width=True,
                disabled=not cal_dir_exists,
                help="Open folder in file explorer" if cal_dir_exists else "Directory doesn't exist"
            ):
                if cal_dir_exists:
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

        # Scan directory for .txt and .raw files
        calibration_files = []
        if calibration_data_dir.exists() and calibration_data_dir.is_dir():
            txt_files = list(calibration_data_dir.glob("*.txt"))
            raw_files = list(calibration_data_dir.glob("*.raw"))
            all_files = txt_files + raw_files
            calibration_files = sorted([f.name for f in all_files])

        if not calibration_files:
            st.warning(f"No .txt or .raw files found in: {calibration_data_dir}")
            calibration_files = ["(no files found)"]

        # Determine default selection - prefer saved selection, fallback to wiki.train.raw
        saved_calibration = config.get("imatrix_calibration_file", "wiki.train.raw")
        default_index = 0

        # Use saved selection if it exists in the file list
        if saved_calibration in calibration_files:
            default_index = calibration_files.index(saved_calibration)
        # Otherwise try wiki.train.raw as fallback
        elif "wiki.train.raw" in calibration_files:
            default_index = calibration_files.index("wiki.train.raw")

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
            st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
            if st.button(
                "Refresh File List",
                key="update_cal_files_btn",
                use_container_width=True,
                help="Rescan directory for calibration files"
            ):
                # Trigger a rerun to rescan the directory
                st.toast("Updated calibration file list")
                st.rerun()

        # File information section
        st.markdown("---")

        # Get the calibration file path
        file_info_cal_dir = config.get("imatrix_calibration_dir", "")
        if file_info_cal_dir:
            file_info_path = Path(file_info_cal_dir) / calibration_selection
        else:
            default_cal_dir = Path(__file__).parent.parent.parent / "calibration_data"
            file_info_path = default_cal_dir / calibration_selection

        if file_info_path.exists() and calibration_selection != "(no files found)":
            try:
                file_size = file_info_path.stat().st_size
                file_size_mb = file_size / (1024 * 1024)

                # Count lines without loading entire file
                with open(file_info_path, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)

                # Basic file info - single line
                st.markdown(f"**File:** {calibration_selection} ({file_size_mb:.2f} MB)")
                st.markdown("")  # Blank line for spacing

                # Side-by-side comparison of full file vs processed data
                col_full, col_processed = st.columns(2)

                # Calculate chunk information (used by both columns)
                from_chunk = int(config.get("imatrix_from_chunk", 0))
                chunks_to_process = int(config.get("imatrix_chunks", 100))
                ctx_size = int(config.get("imatrix_ctx_size", 512))
                estimated_lines_per_chunk = max(1, ctx_size // 5)
                total_chunks = total_lines // estimated_lines_per_chunk

                # Estimate token counts (file_size in bytes ≈ chars for text files)
                # Rough estimate: 1 token ≈ 4 characters
                total_tokens_est = file_size / 4

                # Format token count
                if total_tokens_est >= 1_000_000:
                    total_tokens_str = f"{total_tokens_est / 1_000_000:.1f}M"
                elif total_tokens_est >= 1_000:
                    total_tokens_str = f"{total_tokens_est / 1_000:.1f}K"
                else:
                    total_tokens_str = f"{int(total_tokens_est)}"

                with col_full:
                    st.markdown(f"""**Full File:**
- Lines: {total_lines:,}
- Chunks: ~{total_chunks} (at {ctx_size} ctx)
- Est. tokens: ~{total_tokens_str}""")

                with col_processed:
                    # Calculate what will be processed based on settings
                    start_line = from_chunk * estimated_lines_per_chunk

                    if chunks_to_process > 0:
                        end_line = min(start_line + (chunks_to_process * estimated_lines_per_chunk), total_lines)
                        lines_processed = end_line - start_line
                        actual_chunks = lines_processed // estimated_lines_per_chunk
                    else:
                        # Process all remaining lines
                        end_line = total_lines
                        lines_processed = total_lines - start_line
                        actual_chunks = lines_processed // estimated_lines_per_chunk

                    # Calculate coverage percentage
                    if total_chunks > 0:
                        coverage_pct = (actual_chunks / total_chunks) * 100
                    else:
                        coverage_pct = 0

                    # Estimate tokens in processed range
                    # Estimate based on line ratio
                    processed_chars_est = (lines_processed / total_lines) * file_size if total_lines > 0 else 0
                    processed_tokens_est = processed_chars_est / 4

                    # Format processed token count
                    if processed_tokens_est >= 1_000_000:
                        processed_tokens_str = f"{processed_tokens_est / 1_000_000:.1f}M"
                    elif processed_tokens_est >= 1_000:
                        processed_tokens_str = f"{processed_tokens_est / 1_000:.1f}K"
                    else:
                        processed_tokens_str = f"{int(processed_tokens_est)}"

                    st.markdown(f"""**Will Be Processed:**
- Lines: {start_line+1} to {end_line} ({lines_processed:,})
- Chunks: ~{actual_chunks} (~{coverage_pct:.1f}% coverage)
- Est. tokens: ~{processed_tokens_str}
- Skipped: First {from_chunk} chunks ({start_line:,} lines)""" if from_chunk > 0 else f"""**Will Be Processed:**
- Lines: {start_line+1} to {end_line} ({lines_processed:,})
- Chunks: ~{actual_chunks} (~{coverage_pct:.1f}% coverage)
- Est. tokens: ~{processed_tokens_str}""")

            except Exception as e:
                st.info(f"Could not read file info: {e}")
        else:
            st.info("Select a calibration file to view information")

    with col2:
        st.subheader("Processing Settings")
        st.markdown("Configure importance matrix generation parameters")

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
        imatrix_no_ppl_input = st.checkbox(
            "Disable perplexity calculation",
            value=config.get("imatrix_no_ppl", False),
            help="Skip PPL calculation to speed up processing",
            key=f"imatrix_no_ppl_input_{st.session_state.reset_count}",
            on_change=make_config_saver(config, "imatrix_no_ppl", f"imatrix_no_ppl_input_{st.session_state.reset_count}")
        )

        imatrix_parse_special_input = st.checkbox(
            "Parse special tokens",
            value=config.get("imatrix_parse_special", False),
            help="Parse special tokens like <|im_start|>, <|im_end|>, etc. Recommended for chat models (Qwen, Llama 3, ChatML-based models). Warning: Can significantly slow down imatrix generation.",
            key=f"imatrix_parse_special_input_{st.session_state.reset_count}",
            on_change=make_config_saver(config, "imatrix_parse_special", f"imatrix_parse_special_input_{st.session_state.reset_count}")
        )

        imatrix_collect_output_input = st.checkbox(
            "Collect output.weight tensor",
            value=config.get("imatrix_collect_output_weight", False),
            help="Collect importance matrix data for output.weight tensor. Typically better to leave disabled (default), as the importance matrix is generally not beneficial for this tensor.",
            key=f"imatrix_collect_output_input_{st.session_state.reset_count}",
            on_change=make_config_saver(config, "imatrix_collect_output_weight", f"imatrix_collect_output_input_{st.session_state.reset_count}")
        )

        # GPU offloading (only show if custom binaries enabled)
        if config.get("use_custom_binaries", False):

            # Auto-save callback for ngl
            def save_ngl():
                config["imatrix_ngl"] = int(st.session_state[f"imatrix_ngl_{st.session_state.reset_count}"])
                save_config(config)

            imatrix_ngl_input = st.number_input(
                "GPU layers (-ngl)",
                min_value=0,
                max_value=999,
                value=int(config.get("imatrix_ngl", 0)),
                step=1,
                help="Number of model layers to offload to GPU. 0 = CPU only, 99 = fully offload. Requires GPU-enabled llama.cpp build.",
                key=f"imatrix_ngl_{st.session_state.reset_count}",
                on_change=save_ngl
            )

        # Reset button
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
            config["max_preview_lines"] = defaults["max_preview_lines"]
            config["preview_height"] = defaults["preview_height"]
            save_config(config)
            st.session_state.reset_count += 1
            st.session_state.imatrix_just_reset = True
            st.rerun()

        # Show success message if we just reset
        if st.session_state.get('imatrix_just_reset', False):
            st.success("Reset to defaults! And Saved!")
            st.session_state.imatrix_just_reset = False

    # Calibration file preview section - full width below both columns
    st.markdown("---")

    # Get the calibration file path
    preview_cal_dir = config.get("imatrix_calibration_dir", "")
    if preview_cal_dir:
        preview_calibration_path = Path(preview_cal_dir) / calibration_selection
    else:
        default_cal_dir = Path(__file__).parent.parent.parent / "calibration_data"
        preview_calibration_path = default_cal_dir / calibration_selection

    # Header row with controls - 2 equal columns matching layout above
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Calibration File Preview")

        # Preview mode selection
        preview_mode = st.radio(
            "Preview mode",
            ["No preview", "Full file", "Processed data"],
            index=0,
            help="No preview: File info only. Full file: Show first N lines. Processed data: Show what will be used based on settings.",
            horizontal=True,
            key=f"preview_mode_{st.session_state.reset_count}",
            label_visibility="collapsed"
        )

    with col_right:
        # Sub-columns for max lines and preview height
        col_max_lines, col_height = st.columns([1, 2])

        with col_max_lines:
            # Auto-save callback for max preview lines
            def save_max_preview_lines():
                config["max_preview_lines"] = int(st.session_state[f"max_preview_lines_{st.session_state.reset_count}"])
                save_config(config)

            max_preview_lines = st.number_input(
                "Preview max lines",
                min_value=100,
                max_value=10000,
                value=int(config.get("max_preview_lines", 1000)),
                step=100,
                help="Maximum lines to display in preview. High values may cause lag.",
                key=f"max_preview_lines_{st.session_state.reset_count}",
                on_change=save_max_preview_lines
            )

        with col_height:
            # Auto-save callback for preview height
            def save_preview_height():
                config["preview_height"] = int(st.session_state[f"preview_height_{st.session_state.reset_count}"])
                save_config(config)

            preview_height = st.slider(
                "Preview height",
                min_value=200,
                max_value=1200,
                value=int(config.get("preview_height", 400)),
                step=100,
                help="Adjust preview area height in pixels",
                key=f"preview_height_{st.session_state.reset_count}",
                on_change=save_preview_height
            )

        # Calculate and show truncation info based on preview mode
        if preview_calibration_path.exists() and calibration_selection != "(no files found)" and preview_mode != "No preview":
            try:
                if preview_mode == "Full file":
                    # Count total lines to check for truncation
                    with open(preview_calibration_path, 'r', encoding='utf-8') as f:
                        total_lines = sum(1 for _ in f)

                    if total_lines > max_preview_lines:
                        st.info(f"Showing first {max_preview_lines:,} of {total_lines:,} lines")

                elif preview_mode == "Processed data":
                    # Calculate processed range
                    from_chunk = int(config.get("imatrix_from_chunk", 0))
                    chunks_to_process = int(config.get("imatrix_chunks", 100))
                    ctx_size = int(config.get("imatrix_ctx_size", 512))
                    estimated_lines_per_chunk = max(1, ctx_size // 5)

                    if chunks_to_process > 0:
                        lines_to_read = chunks_to_process * estimated_lines_per_chunk
                        if lines_to_read > max_preview_lines:
                            st.info(f"Showing first {max_preview_lines:,} lines of range")
            except Exception:
                pass

    # Render preview based on mode
    if preview_calibration_path.exists() and calibration_selection != "(no files found)":
        try:
            if preview_mode == "No preview":
                pass  # No preview to show

            elif preview_mode == "Full file":
                # Read only first max_preview_lines lines
                preview_lines = []

                with open(preview_calibration_path, 'r', encoding='utf-8') as f:
                    # Read lines up to limit
                    for i, line in enumerate(f):
                        if i < max_preview_lines:
                            preview_lines.append(line.rstrip('\n\r'))

                preview_content = '\n'.join(preview_lines)

                # Show preview with word wrap
                st.text_area(
                    "Content",
                    value=preview_content,
                    height=preview_height,
                    disabled=True,
                    label_visibility="collapsed"
                )

            elif preview_mode == "Processed data":
                # Calculate what will be processed based on settings
                from_chunk = int(config.get("imatrix_from_chunk", 0))
                chunks_to_process = int(config.get("imatrix_chunks", 100))
                ctx_size = int(config.get("imatrix_ctx_size", 512))

                # Estimate lines per chunk
                estimated_lines_per_chunk = max(1, ctx_size // 5)

                # Calculate line range to read
                start_line = from_chunk * estimated_lines_per_chunk

                if chunks_to_process > 0:
                    lines_to_read = chunks_to_process * estimated_lines_per_chunk
                else:
                    lines_to_read = None  # Read all remaining lines

                # Read only the needed lines
                preview_lines = []

                with open(preview_calibration_path, 'r', encoding='utf-8') as f:
                    # Skip to start_line
                    for _ in itertools.islice(f, start_line):
                        pass

                    # Read the lines we need (up to max_preview_lines)
                    if lines_to_read is None:
                        # Read all remaining lines (up to preview limit)
                        for i, line in enumerate(f):
                            if i < max_preview_lines:
                                preview_lines.append(line.rstrip('\n\r'))
                    else:
                        # Read specific number of lines (up to preview limit)
                        for i, line in enumerate(itertools.islice(f, lines_to_read)):
                            if i < max_preview_lines:
                                preview_lines.append(line.rstrip('\n\r'))

                preview_content = '\n'.join(preview_lines)

                # Show processed data with word wrap
                st.text_area(
                    "Content",
                    value=preview_content,
                    height=preview_height,
                    disabled=True,
                    label_visibility="collapsed"
                )

        except Exception as e:
            st.warning(f"Could not preview calibration file: {e}")
    else:
        st.info("Select a calibration file to preview")
