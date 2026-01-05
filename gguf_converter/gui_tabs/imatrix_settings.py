"""
Imatrix Settings tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import itertools
from typing import Dict, Any, TYPE_CHECKING

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, make_config_saver, get_default_config, TKINTER_AVAILABLE
)

if TYPE_CHECKING:
    from ..converter import GGUFConverter


def render_imatrix_settings_tab(converter: "GGUFConverter", config: Dict[str, Any]) -> None:
    """Render the Imatrix Settings tab"""
    st.header("Importance Matrix Settings")
    st.markdown("Configure how importance matrices are generated for low-bit quantization")

    # Get the default calibration_data directory (one level up from gguf_converter module)
    default_calibration_dir = Path(__file__).parent.parent.parent / "calibration_data"

    # Get the configured directory or use default
    saved_cal_dir = config.get("imatrix_calibration_dir", "")
    if saved_cal_dir:
        calibration_data_dir = Path(saved_cal_dir)
    else:
        calibration_data_dir = default_calibration_dir

    # Main 2-column layout for entire tab
    col_left, col_right = st.columns(2)

    with col_left:
        # --- Calibration Data Section ---
        st.subheader("Calibration Data")
        st.markdown("Select calibration files for importance matrix generation")

        # Directory input field with Select Folder and Open Folder buttons
        if TKINTER_AVAILABLE:
            col_dir, col_dir_browse, col_dir_check = st.columns([4, 1, 1])
        else:
            col_dir, col_dir_check = st.columns([5, 1])
            col_dir_browse = None  # Not used when tkinter unavailable

        with col_dir:
            calibration_dir_input = st.text_input(
                "Calibration files directory",
                value=config.get("imatrix_calibration_dir", ""),
                placeholder=str(default_calibration_dir.resolve()),
                help="Directory containing calibration .txt or .raw files. Leave blank for default."
            )

        if TKINTER_AVAILABLE:
            with col_dir_browse:  # type: ignore[union-attr]
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Select Folder",
                    key="browse_cal_dir_btn",
                    use_container_width=True,
                    help="Select calibration directory"
                ):
                    current_path = calibration_dir_input.strip()
                    initial_dir = current_path if current_path and Path(current_path).exists() else str(default_calibration_dir)
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["imatrix_calibration_dir"] = selected_folder
                        save_config(config)
                        st.rerun()

        with col_dir_check:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            cal_dir_exists = calibration_data_dir.exists() and calibration_data_dir.is_dir()
            if st.button(
                "Open Folder",
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
        calibration_dir_input_clean = strip_quotes(calibration_dir_input).strip()

        # If input is empty, clear config and use default
        if not calibration_dir_input_clean:
            if config.get("imatrix_calibration_dir", "") != "":
                config["imatrix_calibration_dir"] = ""
                save_config(config)
            calibration_data_dir = default_calibration_dir
        # If input changed, update config
        elif calibration_dir_input_clean != str(calibration_data_dir.resolve()):
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
                key = f"imatrix_cal_selection_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["imatrix_calibration_file"] = st.session_state[key]
                    save_config(config)

            # Create dynamic label showing count of calibration files
            if calibration_files and calibration_files[0] != "(no files found)":
                calibration_label = f"Calibration file: {len(calibration_files)} detected"
            else:
                calibration_label = "Calibration file"

            calibration_selection = st.selectbox(
                calibration_label,
                options=calibration_files,
                index=default_index,
                help="Select a calibration file from the directory above",
                key=f"imatrix_cal_selection_{st.session_state.reset_count}",
                on_change=save_calibration_file
            )

            # Sync dropdown value with config if they don't match
            # This handles the case where the dropdown defaults to a different value
            # but the user hasn't manually changed it (so on_change doesn't fire)
            if calibration_selection != saved_calibration:
                config["imatrix_calibration_file"] = calibration_selection
                save_config(config)

        with col_cal_btn:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
            if st.button(
                "Refresh File List",
                key="update_cal_files_btn",
                use_container_width=True,
                help="Rescan directory for calibration files"
            ):
                # Save the currently selected value before rerunning
                current_selection = st.session_state.get(f"imatrix_cal_selection_{st.session_state.reset_count}")
                if current_selection:
                    config["imatrix_calibration_file"] = current_selection
                    save_config(config)
                # Store toast message in session state to show after rerun
                st.session_state.calibration_refresh_toast = "Refreshed - calibration file list"
                st.rerun()

        # Show toast message after refresh (if flag is set)
        if "calibration_refresh_toast" in st.session_state:
            st.toast(st.session_state.calibration_refresh_toast)
            del st.session_state.calibration_refresh_toast

        # --- File Information Section ---

        # Note: calibration_selection already set from selectbox above (line 126)
        # Get the calibration file path
        cal_dir = config.get("imatrix_calibration_dir", "")
        if cal_dir:
            calibration_file_path = Path(cal_dir) / calibration_selection
        else:
            calibration_file_path = default_calibration_dir / calibration_selection

        # Get chunk-related settings (used for stats calculations)
        from_chunk = int(config.get("imatrix_from_chunk", 0))
        chunks_to_process = int(config.get("imatrix_chunks", 100))
        ctx_size = int(config.get("imatrix_ctx_size", 512))
        estimated_lines_per_chunk = max(1, ctx_size // 5)

        if calibration_file_path.exists() and calibration_selection != "(no files found)":
            try:
                file_size = calibration_file_path.stat().st_size
                file_size_mb = file_size / (1024 * 1024)

                # Count lines without loading entire file
                with open(calibration_file_path, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)

                # Side-by-side comparison of full file vs processed data
                col_empty, col_full, col_processed = st.columns(3)

                # Calculate total chunks based on file lines
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

                with col_empty:
                    st.markdown(f"""**File Information:**
- {file_size_mb:.2f} MB""")

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
                st.subheader("File Information")
                st.info(f"Could not read file info: {e}")
        else:
            st.subheader("File Information")
            st.info("Select a calibration file to view information")

        # --- Processing Settings Section ---
        st.markdown("---")
        st.subheader("Processing Settings")
        st.markdown("Configure importance matrix generation parameters")

        # Use 3 sub-columns for processing settings
        col_proc1, col_proc2, col_proc3 = st.columns(3)

        with col_proc1:
            # Auto-save callback for chunks
            def save_chunks():
                key = f"imatrix_chunks_input_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["imatrix_chunks"] = int(st.session_state[key])
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
                key = f"imatrix_ctx_input_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["imatrix_ctx_size"] = int(st.session_state[key])
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
                key = f"imatrix_from_chunk_input_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["imatrix_from_chunk"] = int(st.session_state[key])
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
                key = f"imatrix_output_freq_input_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["imatrix_output_frequency"] = int(st.session_state[key])
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

        with col_proc2:
            # GPU offloading
            # Auto-save callback for num_gpu_layers
            def save_num_gpu_layers():
                key = f"imatrix_num_gpu_layers_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["imatrix_num_gpu_layers"] = int(st.session_state[key])
                    save_config(config)

            imatrix_num_gpu_layers_input = st.number_input(
                "GPU layers (-ngl)",
                min_value=0,
                max_value=999,
                value=int(config.get("imatrix_num_gpu_layers", 0)),
                step=1,
                help="0 = CPU only, >99 = fully offloaded in most cases.",
                key=f"imatrix_num_gpu_layers_{st.session_state.reset_count}",
                on_change=save_num_gpu_layers
            )
            st.caption("Number of model layers to offload to GPU.")
            st.caption("Requires GPU-enabled llama.cpp build.")

        with col_proc3:
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

        # Reset button (full width below processing settings sub-columns)
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
            config["imatrix_num_gpu_layers"] = defaults["imatrix_num_gpu_layers"]
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

    with col_right:
        # --- Calibration File Preview Section ---
        st.subheader("Calibration File Preview")

        # Preview mode selection (full width, horizontal)
        preview_mode = st.radio(
            "Preview mode",
            ["No preview", "Full file", "Processed data"],
            index=0,
            help="No preview: File info only. Full file: Show first N lines. Processed data: Show what will be used based on settings.",
            horizontal=True,
            key=f"preview_mode_{st.session_state.reset_count}",
            label_visibility="collapsed"
        )

        # Preview settings controls in one row
        col_max_lines, col_height = st.columns([1, 2])

        with col_max_lines:
            # Auto-save callback for max preview lines
            def save_max_preview_lines():
                key = f"max_preview_lines_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["max_preview_lines"] = int(st.session_state[key])
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
                key = f"preview_height_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["preview_height"] = int(st.session_state[key])
                    save_config(config)

            preview_height = st.slider(
                "Preview height",
                min_value=200,
                max_value=1200,
                value=int(config.get("preview_height", 700)),
                step=100,
                help="Adjust preview area height in pixels",
                key=f"preview_height_{st.session_state.reset_count}",
                on_change=save_preview_height
            )

        # Note: calibration_file_path, from_chunk, chunks_to_process, ctx_size,
        # and estimated_lines_per_chunk are already defined in File Information section above

        # Calculate and show truncation info based on preview mode
        # Always show info pane for visual consistency
        if not calibration_file_path.exists() or calibration_selection == "(no files found)":
            st.info("No calibration file selected")
        else:
            try:
                if preview_mode == "No preview":
                    # Show placeholder info for visual consistency
                    st.info("Preview disabled")
                elif preview_mode == "Full file":
                    # Count total lines to check for truncation
                    with open(calibration_file_path, 'r', encoding='utf-8') as f:
                        total_lines = sum(1 for _ in f)

                    if total_lines > max_preview_lines:
                        st.info(f"Showing first {max_preview_lines:,} of {total_lines:,} lines")
                    else:
                        st.info(f"Showing all {total_lines:,} lines")

                elif preview_mode == "Processed data":
                    # Calculate if preview needs truncation notice
                    if chunks_to_process > 0:
                        lines_to_read = chunks_to_process * estimated_lines_per_chunk
                        if lines_to_read > max_preview_lines:
                            st.info(f"Showing first {max_preview_lines:,} lines of range")
                        else:
                            st.info(f"Showing all lines in range")
                    else:
                        st.info(f"Showing all remaining lines (up to {max_preview_lines:,})")
            except Exception:
                st.info("Error reading file information")

        # Render preview based on mode
        if calibration_file_path.exists() and calibration_selection != "(no files found)":
            try:
                if preview_mode == "No preview":
                    # Show empty text area for visual consistency
                    st.text_area(
                        "Content",
                        value="",
                        height=preview_height,
                        disabled=True,
                        label_visibility="collapsed",
                        placeholder="Preview disabled"
                    )

                elif preview_mode == "Full file":
                    # Read only first max_preview_lines lines
                    preview_lines = []

                    with open(calibration_file_path, 'r', encoding='utf-8') as f:
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
                    # Calculate line range to read
                    start_line = from_chunk * estimated_lines_per_chunk

                    if chunks_to_process > 0:
                        lines_to_read = chunks_to_process * estimated_lines_per_chunk
                    else:
                        lines_to_read = None  # Read all remaining lines

                    # Read only the needed lines
                    preview_lines = []

                    with open(calibration_file_path, 'r', encoding='utf-8') as f:
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
                # Show empty text area even on error for consistency
                st.text_area(
                    "Content",
                    value="",
                    height=preview_height,
                    disabled=True,
                    label_visibility="collapsed"
                )
        else:
            # Show empty text area for visual consistency when no file selected
            st.text_area(
                "Content",
                value="",
                height=preview_height,
                disabled=True,
                label_visibility="collapsed"
            )
