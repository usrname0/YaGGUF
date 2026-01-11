"""
Convert & Quantize tab for GGUF Converter GUI
"""

import streamlit as st
import re
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from colorama import Style
from ..theme import THEME as theme

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, make_config_saver, path_input_columns,
    get_platform_path
)

from .convert_helpers import (
    sanitize_filename, detect_source_dtype,
    validate_calibration_file, detect_intermediate_gguf_files
)

if TYPE_CHECKING:
    from ..converter import GGUFConverter


def render_convert_tab(
    converter: "GGUFConverter",
    config: Dict[str, Any],
    verbose: bool,
    num_threads: Optional[int]
) -> None:
    """Render the Convert & Quantize tab"""
    st.header("Convert and Quantize Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")

        # Model path with Select Folder and Open Folder buttons
        cols, has_browse = path_input_columns()

        def save_model_path():
            """Save model path to config when changed"""
            raw_value = st.session_state.get("model_path_input", "")
            cleaned_value = strip_quotes(raw_value)
            config["model_path"] = cleaned_value
            save_config(config)

        with cols[0]:
            # Only set value if key not in session state (prevents warning)
            model_path_kwargs = {
                "label": "Model path",
                "placeholder": get_platform_path("C:\\Models\\my-model", "/home/user/Models/my-model"),
                "help": "Local model directory containing config.json and model files.",
                "key": "model_path_input",
                "on_change": save_model_path
            }
            if "model_path_input" not in st.session_state:
                model_path_kwargs["value"] = config.get("model_path", "")
            model_path = st.text_input(**model_path_kwargs)  # type: ignore[arg-type]

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Select Folder",
                    key="browse_model_folder_btn",
                    use_container_width=True,
                    help="Select model directory"
                ):
                    model_path_clean = strip_quotes(model_path)
                    initial_dir = model_path_clean if model_path_clean and Path(model_path_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["model_path"] = selected_folder
                        save_config(config)
                        # Update widget state to show new path
                        st.session_state.pending_model_path = selected_folder
                        # Increment reset_count to refresh imatrix dropdown with new model name
                        st.session_state.reset_count += 1
                        st.rerun()

        with cols[-1]:  # Last column is always the check button
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            model_path_clean = strip_quotes(model_path)
            model_path_exists = bool(model_path_clean and Path(model_path_clean).exists())
            if st.button(
                "Open Folder",
                key="check_model_folder_btn",
                use_container_width=True,
                disabled=not model_path_exists,
                help="Open folder in file explorer" if model_path_exists else "Path doesn't exist yet"
            ):
                if model_path_exists:
                    try:
                        open_folder(model_path_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

        # Initialize session state for reset tracking
        if 'reset_count' not in st.session_state:
            st.session_state.reset_count = 0

        # Strip quotes from paths for detection
        model_path_clean = strip_quotes(model_path)

        # Detect available intermediate files
        intermediate_options = []
        intermediate_info_map = {}  # Maps option string to file info
        source_dtype = None
        config_missing = False

        model_path_valid = bool(model_path_clean and Path(model_path_clean).exists())

        if not model_path_valid:
            intermediate_options = ["Model path invalid - provide valid path above"]
            dropdown_disabled = True
        elif not Path(model_path_clean).is_dir():
            intermediate_options = ["Model path must be a directory"]
            dropdown_disabled = True
        else:
            model_path_obj = Path(model_path_clean)

            # Detect safetensors files
            safetensors_files = list(model_path_obj.glob("*.safetensors"))
            has_safetensors = len(safetensors_files) > 0

            # Detect source dtype if we have safetensors
            if has_safetensors:
                source_dtype = detect_source_dtype(model_path_obj)
                # Check if config.json exists
                if not (model_path_obj / "config.json").exists():
                    config_missing = True

            # Scan for intermediate GGUF files
            detected = detect_intermediate_gguf_files(model_path_obj)

            # Build options
            intermediate_options = []

            # Add safetensors option if found
            if has_safetensors:
                # Check if they're split (look for index files or numbered files)
                split_pattern = re.compile(r'-\d+of\d+\.safetensors$|\.safetensors\.\d+$')
                split_files = [f for f in safetensors_files if split_pattern.search(f.name)]

                # Build dtype part of label
                dtype_part = f", {source_dtype}" if source_dtype else ""

                if split_files:
                    total_size = sum(f.stat().st_size for f in safetensors_files) / (1024**3)
                    option_text = f"safetensors ({len(safetensors_files)} files{dtype_part}, {total_size:.2f} GB)"
                else:
                    total_size = sum(f.stat().st_size for f in safetensors_files) / (1024**3)
                    if len(safetensors_files) == 1:
                        option_text = f"safetensors (single file{dtype_part}, {total_size:.2f} GB)"
                    else:
                        option_text = f"safetensors ({len(safetensors_files)} files{dtype_part}, {total_size:.2f} GB)"

                intermediate_options.append(option_text)
                # Don't add to intermediate_info_map - safetensors needs conversion, not direct use

            # Add intermediate GGUF files
            # Sort by format type to keep consistent ordering
            for key in sorted(detected.keys()):
                info = detected[key]

                # Extract base filename
                if info['type'] == 'single':
                    base_name = info['primary_file'].stem  # Remove .gguf extension
                    option_text = f"{base_name} (single file, {info['total_size_gb']:.2f} GB)"
                else:
                    # Remove shard numbering pattern: -00001-of-00003
                    base_name = re.sub(r'-\d+-of-\d+$', '', info['primary_file'].stem)
                    option_text = f"{base_name} ({info['shard_count']} shards, {info['total_size_gb']:.2f} GB)"

                intermediate_options.append(option_text)
                intermediate_info_map[option_text] = {
                    'format': info['format'],
                    'info': info
                }

            # If no files found, show disabled message
            if not intermediate_options:
                intermediate_options = ["No files found"]
                dropdown_disabled = True
            else:
                dropdown_disabled = False

        # Determine default selection
        default_index = 0
        saved_mode = config.get("custom_intermediate_mode", "")

        if saved_mode and saved_mode in intermediate_options:
            default_index = intermediate_options.index(saved_mode)

        # Intermediate source dropdown with same width as path inputs
        cols, has_browse = path_input_columns()

        # Create dynamic label showing count when files are detected
        if dropdown_disabled:
            model_files_label = "Model files"
        else:
            file_count = len(intermediate_options)
            model_files_label = f"Model files: {file_count} detected"

        with cols[0]:
            intermediate_source = st.selectbox(
                model_files_label,
                options=intermediate_options,
                index=default_index,
                disabled=dropdown_disabled,
                help="Select safetensors to convert or existing intermediate GGUF to quantize directly",
                key=f"intermediate_source_dropdown_{st.session_state.reset_count}"
            )

            # Auto-sync dropdown value with config (similar to imatrix dropdown pattern)
            if intermediate_source in intermediate_info_map:
                # It's a custom intermediate selection - save it
                selected_data = intermediate_info_map[intermediate_source]
                if (config.get("custom_intermediate_mode") != intermediate_source or
                    config.get("custom_intermediate_format") != selected_data['format']):
                    config["custom_intermediate_mode"] = intermediate_source
                    config["custom_intermediate_format"] = selected_data['format']
                    config["custom_intermediate_path"] = str(selected_data['info']['primary_file'])
                    config["custom_intermediate_file_type"] = selected_data['info']['type']  # 'single' or 'split'
                    save_config(config)
            else:
                # Not a custom intermediate - clear settings
                if config.get("custom_intermediate_format") or config.get("custom_intermediate_path"):
                    config["custom_intermediate_mode"] = intermediate_source
                    config["custom_intermediate_format"] = None
                    config["custom_intermediate_path"] = None
                    config["custom_intermediate_file_type"] = None
                    save_config(config)

        # Place button in middle column if tkinter available, otherwise last column
        button_col = cols[1] if has_browse else cols[-1]
        with button_col:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
            if st.button("Refresh File List", key="update_model_files_list", use_container_width=True):
                # Save the current selection before rerunning
                current_selection = st.session_state.get(f"intermediate_source_dropdown_{st.session_state.reset_count}")
                if current_selection:
                    if current_selection in intermediate_info_map:
                        selected_data = intermediate_info_map[current_selection]
                        config["custom_intermediate_mode"] = current_selection
                        config["custom_intermediate_format"] = selected_data['format']
                        config["custom_intermediate_path"] = str(selected_data['info']['primary_file'])
                        config["custom_intermediate_file_type"] = selected_data['info']['type']
                    else:
                        config["custom_intermediate_mode"] = current_selection
                        config["custom_intermediate_format"] = None
                        config["custom_intermediate_path"] = None
                        config["custom_intermediate_file_type"] = None
                    save_config(config)
                st.session_state.model_files_refresh_toast = "Refreshed - model file list"
                st.rerun()

        # Show toast message after refresh (if flag is set)
        if "model_files_refresh_toast" in st.session_state:
            st.toast(st.session_state.model_files_refresh_toast)
            del st.session_state.model_files_refresh_toast

        # Determine if using custom intermediate
        # Check config directly rather than string matching, which is more reliable
        using_custom_intermediate = bool(
            config.get("custom_intermediate_format") and
            config.get("custom_intermediate_path")
        )

        # Get intermediate_type early (needed for checkbox state management)
        if "other_quants" not in config:
            config["other_quants"] = {}
        intermediate_type = config.get("intermediate_type", "F16").upper()

        # Track custom intermediate mode changes
        if 'previous_using_custom_intermediate' not in st.session_state:
            st.session_state.previous_using_custom_intermediate = using_custom_intermediate

        # Handle entering custom intermediate mode - save button selection and clear checkboxes
        if using_custom_intermediate and not st.session_state.previous_using_custom_intermediate:
            # Save the current intermediate type (button selection)
            config["custom_intermediate_saved_intermediate_type"] = intermediate_type

            # Clear all checkboxes (both active and saved states)
            config["other_quants"]["F32"] = False
            config["other_quants"]["F16"] = False
            config["other_quants"]["BF16"] = False

            # Also clear saved states to prevent stale state from being restored
            if "unquantized_saved_states" not in config:
                config["unquantized_saved_states"] = {}
            config["unquantized_saved_states"]["F32"] = False
            config["unquantized_saved_states"]["F16"] = False
            config["unquantized_saved_states"]["BF16"] = False

            st.session_state.previous_using_custom_intermediate = True
            save_config(config)
            st.rerun()

        # Handle exiting custom intermediate mode - restore button selection only
        elif not using_custom_intermediate and st.session_state.previous_using_custom_intermediate:
            # Restore the intermediate type (button selection)
            if "custom_intermediate_saved_intermediate_type" in config:
                saved_intermediate_type = config["custom_intermediate_saved_intermediate_type"]
                config["intermediate_type"] = saved_intermediate_type
                intermediate_type = saved_intermediate_type.upper()
                st.session_state.previous_intermediate = intermediate_type

            # Clear the saved intermediate type
            config["custom_intermediate_saved_intermediate_type"] = None
            st.session_state.previous_using_custom_intermediate = False
            save_config(config)


        # Output directory with Select Folder and Open Folder buttons
        cols, has_browse = path_input_columns()

        with cols[0]:
            output_dir = st.text_input(
                "Output directory",
                value=config.get("output_dir", ""),
                placeholder=get_platform_path("C:\\Models\\output", "/home/user/Models/output"),
                help="Where to save the converted files"
            )

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Select Folder",
                    key="browse_output_folder_btn",
                    use_container_width=True,
                    help="Select output directory"
                ):
                    output_dir_clean = strip_quotes(output_dir)
                    initial_dir = output_dir_clean if output_dir_clean and Path(output_dir_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["output_dir"] = selected_folder
                        save_config(config)
                        st.rerun()

        with cols[-1]:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            output_dir_clean = strip_quotes(output_dir)
            output_dir_exists = bool(output_dir_clean and Path(output_dir_clean).exists())
            if st.button(
                "Open Folder",
                key="check_output_folder_btn",
                use_container_width=True,
                disabled=not output_dir_exists,
                help="Open folder in file explorer" if output_dir_exists else "Path doesn't exist yet"
            ):
                if output_dir_exists:
                    try:
                        open_folder(output_dir_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

        # Strip quotes and update config if changed
        output_dir_clean = strip_quotes(output_dir)
        if output_dir_clean != config.get("output_dir", ""):
            config["output_dir"] = output_dir_clean
            save_config(config)

        # File Handling section
        with st.expander("File Handling"):
            # Three-column layout: radio buttons on left, options in middle, info on right
            radio_col, options_col, info_col = st.columns([1, 3, 2])

            with radio_col:
                # Determine if we should force file mode based on custom intermediate
                force_file_mode = None
                disable_file_mode = False
                custom_file_type = config.get("custom_intermediate_file_type", "none")

                if using_custom_intermediate:
                    if custom_file_type == "single":
                        force_file_mode = "Single files"
                        disable_file_mode = True
                    elif custom_file_type == "split":
                        force_file_mode = "Split files"
                        disable_file_mode = True

                # Use forced mode if available, otherwise use saved config
                if force_file_mode:
                    file_mode_index = 0 if force_file_mode == "Single files" else 1
                else:
                    file_mode_index = 0 if config.get("file_mode", "Single files") == "Single files" else 1

                # Include using_custom_intermediate and custom_file_type in key to force widget recreation
                file_mode_key = f"file_mode_radio_{using_custom_intermediate}_{custom_file_type}"

                # File handling mode callback
                def save_file_mode():
                    if file_mode_key in st.session_state:
                        config["file_mode"] = st.session_state[file_mode_key]
                        save_config(config)

                file_mode = st.radio(
                    "File handling mode",
                    options=["Single files", "Split files"],
                    index=file_mode_index,
                    key=file_mode_key,
                    on_change=save_file_mode,
                    disabled=disable_file_mode
                )

                # Add spacing to prevent size changes when switching modes
                st.markdown("<br><br><br>", unsafe_allow_html=True)

            keep_split = (file_mode == "Split files")

            with options_col:
                if file_mode == "Split files":
                    st.markdown("**Split file options:**")

                    st.markdown("- Split files will always overwrite existing split intermediates and quants.")

                    # Disable shard size input if using split custom intermediate
                    disable_shard_size = (using_custom_intermediate and
                                        config.get("custom_intermediate_file_type") == "split")

                    # Use half-width column for the number input
                    split_size_col1, split_size_col2 = st.columns([1, 1])

                    with split_size_col1:
                        def save_max_shard_size():
                            config["max_shard_size_gb"] = st.session_state.max_shard_size_input
                            save_config(config)

                        max_shard_size_gb = st.number_input(
                            "Max size per shard (GB)",
                            min_value=0.1,
                            value=config.get("max_shard_size_gb", 2.0),
                            step=0.1,
                            format="%.1f",
                            help="Maximum size per shard file in GB. llama.cpp will create as many shards as needed to stay under this limit." if not disable_shard_size else "Disabled because existing intermediate file is already split.",
                            key="max_shard_size_input",
                            on_change=save_max_shard_size,
                            disabled=disable_shard_size
                        )

                    # Split mode doesn't use these settings, but set defaults for the converter call
                    overwrite_intermediates = True  # Always overwrite in split mode
                    overwrite_quants = True  # Always overwrite in split mode
                else:
                    st.markdown("**Single file options:**")

                    # Single file options - show overwrite checkboxes
                    max_shard_size_gb = 0  # Not used in single file mode

                    # Disable "Overwrite intermediates" when using custom intermediate
                    disable_overwrite_intermediates = (using_custom_intermediate and
                                                      config.get("custom_intermediate_file_type") == "single")

                    overwrite_intermediates = st.checkbox(
                        "Overwrite intermediates (F32, F16, BF16)",
                        value=False if disable_overwrite_intermediates else config.get("overwrite_intermediates", True),
                        help="If enabled (default), regenerate intermediate formats even if they exist. If disabled, reuse existing intermediate files to save time." if not disable_overwrite_intermediates else "Disabled when using custom intermediate file.",
                        key=f"overwrite_intermediates_checkbox_{using_custom_intermediate}",
                        on_change=make_config_saver(config, "overwrite_intermediates", f"overwrite_intermediates_checkbox_{using_custom_intermediate}"),
                        disabled=disable_overwrite_intermediates
                    )

                    overwrite_quants = st.checkbox(
                        "Overwrite quants",
                        value=config.get("overwrite_quants", True),
                        help="If enabled (default), regenerate quantized formats even if they exist. If disabled, skip quantization for files that already exist.",
                        key="overwrite_quants_checkbox",
                        on_change=make_config_saver(config, "overwrite_quants", "overwrite_quants_checkbox")
                    )

            with info_col:
                # Show info when using custom intermediate
                if using_custom_intermediate:
                    st.info('Using existing intermediate. Some options disabled.\n\nTo change splits go to the "Split/Merge Shards" tab.')
                # Show info when in split files mode without custom intermediate
                elif file_mode == "Split files":
                    st.info("Intermediate will be split into shards before quantization.\n\nQuantized outputs will then be split accordingly.")
                elif file_mode == "Single files":
                    st.info("Intermediate will be a single file.\n\nQuantized outputs will then be a single file.")

        # Advanced quantization options
        with st.expander("Advanced Quantization Options"):

            st.markdown("**Selective Tensor Quantization:** Keep certain layers at higher precision.")

            # Output tensor type and token embedding type side-by-side
            output_tensor_options = ["Same as quant type (default)", "Unquantized", "Q8_0", "Q6_K", "Q5_K_M", "F16", "F32"]
            token_embedding_options = ["Same as quant type (default)", "Q8_0", "Q6_K", "Q5_K_M", "F16", "F32"]

            tensor_col1, tensor_col2 = st.columns(2)

            with tensor_col1:
                def save_output_tensor_type():
                    config["output_tensor_type"] = st.session_state.output_tensor_type_select
                    save_config(config)

                # Handle migration from old label and old checkbox
                saved_output_value = config.get("output_tensor_type", "Same as quant type (default)")
                if saved_output_value == "Same as quantization type":
                    saved_output_value = "Same as quant type (default)"
                # Migrate from old "leave_output_tensor" checkbox
                if config.get("leave_output_tensor", False) and saved_output_value == "Same as quant type (default)":
                    saved_output_value = "Unquantized"
                    config["output_tensor_type"] = "Unquantized"
                    config["leave_output_tensor"] = False  # Clear old setting
                    save_config(config)

                output_tensor_type = st.selectbox(
                    "Override output tensor type",
                    options=output_tensor_options,
                    index=output_tensor_options.index(saved_output_value),
                    help="Quantization type for output.weight tensor. Select 'Unquantized' or a higher precision type (Q8_0, F16) to improve quality at the cost of slightly larger file size.",
                    key="output_tensor_type_select",
                    on_change=save_output_tensor_type
                )

            with tensor_col2:
                def save_token_embedding_type():
                    config["token_embedding_type"] = st.session_state.token_embedding_type_select
                    save_config(config)

                # Handle migration from old label
                saved_token_value = config.get("token_embedding_type", "Same as quant type (default)")
                if saved_token_value == "Same as quantization type":
                    saved_token_value = "Same as quant type (default)"

                token_embedding_type = st.selectbox(
                    "Override token embedding type",
                    options=token_embedding_options,
                    index=token_embedding_options.index(saved_token_value),
                    help="Quantization type for token embeddings. Keeping this at higher precision can improve accuracy for certain tasks.",
                    key="token_embedding_type_select",
                    on_change=save_token_embedding_type
                )

            pure_quantization = st.checkbox(
                "Pure quantization (disable mixtures)",
                value=config.get("pure_quantization", False),
                help="Disable k-quant mixtures and quantize all tensors to the same type. Results in more uniform quantization.",
                key="pure_quantization_checkbox",
                on_change=make_config_saver(config, "pure_quantization", "pure_quantization_checkbox")
            )

            st.markdown("---")

            mmproj_precision_options = ["F16 (recommended)", "F32", "BF16", "Q8_0"]

            def save_mmproj_precision():
                config["mmproj_precision"] = st.session_state.mmproj_precision_radio
                save_config(config)

            # Default to F16 for compatibility
            saved_mmproj_precision = config.get("mmproj_precision", "F16 (recommended)")

            mmproj_precision = st.radio(
                "Vision Model Settings: mmproj precision",
                options=mmproj_precision_options,
                index=mmproj_precision_options.index(saved_mmproj_precision) if saved_mmproj_precision in mmproj_precision_options else 0,
                horizontal=True,
                help="Precision for vision projector file. F16 recommended for best compatibility (BF16 has known CUDA issues). Only applies to vision/multimodal models.",
                key="mmproj_precision_radio",
                on_change=save_mmproj_precision
            )

        # Track intermediate format changes to restore checkbox states
        if 'previous_intermediate' not in st.session_state:
            st.session_state.previous_intermediate = intermediate_type

        if "unquantized_saved_states" not in config:
            config["unquantized_saved_states"] = {}
        if "other_quants" not in config:
            config["other_quants"] = {}

        # Save current intermediate's state before it gets disabled
        if intermediate_type not in config["unquantized_saved_states"]:
            current_state = config["other_quants"].get(intermediate_type, False)
            config["unquantized_saved_states"][intermediate_type] = current_state
            save_config(config)

        # Handle intermediate format change
        if intermediate_type != st.session_state.previous_intermediate:
            prev_format = st.session_state.previous_intermediate

            if "other_quants" not in config:
                config["other_quants"] = {}
            if "unquantized_saved_states" not in config:
                config["unquantized_saved_states"] = {}

            # Save non-intermediate checkbox states
            for qtype in ["F32", "F16", "BF16"]:
                # Use previous using_custom_intermediate state for the key
                prev_custom = st.session_state.get('previous_using_custom_intermediate', using_custom_intermediate)
                checkbox_key = f"full_{qtype}_{prev_format}_{prev_custom}"
                if checkbox_key in st.session_state and qtype != prev_format:
                    config["other_quants"][qtype] = st.session_state[checkbox_key]

            # Restore previous intermediate's state
            if prev_format in config["unquantized_saved_states"]:
                config["other_quants"][prev_format] = config["unquantized_saved_states"][prev_format]
                del config["unquantized_saved_states"][prev_format]
            elif prev_format not in config["other_quants"]:
                config["other_quants"][prev_format] = False

            # Save new intermediate's state before disabling
            current_new_state = config["other_quants"].get(intermediate_type, False)
            config["unquantized_saved_states"][intermediate_type] = current_new_state

            st.session_state.previous_intermediate = intermediate_type
            config["intermediate_type"] = intermediate_type
            save_config(config)
            st.rerun()

        st.subheader("Importance Matrix (imatrix)")

        # Create columns for checkboxes
        imatrix_col1, imatrix_col2 = st.columns(2)

        with imatrix_col1:
            def save_use_imatrix():
                """Save imatrix checkbox state and custom name"""
                if f"imatrix_custom_name_{st.session_state.reset_count}" in st.session_state:
                    config["imatrix_generate_name"] = st.session_state[f"imatrix_custom_name_{st.session_state.reset_count}"]
                key = f"use_imatrix_checkbox_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["use_imatrix"] = st.session_state[key]
                    save_config(config)

            use_imatrix = st.checkbox(
                "Use imatrix",
                value=config.get("use_imatrix", False),
                help="Use importance matrix for better low-bit quantization (IQ2, IQ3).  \nRequired for best IQ2/IQ3 quality.",
                key=f"use_imatrix_checkbox_{st.session_state.reset_count}",
                on_change=save_use_imatrix
            )

        with imatrix_col2:
            def save_enforce_imatrix():
                """Save enforce imatrix checkbox state"""
                key = f"enforce_imatrix_checkbox_{st.session_state.reset_count}"
                if key in st.session_state:
                    config["ignore_imatrix_warnings"] = not st.session_state[key]
                    save_config(config)

            enforce_imatrix = st.checkbox(
                "Enforce imatrix",
                value=not config.get("ignore_imatrix_warnings", False),
                help="Require importance matrix for IQ quants. Uncheck to allow IQ quants without imatrix (advanced users only).",
                key=f"enforce_imatrix_checkbox_{st.session_state.reset_count}",
                on_change=save_enforce_imatrix
            )

        # Create local variable for backward compatibility
        ignore_imatrix_warnings = not enforce_imatrix

        # Show info when IQ quants are disabled
        if not use_imatrix and not ignore_imatrix_warnings:
            st.info("""
            **Importance Matrix Disabled**

            - Some I Quants (IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M, IQ1_S) require an importance matrix to be generated.
            Enable "Use imatrix" above to unlock these quantization types.

            - Uncheck "Enforce imatrix" to try without an imatrix anyway (advanced users only).
            """)

        # Show imatrix options when enabled
        imatrix_mode = None
        imatrix_generate_name = None
        imatrix_reuse_path = None

        if use_imatrix:
            # Scan output directory for .imatrix files (newest first)
            imatrix_files = []
            if output_dir_clean and Path(output_dir_clean).exists():
                output_path = Path(output_dir_clean)
                # Get files sorted by modification time
                files_with_mtime = [(f.name, f.stat().st_mtime) for f in output_path.glob("*.imatrix")]
                files_with_mtime.sort(key=lambda x: x[1], reverse=True)
                imatrix_files = [f[0] for f in files_with_mtime]

            # Build dropdown: existing files + generate options
            if model_path_clean:
                default_name = f"{Path(model_path_clean).name}.imatrix"
                generate_default_option = f"GENERATE ({default_name})"
            else:
                generate_default_option = "GENERATE (provide model path)"

            generate_custom_option = "GENERATE (custom name)"
            dropdown_options = imatrix_files + [generate_default_option, generate_custom_option]

            # Use same column structure as path inputs
            cols, has_browse = path_input_columns()

            # Determine default selection
            saved_mode = config.get("imatrix_mode", "")
            saved_reuse_path = config.get("imatrix_reuse_path", "")
            default_index = 0
            fallback_happened = False

            if saved_mode == "generate_custom":
                default_index = len(dropdown_options) - 1
            elif saved_mode == "generate":
                default_index = len(imatrix_files)
            elif saved_mode == "reuse" and saved_reuse_path and saved_reuse_path in imatrix_files:
                default_index = imatrix_files.index(saved_reuse_path)
            elif saved_mode == "reuse" and saved_reuse_path:
                # Saved file not found - falling back
                fallback_happened = True
                if imatrix_files:
                    # Auto-select newest existing file
                    default_index = 0
                else:
                    # No files - select generate default
                    default_index = len(imatrix_files)
            elif imatrix_files:
                # Auto-select newest existing file
                default_index = 0
            else:
                # No files - select generate default
                default_index = len(imatrix_files)

            # Create dynamic label showing count of existing imatrix files
            if imatrix_files:
                imatrix_label = f"Imatrix file: {len(imatrix_files)} detected"
            else:
                imatrix_label = "Imatrix file"

            with cols[0]:
                imatrix_selection = st.selectbox(
                    imatrix_label,
                    options=dropdown_options,
                    index=default_index,
                    help="Choose an existing imatrix file, generate with default name, or generate with custom name",
                    key=f"imatrix_dropdown_{st.session_state.reset_count}"
                )

                # Auto-sync dropdown value with config if they don't match
                # This handles the case where the dropdown defaults to a different value
                # but the user hasn't manually changed it (so no manual save has happened)
                if imatrix_selection in imatrix_files:
                    # It's a reuse selection - sync it
                    if saved_mode != "reuse" or saved_reuse_path != imatrix_selection:
                        config["imatrix_mode"] = "reuse"
                        config["imatrix_reuse_path"] = imatrix_selection
                        save_config(config)

                # Track fallback for warning display later
                if fallback_happened:
                    # Only print to terminal if this is a NEW fallback (not already in session state)
                    is_new_fallback = "imatrix_fallback_warning" not in st.session_state

                    st.session_state.imatrix_fallback_warning = {
                        "previous": saved_reuse_path,
                        "current": imatrix_selection,
                        "is_file": imatrix_selection in imatrix_files
                    }

                    # Print to terminal only on first detection
                    if is_new_fallback:
                        if imatrix_selection in imatrix_files:
                            print(f"\n{theme['warning']}Warning: Imatrix file auto-switched{Style.RESET_ALL}")
                            print(f"{theme['warning']}  Previous: {saved_reuse_path} (not found){Style.RESET_ALL}")
                            print(f"{theme['warning']}  Now using: {imatrix_selection}{Style.RESET_ALL}\n")
                        else:
                            print(f"\n{theme['warning']}Warning: Imatrix mode auto-switched{Style.RESET_ALL}")
                            print(f"{theme['warning']}  Previous: {saved_reuse_path} (not found){Style.RESET_ALL}")
                            print(f"{theme['warning']}  Now using: {imatrix_selection}{Style.RESET_ALL}\n")
                else:
                    # Clear warning if no fallback
                    if "imatrix_fallback_warning" in st.session_state:
                        del st.session_state.imatrix_fallback_warning

            # Place button in middle column if tkinter available, otherwise last column
            button_col = cols[1] if has_browse else cols[-1]
            with button_col:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
                if st.button("Refresh File List", key="update_imatrix_file_list", use_container_width=True):
                    # Save the current selection before rerunning
                    current_selection = st.session_state.get(f"imatrix_dropdown_{st.session_state.reset_count}")
                    if current_selection:
                        if current_selection == generate_custom_option:
                            config["imatrix_mode"] = "generate_custom"
                        elif current_selection == generate_default_option:
                            config["imatrix_mode"] = "generate"
                        elif current_selection in imatrix_files:
                            config["imatrix_mode"] = "reuse"
                            config["imatrix_reuse_path"] = current_selection
                        save_config(config)
                    # Clear the fallback warning since user is confirming their selection
                    if "imatrix_fallback_warning" in st.session_state:
                        del st.session_state.imatrix_fallback_warning
                    st.session_state.imatrix_refresh_toast = "Refreshed - imatrix file list"
                    st.rerun()

            # Show toast message after refresh (if flag is set)
            if "imatrix_refresh_toast" in st.session_state:
                st.toast(st.session_state.imatrix_refresh_toast)
                del st.session_state.imatrix_refresh_toast

            # Show imatrix fallback warning right below the dropdown (if applicable)
            # Wrap in cols[0] to match dropdown width
            if "imatrix_fallback_warning" in st.session_state:
                with cols[0]:
                    warning_data = st.session_state.imatrix_fallback_warning
                    if warning_data["is_file"]:
                        st.warning(f"`{warning_data['previous']}` not found.  "
                                 f"Auto-selected newest file: `{warning_data['current']}`.  ")
                    else:
                        st.warning(f"**Imatrix mode changed:** Previously selected `{warning_data['previous']}` not found.  "
                                 f"Switched to: `{warning_data['current']}`.")

            # Handle selection
            if imatrix_selection == generate_custom_option:
                config["imatrix_mode"] = "generate_custom"
                save_config(config)

                def save_custom_imatrix_name():
                    key = f"imatrix_custom_name_{st.session_state.reset_count}"
                    if key in st.session_state:
                        config["imatrix_generate_name"] = st.session_state[key]
                        save_config(config)

                col_imatrix_name, col_imatrix_default = st.columns([5, 1])
                with col_imatrix_name:
                    imatrix_generate_name = st.text_input(
                        "Custom imatrix filename",
                        value=config.get("imatrix_generate_name", ""),
                        placeholder="model.imatrix",
                        help="Filename for the generated imatrix file (saved in output directory). Leave empty to use default naming (model_name.imatrix).",
                        key=f"imatrix_custom_name_{st.session_state.reset_count}",
                        on_change=save_custom_imatrix_name
                    )

                    # Auto-append .imatrix extension and sanitize
                    if imatrix_generate_name:
                        # Sanitize the filename
                        sanitized_name = sanitize_filename(imatrix_generate_name)

                        # Show if sanitization changed the name
                        if sanitized_name != imatrix_generate_name:
                            st.warning(f"Filename cleaned: `{imatrix_generate_name}` → `{sanitized_name}`")

                        # Add .imatrix extension if needed
                        if sanitized_name and not sanitized_name.endswith('.imatrix'):
                            final_name = f"{sanitized_name}.imatrix"
                            st.info(f"Will be saved as: `{final_name}`")
                        elif sanitized_name:
                            final_name = sanitized_name
                            st.info(f"Will be saved as: `{final_name}`")
                        else:
                            # Sanitization resulted in empty string - fall back to default
                            if model_path_clean:
                                final_name = f"{Path(model_path_clean).name}.imatrix"
                                st.warning(f"Invalid filename - using default: `{final_name}`")
                            else:
                                final_name = None

                        # Warn if file exists
                        if final_name and output_dir_clean:
                            imatrix_file_path = Path(output_dir_clean) / final_name
                            if imatrix_file_path.exists():
                                st.warning(f"WARNING: File already exists and will be overwritten: `{final_name}`")
                    else:
                        # Show default name when field is blank
                        if model_path_clean:
                            default_name = f"{Path(model_path_clean).name}.imatrix"
                            st.info(f"Will use default name: `{default_name}`")

                            # Warn if default file exists
                            if output_dir_clean:
                                imatrix_file_path = Path(output_dir_clean) / default_name
                                if imatrix_file_path.exists():
                                    st.warning(f"WARNING: File already exists and will be overwritten: `{default_name}`")

                with col_imatrix_default:
                    st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                    if st.button("Set to default", key="set_default_imatrix_name", use_container_width=True):
                        if model_path_clean:
                            default_name = f"{Path(model_path_clean).name}.imatrix"
                            config["imatrix_generate_name"] = default_name
                        else:
                            config["imatrix_generate_name"] = ""
                        save_config(config)
                        st.session_state.reset_count += 1
                        st.rerun()

                imatrix_reuse_path = None
                imatrix_mode = "GENERATE (custom name)"

            elif imatrix_selection == generate_default_option:
                imatrix_generate_name = ""
                imatrix_reuse_path = None
                imatrix_mode = "GENERATE (default name)"
                config["imatrix_mode"] = "generate"
                save_config(config)

                # Warn if file exists (in same column as selectbox for consistent width)
                with cols[0]:
                    if output_dir_clean and model_path_clean:
                        imatrix_file_path = Path(output_dir_clean) / default_name
                        if imatrix_file_path.exists():
                            st.warning(f"WARNING: File already exists and will be overwritten: `{default_name}`")

            else:
                imatrix_generate_name = None
                imatrix_reuse_path = imatrix_selection
                imatrix_mode = "Reuse existing"
                config["imatrix_mode"] = "reuse"
                config["imatrix_reuse_path"] = imatrix_selection
                save_config(config)

    with col2:
        st.subheader("Output Types")
        st.markdown("Select output formats (one intermediate format is required):")

        IMATRIX_REQUIRED_TYPES = [
            "IQ1_S", "IQ1_M",
            "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
            "IQ3_XXS", "IQ3_XS"
        ]

        # Show label and custom intermediate info when applicable
        if using_custom_intermediate:
            custom_intermediate_mode = config.get("custom_intermediate_mode", "")
            st.markdown(f'**Intermediate File:** `{custom_intermediate_mode}`')

        else:
            # Build header with source dtype info if available
            if source_dtype:
                header = f"**Intermediate Formats** (source: `{source_dtype}`):"
            elif config_missing:
                header = "**Intermediate Formats** (config.json is missing!)"
            else:
                header = "**Intermediate Formats:**"
            st.markdown(header)

        # Override intermediate_type if using custom intermediate
        if using_custom_intermediate:
            custom_format = config.get("custom_intermediate_format")
            if custom_format:
                intermediate_type = custom_format

        # 3 columns for the 3 formats
        format_cols = st.columns(3)
        full_quants = {
            "F32": "32-bit float (full precision)",
            "F16": "16-bit float (half precision)",
            "BF16": "16-bit bfloat (brain float)",
        }
        full_checkboxes = {}
        selected_format = None

        # First row: buttons to select intermediate format
        for idx, (qtype, tooltip) in enumerate(full_quants.items()):
            with format_cols[idx]:
                is_selected = qtype == intermediate_type
                button_type = "primary" if is_selected else "secondary"
                button_label = f"{qtype} Intermediate"
                if st.button(
                    button_label,
                    key=f"intermediate_btn_{qtype}",
                    type=button_type,
                    disabled=using_custom_intermediate
                ):
                    selected_format = qtype

        # Second row: checkboxes to enable output
        for idx, (qtype, tooltip) in enumerate(full_quants.items()):
            with format_cols[idx]:
                is_intermediate = qtype == intermediate_type

                if is_intermediate and not using_custom_intermediate:
                    # Intermediate format is checked and disabled (only in normal mode)
                    checkbox_value = True
                    checkbox_disabled = True
                elif using_custom_intermediate:
                    # In custom intermediate mode, all checkboxes are disabled and unchecked
                    checkbox_value = config.get("other_quants", {}).get(qtype, False)
                    checkbox_disabled = True
                else:
                    # Normal mode: show saved state and are enabled
                    checkbox_value = config.get("other_quants", {}).get(qtype, False)
                    checkbox_disabled = False

                def save_full_selection(qt, inter_type):
                    def save_to_config():
                        key = f"full_{qt}_{inter_type}_{using_custom_intermediate}"
                        if key in st.session_state and not st.session_state[key]:
                            config["other_quants"][qt] = False
                            save_config(config)
                    return save_to_config

                # Include using_custom_intermediate in key to force widget recreation when switching modes
                widget_key = f"full_{qtype}_{intermediate_type}_{using_custom_intermediate}"

                full_checkboxes[qtype] = st.checkbox(
                    qtype,
                    value=checkbox_value,
                    help=tooltip,
                    key=widget_key,
                    disabled=checkbox_disabled,
                    on_change=save_full_selection(qtype, intermediate_type) if not checkbox_disabled else None
                )

        # Handle intermediate format change
        if selected_format and selected_format != intermediate_type:
            if "unquantized_saved_states" not in config:
                config["unquantized_saved_states"] = {}
            if "other_quants" not in config:
                config["other_quants"] = {}

            # Save checkbox states before switching
            for qtype in ["F32", "F16", "BF16"]:
                checkbox_key = f"full_{qtype}_{intermediate_type}_{using_custom_intermediate}"
                if checkbox_key in st.session_state:
                    if qtype != intermediate_type:
                        config["other_quants"][qtype] = st.session_state[checkbox_key]

            # Restore old intermediate's state
            if intermediate_type in config["unquantized_saved_states"]:
                config["other_quants"][intermediate_type] = config["unquantized_saved_states"][intermediate_type]
                del config["unquantized_saved_states"][intermediate_type]

            # Save new intermediate's state before making it intermediate
            config["unquantized_saved_states"][selected_format] = config.get("other_quants", {}).get(selected_format, False)

            config["intermediate_type"] = selected_format
            intermediate_type = selected_format
            st.session_state.previous_intermediate = selected_format
            save_config(config)
            st.rerun()

        st.markdown("**Legacy Quants:**")
        def save_trad_selection(qtype, widget_key):
            def save_to_config():
                if widget_key in st.session_state:
                    config["other_quants"][qtype] = st.session_state[widget_key]
                    save_config(config)
            return save_to_config

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
                checkbox_value = config.get("other_quants", {}).get(qtype, qtype == "Q8_0" if qtype == "Q8_0" else False)
                widget_key = f"trad_{qtype}"

                trad_checkboxes[qtype] = st.checkbox(
                    qtype,
                    value=checkbox_value,
                    help=tooltip,
                    key=widget_key,
                    on_change=save_trad_selection(qtype, widget_key)
                )

        def save_quant_selection(qtype, widget_key):
            def save_to_config():
                if widget_key in st.session_state:
                    config["other_quants"][qtype] = st.session_state[widget_key]
                    save_config(config)
            return save_to_config

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
                checkbox_value = config.get("other_quants", {}).get(qtype, qtype == "Q4_K_M")
                widget_key = f"k_{qtype}"

                k_checkboxes[qtype] = st.checkbox(
                    qtype,
                    value=checkbox_value,
                    help=tooltip,
                    key=widget_key,
                    on_change=save_quant_selection(qtype, widget_key)
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
                is_disabled = (qtype in IMATRIX_REQUIRED_TYPES) and not use_imatrix and not ignore_imatrix_warnings
                widget_key = f"i_{qtype}_{use_imatrix}_{ignore_imatrix_warnings}"

                if is_disabled:
                    checkbox_value = False
                else:
                    checkbox_value = config.get("other_quants", {}).get(qtype, False)

                help_text = tooltip
                if is_disabled:
                    help_text += " (Requires importance matrix)"

                i_checkboxes[qtype] = st.checkbox(
                    qtype,
                    value=checkbox_value,
                    help=help_text,
                    key=widget_key,
                    disabled=is_disabled,
                    on_change=save_quant_selection(qtype, widget_key) if not is_disabled else None
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
        # Check if there's an imatrix auto-switch warning - block conversion if so
        if "imatrix_fallback_warning" in st.session_state:
            warning_data = st.session_state.imatrix_fallback_warning
            st.error(f"`{warning_data['previous']}` not found.  `{warning_data['current']}` auto-selected. "
                   f"Please confirm your selection, then try again.")
            print(f"\n{theme['error']}Conversion blocked: Imatrix auto-switch detected{Style.RESET_ALL}")
            print(f"{theme['error']}  User must confirm selection before proceeding{Style.RESET_ALL}\n")
            # Clear the warning since user has now seen the blocking error
            del st.session_state.imatrix_fallback_warning
            return

        # Strip quotes from paths
        model_path_clean = strip_quotes(model_path)
        output_dir_clean = strip_quotes(output_dir)

        if not model_path_clean:
            st.error("Please provide a model path")
        elif not output_dir_clean:
            st.error("Please provide an output directory")
        elif not selected_quants:
            st.error("Please select at least one quantization type")
        else:
            # Update config with current values
            config["verbose"] = verbose
            config["num_threads"] = num_threads
            config["use_imatrix"] = use_imatrix
            config["model_path"] = model_path_clean
            config["output_dir"] = output_dir_clean
            config["intermediate_type"] = intermediate_type

            # Save all quantization selections in other_quants (excluding intermediate format)
            all_quant_selections = {}
            all_quant_selections.update(full_checkboxes)
            all_quant_selections.update(trad_checkboxes)
            all_quant_selections.update(k_checkboxes)
            # For I quants, save the actual session state values (not the disabled ones)
            if "iq_checkbox_states" in st.session_state:
                all_quant_selections.update(st.session_state.iq_checkbox_states)
            else:
                all_quant_selections.update(i_checkboxes)

            # Remove intermediate format from saved selections (tracked separately)
            if intermediate_type in all_quant_selections:
                del all_quant_selections[intermediate_type]

            config["other_quants"] = all_quant_selections

            save_config(config)

            try:

                # Determine imatrix parameters based on mode
                generate_imatrix_flag = False
                imatrix_path_to_use = None
                calibration_file_path = None
                imatrix_output_filename = None

                if use_imatrix:
                    # Determine if we're generating or reusing based on mode
                    if imatrix_mode == "Reuse existing":
                        # Reuse existing file
                        generate_imatrix_flag = False
                        if imatrix_reuse_path:
                            imatrix_path_to_use = Path(output_dir_clean) / imatrix_reuse_path

                            # Validate that the imatrix file actually exists
                            if not imatrix_path_to_use.exists():
                                # Print detailed error to terminal
                                print(f"\n{theme['error']}Imatrix file not found:{Style.RESET_ALL}")
                                print(f"{theme['error']}  Looking for: {imatrix_path_to_use}{Style.RESET_ALL}")
                                print(f"{theme['error']}  Config has: {imatrix_reuse_path}{Style.RESET_ALL}")
                                print(f"{theme['error']}  In directory: {output_dir_clean}{Style.RESET_ALL}")
                                print(f"{theme['warning']}  Tip: Click 'Refresh File List' to update the imatrix dropdown{Style.RESET_ALL}\n")

                                # Show detailed error in GUI
                                st.error(f"**Imatrix file not found:** `{imatrix_reuse_path}`\n\n"
                                       f"Expected path: `{imatrix_path_to_use}`\n\n"
                                       f"**Fix:** Click **Refresh File List** to update the imatrix dropdown, "
                                       f"generate a new imatrix, or turn off imatrix.")
                                return  # Stop processing but keep UI working
                        config["imatrix_mode"] = "reuse"
                        config["imatrix_reuse_path"] = imatrix_reuse_path
                    elif imatrix_mode == "GENERATE (default name)":
                        # Generate with default name
                        generate_imatrix_flag = True
                        imatrix_output_filename = None  # Use default naming
                        config["imatrix_mode"] = "generate"

                        # Validate calibration file
                        calibration_file_path = validate_calibration_file(config)
                        if calibration_file_path is None:
                            return  # Stop processing but keep UI working
                    elif imatrix_mode == "GENERATE (custom name)":
                        # Generate with custom name
                        generate_imatrix_flag = True
                        config["imatrix_mode"] = "generate_custom"

                        # Sanitize and normalize filename to ensure .imatrix extension
                        if imatrix_generate_name:
                            sanitized_name = sanitize_filename(imatrix_generate_name)
                            if sanitized_name:
                                if not sanitized_name.endswith('.imatrix'):
                                    imatrix_output_filename = f"{sanitized_name}.imatrix"
                                else:
                                    imatrix_output_filename = sanitized_name
                                config["imatrix_generate_name"] = imatrix_generate_name
                            else:
                                # Sanitization resulted in empty string - use default
                                imatrix_output_filename = None
                        else:
                            imatrix_output_filename = None

                        # Validate calibration file
                        calibration_file_path = validate_calibration_file(config)
                        if calibration_file_path is None:
                            return  # Stop processing but keep UI working

                    save_config(config)

                with st.spinner("Converting and quantizing... This may take a while."):
                    # Use GPU offloading if configured
                    num_gpu_layers_value = int(config.get("imatrix_num_gpu_layers", 0))
                    use_num_gpu_layers = num_gpu_layers_value if num_gpu_layers_value > 0 else None

                    # Format split size for llama.cpp
                    split_size_override = None
                    if keep_split and max_shard_size_gb > 0:
                        # Always convert to MB (e.g., 1.1G -> 1100M, 2.5G -> 2500M)
                        split_size_override = f"{round(max_shard_size_gb * 1000)}M"

                    # Handle output tensor type selection
                    leave_output_tensor = (output_tensor_type == "Unquantized")
                    output_tensor_type_param = None if output_tensor_type in ["Same as quant type (default)", "Unquantized"] else output_tensor_type

                    # Handle token embedding type selection
                    token_embedding_type_param = None if token_embedding_type in ["Same as quant type (default)", "Unquantized"] else token_embedding_type

                    # Get custom intermediate parameters if selected
                    custom_intermediate_path = None
                    custom_intermediate_format = None
                    if using_custom_intermediate:
                        custom_intermediate_path = config.get("custom_intermediate_path")
                        custom_intermediate_format = config.get("custom_intermediate_format")

                    # Handle mmproj precision selection
                    mmproj_precision_raw = config.get("mmproj_precision", "F16 (recommended)")
                    # Strip the "(recommended)" suffix if present
                    mmproj_precision_clean = mmproj_precision_raw.replace(" (recommended)", "").strip()

                    output_files = converter.convert_and_quantize(
                        model_path=model_path_clean,
                        output_dir=output_dir_clean,
                        quantization_types=selected_quants,
                        intermediate_type=intermediate_type,
                        custom_intermediate_path=custom_intermediate_path,
                        custom_intermediate_format=custom_intermediate_format,
                        verbose=verbose,
                        generate_imatrix=generate_imatrix_flag,
                        imatrix_path=imatrix_path_to_use,
                        num_threads=num_threads,
                        imatrix_ctx_size=int(config.get("imatrix_ctx_size", 512)),
                        imatrix_chunks=int(config.get("imatrix_chunks", 100)) if config.get("imatrix_chunks", 100) > 0 else None,
                        imatrix_collect_output=config.get("imatrix_collect_output_weight", False),
                        imatrix_calibration_file=calibration_file_path,
                        imatrix_output_name=imatrix_output_filename,
                        imatrix_num_gpu_layers=use_num_gpu_layers,
                        ignore_imatrix_warnings=ignore_imatrix_warnings,
                        leave_output_tensor=leave_output_tensor,
                        pure_quantization=pure_quantization,
                        keep_split=keep_split,
                        output_tensor_type=output_tensor_type_param,
                        token_embedding_type=token_embedding_type_param,
                        overwrite_intermediates=config.get("overwrite_intermediates", False),
                        overwrite_quants=config.get("overwrite_quants", True),
                        split_max_size=split_size_override,
                        mmproj_precision=mmproj_precision_clean
                    )

                st.success(f"Successfully processed {len(output_files)} files!")

                # Update imatrix statistics tab with this conversion's output directory
                config["imatrix_stats_output_dir"] = output_dir_clean

                # If imatrix was used, save paths for statistics tab
                if use_imatrix:
                    # Determine which imatrix file was used
                    actual_imatrix_path = None
                    if imatrix_mode == "Reuse existing":
                        # Reused existing file
                        actual_imatrix_path = imatrix_path_to_use
                    elif imatrix_mode == "GENERATE (default name)":
                        # Generated with default name
                        model_name = Path(model_path_clean).name
                        actual_imatrix_path = Path(output_dir_clean) / f"{model_name}.imatrix"
                    elif imatrix_mode == "GENERATE (custom name)":
                        # Generated with custom name
                        if imatrix_output_filename:
                            actual_imatrix_path = Path(output_dir_clean) / imatrix_output_filename
                        else:
                            model_name = Path(model_path_clean).name
                            actual_imatrix_path = Path(output_dir_clean) / f"{model_name}.imatrix"

                    # Save paths for statistics tab
                    intermediate_path = Path(output_dir_clean) / f"{Path(model_path_clean).name}_{intermediate_type.upper()}.gguf"

                    if actual_imatrix_path and actual_imatrix_path.exists():
                        config["imatrix_stats_path"] = str(actual_imatrix_path)
                    if intermediate_path.exists():
                        config["imatrix_stats_model"] = str(intermediate_path)
                    save_config(config)

                st.subheader("Output Files")
                for file_path in output_files:
                    file_size = file_path.stat().st_size / (1024**3)  # GB
                    st.write(f"`{file_path}` ({file_size:.2f} GB)")

            except Exception as e:
                # Show in Streamlit UI
                st.error(f"Error: {e}")
                if verbose:
                    st.markdown("**Verbose output:**")
                    st.exception(e)

                # Print to terminal - traceback only for unexpected errors
                import traceback
                import sys
                exc_type, exc_value, exc_tb = sys.exc_info()

                # Expected user errors - just print the message, no traceback
                if isinstance(e, (RuntimeError, ValueError, FileNotFoundError)):
                    print(f"\n{theme['error']}{exc_type.__name__}: {exc_value}{Style.RESET_ALL}\n")
                else:
                    # Unexpected errors - print full traceback for debugging
                    traceback.print_tb(exc_tb)
                    print(f"{theme['error']}{exc_type.__name__}: {exc_value}{Style.RESET_ALL}")
