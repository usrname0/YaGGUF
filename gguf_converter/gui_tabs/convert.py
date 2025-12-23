"""
Convert & Quantize tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, make_config_saver, path_input_columns
)


def render_convert_tab(converter, config, verbose, nthreads, ignore_incompatibilities, ignore_imatrix_warnings):
    """Render the Convert & Quantize tab"""
    st.header("Convert and Quantize Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")

        # Model path with Browse and Check Folder buttons
        cols, has_browse = path_input_columns()

        with cols[0]:
            # Only set value if key not in session state (prevents warning)
            model_path_kwargs = {
                "label": "Model path",
                "placeholder": "E:/Models/my-model",
                "help": "Local model directory containing config.json and model files.",
                "key": "model_path_input"
            }
            if "model_path_input" not in st.session_state:
                model_path_kwargs["value"] = config.get("model_path", "")
            model_path = st.text_input(**model_path_kwargs)  # type: ignore[arg-type]

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Browse",
                    key="browse_model_folder_btn",
                    use_container_width=True,
                    help="Browse for model directory"
                ):
                    model_path_clean = strip_quotes(model_path)
                    initial_dir = model_path_clean if model_path_clean and Path(model_path_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["model_path"] = selected_folder
                        save_config(config)
                        # Update widget state to show new path
                        st.session_state.pending_model_path = selected_folder
                        st.rerun()

        with cols[-1]:  # Last column is always the check button
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            model_path_clean = strip_quotes(model_path)
            model_path_exists = bool(model_path_clean and Path(model_path_clean).exists())
            if st.button(
                "Check Folder",
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

        # Output directory with Browse and Check Folder buttons
        cols, has_browse = path_input_columns()

        with cols[0]:
            output_dir = st.text_input(
                "Output directory",
                value=config.get("output_dir", ""),
                placeholder="E:/Models/converted",
                help="Where to save the converted files"
            )

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Browse",
                    key="browse_output_folder_btn",
                    use_container_width=True,
                    help="Browse for output directory"
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
                "Check Folder",
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

        # Strip quotes from paths
        model_path_clean = strip_quotes(model_path)
        output_dir_clean = strip_quotes(output_dir)

        # Check for model incompatibilities
        incompatible_quants = []
        if model_path_clean and Path(model_path_clean).exists() and Path(model_path_clean).is_dir():
            config_json = Path(model_path_clean) / "config.json"
            if config_json.exists() and not ignore_incompatibilities:
                try:
                    incompat_info = converter.get_incompatibility_info(model_path_clean)
                    if incompat_info["has_incompatibilities"]:
                        incompatible_quants = incompat_info["incompatible_quants"]

                        # Display warning banner
                        st.info(f"""
**Some quants are disabled due to:** {', '.join(incompat_info['types'])}

{chr(10).join('- ' + reason for reason in incompat_info['reasons'])}

                        """)
                except (ValueError, OSError, KeyError, UnicodeDecodeError):
                    # Ignore config parsing errors - compatibility check is optional metadata display
                    pass

        # Advanced quantization options
        with st.expander("Advanced Quantization Options"):
            st.markdown("These options affect how llama.cpp quantizes your model. Most users won't need these.")

            allow_requantize = st.checkbox(
                "Allow requantize",
                value=config.get("allow_requantize", False),
                help="Allow quantizing models that are already quantized. Warning: This can severely reduce quality compared to quantizing from F16/F32.",
                key="allow_requantize_checkbox",
                on_change=make_config_saver(config, "allow_requantize", "allow_requantize_checkbox")
            )

            leave_output_tensor = st.checkbox(
                "Leave output tensor unquantized",
                value=config.get("leave_output_tensor", False),
                help="Keep output.weight at higher precision for better quality (increases model size slightly). Especially useful when requantizing.",
                key="leave_output_tensor_checkbox",
                on_change=make_config_saver(config, "leave_output_tensor", "leave_output_tensor_checkbox")
            )

            pure_quantization = st.checkbox(
                "Pure quantization (disable mixtures)",
                value=config.get("pure_quantization", False),
                help="Disable k-quant mixtures and quantize all tensors to the same type. Results in more uniform quantization.",
                key="pure_quantization_checkbox",
                on_change=make_config_saver(config, "pure_quantization", "pure_quantization_checkbox")
            )

            keep_split = st.checkbox(
                "Keep split files",
                value=config.get("keep_split", False),
                help="Generate quantized model in the same shards as the input model (for multi-file models).",
                key="keep_split_checkbox",
                on_change=make_config_saver(config, "keep_split", "keep_split_checkbox")
            )

            st.markdown("---")
            st.markdown("**Selective Tensor Quantization:** Override quantization type for specific tensors. Use these to keep certain layers at higher precision.")

            # Output tensor type
            tensor_type_options = ["Same as quantization type", "Q8_0", "Q6_K", "Q5_K_M", "F16", "F32"]

            def save_output_tensor_type():
                config["output_tensor_type"] = st.session_state.output_tensor_type_select
                save_config(config)

            output_tensor_type = st.selectbox(
                "Output tensor type",
                options=tensor_type_options,
                index=tensor_type_options.index(config.get("output_tensor_type", "Same as quantization type")),
                help="Quantization type for output.weight tensor. Keeping this at higher precision (Q8_0 or F16) can improve quality at the cost of slightly larger file size.",
                key="output_tensor_type_select",
                on_change=save_output_tensor_type
            )

            # Token embedding type
            def save_token_embedding_type():
                config["token_embedding_type"] = st.session_state.token_embedding_type_select
                save_config(config)

            token_embedding_type = st.selectbox(
                "Token embedding type",
                options=tensor_type_options,
                index=tensor_type_options.index(config.get("token_embedding_type", "Same as quantization type")),
                help="Quantization type for token embeddings. Keeping this at higher precision can improve accuracy for certain tasks.",
                key="token_embedding_type_select",
                on_change=save_token_embedding_type
            )

        # Save/restore quant selections when incompatibility status changes
        if 'previous_incompatible_quants' not in st.session_state:
            st.session_state.previous_incompatible_quants = []

        if "incompatible_saved_states" not in config:
            config["incompatible_saved_states"] = {}
        if "other_quants" not in config:
            config["other_quants"] = {}

        # Initialize IQ checkbox states
        if "iq_checkbox_states" not in st.session_state:
            st.session_state.iq_checkbox_states = config.get("other_quants", {}).copy()

        current_incompatible = set(incompatible_quants)
        previous_incompatible = set(st.session_state.previous_incompatible_quants)

        if current_incompatible != previous_incompatible:
            newly_incompatible = current_incompatible - previous_incompatible
            newly_compatible = previous_incompatible - current_incompatible

            # Save state before disabling newly incompatible quants
            for qtype in newly_incompatible:
                current_state = config["other_quants"].get(qtype, False)
                config["incompatible_saved_states"][qtype] = current_state
                config["other_quants"][qtype] = False
                if qtype in st.session_state.iq_checkbox_states:
                    st.session_state.iq_checkbox_states[qtype] = False

            # Restore state for newly compatible quants
            for qtype in newly_compatible:
                if qtype in config["incompatible_saved_states"]:
                    restored_value = config["incompatible_saved_states"][qtype]
                    config["other_quants"][qtype] = restored_value
                    st.session_state.iq_checkbox_states[qtype] = restored_value
                    del config["incompatible_saved_states"][qtype]

            st.session_state.previous_incompatible_quants = incompatible_quants.copy()

            if newly_incompatible or newly_compatible:
                save_config(config)

        intermediate_type = config.get("intermediate_type", "F16").upper()

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
                checkbox_key = f"full_{qtype}_{prev_format}"
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

        def save_use_imatrix():
            """Save imatrix checkbox state and custom name"""
            if f"imatrix_custom_name_{st.session_state.reset_count}" in st.session_state:
                config["imatrix_generate_name"] = st.session_state[f"imatrix_custom_name_{st.session_state.reset_count}"]
            config["use_imatrix"] = st.session_state[f"use_imatrix_checkbox_{st.session_state.reset_count}"]
            save_config(config)

        use_imatrix = st.checkbox(
            "Use importance matrix",
            value=config.get("use_imatrix", False),
            help="Use importance matrix for better low-bit quantization (IQ2, IQ3).  \nRequired for best IQ2/IQ3 quality.",
            key=f"use_imatrix_checkbox_{st.session_state.reset_count}",
            on_change=save_use_imatrix
        )

        # Show info when IQ quants are disabled
        if not use_imatrix and not ignore_imatrix_warnings:
            st.info("""
            **Importance Matrix Disabled**

            Some I Quants (IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M, IQ1_S) require an importance matrix to be generated.
            Enable "Use importance matrix" above to unlock these quantization types.
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
            col_imatrix_select, col_imatrix_update = st.columns([5, 1])
            with col_imatrix_select:
                # Determine default selection
                saved_mode = config.get("imatrix_mode", "")
                saved_reuse_path = config.get("imatrix_reuse_path", "")
                default_index = 0

                if saved_mode == "generate_custom":
                    default_index = len(dropdown_options) - 1
                elif saved_mode == "generate":
                    default_index = len(imatrix_files)
                elif saved_mode == "reuse" and saved_reuse_path and saved_reuse_path in imatrix_files:
                    default_index = imatrix_files.index(saved_reuse_path)
                elif imatrix_files:
                    # Auto-select newest existing file
                    default_index = 0
                else:
                    # No files - select generate default
                    default_index = len(imatrix_files)

                imatrix_selection = st.selectbox(
                    "Imatrix file",
                    options=dropdown_options,
                    index=default_index,
                    help="Choose an existing imatrix file, generate with default name, or generate with custom name",
                    key=f"imatrix_dropdown_{st.session_state.reset_count}"
                )
            with col_imatrix_update:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
                if st.button("Refresh File List", key="update_imatrix_file_list", use_container_width=True):
                    st.rerun()

            # Handle selection
            if imatrix_selection == generate_custom_option:
                config["imatrix_mode"] = "generate_custom"
                save_config(config)
                col_imatrix_name, col_imatrix_default = st.columns([5, 1])
                with col_imatrix_name:
                    imatrix_generate_name = st.text_input(
                        "Custom imatrix filename",
                        value=config.get("imatrix_generate_name", ""),
                        placeholder="model.imatrix",
                        help="Filename for the generated imatrix file (saved in output directory). Leave empty to use default naming (model_name.imatrix).",
                        key=f"imatrix_custom_name_{st.session_state.reset_count}"
                    )
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

                # Auto-append .imatrix extension
                if imatrix_generate_name:
                    if not imatrix_generate_name.endswith('.imatrix'):
                        final_name = f"{imatrix_generate_name}.imatrix"
                        st.info(f"Will be saved as: `{final_name}`")
                    else:
                        final_name = imatrix_generate_name

                    # Warn if file exists
                    if output_dir_clean:
                        imatrix_file_path = Path(output_dir_clean) / final_name
                        if imatrix_file_path.exists():
                            st.warning(f"WARNING: File already exists and will be overwritten: `{final_name}`")

                imatrix_reuse_path = None
                imatrix_mode = "GENERATE (custom name)"

            elif imatrix_selection == generate_default_option:
                imatrix_generate_name = ""
                imatrix_reuse_path = None
                imatrix_mode = "GENERATE (default name)"
                config["imatrix_mode"] = "generate"
                save_config(config)

                # Warn if file exists
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
        st.markdown("**Intermediate Formats:**")

        # 6 columns: [checkbox:1, button:2, checkbox:1, button:2, checkbox:1, button:2]
        all_cols = st.columns([1, 2, 1, 2, 1, 2])
        full_quants = {
            "F32": "32-bit float (full precision)",
            "F16": "16-bit float (half precision)",
            "BF16": "16-bit bfloat (brain float)",
        }
        full_checkboxes = {}
        selected_format = None

        for idx, (qtype, tooltip) in enumerate(full_quants.items()):
            with all_cols[idx * 2]:
                is_intermediate = qtype == intermediate_type

                if is_intermediate:
                    checkbox_value = True
                    checkbox_disabled = True
                else:
                    checkbox_value = config.get("other_quants", {}).get(qtype, False)
                    checkbox_disabled = False

                def save_full_selection(qt, inter_type):
                    def callback():
                        if not st.session_state[f"full_{qt}_{inter_type}"]:
                            config["other_quants"][qt] = False
                            save_config(config)
                    return callback

                full_checkboxes[qtype] = st.checkbox(
                    qtype,
                    value=checkbox_value,
                    help=tooltip,
                    key=f"full_{qtype}_{intermediate_type}",
                    disabled=checkbox_disabled,
                    on_change=save_full_selection(qtype, intermediate_type) if not checkbox_disabled else None
                )

            with all_cols[idx * 2 + 1]:
                is_selected = qtype == intermediate_type
                button_type = "primary" if is_selected else "secondary"
                button_label = f"{qtype} Intermediate"
                if st.button(
                    button_label,
                    key=f"intermediate_btn_{qtype}",
                    type=button_type
                ):
                    selected_format = qtype

        # Handle intermediate format change
        if selected_format and selected_format != intermediate_type:
            if "unquantized_saved_states" not in config:
                config["unquantized_saved_states"] = {}
            if "other_quants" not in config:
                config["other_quants"] = {}

            # Save checkbox states before switching
            for qtype in ["F32", "F16", "BF16"]:
                checkbox_key = f"full_{qtype}_{intermediate_type}"
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
            def callback():
                config["other_quants"][qtype] = st.session_state[widget_key]
                save_config(config)
            return callback

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
                is_incompatible = qtype in incompatible_quants

                if is_incompatible:
                    checkbox_value = False
                else:
                    checkbox_value = config.get("other_quants", {}).get(qtype, qtype == "Q8_0" if qtype == "Q8_0" else False)
                help_text = tooltip
                if is_incompatible:
                    help_text += " (Incompatible with this model - see info banner above)"

                widget_key = f"trad_{qtype}_{is_incompatible}"

                trad_checkboxes[qtype] = st.checkbox(
                    qtype,
                    value=checkbox_value,
                    help=help_text,
                    key=widget_key,
                    disabled=is_incompatible,
                    on_change=save_trad_selection(qtype, widget_key) if not is_incompatible else None
                )

        def save_quant_selection(qtype, widget_key):
            def callback():
                config["other_quants"][qtype] = st.session_state[widget_key]
                save_config(config)
            return callback

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
                is_incompatible = qtype in incompatible_quants

                if is_incompatible:
                    checkbox_value = False
                else:
                    checkbox_value = config.get("other_quants", {}).get(qtype, qtype == "Q4_K_M")
                help_text = tooltip
                if is_incompatible:
                    help_text += " (Incompatible with this model - see info banner above)"

                widget_key = f"k_{qtype}_{is_incompatible}"

                k_checkboxes[qtype] = st.checkbox(
                    qtype,
                    value=checkbox_value,
                    help=help_text,
                    key=widget_key,
                    disabled=is_incompatible,
                    on_change=save_quant_selection(qtype, widget_key) if not is_incompatible else None
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
                imatrix_disabled = (qtype in IMATRIX_REQUIRED_TYPES) and not use_imatrix and not ignore_imatrix_warnings
                incompatible_disabled = qtype in incompatible_quants
                is_disabled = imatrix_disabled or incompatible_disabled
                widget_key = f"i_{qtype}_{use_imatrix}_{incompatible_disabled}_{ignore_imatrix_warnings}"
                prev_key = f"i_{qtype}_{not use_imatrix}_{incompatible_disabled}_{not ignore_imatrix_warnings}"

                # Save previous state when transitioning enabled→disabled
                prev_was_enabled = (qtype not in IMATRIX_REQUIRED_TYPES) or (not use_imatrix) or ignore_imatrix_warnings
                if prev_key in st.session_state and prev_was_enabled:
                    st.session_state.iq_checkbox_states[qtype] = st.session_state[prev_key]
                if is_disabled:
                    checkbox_value = False
                else:
                    checkbox_value = st.session_state.iq_checkbox_states.get(qtype, False)

                def save_iq_selection(qt, key):
                    def callback():
                        val = st.session_state[key]
                        st.session_state.iq_checkbox_states[qt] = val
                        config["other_quants"][qt] = val
                        save_config(config)
                    return callback

                help_text = tooltip
                if incompatible_disabled:
                    help_text += " (Incompatible with this model - see info banner above)"
                elif imatrix_disabled:
                    help_text += " (Requires importance matrix)"

                i_checkboxes[qtype] = st.checkbox(
                    qtype,
                    value=checkbox_value,
                    help=help_text,
                    key=widget_key,
                    disabled=is_disabled,
                    on_change=save_iq_selection(qtype, widget_key) if not is_disabled else None
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
                        config["imatrix_mode"] = "reuse"
                        config["imatrix_reuse_path"] = imatrix_reuse_path
                    elif imatrix_mode == "GENERATE (default name)":
                        # Generate with default name
                        generate_imatrix_flag = True
                        imatrix_output_filename = None  # Use default naming
                        config["imatrix_mode"] = "generate"

                        # Build calibration file path for generation
                        cal_dir = config.get("imatrix_calibration_dir", "")
                        cal_file = config.get("imatrix_calibration_file", "wiki.train.raw")

                        if cal_dir:
                            calibration_file_path = Path(cal_dir) / cal_file
                        else:
                            # Use default calibration_data directory (one level up from gguf_converter module)
                            default_cal_dir = Path(__file__).parent.parent.parent / "calibration_data"
                            calibration_file_path = default_cal_dir / cal_file

                        # Validate calibration file exists
                        if not calibration_file_path.exists():
                            st.error("**Calibration file not available:** Add a file to the calibration_data folder, use an existing imatrix or turn off imatrix.")
                            return  # Stop processing but keep UI working
                    elif imatrix_mode == "GENERATE (custom name)":
                        # Generate with custom name
                        generate_imatrix_flag = True
                        config["imatrix_mode"] = "generate_custom"

                        # Normalize filename to ensure .imatrix extension
                        if imatrix_generate_name:
                            if not imatrix_generate_name.endswith('.imatrix'):
                                imatrix_output_filename = f"{imatrix_generate_name}.imatrix"
                            else:
                                imatrix_output_filename = imatrix_generate_name
                            config["imatrix_generate_name"] = imatrix_generate_name
                        else:
                            imatrix_output_filename = None

                        # Build calibration file path for generation
                        cal_dir = config.get("imatrix_calibration_dir", "")
                        cal_file = config.get("imatrix_calibration_file", "wiki.train.raw")

                        if cal_dir:
                            calibration_file_path = Path(cal_dir) / cal_file
                        else:
                            # Use default calibration_data directory (one level up from gguf_converter module)
                            default_cal_dir = Path(__file__).parent.parent.parent / "calibration_data"
                            calibration_file_path = default_cal_dir / cal_file

                        # Validate calibration file exists
                        if not calibration_file_path.exists():
                            st.error("**Calibration file not available:** Add a file to the calibration_data folder, use an existing imatrix or turn off imatrix.")
                            return  # Stop processing but keep UI working

                    save_config(config)

                with st.spinner("Converting and quantizing... This may take a while."):
                    # Only use GPU offloading if custom binaries are enabled
                    # YaGUFF binaries are CPU-only and don't support -ngl
                    use_ngl = None
                    if config.get("use_custom_binaries", False):
                        ngl_value = int(config.get("imatrix_ngl", 0))
                        use_ngl = ngl_value if ngl_value > 0 else None

                    output_files = converter.convert_and_quantize(
                        model_path=model_path_clean,
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
                        imatrix_output_name=imatrix_output_filename,
                        imatrix_ngl=use_ngl,
                        ignore_incompatibilities=ignore_incompatibilities,
                        ignore_imatrix_warnings=ignore_imatrix_warnings,
                        allow_requantize=allow_requantize,
                        leave_output_tensor=leave_output_tensor,
                        pure_quantization=pure_quantization,
                        keep_split=keep_split,
                        output_tensor_type=output_tensor_type if output_tensor_type != "Same as quantization type" else None,
                        token_embedding_type=token_embedding_type if token_embedding_type != "Same as quantization type" else None
                    )

                st.success(f"Successfully processed {len(output_files)} files!")

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
