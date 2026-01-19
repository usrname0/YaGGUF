"""
VRAM Calculator tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING, cast

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, TKINTER_AVAILABLE, get_platform_path,
    path_input_columns, detect_all_model_files
)
from ..vram_calc import (
    get_all_gpus, get_gguf_model_info, calculate_vram, format_size,
    get_system_ram_mb, GGUFModelInfo
)

if TYPE_CHECKING:
    from ..converter import GGUFConverter


def render_vram_calc_tab(converter: "GGUFConverter", config: Dict[str, Any]) -> None:
    """Render the VRAM Calculator tab."""
    st.header("VRAM Calculator")
    st.markdown("Estimate VRAM usage and recommended GPU layers (-ngl) for GGUF models")

    # =========================================================================
    # Row 1: GPU Detection (left) | GPU Stats (right)
    # =========================================================================
    col_gpu_detect, col_gpu_stats = st.columns(2)

    # Detect GPUs
    if "detected_gpus" not in st.session_state:
        st.session_state.detected_gpus = get_all_gpus()

    gpus = st.session_state.detected_gpus

    with col_gpu_detect:
        st.subheader("GPU Detection")

        # Build GPU options with manual option at the end
        gpu_options = []
        for gpu in gpus:
            vendor_label = gpu.vendor.upper()
            gpu_options.append(
                f"[{vendor_label}] {gpu.name} - {format_size(gpu.total_mb)} "
                f"({format_size(gpu.free_mb)} free)"
            )
        gpu_options.append("Manually specify VRAM")

        # Build label with count
        gpu_count = len(gpus)
        if gpu_count == 0:
            gpu_label = "Select GPU: none detected"
        elif gpu_count == 1:
            gpu_label = "Select GPU: 1 detected"
        else:
            gpu_label = f"Select GPU: {gpu_count} detected"

        col_gpu_select, col_gpu_refresh = st.columns([5, 1])
        with col_gpu_select:
            selected_gpu_idx = st.selectbox(
                gpu_label,
                options=range(len(gpu_options)),
                format_func=lambda i: gpu_options[i],
                help="Select the GPU to use for VRAM calculations"
            )
        with col_gpu_refresh:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "Refresh",
                key="refresh_gpus_btn",
                use_container_width=True,
                help="Re-detect available GPUs"
            ):
                st.session_state.detected_gpus = get_all_gpus()
                st.rerun()

        # Check if manual option selected (last item)
        use_manual_vram = selected_gpu_idx == len(gpus)

        if use_manual_vram:
            selected_gpu = None
            default_vram = max(512, config.get("vram_calc_manual_vram", 8192))
        else:
            selected_gpu = gpus[selected_gpu_idx]
            default_vram = max(512, selected_gpu.free_mb)

        # Each input field with its own conversion
        vram_label = "Manually Enter Available VRAM (MB)" if use_manual_vram else "GPU's Available VRAM (MB)"
        col_vram_input, col_vram_conv = st.columns(2)
        with col_vram_input:
            manual_vram = st.number_input(
                vram_label,
                min_value=512,
                max_value=524288,
                value=default_vram,
                step=512,
                help="Manually specify available VRAM in MB" if use_manual_vram else "Select 'Manually specify VRAM' to edit",
                disabled=not use_manual_vram
            )
        available_vram = manual_vram if use_manual_vram else selected_gpu.free_mb
        with col_vram_conv:
            st.text_input("Available VRAM", value=f"{available_vram/1000:.1f} GB / {available_vram/1024:.1f} GiB", disabled=True)

        col_head_input, col_head_conv = st.columns(2)
        with col_head_input:
            headroom_mb = st.number_input(
                "VRAM Headroom (MB)",
                min_value=0,
                max_value=16384,
                value=int(config.get("vram_calc_headroom_mb", 2048)),
                step=256,
                help="VRAM to reserve for other applications"
            )
        with col_head_conv:
            st.text_input("Headroom", value=f"{headroom_mb/1000:.1f} GB / {headroom_mb/1024:.1f} GiB", disabled=True)

        col_ctx_input, col_ctx_conv = st.columns(2)
        with col_ctx_input:
            context_size = st.number_input(
                "Context Size",
                min_value=512,
                max_value=131072,
                value=int(config.get("vram_calc_context_size", 4096)),
                step=512,
                help="Context window size affects KV cache VRAM"
            )
        with col_ctx_conv:
            approx_words = int(context_size * 0.75)
            approx_pages = context_size / 500
            st.text_input("Context", value=f"~{approx_words:,} words / ~{approx_pages:.0f} pages", disabled=True)

        col_kv_quant, col_kv_info = st.columns(2)
        with col_kv_quant:
            kv_quant_options = ["F16 (Default)", "Q8_0", "Q4_0"]
            saved_kv_quant = config.get("vram_calc_kv_quant", "F16 (Default)")
            kv_quant_index = kv_quant_options.index(saved_kv_quant) if saved_kv_quant in kv_quant_options else 0
            kv_quant_selected = st.selectbox(
                "KV Cache Quantization",
                options=kv_quant_options,
                index=kv_quant_index,
                help="Quantized KV cache reduces context VRAM. Use -ctk and -ctv flags in llama.cpp"
            )
        with col_kv_info:
            kv_savings = {"F16 (Default)": "1x", "Q8_0": "~2x smaller", "Q4_0": "~4x smaller"}
            st.text_input("KV Cache", value=kv_savings.get(kv_quant_selected, ""), disabled=True)

    with col_gpu_stats:
        if selected_gpu:
            st.subheader(f"{selected_gpu.name} Stats")
            col_total, col_used, col_free = st.columns(3)
            with col_total:
                st.metric("Total VRAM", format_size(selected_gpu.total_mb))
            with col_used:
                st.metric("Currently Used VRAM", format_size(selected_gpu.used_mb))
            with col_free:
                st.metric("Available VRAM", format_size(selected_gpu.free_mb))
        else:
            st.subheader("Manually Entered VRAM Stats")
            st.metric("Available VRAM", format_size(available_vram))

        # VRAM Breakdown (updates dynamically)
        usable_after_headroom = available_vram - headroom_mb
        st.markdown(
            f"Available: **{format_size(available_vram)}** - "
            f"Headroom: **{format_size(headroom_mb)}** = "
            f"Usable: **{format_size(max(0, usable_after_headroom))}**"
        )
        
        st.markdown("---")
        
        # System RAM Stats
        ram_total, ram_used, ram_avail = get_system_ram_mb()
        st.subheader("System RAM Stats")
        col_ram_total, col_ram_used, col_ram_free = st.columns(3)
        with col_ram_total:
            st.metric("Total RAM", format_size(ram_total))
        with col_ram_used:
            st.metric("Currently Used RAM", format_size(ram_used))
        with col_ram_free:
            st.metric("Available RAM", format_size(ram_avail))

        # RAM Breakdown
        st.markdown(
            f"Total: **{format_size(ram_total)}** - "
            f"Used: **{format_size(ram_used)}** = "
            f"Available: **{format_size(ram_avail)}**"
        )

    st.markdown("---")

    # =========================================================================
    # Row 2: Model Selection (left) | Model Info (right)
    # =========================================================================
    col_model_select, col_model_info = st.columns(2)

    with col_model_select:
        st.subheader("Model Selection")

        # Directory selector using shared component
        cols, has_browse = path_input_columns()
        
        with cols[0]:
            model_dir = st.text_input(
                "GGUF Directory",
                value=config.get("vram_calc_model_dir", config.get("output_dir", "")),
                placeholder=get_platform_path("C:\\Models\\output", "/home/user/models"),
                help="Directory containing GGUF model files"
            )

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(
                    "Select Folder",
                    key="browse_vram_model_dir_btn",
                    use_container_width=True,
                    help="Browse for directory"
                ):
                    dir_clean = strip_quotes(model_dir)
                    initial_dir = dir_clean if dir_clean and Path(dir_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["vram_calc_model_dir"] = selected_folder
                        save_config(config)
                        st.rerun()

        with cols[-1]:
            st.markdown("<br>", unsafe_allow_html=True)
            dir_clean = strip_quotes(model_dir)
            dir_exists = bool(dir_clean and Path(dir_clean).exists())
            if st.button(
                "Open Folder",
                key="open_vram_model_dir_btn",
                use_container_width=True,
                disabled=not dir_exists,
                help="Open folder in file explorer" if dir_exists else "Path doesn't exist"
            ):
                if dir_exists:
                    try:
                        open_folder(dir_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

        # Detect GGUF files
        model_files_options = []
        model_files_map = {}
        
        dir_clean = strip_quotes(model_dir)
        if dir_clean and Path(dir_clean).exists() and Path(dir_clean).is_dir():
            # Use shared detection logic
            detected = detect_all_model_files(Path(dir_clean))
            
            # Filter for GGUF files only
            for key in sorted(detected.keys()):
                info = detected[key]
                if info['extension'] == 'gguf':
                    display_name = info['display_name']
                    model_files_options.append(display_name)
                    model_files_map[display_name] = info

        if not model_files_options:
            if not dir_clean or not Path(dir_clean).exists():
                model_files_options = ["(Invalid directory)"]
            else:
                model_files_options = ["(no .gguf files found)"]
            dropdown_disabled = True
            no_files = True
        else:
            dropdown_disabled = False
            no_files = False

        # File selector with counts in label
        cols, has_browse = path_input_columns()
        
        # Create dynamic label
        if dropdown_disabled:
            model_files_label = "Model files"
        else:
            file_count = len(model_files_options)
            model_files_label = f"Model files: {file_count} detected"

        with cols[0]:
            selected_display_name = st.selectbox(
                model_files_label,
                options=model_files_options,
                disabled=dropdown_disabled,
                help="Select a GGUF model file (or split set) to analyze"
            )

        # Dynamic Model Info Update
        # Update session state immediately when selection changes
        if selected_display_name and selected_display_name in model_files_map:
            selected_info = model_files_map[selected_display_name]
            target_file = selected_info['primary_file']
            
            # Get current info from session state
            current_info = st.session_state.get("vram_calc_model_info")
            
            # If changed or not set, update it
            if not current_info or current_info.file_path != target_file:
                try:
                    # Update with partial info immediately (avoid reading file header for speed)
                    # Full info will be populated when user clicks Calculate
                    partial_info = GGUFModelInfo(
                        file_path=target_file,
                        file_size_mb=selected_info['total_size_gb'] * 1024,
                        num_layers=None,
                        architecture=None,
                        context_length=None,
                        embedding_length=None,
                        head_count=None,
                        head_count_kv=None,
                        vocab_size=None,
                        quantization_version=None,
                        file_type=None
                    )
                    st.session_state.vram_calc_model_info = cast(GGUFModelInfo, partial_info)
                    
                    # Clear previous results as they don't match the new model
                    st.session_state.vram_calc_result = None
                    st.session_state.vram_calc_error = None
                    
                    # Force rerun to update the UI immediately
                    st.rerun()
                except Exception:
                    # If anything fails, clear info
                    st.session_state.vram_calc_model_info = None
        else:
            # No valid selection, clear info if it exists
            if st.session_state.get("vram_calc_model_info"):
                st.session_state.vram_calc_model_info = None
                st.session_state.vram_calc_result = None
                st.rerun()

        # Refresh button in middle column (if tkinter available) or last column
        button_col = cols[1] if has_browse else cols[-1]
        with button_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "Refresh",
                key="refresh_vram_files_btn",
                use_container_width=True,
                help="Rescan directory for GGUF files"
            ):
                st.toast("Refreshed file list")
                st.rerun()

    with col_model_info:
        if "vram_calc_model_info" in st.session_state and st.session_state.vram_calc_model_info:
            model_info = st.session_state.vram_calc_model_info
            st.subheader(f"Model Info: {model_info.file_path.name}")

            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown(f"**Size:** {format_size(model_info.file_size_mb)}")
                if model_info.architecture:
                    st.markdown(f"**Arch:** {model_info.architecture}")
                else:
                    st.markdown("**Arch:** (calc needed)")
                if model_info.file_type:
                    st.markdown(f"**Quant:** {model_info.file_type}")
                else:
                    st.markdown("**Quant:** (calc needed)")
            with col_info2:
                if model_info.num_layers:
                    st.markdown(f"**Layers:** {model_info.num_layers}")
                else:
                    st.markdown("**Layers:** -")
                if model_info.context_length:
                    st.markdown(f"**Native Context:** {model_info.context_length:,}")
                else:
                    st.markdown("**Native Context:** -")
                if model_info.vocab_size:
                    st.markdown(f"**Vocab Size:** {model_info.vocab_size:,}")
                else:
                    st.markdown("**Vocab Size:** -")
            with col_info3:
                if model_info.embedding_length:
                    st.markdown(f"**Embedding:** {model_info.embedding_length:,}")
                else:
                    st.markdown("**Embedding:** -")
                if model_info.head_count and model_info.head_count_kv:
                    if model_info.head_count == model_info.head_count_kv:
                        st.markdown(f"**Attention:** MHA ({model_info.head_count})")
                    else:
                        st.markdown(f"**Attention:** GQA ({model_info.head_count}/{model_info.head_count_kv})")
                else:
                    st.markdown("**Attention:** -")
        else:
            st.subheader("Model Info")
            st.info("Calculate VRAM to see model details")

    st.markdown("---")

    # =========================================================================
    # Row 3: Calculate Button & Results
    # =========================================================================
    
    # Calculate button (Full width)
    if st.button(
        "Calculate VRAM",
        use_container_width=True,
        disabled=no_files,
        type="primary"
    ):
        if no_files or selected_display_name not in model_files_map:
            st.error("Please select a valid GGUF file")
        else:
            try:
                selected_info = model_files_map[selected_display_name]
                # Use primary file (works for both single and split)
                target_file = selected_info['primary_file']

                # Check if we already have full model info for this file
                existing_info = st.session_state.get("vram_calc_model_info")
                has_full_info = (
                    existing_info is not None
                    and existing_info.file_path == target_file
                    and existing_info.num_layers is not None
                )

                if has_full_info and existing_info is not None:
                    # Reuse existing model info (fast path)
                    model_info = cast(GGUFModelInfo, existing_info)
                else:
                    # Extract model info (slow path - first run or model changed)
                    with st.spinner("Analyzing model..."):
                        model_info = get_gguf_model_info(target_file)

                # Convert display name to internal format
                kv_quant_map = {"F16 (Default)": "f16", "Q8_0": "q8_0", "Q4_0": "q4_0"}
                kv_quant_value = kv_quant_map.get(kv_quant_selected, "f16")
                result = calculate_vram(
                    model_info,
                    available_vram_mb=available_vram,
                    headroom_mb=headroom_mb,
                    context_size=context_size,
                    kv_cache_quant=kv_quant_value
                )

                st.session_state.vram_calc_result = result
                st.session_state.vram_calc_model_info = cast(GGUFModelInfo, model_info)
                st.session_state.vram_calc_context_size = context_size
                st.session_state.vram_calc_kv_quant = kv_quant_selected
                st.session_state.vram_calc_error = None

                # Save settings
                config["vram_calc_model_dir"] = dir_clean
                config["vram_calc_headroom_mb"] = headroom_mb
                config["vram_calc_context_size"] = context_size
                config["vram_calc_kv_quant"] = kv_quant_selected
                if use_manual_vram:
                    config["vram_calc_manual_vram"] = available_vram
                save_config(config)
                
                # Rerun to update the Model Info column immediately
                st.rerun()

            except Exception as e:
                st.session_state.vram_calc_result = None
                st.session_state.vram_calc_model_info = None
                st.session_state.vram_calc_error = str(e)

    # Results Section
    if ("vram_calc_error" in st.session_state and st.session_state.vram_calc_error) or \
       ("vram_calc_result" in st.session_state and st.session_state.vram_calc_result):
        st.markdown("---")

        if "vram_calc_error" in st.session_state and st.session_state.vram_calc_error:
            st.error(f"Error: {st.session_state.vram_calc_error}")

        elif "vram_calc_result" in st.session_state and st.session_state.vram_calc_result:
            result = st.session_state.vram_calc_result
            context_size = st.session_state.get("vram_calc_context_size", 4096)

            # Calculate RAM offload info for partial loading
            layers_on_cpu = result.total_layers - result.recommended_layers
            ram_offload_mb = layers_on_cpu * result.mb_per_layer
            ram_total, ram_used, ram_avail = get_system_ram_mb()
            ram_offload_pct = (ram_offload_mb / ram_avail * 100) if ram_avail > 0 else 0

            col_res_left, col_res_right = st.columns(2)

            with col_res_left:
                st.subheader("Results")

                # Main recommendation
                if result.error:
                    st.error(result.error)
                else:
                    # Calculate VRAM usage percentage
                    vram_usage_pct = (result.estimated_usage_mb / result.available_vram_mb * 100) if result.available_vram_mb > 0 else 0

                    if result.fits_entirely:
                        # Calculate max context estimate based on remaining VRAM
                        remaining_vram = result.available_vram_mb - result.headroom_mb - result.estimated_usage_mb
                        if result.context_overhead_mb > 0 and context_size > 0:
                            mb_per_ctx_token = result.context_overhead_mb / context_size
                            extra_context = int(remaining_vram / mb_per_ctx_token) if mb_per_ctx_token > 0 else 0
                            max_context = context_size + extra_context
                            # Cap at model's native context if available
                            model_info = st.session_state.vram_calc_model_info
                            native_ctx = model_info.context_length if model_info and model_info.context_length else None
                            max_context_str = f"  \nMax context estimate: **~{max_context:,}**"
                            if native_ctx and max_context > native_ctx:
                                max_context_str += f" (exceeds native {native_ctx:,})"
                        else:
                            max_context_str = ""

                        st.success(
                            f"Full offload: **-ngl {result.recommended_layers}** "
                            f"({vram_usage_pct:.0f}% of available VRAM)"
                            f"{max_context_str}"
                        )
                    else:
                        # Calculate what context would allow full offload
                        usable_vram = result.available_vram_mb - result.headroom_mb
                        full_model_vram = result.total_layers * result.mb_per_layer
                        remaining_for_context = usable_vram - full_model_vram

                        context_hint = ""
                        if remaining_for_context > 0 and result.context_overhead_mb > 0 and context_size > 0:
                            mb_per_ctx_token = result.context_overhead_mb / context_size
                            if mb_per_ctx_token > 0:
                                max_ctx_for_full = int(remaining_for_context / mb_per_ctx_token)
                                if max_ctx_for_full > 0:
                                    context_hint = f"  \nReduce context to **~{max_ctx_for_full:,}** for full GPU offload"

                        # Check if RAM can handle the offload
                        if ram_offload_mb > ram_avail:
                            st.error(
                                f"Partial: **-ngl {result.recommended_layers}** "
                                f"({vram_usage_pct:.0f}% of available VRAM)  \n"
                                f"RAM offload: **{format_size(ram_offload_mb)}** needed but only "
                                f"**{format_size(ram_avail)}** available - model may not fit!"
                                f"{context_hint}"
                            )
                        else:
                            st.warning(
                                f"Partial: **-ngl {result.recommended_layers}** "
                                f"({vram_usage_pct:.0f}% of available VRAM)  \n"
                                f"RAM offload: **{format_size(ram_offload_mb)}** "
                                f"({ram_offload_pct:.0f}% of available RAM)"
                                f"{context_hint}"
                            )

                    # Detailed metrics
                    col_m1, col_m2, col_m3 = st.columns(3)

                    with col_m1:
                        st.metric("Layers on GPU", result.recommended_layers)

                    with col_m2:
                        st.metric("VRAM per Layer", format_size(result.mb_per_layer))

                    with col_m3:
                        st.metric("Total Est. VRAM", format_size(result.estimated_usage_mb))

                    # RAM offload metrics (only show if partial offload)
                    if not result.fits_entirely:
                        col_r1, col_r2, col_r3 = st.columns(3)

                        with col_r1:
                            st.metric("Layers on CPU", layers_on_cpu)

                        with col_r2:
                            st.metric("Est. RAM Usage", format_size(ram_offload_mb))

                        with col_r3:
                            st.metric("% of Available RAM", f"{ram_offload_pct:.0f}%")

                    # Context row
                    model_info = st.session_state.vram_calc_model_info
                    native_ctx = model_info.context_length if model_info and model_info.context_length else None

                    col_c1, col_c2, col_c3 = st.columns(3)

                    with col_c1:
                        if native_ctx:
                            st.metric("Context (Selected/Native)", f"{context_size:,} / {native_ctx:,}")
                        else:
                            st.metric("Context (Selected)", f"{context_size:,}")

                    with col_c2:
                        if native_ctx:
                            ctx_pct = (context_size / native_ctx) * 100
                            st.metric("% of Native Context", f"{ctx_pct:.0f}%")
                        else:
                            st.metric("% of Native Context", "N/A")

                    with col_c3:
                        st.metric("KV Cache VRAM (Est.)", format_size(result.context_overhead_mb))

            with col_res_right:
                st.subheader("Example Commands")
                model_info = st.session_state.vram_calc_model_info
                model_path = str(model_info.file_path) if model_info else "model.gguf"

                # Build KV cache flags if quantization is used
                kv_quant_used = st.session_state.get("vram_calc_kv_quant", "F16 (Default)")
                kv_flags = ""
                if kv_quant_used == "Q8_0":
                    kv_flags = " -ctk q8_0 -ctv q8_0"
                elif kv_quant_used == "Q4_0":
                    kv_flags = " -ctk q4_0 -ctv q4_0"

                st.code(
                    f'llama-cli -m "{model_path}" -ngl {result.recommended_layers} -c {context_size}{kv_flags}',
                    language="bash"
                )
                st.code(
                    f'llama-server -m "{model_path}" -ngl {result.recommended_layers} -c {context_size}{kv_flags}',
                    language="bash"
                )

