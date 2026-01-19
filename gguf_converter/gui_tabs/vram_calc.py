"""
VRAM Calculator tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, TKINTER_AVAILABLE, get_platform_path
)
from ..vram_calc import (
    get_all_gpus, get_gguf_model_info, calculate_vram, format_size
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

        if gpus:
            # Build GPU options with manual option at the end
            gpu_options = []
            for gpu in gpus:
                vendor_label = gpu.vendor.upper()
                gpu_options.append(
                    f"[{vendor_label}] {gpu.name} - {format_size(gpu.total_mb)} "
                    f"({format_size(gpu.free_mb)} free)"
                )
            gpu_options.append("Manually specify VRAM")

            col_gpu_select, col_gpu_refresh = st.columns([5, 1])
            with col_gpu_select:
                selected_gpu_idx = st.selectbox(
                    "Select GPU",
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
                default_vram = config.get("vram_calc_manual_vram", 8192)
            else:
                selected_gpu = gpus[selected_gpu_idx]
                default_vram = selected_gpu.free_mb

            vram_label = "Manually Enter Available VRAM (MB)" if use_manual_vram else "GPU's Available VRAM (MB)"
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

            # Parameters
            col_headroom, col_context = st.columns(2)
            with col_headroom:
                headroom_mb = st.number_input(
                    "VRAM Headroom (MB)",
                    min_value=0,
                    max_value=16384,
                    value=int(config.get("vram_calc_headroom_mb", 2048)),
                    step=256,
                    help="VRAM to reserve for other applications"
                )
            with col_context:
                context_size = st.number_input(
                    "Context Size",
                    min_value=512,
                    max_value=131072,
                    value=int(config.get("vram_calc_context_size", 4096)),
                    step=512,
                    help="Context window size affects KV cache VRAM"
                )

        else:
            col_no_gpu, col_no_gpu_refresh = st.columns([5, 1])
            with col_no_gpu:
                st.warning(
                    "No GPUs detected. Make sure nvidia-smi (NVIDIA) or rocm-smi (AMD) "
                    "is available, or enter VRAM manually below."
                )
            with col_no_gpu_refresh:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(
                    "Refresh",
                    key="refresh_gpus_btn",
                    use_container_width=True,
                    help="Re-detect available GPUs"
                ):
                    st.session_state.detected_gpus = get_all_gpus()
                    st.rerun()

            manual_vram = st.number_input(
                "Manually Enter Available VRAM (MB)",
                min_value=512,
                max_value=524288,
                value=config.get("vram_calc_manual_vram", 8192),
                step=512,
                help="Manually specify available VRAM in MB"
            )
            available_vram = manual_vram
            use_manual_vram = True
            selected_gpu = None

            # Parameters
            col_headroom, col_context = st.columns(2)
            with col_headroom:
                headroom_mb = st.number_input(
                    "VRAM Headroom (MB)",
                    min_value=0,
                    max_value=16384,
                    value=int(config.get("vram_calc_headroom_mb", 2048)),
                    step=256,
                    help="VRAM to reserve for other applications"
                )
            with col_context:
                context_size = st.number_input(
                    "Context Size",
                    min_value=512,
                    max_value=131072,
                    value=int(config.get("vram_calc_context_size", 4096)),
                    step=512,
                    help="Context window size affects KV cache VRAM"
                )

    with col_gpu_stats:
        if selected_gpu:
            st.subheader(f"{selected_gpu.name} Stats")
            col_total, col_used, col_free = st.columns(3)
            with col_total:
                st.metric("Total VRAM", format_size(selected_gpu.total_mb))
            with col_used:
                st.metric("Currently Used", format_size(selected_gpu.used_mb))
            with col_free:
                st.metric("Currently Free", format_size(selected_gpu.free_mb))
        else:
            st.subheader("Manually Entered VRAM Stats")
            st.metric("Available VRAM", format_size(available_vram))

        # VRAM Breakdown (updates dynamically)
        st.markdown("**VRAM Breakdown:**")
        usable_after_headroom = available_vram - headroom_mb
        st.markdown(f"- Headroom Reserved: -{format_size(headroom_mb)}")
        st.markdown(f"- **Usable for Model: {format_size(max(0, usable_after_headroom))}**")

    st.markdown("---")

    # =========================================================================
    # Row 2: Model Selection (left) | Results (right)
    # =========================================================================
    col_model, col_results = st.columns(2)

    with col_model:
        st.subheader("Model Selection")

        # Directory selector
        if TKINTER_AVAILABLE:
            col_dir, col_dir_browse, col_dir_open = st.columns([4, 1, 1])
        else:
            col_dir, col_dir_open = st.columns([5, 1])
            col_dir_browse = None

        with col_dir:
            model_dir = st.text_input(
                "GGUF Directory",
                value=config.get("vram_calc_model_dir", config.get("output_dir", "")),
                placeholder=get_platform_path("C:\\Models\\output", "/home/user/models"),
                help="Directory containing GGUF model files"
            )

        if TKINTER_AVAILABLE and col_dir_browse is not None:
            with col_dir_browse:
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

        with col_dir_open:
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

        # Scan for GGUF files
        dir_clean = strip_quotes(model_dir)
        gguf_files = []
        if dir_clean and Path(dir_clean).exists() and Path(dir_clean).is_dir():
            gguf_files = sorted([str(f) for f in Path(dir_clean).glob("*.gguf")])

        if not gguf_files:
            gguf_files = ["(no .gguf files found)"]
            no_files = True
        else:
            no_files = False

        # File selector
        col_file, col_refresh_files = st.columns([5, 1])
        with col_file:
            selected_file = st.selectbox(
                "Select GGUF Model",
                options=gguf_files,
                help="Select a GGUF model file to analyze"
            )
        with col_refresh_files:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "Refresh",
                key="refresh_vram_files_btn",
                use_container_width=True,
                help="Rescan directory for GGUF files"
            ):
                st.toast("Refreshed file list")
                st.rerun()

        # Calculate button
        if st.button(
            "Calculate VRAM",
            use_container_width=True,
            disabled=no_files,
            type="primary"
        ):
            if no_files or selected_file == "(no .gguf files found)":
                st.error("Please select a valid GGUF file")
            else:
                try:
                    with st.spinner("Analyzing model..."):
                        model_info = get_gguf_model_info(Path(selected_file))
                        result = calculate_vram(
                            model_info,
                            available_vram_mb=available_vram,
                            headroom_mb=headroom_mb,
                            context_size=context_size
                        )

                    st.session_state.vram_calc_result = result
                    st.session_state.vram_calc_model_info = model_info
                    st.session_state.vram_calc_context_size = context_size
                    st.session_state.vram_calc_error = None

                    # Save settings
                    config["vram_calc_model_dir"] = dir_clean
                    config["vram_calc_headroom_mb"] = headroom_mb
                    config["vram_calc_context_size"] = context_size
                    if use_manual_vram:
                        config["vram_calc_manual_vram"] = available_vram
                    save_config(config)

                except Exception as e:
                    st.session_state.vram_calc_result = None
                    st.session_state.vram_calc_model_info = None
                    st.session_state.vram_calc_error = str(e)

    with col_results:
        st.subheader("Results")

        if "vram_calc_error" in st.session_state and st.session_state.vram_calc_error:
            st.error(f"Error: {st.session_state.vram_calc_error}")

        elif "vram_calc_result" in st.session_state and st.session_state.vram_calc_result:
            result = st.session_state.vram_calc_result
            model_info = st.session_state.vram_calc_model_info
            context_size = st.session_state.get("vram_calc_context_size", 4096)

            # Model info
            st.markdown(f"**Model:** {model_info.file_path.name}")

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"**Size:** {format_size(model_info.file_size_mb)}")
                st.markdown(f"**Layers:** {model_info.num_layers}")
            with col_info2:
                if model_info.architecture:
                    st.markdown(f"**Arch:** {model_info.architecture}")
                if model_info.file_type:
                    st.markdown(f"**Quant:** {model_info.file_type}")

            st.markdown("---")

            # Main recommendation
            if result.error:
                st.error(result.error)
            else:
                if result.fits_entirely:
                    st.success(
                        f"Full offload: **-ngl {result.recommended_layers}**"
                    )
                else:
                    st.warning(
                        f"Partial: **-ngl {result.recommended_layers}** "
                        f"({result.offload_percentage:.0f}% of {result.total_layers})"
                    )

                # Detailed metrics
                col_m1, col_m2 = st.columns(2)

                with col_m1:
                    st.metric("Recommended -ngl", result.recommended_layers)
                    st.metric("Per Layer", format_size(result.mb_per_layer))

                with col_m2:
                    st.metric("Est. Usage", format_size(result.estimated_usage_mb))
                    st.metric("KV Cache", format_size(result.context_overhead_mb))

                # Command line example
                st.markdown("---")
                st.code(
                    f"llama-cli -m model.gguf -ngl {result.recommended_layers} -c {context_size}",
                    language="bash"
                )

        else:
            st.info("Select a model and click 'Calculate VRAM' to see results")
