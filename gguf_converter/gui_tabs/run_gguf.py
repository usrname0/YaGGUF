"""
Run GGUF tab — launch llama-server, llama-cli, or llama-bench against a local GGUF file.
"""

import platform
import streamlit as st
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

from ..gui_utils import (
    strip_quotes, browse_folder, open_folder, save_config, path_input_columns,
    get_platform_path, detect_all_model_files, launch_in_terminal,
    load_presets, save_preset, delete_preset, get_default_config, make_config_saver,
    TKINTER_AVAILABLE,
)

if TYPE_CHECKING:
    from ..converter import GGUFConverter


# ---------------------------------------------------------------------------
# Pure command builders
# ---------------------------------------------------------------------------

def _build_base_cmd(binary_path: str, model_file: str, params: Dict[str, Any]) -> List[str]:
    cmd = [binary_path, "-m", model_file]
    if params.get("auto_fit", True):
        if params.get("fit_target", 0) > 0:
            cmd += ["--fit-target", str(params["fit_target"])]
        if params.get("fit_ctx", 0) > 0:
            cmd += ["--fit-ctx", str(params["fit_ctx"])]
    else:
        cmd += ["-ngl", str(params["ngl"])]
        cmd += ["-c", str(params["ctx"])]
        if params.get("flash_attn"):
            cmd += ["-fa", "on"]
    if params.get("cache_type_k", "f16") != "f16":
        cmd += ["-ctk", params["cache_type_k"]]
    if params.get("cache_type_v", "f16") != "f16":
        cmd += ["-ctv", params["cache_type_v"]]
    return cmd


def _append_sampling_args(cmd: List[str], params: Dict[str, Any]) -> None:
    cmd += ["--temp", str(params["temp"])]
    cmd += ["--top-k", str(params["top_k"])]
    cmd += ["--top-p", str(params["top_p"])]
    cmd += ["--min-p", str(params["min_p"])]
    cmd += ["--presence-penalty", str(params["presence_penalty"])]
    if params.get("jinja"):
        cmd += ["--jinja"]
        if params.get("reasoning") and params["reasoning"] != "auto":
            cmd += ["--reasoning", params["reasoning"]]


def build_server_cmd(binary_path: str, model_file: str, params: Dict[str, Any]) -> List[str]:
    cmd = _build_base_cmd(binary_path, model_file, params)
    if params.get("mmproj_enabled") and params.get("mmproj_path"):
        cmd += ["--mmproj", params["mmproj_path"]]
    cmd += ["--port", str(params["port"])]
    cmd += ["--host", params["host"]]
    cmd += ["-np", str(params["parallel"])]
    _append_sampling_args(cmd, params)
    return cmd


def build_cli_cmd(binary_path: str, model_file: str, params: Dict[str, Any]) -> List[str]:
    cmd = _build_base_cmd(binary_path, model_file, params)
    if params.get("mmproj_enabled") and params.get("mmproj_path"):
        cmd += ["--mmproj", params["mmproj_path"]]
    _append_sampling_args(cmd, params)
    cmd += ["-cnv"]
    return cmd


def build_bench_cmd(binary_path: str, model_file: str, params: Dict[str, Any]) -> List[str]:
    cmd = [binary_path, "-m", model_file]
    if not params.get("auto_fit", True):
        cmd += ["-ngl", str(params["ngl"])]
    cmd += ["-p", str(params["bench_n_prompt"])]
    cmd += ["-n", str(params["bench_n_gen"])]
    cmd += ["-b", str(params["bench_batch_size"])]
    cmd += ["-ub", str(params["bench_ubatch_size"])]
    cmd += ["-r", str(params["bench_repetitions"])]
    if params.get("bench_threads", 0) > 0:
        cmd += ["-t", str(params["bench_threads"])]
    cmd += ["-ctk", params["bench_cache_type_k"]]
    cmd += ["-ctv", params["bench_cache_type_v"]]
    cmd += ["-fa", "1" if params.get("flash_attn") else "0"]
    cmd += ["-o", params["bench_output"]]
    if params.get("bench_no_warmup"):
        cmd += ["--no-warmup"]
    if params.get("bench_progress"):
        cmd += ["--progress"]
    return cmd


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

@st.dialog("Save preset")
def _save_preset_dialog(config: Dict[str, Any]) -> None:
    name = st.text_input("Preset name", placeholder="My preset", key="run_preset_name_dialog")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save", use_container_width=True, type="primary"):
            stripped = name.strip()
            if stripped:
                params = _collect_params(config)
                save_preset(stripped, params)
                st.toast(f"Preset '{stripped}' saved.")
                st.rerun()
            else:
                st.warning("Enter a preset name.")
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ---------------------------------------------------------------------------
# Tab render function
# ---------------------------------------------------------------------------

def render_run_gguf_tab(converter: "GGUFConverter", config: Dict[str, Any]) -> None:
    """Render the Run GGUF tab."""
    st.header("Run GGUF")

    # Apply pending preset BEFORE any widgets render so session state keys can be set safely.
    if "pending_preset_load" in st.session_state:
        _apply_pending_preset(st.session_state.pop("pending_preset_load"), config)

    if "pending_fit_apply" in st.session_state:
        ngl_apply, ctx_apply = st.session_state.pop("pending_fit_apply")
        st.session_state["run_ngl_input"] = ngl_apply
        st.session_state["run_ctx_input"] = ctx_apply
        config["run_ngl"] = ngl_apply
        config["run_ctx"] = ctx_apply
        save_config(config)

    col1, col2 = st.columns(2)

    # ------------------------------------------------------------------
    # Left column: model selection, mode radio, launch
    # ------------------------------------------------------------------
    with col1:
        st.subheader("Model")

        cols, has_browse = path_input_columns()

        with cols[0]:
            run_model_path_str = st.text_input(
                "Model folder",
                value=config.get("run_model_path", ""),
                placeholder=get_platform_path(
                    "C:\\Models\\my-model", "/home/user/Models/my-model"
                ),
                help="Directory containing GGUF file(s).",
            )

        model_path_clean = strip_quotes(run_model_path_str)
        model_path_exists = bool(model_path_clean and Path(model_path_clean).exists())

        if model_path_clean != config.get("run_model_path", ""):
            config["run_model_path"] = model_path_clean
            config["run_model_file"] = ""
            save_config(config)

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(
                    "Select Folder",
                    key="run_browse_folder_btn",
                    use_container_width=True,
                ):
                    selected = browse_folder(model_path_clean if model_path_exists else None)
                    if selected:
                        config["run_model_path"] = selected
                        config["run_model_file"] = ""
                        save_config(config)
                        st.rerun()

        with cols[-1]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "Open Folder",
                key="run_open_folder_btn",
                use_container_width=True,
                disabled=not model_path_exists,
                help="Open folder in file explorer" if model_path_exists else "Path doesn't exist yet",
            ):
                try:
                    open_folder(model_path_clean)
                    st.toast("Opened folder")
                except Exception as e:
                    st.error(str(e))

        model_path = Path(model_path_clean) if model_path_clean else None

        gguf_files: Dict[str, Any] = {}
        mmproj_path: Optional[str] = None
        if model_path and model_path.exists():
            all_files = detect_all_model_files(model_path)
            gguf_files = {
                k: v for k, v in all_files.items()
                if v["extension"] == "gguf"
                and "mmproj" not in v["display_name"].lower()
            }
            mmproj_hits = [
                v["primary_file"] for v in all_files.values()
                if v["extension"] == "gguf" and "mmproj" in v["display_name"].lower()
            ]
            if mmproj_hits:
                mmproj_path = str(mmproj_hits[0])

        no_files = not gguf_files
        if no_files:
            file_options = ["No files found"]
            file_keys: List[Optional[str]] = [None]
        else:
            file_options = [v["display_name"] for v in gguf_files.values()]
            file_keys = list(gguf_files.keys())

        saved_file = config.get("run_model_file", "")
        default_idx = 0
        if saved_file and not no_files:
            for i, fkey in enumerate(file_keys):
                if fkey and gguf_files.get(fkey, {}).get("primary_file") and \
                        str(gguf_files[fkey]["primary_file"].name) == saved_file:
                    default_idx = i
                    break

        gguf_label = "GGUF file" if no_files else f"GGUF file: {len(gguf_files)} detected"
        gguf_cols = st.columns([4, 2] if TKINTER_AVAILABLE else [5, 1])

        with gguf_cols[0]:
            selected_display = st.selectbox(
                gguf_label,
                options=file_options,
                index=default_idx,
                help="Select the GGUF file to run.",
                disabled=no_files,
            )

        selected_file_path: Optional[str] = None
        if not no_files and selected_display and selected_display in file_options:
            idx = file_options.index(selected_display)
            fkey = file_keys[idx]
            if fkey and fkey in gguf_files:
                selected_file_path = str(gguf_files[fkey]["primary_file"])
                new_file = str(gguf_files[fkey]["primary_file"].name)
                if new_file != config.get("run_model_file", ""):
                    config["run_model_file"] = new_file
                    save_config(config)

        with gguf_cols[1]:
            if mmproj_path is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                mmproj_kwargs: Dict[str, Any] = {
                    "label": "Enable multimodal (--mmproj)",
                    "help": f"Loads the multimodal projector ({Path(mmproj_path).name}), enabling image, video, or audio input depending on the model. Passes --mmproj to the binary.",
                    "key": "run_mmproj_checkbox",
                    "on_change": make_config_saver(config, "run_mmproj_enabled", "run_mmproj_checkbox"),
                }
                if "run_mmproj_checkbox" not in st.session_state:
                    mmproj_kwargs["value"] = bool(config.get("run_mmproj_enabled", True))
                st.checkbox(**mmproj_kwargs)  # type: ignore[arg-type]

        binary_dir = _get_binary_dir(converter)

        # ------------------------------------------------------------------
        # Fit estimator
        # ------------------------------------------------------------------
        with st.expander("Hardware fit estimate", expanded=False):
            if "run_fit_calc_result" in st.session_state:
                if st.session_state["run_fit_calc_result"].get("model") != selected_file_path:
                    del st.session_state["run_fit_calc_result"]

            if st.button(
                "Estimate fit for selected model",
                key="run_fit_calc_btn",
                use_container_width=True,
                disabled=not selected_file_path,
                help="Run llama-fit-params to calculate optimal -ngl and -c values for your hardware.",
            ):
                cur_fit_target = int(st.session_state.get("run_fit_target_input", config.get("run_fit_target", 1024)))
                cur_fit_ctx = int(st.session_state.get("run_fit_ctx_input", config.get("run_fit_ctx", 0)))
                with st.spinner("Calculating fit…"):
                    stdout, stderr, rc = _run_fit_params(
                        binary_dir, selected_file_path, cur_fit_target, cur_fit_ctx  # type: ignore[arg-type]
                    )
                st.session_state["run_fit_calc_result"] = {
                    "stdout": stdout, "stderr": stderr, "rc": rc, "model": selected_file_path,
                }

            if "run_fit_calc_result" in st.session_state:
                result = st.session_state["run_fit_calc_result"]
                if result["rc"] == 0 and result["stdout"]:
                    fit_ngl, fit_ctx_val = _parse_fit_output(result["stdout"])
                    cur_auto_fit = config.get("run_auto_fit", True)
                    radio_val = st.session_state.get("run_auto_fit_radio")
                    if radio_val is not None:
                        cur_auto_fit = radio_val == "Auto"
                    if fit_ngl is not None and fit_ctx_val is not None:
                        res_col, btn_col = st.columns([3, 1])
                        with res_col:
                            st.success(f"Fitted params: `{result['stdout']}`")
                        with btn_col:
                            apply_disabled = cur_auto_fit
                            apply_label = "(Auto)" if cur_auto_fit else "Apply"
                            apply_help = "Switch to Manual mode to apply these values manually." if cur_auto_fit else None
                            if st.button(apply_label, key="run_fit_apply_btn", use_container_width=True, disabled=apply_disabled, help=apply_help):
                                st.session_state["pending_fit_apply"] = (fit_ngl, fit_ctx_val)
                                st.rerun()
                    else:
                        st.success(f"Fitted params: `{result['stdout']}`")
                    with st.expander("Memory breakdown", expanded=False):
                        st.code(result["stderr"], language=None)
                else:
                    st.error("llama-fit-params failed.")
                    if result["stderr"]:
                        st.code(result["stderr"], language=None)

            if not selected_file_path:
                st.caption("Select a GGUF file above to enable.")

        # ------------------------------------------------------------------
        # Mode radio
        # ------------------------------------------------------------------
        st.markdown("---")
        mode_options = ["llama-server", "llama-cli", "llama-bench"]
        mode_kwargs: Dict[str, Any] = {
            "label": "Mode",
            "options": mode_options,
            "horizontal": True,
            "key": "run_mode_radio",
            "on_change": make_config_saver(config, "run_mode", "run_mode_radio"),
        }
        if "run_mode_radio" not in st.session_state:
            saved_mode = config.get("run_mode", "llama-server")
            mode_kwargs["index"] = mode_options.index(saved_mode) if saved_mode in mode_options else 0
        st.radio(**mode_kwargs)  # type: ignore[arg-type]
        current_mode = str(st.session_state.get("run_mode_radio", config.get("run_mode", "llama-server")))

        # ------------------------------------------------------------------
        # Launch
        # ------------------------------------------------------------------
        params = _collect_params(config, mmproj_path)

        if current_mode in ("llama-server", "llama-cli") and not params.get("auto_fit", True):
            if params.get("cache_type_v", "f16") != "f16" and not params.get("flash_attn"):
                st.warning(
                    f"-ctv {params['cache_type_v']} requires Flash Attention (-fa). "
                    "Enable it in Hardware."
                )

        st.subheader("Launch")
        cmd: List[str] = []

        if current_mode == "llama-server":
            binary = _find_binary(binary_dir, "llama-server")
            if selected_file_path:
                cmd = build_server_cmd(binary, selected_file_path, params)
            if st.button(
                "Launch llama-server",
                key="run_launch_server_btn",
                use_container_width=True,
                disabled=not selected_file_path,
                help="Start llama-server in a new terminal window",
            ):
                ok = launch_in_terminal(
                    cmd, window_title="llama-server",
                    announce_title="llama-server",
                    announce_subtitle=Path(selected_file_path or "").name,
                )
                if ok:
                    st.toast("llama-server launched.")
                    if params.get("open_browser", True):
                        webbrowser.open(f"http://{params['host']}:{params['port']}")
                else:
                    st.error("Could not find a terminal emulator to launch llama-server.")

        elif current_mode == "llama-cli":
            binary = _find_binary(binary_dir, "llama-cli")
            if selected_file_path:
                cmd = build_cli_cmd(binary, selected_file_path, params)
            if st.button(
                "Launch llama-cli",
                key="run_launch_cli_btn",
                use_container_width=True,
                disabled=not selected_file_path,
                help="Start an interactive chat session in a new terminal window",
            ):
                ok = launch_in_terminal(
                    cmd, window_title="llama-cli",
                    announce_title="llama-cli",
                    announce_subtitle=Path(selected_file_path or "").name,
                )
                if ok:
                    st.toast("llama-cli launched.")
                else:
                    st.error("Could not find a terminal emulator to launch llama-cli.")

        elif current_mode == "llama-bench":
            binary = _find_binary(binary_dir, "llama-bench")
            if selected_file_path:
                cmd = build_bench_cmd(binary, selected_file_path, params)
            if st.button(
                "Launch llama-bench",
                key="run_launch_bench_btn",
                use_container_width=True,
                disabled=not selected_file_path,
                help="Run benchmarks in a new terminal window",
            ):
                ok = launch_in_terminal(
                    cmd, window_title="llama-bench",
                    announce_title="llama-bench",
                    announce_subtitle=Path(selected_file_path or "").name,
                )
                if ok:
                    st.toast("llama-bench launched.")
                else:
                    st.error("Could not find a terminal emulator to launch llama-bench.")

        if cmd:
            st.code(" ".join(cmd), language="bash")
        else:
            st.caption("Select a GGUF file to see the command.")

    # ------------------------------------------------------------------
    # Right column: presets, hardware, mode-specific settings
    # ------------------------------------------------------------------
    with col2:

        # ------------------------------------------------------------------
        # Presets
        # ------------------------------------------------------------------
        with st.expander("Presets", expanded=False):
            presets = load_presets()
            preset_names = list(presets.keys())

            if st.button("Save as...", key="run_preset_save_btn", use_container_width=True):
                _save_preset_dialog(config)

            st.markdown("---")

            dc1, dc2, dc3 = st.columns([4, 1, 1])
            with dc1:
                st.markdown("*(defaults)*")
            with dc2:
                if st.button("Load", key="run_preset_load_defaults", use_container_width=True):
                    st.session_state.pending_preset_load = "__defaults__"
                    st.rerun()
            with dc3:
                st.button("Delete", key="run_preset_delete_defaults", disabled=True, use_container_width=True)

            for i, name in enumerate(preset_names):
                pc1, pc2, pc3 = st.columns([4, 1, 1])
                with pc1:
                    st.markdown(name)
                with pc2:
                    if st.button("Load", key=f"run_preset_load_{i}", use_container_width=True):
                        st.session_state.pending_preset_load = presets[name]
                        st.rerun()
                with pc3:
                    if st.button("Delete", key=f"run_preset_delete_{i}", use_container_width=True):
                        delete_preset(name)
                        st.toast(f"Preset '{name}' deleted.")
                        st.rerun()

            if not preset_names:
                st.caption("No saved presets yet.")

        # ------------------------------------------------------------------
        # Hardware (all modes)
        # ------------------------------------------------------------------
        with st.expander("Hardware", expanded=True):
            auto_fit_options = ["Auto", "Manual"]

            def _save_auto_fit():
                config["run_auto_fit"] = st.session_state.get("run_auto_fit_radio") == "Auto"
                save_config(config)

            auto_fit_kwargs: Dict[str, Any] = {
                "label": "GPU configuration",
                "options": auto_fit_options,
                "horizontal": True,
                "key": "run_auto_fit_radio",
                "on_change": _save_auto_fit,
                "help": (
                    "**Auto**: llama.cpp auto-fits GPU layers, context, and flash attention "
                    "to your hardware. Use --fit-target to control VRAM headroom.\n\n"
                    "**Manual**: Explicitly set GPU layers, context size, and flash attention."
                ),
            }
            if "run_auto_fit_radio" not in st.session_state:
                saved_auto_fit = config.get("run_auto_fit", True)
                auto_fit_kwargs["index"] = 0 if saved_auto_fit else 1
            st.radio(**auto_fit_kwargs)  # type: ignore[arg-type]
            current_auto_fit = st.session_state.get("run_auto_fit_radio", auto_fit_options[0]) == "Auto"

            st.markdown("---")

            if current_auto_fit:
                hw_c1, hw_c2, _ = st.columns(3)

                with hw_c1:
                    fit_target_kwargs: Dict[str, Any] = {
                        "label": "VRAM headroom (--fit-target, MiB)",
                        "min_value": 0,
                        "max_value": 4095,
                        "step": 256,
                        "help": (
                            "Free VRAM to leave on each GPU (MiB) when auto-fitting. "
                            "llama.cpp default: 1024 MiB. "
                            "Lower values pack more onto GPU; raise if hitting OOM. "
                            "0 = omit flag (use llama.cpp default). "
                            "Capped at 4095 MiB due to a Windows overflow bug in older builds."
                        ),
                        "key": "run_fit_target_input",
                        "on_change": make_config_saver(config, "run_fit_target", "run_fit_target_input"),
                    }
                    if "run_fit_target_input" not in st.session_state:
                        fit_target_kwargs["value"] = int(config.get("run_fit_target", 1024))
                    st.number_input(**fit_target_kwargs)  # type: ignore[arg-type]

                with hw_c2:
                    fit_ctx_kwargs: Dict[str, Any] = {
                        "label": "Min context (--fit-ctx, tokens)",
                        "min_value": 0,
                        "max_value": 131072,
                        "step": 512,
                        "help": (
                            "Minimum context size that --fit is allowed to set. "
                            "llama.cpp default: 4096. "
                            "0 = omit flag (use llama.cpp default). "
                            "Raise this if you want auto-fit to guarantee a larger context."
                        ),
                        "key": "run_fit_ctx_input",
                        "on_change": make_config_saver(config, "run_fit_ctx", "run_fit_ctx_input"),
                    }
                    if "run_fit_ctx_input" not in st.session_state:
                        fit_ctx_kwargs["value"] = int(config.get("run_fit_ctx", 0))
                    st.number_input(**fit_ctx_kwargs)  # type: ignore[arg-type]

            else:
                hw_c1, hw_c2, _ = st.columns(3)

                with hw_c1:
                    ngl_kwargs: Dict[str, Any] = {
                        "label": "GPU Layers (-ngl)",
                        "min_value": -1,
                        "max_value": 999,
                        "step": 1,
                        "help": "Number of layers to offload to GPU. -1 = all layers (OOM if model doesn't fit). 0 = CPU only.",
                        "key": "run_ngl_input",
                        "on_change": make_config_saver(config, "run_ngl", "run_ngl_input"),
                    }
                    if "run_ngl_input" not in st.session_state:
                        ngl_kwargs["value"] = int(config.get("run_ngl", 99))
                    st.number_input(**ngl_kwargs)  # type: ignore[arg-type]

                with hw_c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    fa_kwargs: Dict[str, Any] = {
                        "label": "Flash Attention (-fa on)",
                        "help": "Enable flash attention for faster inference (requires compatible GPU).",
                        "key": "run_flash_attn_checkbox",
                        "on_change": make_config_saver(config, "run_flash_attn", "run_flash_attn_checkbox"),
                    }
                    if "run_flash_attn_checkbox" not in st.session_state:
                        fa_kwargs["value"] = bool(config.get("run_flash_attn", False))
                    st.checkbox(**fa_kwargs)  # type: ignore[arg-type]

        # ------------------------------------------------------------------
        # Mode-specific settings
        # ------------------------------------------------------------------
        cache_type_options = ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "bf16"]

        if current_mode == "llama-server":
            with st.expander("Server Settings", expanded=True):
                _render_ctx_section(config, current_auto_fit)
                st.markdown("---")
                _render_kv_cache_widgets(config, cache_type_options)
                st.markdown("---")

                r3c1, r3c2, r3c3 = st.columns(3)

                with r3c1:
                    port_kwargs: Dict[str, Any] = {
                        "label": "Port (--port)",
                        "min_value": 1,
                        "max_value": 65535,
                        "step": 1,
                        "help": "Port for llama-server to listen on.",
                        "key": "run_port_input",
                        "on_change": make_config_saver(config, "run_port", "run_port_input"),
                    }
                    if "run_port_input" not in st.session_state:
                        port_kwargs["value"] = int(config.get("run_port", 8080))
                    st.number_input(**port_kwargs)  # type: ignore[arg-type]

                with r3c2:
                    host_kwargs: Dict[str, Any] = {
                        "label": "Host (--host)",
                        "help": "Host address for llama-server. Use 0.0.0.0 to expose on LAN.",
                        "key": "run_host_input",
                        "on_change": make_config_saver(config, "run_host", "run_host_input"),
                    }
                    if "run_host_input" not in st.session_state:
                        host_kwargs["value"] = config.get("run_host", "127.0.0.1")
                    st.text_input(**host_kwargs)  # type: ignore[arg-type]

                with r3c3:
                    parallel_kwargs: Dict[str, Any] = {
                        "label": "Parallel slots (-np)",
                        "min_value": 1,
                        "max_value": 64,
                        "step": 1,
                        "help": (
                            "Number of parallel inference slots for llama-server. "
                            "KV cache is shared across all slots, so effective per-slot context = "
                            "context size ÷ slots. Use 1 for lowest latency (single user). "
                            "Increase only if serving multiple concurrent users."
                        ),
                        "key": "run_parallel_input",
                        "on_change": make_config_saver(config, "run_parallel", "run_parallel_input"),
                    }
                    if "run_parallel_input" not in st.session_state:
                        parallel_kwargs["value"] = int(config.get("run_parallel", 1))
                    st.number_input(**parallel_kwargs)  # type: ignore[arg-type]

                browser_kwargs: Dict[str, Any] = {
                    "label": "Open browser on launch",
                    "help": "Automatically open the browser when launching llama-server.",
                    "key": "run_open_browser_checkbox",
                    "on_change": make_config_saver(config, "run_open_browser", "run_open_browser_checkbox"),
                }
                if "run_open_browser_checkbox" not in st.session_state:
                    browser_kwargs["value"] = bool(config.get("run_open_browser", True))
                st.checkbox(**browser_kwargs)  # type: ignore[arg-type]

                st.markdown("---")
                _render_sampling_widgets(config)
                st.markdown("---")
                _render_jinja_reasoning_widgets(config)

        elif current_mode == "llama-cli":
            with st.expander("CLI Settings", expanded=True):
                _render_ctx_section(config, current_auto_fit)
                st.markdown("---")
                _render_kv_cache_widgets(config, cache_type_options)
                st.markdown("---")
                _render_sampling_widgets(config)
                st.markdown("---")
                _render_jinja_reasoning_widgets(config)

        elif current_mode == "llama-bench":
            with st.expander("Bench Settings", expanded=True):
                b1c1, b1c2, b1c3 = st.columns(3)

                with b1c1:
                    bp_kwargs: Dict[str, Any] = {
                        "label": "Prompt tokens (-p)",
                        "min_value": 0,
                        "max_value": 65536,
                        "step": 64,
                        "help": "Number of prompt tokens to process.",
                        "key": "run_bench_n_prompt_input",
                        "on_change": make_config_saver(config, "run_bench_n_prompt", "run_bench_n_prompt_input"),
                    }
                    if "run_bench_n_prompt_input" not in st.session_state:
                        bp_kwargs["value"] = int(config.get("run_bench_n_prompt", 512))
                    st.number_input(**bp_kwargs)  # type: ignore[arg-type]

                with b1c2:
                    bg_kwargs: Dict[str, Any] = {
                        "label": "Gen tokens (-n)",
                        "min_value": 0,
                        "max_value": 65536,
                        "step": 16,
                        "help": "Number of tokens to generate.",
                        "key": "run_bench_n_gen_input",
                        "on_change": make_config_saver(config, "run_bench_n_gen", "run_bench_n_gen_input"),
                    }
                    if "run_bench_n_gen_input" not in st.session_state:
                        bg_kwargs["value"] = int(config.get("run_bench_n_gen", 128))
                    st.number_input(**bg_kwargs)  # type: ignore[arg-type]

                with b1c3:
                    br_kwargs: Dict[str, Any] = {
                        "label": "Repetitions (-r)",
                        "min_value": 1,
                        "max_value": 50,
                        "step": 1,
                        "help": "Number of times to repeat each test.",
                        "key": "run_bench_repetitions_input",
                        "on_change": make_config_saver(config, "run_bench_repetitions", "run_bench_repetitions_input"),
                    }
                    if "run_bench_repetitions_input" not in st.session_state:
                        br_kwargs["value"] = int(config.get("run_bench_repetitions", 5))
                    st.number_input(**br_kwargs)  # type: ignore[arg-type]

                b2c1, b2c2, b2c3 = st.columns(3)

                with b2c1:
                    bb_kwargs: Dict[str, Any] = {
                        "label": "Batch size (-b)",
                        "min_value": 1,
                        "max_value": 65536,
                        "step": 64,
                        "help": "Batch size for prompt processing.",
                        "key": "run_bench_batch_size_input",
                        "on_change": make_config_saver(config, "run_bench_batch_size", "run_bench_batch_size_input"),
                    }
                    if "run_bench_batch_size_input" not in st.session_state:
                        bb_kwargs["value"] = int(config.get("run_bench_batch_size", 2048))
                    st.number_input(**bb_kwargs)  # type: ignore[arg-type]

                with b2c2:
                    bub_kwargs: Dict[str, Any] = {
                        "label": "Ubatch size (-ub)",
                        "min_value": 1,
                        "max_value": 65536,
                        "step": 64,
                        "help": "Micro-batch size for prompt processing.",
                        "key": "run_bench_ubatch_size_input",
                        "on_change": make_config_saver(config, "run_bench_ubatch_size", "run_bench_ubatch_size_input"),
                    }
                    if "run_bench_ubatch_size_input" not in st.session_state:
                        bub_kwargs["value"] = int(config.get("run_bench_ubatch_size", 512))
                    st.number_input(**bub_kwargs)  # type: ignore[arg-type]

                with b2c3:
                    bt_kwargs: Dict[str, Any] = {
                        "label": "Threads (-t)",
                        "min_value": 0,
                        "max_value": 256,
                        "step": 1,
                        "help": "CPU threads. 0 = use llama-bench default.",
                        "key": "run_bench_threads_input",
                        "on_change": make_config_saver(config, "run_bench_threads", "run_bench_threads_input"),
                    }
                    if "run_bench_threads_input" not in st.session_state:
                        bt_kwargs["value"] = int(config.get("run_bench_threads", 0))
                    st.number_input(**bt_kwargs)  # type: ignore[arg-type]

                b3c1, b3c2, b3c3 = st.columns(3)

                with b3c1:
                    bctk_kwargs: Dict[str, Any] = {
                        "label": "Cache type K (-ctk)",
                        "options": cache_type_options,
                        "help": "KV cache type for keys. Useful for benchmarking KV quantization.",
                        "key": "run_bench_cache_type_k_select",
                        "on_change": make_config_saver(config, "run_bench_cache_type_k", "run_bench_cache_type_k_select"),
                    }
                    saved_bctk = config.get("run_bench_cache_type_k", "f16")
                    if "run_bench_cache_type_k_select" not in st.session_state:
                        bctk_kwargs["index"] = cache_type_options.index(saved_bctk) \
                            if saved_bctk in cache_type_options else 0
                    st.selectbox(**bctk_kwargs)  # type: ignore[arg-type]

                with b3c2:
                    bctv_kwargs: Dict[str, Any] = {
                        "label": "Cache type V (-ctv)",
                        "options": cache_type_options,
                        "help": "KV cache type for values.",
                        "key": "run_bench_cache_type_v_select",
                        "on_change": make_config_saver(config, "run_bench_cache_type_v", "run_bench_cache_type_v_select"),
                    }
                    saved_bctv = config.get("run_bench_cache_type_v", "f16")
                    if "run_bench_cache_type_v_select" not in st.session_state:
                        bctv_kwargs["index"] = cache_type_options.index(saved_bctv) \
                            if saved_bctv in cache_type_options else 0
                    st.selectbox(**bctv_kwargs)  # type: ignore[arg-type]

                with b3c3:
                    output_options = ["md", "csv", "json", "jsonl", "sql"]
                    bo_kwargs: Dict[str, Any] = {
                        "label": "Output format (-o)",
                        "options": output_options,
                        "help": "Output format for benchmark results.",
                        "key": "run_bench_output_select",
                        "on_change": make_config_saver(config, "run_bench_output", "run_bench_output_select"),
                    }
                    saved_bo = config.get("run_bench_output", "md")
                    if "run_bench_output_select" not in st.session_state:
                        bo_kwargs["index"] = output_options.index(saved_bo) \
                            if saved_bo in output_options else 0
                    st.selectbox(**bo_kwargs)  # type: ignore[arg-type]

                b4c1, b4c2, _ = st.columns(3)

                with b4c1:
                    nw_kwargs: Dict[str, Any] = {
                        "label": "No warmup (--no-warmup)",
                        "help": "Skip warmup runs before benchmarking.",
                        "key": "run_bench_no_warmup_checkbox",
                        "on_change": make_config_saver(config, "run_bench_no_warmup", "run_bench_no_warmup_checkbox"),
                    }
                    if "run_bench_no_warmup_checkbox" not in st.session_state:
                        nw_kwargs["value"] = bool(config.get("run_bench_no_warmup", False))
                    st.checkbox(**nw_kwargs)  # type: ignore[arg-type]

                with b4c2:
                    prog_kwargs: Dict[str, Any] = {
                        "label": "Progress (--progress)",
                        "help": "Print test progress indicators.",
                        "key": "run_bench_progress_checkbox",
                        "on_change": make_config_saver(config, "run_bench_progress", "run_bench_progress_checkbox"),
                    }
                    if "run_bench_progress_checkbox" not in st.session_state:
                        prog_kwargs["value"] = bool(config.get("run_bench_progress", False))
                    st.checkbox(**prog_kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shared widget helpers (server / CLI settings)
# ---------------------------------------------------------------------------

def _render_ctx_section(config: Dict[str, Any], current_auto_fit: bool) -> None:
    if current_auto_fit:
        fit_ctx_val = int(st.session_state.get("run_fit_ctx_input", config.get("run_fit_ctx", 0)))
        min_ctx_str = f"{fit_ctx_val:,}" if fit_ctx_val > 0 else "4,096 (llama.cpp default)"
        st.caption(f"Context size: auto-configured by --fit (minimum: {min_ctx_str} tokens). Set in Hardware.")
    else:
        ctx_kwargs: Dict[str, Any] = {
            "label": "Context Size (-c)",
            "min_value": 0,
            "max_value": 131072,
            "step": 512,
            "help": "Context window size in tokens. 0 = use the model's native context length from its metadata.",
            "key": "run_ctx_input",
            "on_change": make_config_saver(config, "run_ctx", "run_ctx_input"),
        }
        if "run_ctx_input" not in st.session_state:
            ctx_kwargs["value"] = int(config.get("run_ctx", 4096))
        st.number_input(**ctx_kwargs)  # type: ignore[arg-type]


def _render_kv_cache_widgets(config: Dict[str, Any], cache_type_options: List[str]) -> None:
    c1, c2, _ = st.columns(3)
    with c1:
        ctk_kwargs: Dict[str, Any] = {
            "label": "KV cache K type (-ctk)",
            "options": cache_type_options,
            "help": "Quantize the KV cache keys. q8_0 halves memory vs f16 with minimal quality loss.",
            "key": "run_cache_type_k_select",
            "on_change": make_config_saver(config, "run_cache_type_k", "run_cache_type_k_select"),
        }
        saved_ctk = config.get("run_cache_type_k", "f16")
        if "run_cache_type_k_select" not in st.session_state:
            ctk_kwargs["index"] = cache_type_options.index(saved_ctk) if saved_ctk in cache_type_options else 0
        st.selectbox(**ctk_kwargs)  # type: ignore[arg-type]
    with c2:
        ctv_kwargs: Dict[str, Any] = {
            "label": "KV cache V type (-ctv)",
            "options": cache_type_options,
            "help": (
                "Quantize the KV cache values. q8_0 halves memory vs f16 with minimal quality loss. "
                "Quantized V types require Flash Attention (-fa)."
            ),
            "key": "run_cache_type_v_select",
            "on_change": make_config_saver(config, "run_cache_type_v", "run_cache_type_v_select"),
        }
        saved_ctv = config.get("run_cache_type_v", "f16")
        if "run_cache_type_v_select" not in st.session_state:
            ctv_kwargs["index"] = cache_type_options.index(saved_ctv) if saved_ctv in cache_type_options else 0
        st.selectbox(**ctv_kwargs)  # type: ignore[arg-type]


def _render_sampling_widgets(config: Dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        temp_kwargs: Dict[str, Any] = {
            "label": "Temp (--temp)",
            "min_value": 0.0,
            "max_value": 5.0,
            "step": 0.05,
            "format": "%.2f",
            "help": "Sampling temperature.",
            "key": "run_temp_input",
            "on_change": make_config_saver(config, "run_temp", "run_temp_input"),
        }
        if "run_temp_input" not in st.session_state:
            temp_kwargs["value"] = float(config.get("run_temp", 0.8))
        st.number_input(**temp_kwargs)  # type: ignore[arg-type]
    with c2:
        topk_kwargs: Dict[str, Any] = {
            "label": "Top-K (--top-k)",
            "min_value": 0,
            "max_value": 1000,
            "step": 1,
            "help": "Top-K sampling. 0 = disabled.",
            "key": "run_top_k_input",
            "on_change": make_config_saver(config, "run_top_k", "run_top_k_input"),
        }
        if "run_top_k_input" not in st.session_state:
            topk_kwargs["value"] = int(config.get("run_top_k", 40))
        st.number_input(**topk_kwargs)  # type: ignore[arg-type]
    with c3:
        topp_kwargs: Dict[str, Any] = {
            "label": "Top-P (--top-p)",
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.01,
            "format": "%.2f",
            "help": "Top-P (nucleus) sampling.",
            "key": "run_top_p_input",
            "on_change": make_config_saver(config, "run_top_p", "run_top_p_input"),
        }
        if "run_top_p_input" not in st.session_state:
            topp_kwargs["value"] = float(config.get("run_top_p", 0.95))
        st.number_input(**topp_kwargs)  # type: ignore[arg-type]

    c1, c2, _ = st.columns(3)
    with c1:
        minp_kwargs: Dict[str, Any] = {
            "label": "Min-P (--min-p)",
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.01,
            "format": "%.2f",
            "help": "Min-P sampling threshold.",
            "key": "run_min_p_input",
            "on_change": make_config_saver(config, "run_min_p", "run_min_p_input"),
        }
        if "run_min_p_input" not in st.session_state:
            minp_kwargs["value"] = float(config.get("run_min_p", 0.05))
        st.number_input(**minp_kwargs)  # type: ignore[arg-type]
    with c2:
        pp_kwargs: Dict[str, Any] = {
            "label": "Presence (--presence-penalty)",
            "min_value": -2.0,
            "max_value": 2.0,
            "step": 0.05,
            "format": "%.2f",
            "help": "Presence penalty for repetition reduction.",
            "key": "run_presence_penalty_input",
            "on_change": make_config_saver(config, "run_presence_penalty", "run_presence_penalty_input"),
        }
        if "run_presence_penalty_input" not in st.session_state:
            pp_kwargs["value"] = float(config.get("run_presence_penalty", 0.0))
        st.number_input(**pp_kwargs)  # type: ignore[arg-type]


def _render_jinja_reasoning_widgets(config: Dict[str, Any]) -> None:
    c1, c2, _ = st.columns(3)
    with c1:
        st.markdown("<br>", unsafe_allow_html=True)
        jinja_kwargs: Dict[str, Any] = {
            "label": "Jinja templates (--jinja)",
            "help": "Enable Jinja2 chat template rendering. Required for --reasoning.",
            "key": "run_jinja_checkbox",
            "on_change": make_config_saver(config, "run_jinja", "run_jinja_checkbox"),
        }
        if "run_jinja_checkbox" not in st.session_state:
            jinja_kwargs["value"] = bool(config.get("run_jinja", True))
        jinja_enabled = st.checkbox(**jinja_kwargs)  # type: ignore[arg-type]
    with c2:
        reasoning_options = ["off", "on", "auto"]
        reasoning_kwargs: Dict[str, Any] = {
            "label": "Reasoning (--reasoning)",
            "options": reasoning_options,
            "help": "Reasoning mode. Only emitted when Jinja is enabled. 'auto' is llama.cpp's default (omitted from command).",
            "key": "run_reasoning_select",
            "on_change": make_config_saver(config, "run_reasoning", "run_reasoning_select"),
            "disabled": not jinja_enabled,
        }
        if "run_reasoning_select" not in st.session_state:
            saved_reasoning = config.get("run_reasoning", "off")
            reasoning_kwargs["index"] = reasoning_options.index(saved_reasoning) \
                if saved_reasoning in reasoning_options else 0
        st.selectbox(**reasoning_kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_params(config: Dict[str, Any], mmproj_path: Optional[str] = None) -> Dict[str, Any]:
    """Collect current inference parameters from session state / config."""
    saved_auto_fit = config.get("run_auto_fit", True)
    radio_val = st.session_state.get("run_auto_fit_radio", "Auto" if saved_auto_fit else "Manual")
    return {
        "mode": str(st.session_state.get("run_mode_radio", config.get("run_mode", "llama-server"))),
        "auto_fit": radio_val == "Auto",
        "ngl": int(st.session_state.get("run_ngl_input", config.get("run_ngl", 99))),
        "ctx": int(st.session_state.get("run_ctx_input", config.get("run_ctx", 4096))),
        "flash_attn": bool(st.session_state.get("run_flash_attn_checkbox", config.get("run_flash_attn", False))),
        "fit_target": int(st.session_state.get("run_fit_target_input", config.get("run_fit_target", 1024))),
        "fit_ctx": int(st.session_state.get("run_fit_ctx_input", config.get("run_fit_ctx", 0))),
        "cache_type_k": str(st.session_state.get("run_cache_type_k_select", config.get("run_cache_type_k", "f16"))),
        "cache_type_v": str(st.session_state.get("run_cache_type_v_select", config.get("run_cache_type_v", "f16"))),
        "port": int(st.session_state.get("run_port_input", config.get("run_port", 8080))),
        "host": str(st.session_state.get("run_host_input", config.get("run_host", "127.0.0.1"))),
        "parallel": int(st.session_state.get("run_parallel_input", config.get("run_parallel", 1))),
        "open_browser": bool(st.session_state.get("run_open_browser_checkbox", config.get("run_open_browser", True))),
        "temp": float(st.session_state.get("run_temp_input", config.get("run_temp", 0.8))),
        "top_k": int(st.session_state.get("run_top_k_input", config.get("run_top_k", 40))),
        "top_p": float(st.session_state.get("run_top_p_input", config.get("run_top_p", 0.95))),
        "min_p": float(st.session_state.get("run_min_p_input", config.get("run_min_p", 0.05))),
        "presence_penalty": float(st.session_state.get("run_presence_penalty_input", config.get("run_presence_penalty", 0.0))),
        "jinja": bool(st.session_state.get("run_jinja_checkbox", config.get("run_jinja", True))),
        "reasoning": str(st.session_state.get("run_reasoning_select", config.get("run_reasoning", "off"))),
        "bench_n_prompt": int(st.session_state.get("run_bench_n_prompt_input", config.get("run_bench_n_prompt", 512))),
        "bench_n_gen": int(st.session_state.get("run_bench_n_gen_input", config.get("run_bench_n_gen", 128))),
        "bench_batch_size": int(st.session_state.get("run_bench_batch_size_input", config.get("run_bench_batch_size", 2048))),
        "bench_ubatch_size": int(st.session_state.get("run_bench_ubatch_size_input", config.get("run_bench_ubatch_size", 512))),
        "bench_repetitions": int(st.session_state.get("run_bench_repetitions_input", config.get("run_bench_repetitions", 5))),
        "bench_threads": int(st.session_state.get("run_bench_threads_input", config.get("run_bench_threads", 0))),
        "bench_cache_type_k": str(st.session_state.get("run_bench_cache_type_k_select", config.get("run_bench_cache_type_k", "f16"))),
        "bench_cache_type_v": str(st.session_state.get("run_bench_cache_type_v_select", config.get("run_bench_cache_type_v", "f16"))),
        "bench_output": str(st.session_state.get("run_bench_output_select", config.get("run_bench_output", "md"))),
        "bench_no_warmup": bool(st.session_state.get("run_bench_no_warmup_checkbox", config.get("run_bench_no_warmup", False))),
        "bench_progress": bool(st.session_state.get("run_bench_progress_checkbox", config.get("run_bench_progress", False))),
        "mmproj_enabled": bool(st.session_state.get("run_mmproj_checkbox", config.get("run_mmproj_enabled", True))),
        "mmproj_path": mmproj_path,
    }


RUN_GGUF_KEY_MAP: List[tuple] = [
    ("mode",             "run_mode",             "run_mode_radio"),
    ("auto_fit",         "run_auto_fit",         "run_auto_fit_radio"),
    ("ngl",              "run_ngl",              "run_ngl_input"),
    ("ctx",              "run_ctx",              "run_ctx_input"),
    ("flash_attn",       "run_flash_attn",       "run_flash_attn_checkbox"),
    ("fit_target",       "run_fit_target",       "run_fit_target_input"),
    ("fit_ctx",          "run_fit_ctx",          "run_fit_ctx_input"),
    ("cache_type_k",     "run_cache_type_k",     "run_cache_type_k_select"),
    ("cache_type_v",     "run_cache_type_v",     "run_cache_type_v_select"),
    ("port",             "run_port",             "run_port_input"),
    ("host",             "run_host",             "run_host_input"),
    ("parallel",         "run_parallel",         "run_parallel_input"),
    ("open_browser",     "run_open_browser",     "run_open_browser_checkbox"),
    ("temp",             "run_temp",             "run_temp_input"),
    ("top_k",            "run_top_k",            "run_top_k_input"),
    ("top_p",            "run_top_p",            "run_top_p_input"),
    ("min_p",            "run_min_p",            "run_min_p_input"),
    ("presence_penalty", "run_presence_penalty", "run_presence_penalty_input"),
    ("jinja",            "run_jinja",            "run_jinja_checkbox"),
    ("reasoning",        "run_reasoning",        "run_reasoning_select"),
    ("bench_n_prompt",   "run_bench_n_prompt",   "run_bench_n_prompt_input"),
    ("bench_n_gen",      "run_bench_n_gen",      "run_bench_n_gen_input"),
    ("bench_batch_size", "run_bench_batch_size", "run_bench_batch_size_input"),
    ("bench_ubatch_size","run_bench_ubatch_size","run_bench_ubatch_size_input"),
    ("bench_repetitions","run_bench_repetitions","run_bench_repetitions_input"),
    ("bench_threads",    "run_bench_threads",    "run_bench_threads_input"),
    ("bench_cache_type_k","run_bench_cache_type_k","run_bench_cache_type_k_select"),
    ("bench_cache_type_v","run_bench_cache_type_v","run_bench_cache_type_v_select"),
    ("bench_output",     "run_bench_output",     "run_bench_output_select"),
    ("bench_no_warmup",  "run_bench_no_warmup",  "run_bench_no_warmup_checkbox"),
    ("bench_progress",   "run_bench_progress",   "run_bench_progress_checkbox"),
    ("mmproj_enabled",   "run_mmproj_enabled",   "run_mmproj_checkbox"),
]


def _apply_pending_preset(pending: Any, config: Dict[str, Any]) -> None:
    """Apply a preset (or defaults) to config and session state.

    Must be called BEFORE any widgets render so session state keys can be set
    directly (setting a key after its widget renders raises StreamlitAPIException).
    """
    if pending == "__defaults__":
        defaults = get_default_config()
        preset = {param_key: defaults[cfg_key]
                  for param_key, cfg_key, _ in RUN_GGUF_KEY_MAP
                  if cfg_key in defaults}
    else:
        preset = pending

    for param_key, cfg_key, session_key in RUN_GGUF_KEY_MAP:
        if param_key in preset:
            val = preset[param_key]
            config[cfg_key] = val
            # Radio widgets need their string option value, not the stored bool
            if param_key == "auto_fit":
                st.session_state[session_key] = "Auto" if val else "Manual"
            else:
                st.session_state[session_key] = val

    save_config(config)


def _get_binary_dir(converter: Any) -> Optional[Path]:
    """Return the directory containing llama.cpp binaries."""
    try:
        return converter.llama_cpp_manager.bin_dir
    except AttributeError:
        return None


def _find_binary(binary_dir: Optional[Path], name: str) -> str:
    """Find a binary by name, returning its full path or just the name as fallback."""
    if binary_dir is None:
        return name
    suffix = ".exe" if platform.system() == "Windows" else ""
    candidate = binary_dir / f"{name}{suffix}"
    if candidate.exists():
        return str(candidate)
    return name


def _run_fit_params(
    binary_dir: Optional[Path],
    model_path: str,
    fit_target: int,
    fit_ctx: int,
) -> Tuple[str, str, int]:
    """Run llama-fit-params and return (stdout, stderr, returncode)."""
    binary = _find_binary(binary_dir, "llama-fit-params")
    cmd = [binary, "-m", model_path]
    if fit_target > 0:
        cmd += ["--fit-target", str(fit_target)]
    if fit_ctx > 0:
        cmd += ["--fit-ctx", str(fit_ctx)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except FileNotFoundError:
        return "", f"Binary not found: {binary}", 1


def _parse_fit_output(stdout: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse '-c 40192 -ngl -1' style fit output into (ngl, ctx)."""
    ngl: Optional[int] = None
    ctx: Optional[int] = None
    parts = stdout.split()
    for i, part in enumerate(parts):
        if part == "-ngl" and i + 1 < len(parts):
            try:
                ngl = int(parts[i + 1])
            except ValueError:
                pass
        elif part == "-c" and i + 1 < len(parts):
            try:
                ctx = int(parts[i + 1])
            except ValueError:
                pass
    return ngl, ctx
