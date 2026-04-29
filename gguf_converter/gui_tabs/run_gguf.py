"""
Run GGUF tab — launch llama-server, llama-cli, or llama-bench against a local GGUF file.
"""

import streamlit as st
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ..gui_utils import (
    strip_quotes, browse_folder, open_folder, save_config, path_input_columns,
    get_platform_path, detect_all_model_files, launch_in_terminal,
    load_presets, save_preset, delete_preset, get_default_config,
)

if TYPE_CHECKING:
    from ..converter import GGUFConverter


# ---------------------------------------------------------------------------
# Pure command builders
# ---------------------------------------------------------------------------

def build_server_cmd(binary_path: str, model_file: str, params: Dict[str, Any]) -> List[str]:
    cmd = [binary_path, "-m", model_file]
    cmd += ["-ngl", str(params["ngl"])]
    cmd += ["-c", str(params["ctx"])]
    if params.get("flash_attn"):
        cmd += ["-fa"]
    if params.get("fit_target", 0) > 0:
        cmd += ["--fit-target", str(params["fit_target"])]
    if params.get("cache_type_k", "f16") != "f16":
        cmd += ["-ctk", params["cache_type_k"]]
    if params.get("cache_type_v", "f16") != "f16":
        cmd += ["-ctv", params["cache_type_v"]]
    cmd += ["--port", str(params["port"])]
    cmd += ["--host", params["host"]]
    cmd += ["-np", str(params["parallel"])]
    cmd += ["--temp", str(params["temp"])]
    cmd += ["--top-k", str(params["top_k"])]
    cmd += ["--top-p", str(params["top_p"])]
    cmd += ["--min-p", str(params["min_p"])]
    cmd += ["--presence-penalty", str(params["presence_penalty"])]
    if params.get("jinja"):
        cmd += ["--jinja"]
        if params.get("reasoning") and params["reasoning"] != "auto":
            cmd += ["--reasoning", params["reasoning"]]
    return cmd


def build_cli_cmd(binary_path: str, model_file: str, params: Dict[str, Any]) -> List[str]:
    cmd = [binary_path, "-m", model_file]
    cmd += ["-ngl", str(params["ngl"])]
    cmd += ["-c", str(params["ctx"])]
    if params.get("flash_attn"):
        cmd += ["-fa"]
    if params.get("fit_target", 0) > 0:
        cmd += ["--fit-target", str(params["fit_target"])]
    if params.get("cache_type_k", "f16") != "f16":
        cmd += ["-ctk", params["cache_type_k"]]
    if params.get("cache_type_v", "f16") != "f16":
        cmd += ["-ctv", params["cache_type_v"]]
    cmd += ["--temp", str(params["temp"])]
    cmd += ["--top-k", str(params["top_k"])]
    cmd += ["--top-p", str(params["top_p"])]
    cmd += ["--min-p", str(params["min_p"])]
    cmd += ["--presence-penalty", str(params["presence_penalty"])]
    if params.get("jinja"):
        cmd += ["--jinja"]
        if params.get("reasoning") and params["reasoning"] != "auto":
            cmd += ["--reasoning", params["reasoning"]]
    cmd += ["-cnv"]
    return cmd


def build_bench_cmd(binary_path: str, model_file: str, params: Dict[str, Any]) -> List[str]:
    cmd = [binary_path, "-m", model_file]
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

    def _make_saver(cfg_key: str, session_key: str):
        def _save():
            val = st.session_state.get(session_key)
            if val is not None:
                config[cfg_key] = val
                save_config(config)
        return _save

    col1, col2 = st.columns(2)

    # ------------------------------------------------------------------
    # Left column: model selection, mode radio, launch
    # ------------------------------------------------------------------
    with col1:
        st.subheader("Model")

        cols, has_browse = path_input_columns()

        with cols[0]:
            path_kwargs: Dict[str, Any] = {
                "label": "Model folder",
                "placeholder": get_platform_path(
                    "C:\\Models\\my-model", "/home/user/Models/my-model"
                ),
                "help": "Directory containing GGUF file(s).",
                "key": "run_model_path_input",
            }
            if "run_model_path_input" not in st.session_state:
                path_kwargs["value"] = config.get("run_model_path", "")
            run_model_path_str = st.text_input(**path_kwargs)  # type: ignore[arg-type]

        cleaned = strip_quotes(run_model_path_str)
        if cleaned != config.get("run_model_path", ""):
            config["run_model_path"] = cleaned
            config["run_model_file"] = ""
            save_config(config)
            st.session_state.run_reset_count += 1

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(
                    "Select Folder",
                    key="run_browse_folder_btn",
                    use_container_width=True,
                ):
                    current = config.get("run_model_path", "")
                    initial = current if current and Path(current).exists() else None
                    selected = browse_folder(initial)
                    if selected:
                        config["run_model_path"] = selected
                        config["run_model_file"] = ""
                        save_config(config)
                        st.session_state.pending_run_model_path = selected
                        st.rerun()

        with cols[-1]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open Folder", key="run_open_folder_btn", use_container_width=True):
                path_clean = strip_quotes(run_model_path_str)
                if path_clean and Path(path_clean).exists():
                    try:
                        open_folder(path_clean)
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("Folder not found.")

        model_path_clean = strip_quotes(run_model_path_str)
        model_path = Path(model_path_clean) if model_path_clean else None

        gguf_files: Dict[str, Any] = {}
        if model_path and model_path.exists():
            all_files = detect_all_model_files(model_path)
            gguf_files = {
                k: v for k, v in all_files.items()
                if v["extension"] == "gguf"
                and "mmproj" not in v["display_name"].lower()
            }

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
            for i, key in enumerate(file_keys):
                if key and gguf_files.get(key, {}).get("primary_file") and \
                        str(gguf_files[key]["primary_file"].name) == saved_file:
                    default_idx = i
                    break

        current_select_key = f"run_model_file_select_{st.session_state.run_reset_count}"

        def save_run_model_file():
            val = st.session_state.get(current_select_key)
            if val not in file_options:
                return
            assert isinstance(val, str)
            idx = file_options.index(val)
            selected_key = file_keys[idx]
            if selected_key and selected_key in gguf_files:
                config["run_model_file"] = str(gguf_files[selected_key]["primary_file"].name)
            else:
                config["run_model_file"] = ""
            save_config(config)

        gguf_label = "GGUF file" if no_files else f"GGUF file: {len(gguf_files)} detected"
        st.selectbox(
            gguf_label,
            options=file_options,
            index=default_idx,
            key=current_select_key,
            on_change=save_run_model_file,
            help="Select the GGUF file to run.",
            disabled=no_files,
        )

        selected_display = st.session_state.get(current_select_key, "")
        selected_file_path: Optional[str] = None
        if not no_files and selected_display and selected_display in file_options:
            idx = file_options.index(selected_display)
            key = file_keys[idx]
            if key and key in gguf_files:
                selected_file_path = str(gguf_files[key]["primary_file"])

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
            "on_change": _make_saver("run_mode", "run_mode_radio"),
        }
        if "run_mode_radio" not in st.session_state:
            saved_mode = config.get("run_mode", "llama-server")
            mode_kwargs["index"] = mode_options.index(saved_mode) if saved_mode in mode_options else 0
        st.radio(**mode_kwargs)  # type: ignore[arg-type]
        current_mode = str(st.session_state.get("run_mode_radio", config.get("run_mode", "llama-server")))

        # ------------------------------------------------------------------
        # Launch
        # ------------------------------------------------------------------
        params = _collect_params(config)
        binary_dir = _get_binary_dir(converter)

        if current_mode in ("llama-server", "llama-cli"):
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
                assert selected_file_path
                ok = launch_in_terminal(
                    build_server_cmd(binary, selected_file_path, params),
                    window_title="llama-server",
                )
                if ok:
                    st.toast("llama-server launched.")
                    if config.get("run_open_browser", True):
                        host = config.get("run_host", "127.0.0.1")
                        port = config.get("run_port", 8080)
                        webbrowser.open(f"http://{host}:{port}")
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
                assert selected_file_path
                ok = launch_in_terminal(
                    build_cli_cmd(binary, selected_file_path, params),
                    window_title="llama-cli",
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
                assert selected_file_path
                ok = launch_in_terminal(
                    build_bench_cmd(binary, selected_file_path, params),
                    window_title="llama-bench",
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

            # Defaults row
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
        # Section: Hardware (all modes)
        # ------------------------------------------------------------------
        with st.expander("Hardware", expanded=True):
            hw_c1, hw_c2, _ = st.columns(3)

            with hw_c1:
                ngl_kwargs: Dict[str, Any] = {
                    "label": "GPU Layers (-ngl)",
                    "min_value": 0,
                    "max_value": 999,
                    "step": 1,
                    "help": "Number of layers to offload to GPU. 99 = all layers.",
                    "key": "run_ngl_input",
                    "on_change": _make_saver("run_ngl", "run_ngl_input"),
                }
                if "run_ngl_input" not in st.session_state:
                    ngl_kwargs["value"] = int(config.get("run_ngl", 99))
                st.number_input(**ngl_kwargs)  # type: ignore[arg-type]

            with hw_c2:
                st.markdown("<br>", unsafe_allow_html=True)
                fa_kwargs: Dict[str, Any] = {
                    "label": "Flash Attention (-fa)",
                    "help": "Enable flash attention for faster inference (requires compatible GPU).",
                    "key": "run_flash_attn_checkbox",
                    "on_change": _make_saver("run_flash_attn", "run_flash_attn_checkbox"),
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
                r1c1, r1c2, _ = st.columns(3)

                with r1c1:
                    ctx_kwargs: Dict[str, Any] = {
                        "label": "Context Size (-c)",
                        "min_value": 512,
                        "max_value": 131072,
                        "step": 512,
                        "help": "Context window size in tokens.",
                        "key": "run_ctx_input",
                        "on_change": _make_saver("run_ctx", "run_ctx_input"),
                    }
                    if "run_ctx_input" not in st.session_state:
                        ctx_kwargs["value"] = int(config.get("run_ctx", 4096))
                    st.number_input(**ctx_kwargs)  # type: ignore[arg-type]

                with r1c2:
                    fit_target_kwargs: Dict[str, Any] = {
                        "label": "VRAM headroom (--fit-target, MiB)",
                        "min_value": 0,
                        "max_value": 4095,
                        "step": 256,
                        "help": (
                            "Free VRAM to leave on each GPU (MiB) when auto-fitting. "
                            "0 = omit flag (llama.cpp default: 1024 MiB). "
                            "Lower values pack more onto GPU; raise if hitting OOM. "
                            "Has no effect when -ngl is set (auto-fit requires omitting -ngl). "
                            "Capped at 4095 MiB due to a Windows overflow bug in older llama.cpp builds."
                        ),
                        "key": "run_fit_target_input",
                        "on_change": _make_saver("run_fit_target", "run_fit_target_input"),
                    }
                    if "run_fit_target_input" not in st.session_state:
                        fit_target_kwargs["value"] = int(config.get("run_fit_target", 0))
                    st.number_input(**fit_target_kwargs)  # type: ignore[arg-type]

                r2c1, r2c2, _ = st.columns(3)

                with r2c1:
                    ctk_kwargs: Dict[str, Any] = {
                        "label": "KV cache K type (-ctk)",
                        "options": cache_type_options,
                        "help": "Quantize the KV cache keys. q8_0 halves memory vs f16 with minimal quality loss.",
                        "key": "run_cache_type_k_select",
                        "on_change": _make_saver("run_cache_type_k", "run_cache_type_k_select"),
                    }
                    saved_ctk = config.get("run_cache_type_k", "f16")
                    if "run_cache_type_k_select" not in st.session_state:
                        ctk_kwargs["index"] = cache_type_options.index(saved_ctk) \
                            if saved_ctk in cache_type_options else 0
                    st.selectbox(**ctk_kwargs)  # type: ignore[arg-type]

                with r2c2:
                    ctv_kwargs: Dict[str, Any] = {
                        "label": "KV cache V type (-ctv)",
                        "options": cache_type_options,
                        "help": (
                            "Quantize the KV cache values. q8_0 halves memory vs f16 with minimal quality loss. "
                            "Quantized V types require Flash Attention (-fa)."
                        ),
                        "key": "run_cache_type_v_select",
                        "on_change": _make_saver("run_cache_type_v", "run_cache_type_v_select"),
                    }
                    saved_ctv = config.get("run_cache_type_v", "f16")
                    if "run_cache_type_v_select" not in st.session_state:
                        ctv_kwargs["index"] = cache_type_options.index(saved_ctv) \
                            if saved_ctv in cache_type_options else 0
                    st.selectbox(**ctv_kwargs)  # type: ignore[arg-type]

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
                        "on_change": _make_saver("run_port", "run_port_input"),
                    }
                    if "run_port_input" not in st.session_state:
                        port_kwargs["value"] = int(config.get("run_port", 8080))
                    st.number_input(**port_kwargs)  # type: ignore[arg-type]

                with r3c2:
                    host_kwargs: Dict[str, Any] = {
                        "label": "Host (--host)",
                        "help": "Host address for llama-server. Use 0.0.0.0 to expose on LAN.",
                        "key": "run_host_input",
                        "on_change": _make_saver("run_host", "run_host_input"),
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
                        "on_change": _make_saver("run_parallel", "run_parallel_input"),
                    }
                    if "run_parallel_input" not in st.session_state:
                        parallel_kwargs["value"] = int(config.get("run_parallel", 1))
                    st.number_input(**parallel_kwargs)  # type: ignore[arg-type]

                browser_kwargs: Dict[str, Any] = {
                    "label": "Open browser on launch",
                    "help": "Automatically open the browser when launching llama-server.",
                    "key": "run_open_browser_checkbox",
                    "on_change": _make_saver("run_open_browser", "run_open_browser_checkbox"),
                }
                if "run_open_browser_checkbox" not in st.session_state:
                    browser_kwargs["value"] = bool(config.get("run_open_browser", True))
                st.checkbox(**browser_kwargs)  # type: ignore[arg-type]

                st.markdown("---")

                r4c1, r4c2, r4c3 = st.columns(3)

                with r4c1:
                    temp_kwargs: Dict[str, Any] = {
                        "label": "Temp (--temp)",
                        "min_value": 0.0,
                        "max_value": 5.0,
                        "step": 0.05,
                        "format": "%.2f",
                        "help": "Sampling temperature.",
                        "key": "run_temp_input",
                        "on_change": _make_saver("run_temp", "run_temp_input"),
                    }
                    if "run_temp_input" not in st.session_state:
                        temp_kwargs["value"] = float(config.get("run_temp", 0.8))
                    st.number_input(**temp_kwargs)  # type: ignore[arg-type]

                with r4c2:
                    topk_kwargs: Dict[str, Any] = {
                        "label": "Top-K (--top-k)",
                        "min_value": 0,
                        "max_value": 1000,
                        "step": 1,
                        "help": "Top-K sampling. 0 = disabled.",
                        "key": "run_top_k_input",
                        "on_change": _make_saver("run_top_k", "run_top_k_input"),
                    }
                    if "run_top_k_input" not in st.session_state:
                        topk_kwargs["value"] = int(config.get("run_top_k", 40))
                    st.number_input(**topk_kwargs)  # type: ignore[arg-type]

                with r4c3:
                    topp_kwargs: Dict[str, Any] = {
                        "label": "Top-P (--top-p)",
                        "min_value": 0.0,
                        "max_value": 1.0,
                        "step": 0.01,
                        "format": "%.2f",
                        "help": "Top-P (nucleus) sampling.",
                        "key": "run_top_p_input",
                        "on_change": _make_saver("run_top_p", "run_top_p_input"),
                    }
                    if "run_top_p_input" not in st.session_state:
                        topp_kwargs["value"] = float(config.get("run_top_p", 0.95))
                    st.number_input(**topp_kwargs)  # type: ignore[arg-type]

                r5c1, r5c2, _ = st.columns(3)

                with r5c1:
                    minp_kwargs: Dict[str, Any] = {
                        "label": "Min-P (--min-p)",
                        "min_value": 0.0,
                        "max_value": 1.0,
                        "step": 0.01,
                        "format": "%.2f",
                        "help": "Min-P sampling threshold.",
                        "key": "run_min_p_input",
                        "on_change": _make_saver("run_min_p", "run_min_p_input"),
                    }
                    if "run_min_p_input" not in st.session_state:
                        minp_kwargs["value"] = float(config.get("run_min_p", 0.05))
                    st.number_input(**minp_kwargs)  # type: ignore[arg-type]

                with r5c2:
                    pp_kwargs: Dict[str, Any] = {
                        "label": "Presence (--presence-penalty)",
                        "min_value": -2.0,
                        "max_value": 2.0,
                        "step": 0.05,
                        "format": "%.2f",
                        "help": "Presence penalty for repetition reduction.",
                        "key": "run_presence_penalty_input",
                        "on_change": _make_saver("run_presence_penalty", "run_presence_penalty_input"),
                    }
                    if "run_presence_penalty_input" not in st.session_state:
                        pp_kwargs["value"] = float(config.get("run_presence_penalty", 0.0))
                    st.number_input(**pp_kwargs)  # type: ignore[arg-type]

                st.markdown("---")

                r6c1, r6c2, _ = st.columns(3)

                with r6c1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    jinja_kwargs: Dict[str, Any] = {
                        "label": "Jinja templates (--jinja)",
                        "help": "Enable Jinja2 chat template rendering. Required for --reasoning.",
                        "key": "run_jinja_checkbox",
                        "on_change": _make_saver("run_jinja", "run_jinja_checkbox"),
                    }
                    if "run_jinja_checkbox" not in st.session_state:
                        jinja_kwargs["value"] = bool(config.get("run_jinja", True))
                    jinja_enabled = st.checkbox(**jinja_kwargs)  # type: ignore[arg-type]
                    if "run_jinja_checkbox" in st.session_state:
                        jinja_enabled = st.session_state["run_jinja_checkbox"]

                with r6c2:
                    reasoning_options = ["off", "on", "auto"]
                    reasoning_kwargs: Dict[str, Any] = {
                        "label": "Reasoning (--reasoning)",
                        "options": reasoning_options,
                        "help": "Reasoning mode. Only emitted when Jinja is enabled. 'auto' is llama.cpp's default (omitted from command).",
                        "key": "run_reasoning_select",
                        "on_change": _make_saver("run_reasoning", "run_reasoning_select"),
                        "disabled": not jinja_enabled,
                    }
                    if "run_reasoning_select" not in st.session_state:
                        saved_reasoning = config.get("run_reasoning", "off")
                        reasoning_kwargs["index"] = reasoning_options.index(saved_reasoning) \
                            if saved_reasoning in reasoning_options else 0
                    st.selectbox(**reasoning_kwargs)  # type: ignore[arg-type]

        elif current_mode == "llama-cli":
            with st.expander("CLI Settings", expanded=True):
                r1c1, r1c2, _ = st.columns(3)

                with r1c1:
                    ctx_kwargs = {
                        "label": "Context Size (-c)",
                        "min_value": 512,
                        "max_value": 131072,
                        "step": 512,
                        "help": "Context window size in tokens.",
                        "key": "run_ctx_input",
                        "on_change": _make_saver("run_ctx", "run_ctx_input"),
                    }
                    if "run_ctx_input" not in st.session_state:
                        ctx_kwargs["value"] = int(config.get("run_ctx", 4096))
                    st.number_input(**ctx_kwargs)  # type: ignore[arg-type]

                with r1c2:
                    fit_target_kwargs = {
                        "label": "VRAM headroom (--fit-target, MiB)",
                        "min_value": 0,
                        "max_value": 4095,
                        "step": 256,
                        "help": (
                            "Free VRAM to leave on each GPU (MiB) when auto-fitting. "
                            "0 = omit flag (llama.cpp default: 1024 MiB). "
                            "Has no effect when -ngl is set. "
                            "Capped at 4095 MiB due to a Windows overflow bug in older llama.cpp builds."
                        ),
                        "key": "run_fit_target_input",
                        "on_change": _make_saver("run_fit_target", "run_fit_target_input"),
                    }
                    if "run_fit_target_input" not in st.session_state:
                        fit_target_kwargs["value"] = int(config.get("run_fit_target", 0))
                    st.number_input(**fit_target_kwargs)  # type: ignore[arg-type]

                r2c1, r2c2, _ = st.columns(3)

                with r2c1:
                    ctk_kwargs = {
                        "label": "KV cache K type (-ctk)",
                        "options": cache_type_options,
                        "help": "Quantize the KV cache keys. q8_0 halves memory vs f16 with minimal quality loss.",
                        "key": "run_cache_type_k_select",
                        "on_change": _make_saver("run_cache_type_k", "run_cache_type_k_select"),
                    }
                    saved_ctk = config.get("run_cache_type_k", "f16")
                    if "run_cache_type_k_select" not in st.session_state:
                        ctk_kwargs["index"] = cache_type_options.index(saved_ctk) \
                            if saved_ctk in cache_type_options else 0
                    st.selectbox(**ctk_kwargs)  # type: ignore[arg-type]

                with r2c2:
                    ctv_kwargs = {
                        "label": "KV cache V type (-ctv)",
                        "options": cache_type_options,
                        "help": (
                            "Quantize the KV cache values. q8_0 halves memory vs f16 with minimal quality loss. "
                            "Quantized V types require Flash Attention (-fa)."
                        ),
                        "key": "run_cache_type_v_select",
                        "on_change": _make_saver("run_cache_type_v", "run_cache_type_v_select"),
                    }
                    saved_ctv = config.get("run_cache_type_v", "f16")
                    if "run_cache_type_v_select" not in st.session_state:
                        ctv_kwargs["index"] = cache_type_options.index(saved_ctv) \
                            if saved_ctv in cache_type_options else 0
                    st.selectbox(**ctv_kwargs)  # type: ignore[arg-type]

                st.markdown("---")

                r3c1, r3c2, r3c3 = st.columns(3)

                with r3c1:
                    temp_kwargs = {
                        "label": "Temp (--temp)",
                        "min_value": 0.0,
                        "max_value": 5.0,
                        "step": 0.05,
                        "format": "%.2f",
                        "help": "Sampling temperature.",
                        "key": "run_temp_input",
                        "on_change": _make_saver("run_temp", "run_temp_input"),
                    }
                    if "run_temp_input" not in st.session_state:
                        temp_kwargs["value"] = float(config.get("run_temp", 0.8))
                    st.number_input(**temp_kwargs)  # type: ignore[arg-type]

                with r3c2:
                    topk_kwargs = {
                        "label": "Top-K (--top-k)",
                        "min_value": 0,
                        "max_value": 1000,
                        "step": 1,
                        "help": "Top-K sampling. 0 = disabled.",
                        "key": "run_top_k_input",
                        "on_change": _make_saver("run_top_k", "run_top_k_input"),
                    }
                    if "run_top_k_input" not in st.session_state:
                        topk_kwargs["value"] = int(config.get("run_top_k", 40))
                    st.number_input(**topk_kwargs)  # type: ignore[arg-type]

                with r3c3:
                    topp_kwargs = {
                        "label": "Top-P (--top-p)",
                        "min_value": 0.0,
                        "max_value": 1.0,
                        "step": 0.01,
                        "format": "%.2f",
                        "help": "Top-P (nucleus) sampling.",
                        "key": "run_top_p_input",
                        "on_change": _make_saver("run_top_p", "run_top_p_input"),
                    }
                    if "run_top_p_input" not in st.session_state:
                        topp_kwargs["value"] = float(config.get("run_top_p", 0.95))
                    st.number_input(**topp_kwargs)  # type: ignore[arg-type]

                r4c1, r4c2, _ = st.columns(3)

                with r4c1:
                    minp_kwargs = {
                        "label": "Min-P (--min-p)",
                        "min_value": 0.0,
                        "max_value": 1.0,
                        "step": 0.01,
                        "format": "%.2f",
                        "help": "Min-P sampling threshold.",
                        "key": "run_min_p_input",
                        "on_change": _make_saver("run_min_p", "run_min_p_input"),
                    }
                    if "run_min_p_input" not in st.session_state:
                        minp_kwargs["value"] = float(config.get("run_min_p", 0.05))
                    st.number_input(**minp_kwargs)  # type: ignore[arg-type]

                with r4c2:
                    pp_kwargs = {
                        "label": "Presence (--presence-penalty)",
                        "min_value": -2.0,
                        "max_value": 2.0,
                        "step": 0.05,
                        "format": "%.2f",
                        "help": "Presence penalty for repetition reduction.",
                        "key": "run_presence_penalty_input",
                        "on_change": _make_saver("run_presence_penalty", "run_presence_penalty_input"),
                    }
                    if "run_presence_penalty_input" not in st.session_state:
                        pp_kwargs["value"] = float(config.get("run_presence_penalty", 0.0))
                    st.number_input(**pp_kwargs)  # type: ignore[arg-type]

                st.markdown("---")

                r5c1, r5c2, _ = st.columns(3)

                with r5c1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    jinja_kwargs = {
                        "label": "Jinja templates (--jinja)",
                        "help": "Enable Jinja2 chat template rendering. Required for --reasoning.",
                        "key": "run_jinja_checkbox",
                        "on_change": _make_saver("run_jinja", "run_jinja_checkbox"),
                    }
                    if "run_jinja_checkbox" not in st.session_state:
                        jinja_kwargs["value"] = bool(config.get("run_jinja", True))
                    jinja_enabled = st.checkbox(**jinja_kwargs)  # type: ignore[arg-type]
                    if "run_jinja_checkbox" in st.session_state:
                        jinja_enabled = st.session_state["run_jinja_checkbox"]

                with r5c2:
                    reasoning_options = ["off", "on", "auto"]
                    reasoning_kwargs = {
                        "label": "Reasoning (--reasoning)",
                        "options": reasoning_options,
                        "help": "Reasoning mode. Only emitted when Jinja is enabled. 'auto' is llama.cpp's default (omitted from command).",
                        "key": "run_reasoning_select",
                        "on_change": _make_saver("run_reasoning", "run_reasoning_select"),
                        "disabled": not jinja_enabled,
                    }
                    if "run_reasoning_select" not in st.session_state:
                        saved_reasoning = config.get("run_reasoning", "off")
                        reasoning_kwargs["index"] = reasoning_options.index(saved_reasoning) \
                            if saved_reasoning in reasoning_options else 0
                    st.selectbox(**reasoning_kwargs)  # type: ignore[arg-type]

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
                        "on_change": _make_saver("run_bench_n_prompt", "run_bench_n_prompt_input"),
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
                        "on_change": _make_saver("run_bench_n_gen", "run_bench_n_gen_input"),
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
                        "on_change": _make_saver("run_bench_repetitions", "run_bench_repetitions_input"),
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
                        "on_change": _make_saver("run_bench_batch_size", "run_bench_batch_size_input"),
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
                        "on_change": _make_saver("run_bench_ubatch_size", "run_bench_ubatch_size_input"),
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
                        "on_change": _make_saver("run_bench_threads", "run_bench_threads_input"),
                    }
                    if "run_bench_threads_input" not in st.session_state:
                        bt_kwargs["value"] = int(config.get("run_bench_threads", 0))
                    st.number_input(**bt_kwargs)  # type: ignore[arg-type]

                b3c1, b3c2, b3c3 = st.columns(3)

                with b3c1:
                    ctk_kwargs = {
                        "label": "Cache type K (-ctk)",
                        "options": cache_type_options,
                        "help": "KV cache type for keys. Useful for benchmarking KV quantization.",
                        "key": "run_bench_cache_type_k_select",
                        "on_change": _make_saver("run_bench_cache_type_k", "run_bench_cache_type_k_select"),
                    }
                    saved_ctk = config.get("run_bench_cache_type_k", "f16")
                    if "run_bench_cache_type_k_select" not in st.session_state:
                        ctk_kwargs["index"] = cache_type_options.index(saved_ctk) \
                            if saved_ctk in cache_type_options else 0
                    st.selectbox(**ctk_kwargs)  # type: ignore[arg-type]

                with b3c2:
                    ctv_kwargs = {
                        "label": "Cache type V (-ctv)",
                        "options": cache_type_options,
                        "help": "KV cache type for values.",
                        "key": "run_bench_cache_type_v_select",
                        "on_change": _make_saver("run_bench_cache_type_v", "run_bench_cache_type_v_select"),
                    }
                    saved_ctv = config.get("run_bench_cache_type_v", "f16")
                    if "run_bench_cache_type_v_select" not in st.session_state:
                        ctv_kwargs["index"] = cache_type_options.index(saved_ctv) \
                            if saved_ctv in cache_type_options else 0
                    st.selectbox(**ctv_kwargs)  # type: ignore[arg-type]

                with b3c3:
                    output_options = ["md", "csv", "json", "jsonl", "sql"]
                    bo_kwargs: Dict[str, Any] = {
                        "label": "Output format (-o)",
                        "options": output_options,
                        "help": "Output format for benchmark results.",
                        "key": "run_bench_output_select",
                        "on_change": _make_saver("run_bench_output", "run_bench_output_select"),
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
                        "on_change": _make_saver("run_bench_no_warmup", "run_bench_no_warmup_checkbox"),
                    }
                    if "run_bench_no_warmup_checkbox" not in st.session_state:
                        nw_kwargs["value"] = bool(config.get("run_bench_no_warmup", False))
                    st.checkbox(**nw_kwargs)  # type: ignore[arg-type]

                with b4c2:
                    prog_kwargs: Dict[str, Any] = {
                        "label": "Progress (--progress)",
                        "help": "Print test progress indicators.",
                        "key": "run_bench_progress_checkbox",
                        "on_change": _make_saver("run_bench_progress", "run_bench_progress_checkbox"),
                    }
                    if "run_bench_progress_checkbox" not in st.session_state:
                        prog_kwargs["value"] = bool(config.get("run_bench_progress", False))
                    st.checkbox(**prog_kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect current inference parameters from session state / config."""
    return {
        "mode": str(st.session_state.get("run_mode_radio", config.get("run_mode", "llama-server"))),
        "ngl": int(st.session_state.get("run_ngl_input", config.get("run_ngl", 99))),
        "ctx": int(st.session_state.get("run_ctx_input", config.get("run_ctx", 4096))),
        "flash_attn": bool(st.session_state.get("run_flash_attn_checkbox", config.get("run_flash_attn", False))),
        "fit_target": int(st.session_state.get("run_fit_target_input", config.get("run_fit_target", 0))),
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
    }


_PRESET_KEY_MAP: List[tuple] = [
    ("mode",             "run_mode",             "run_mode_radio"),
    ("ngl",              "run_ngl",              "run_ngl_input"),
    ("ctx",              "run_ctx",              "run_ctx_input"),
    ("flash_attn",       "run_flash_attn",       "run_flash_attn_checkbox"),
    ("fit_target",       "run_fit_target",       "run_fit_target_input"),
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
]


def _apply_pending_preset(pending: Any, config: Dict[str, Any]) -> None:
    """Apply a preset (or defaults) to config and session state.

    Must be called BEFORE any widgets render so session state keys can be set
    directly (setting a key after its widget renders raises StreamlitAPIException).
    """
    if pending == "__defaults__":
        defaults = get_default_config()
        preset = {param_key: defaults[cfg_key]
                  for param_key, cfg_key, _ in _PRESET_KEY_MAP
                  if cfg_key in defaults}
    else:
        preset = pending

    for param_key, cfg_key, session_key in _PRESET_KEY_MAP:
        if param_key in preset:
            config[cfg_key] = preset[param_key]
            st.session_state[session_key] = preset[param_key]

    save_config(config)


def _get_binary_dir(converter: Any) -> Optional[Path]:
    """Return the directory containing llama.cpp binaries."""
    try:
        return converter.llama_cpp_manager.bin_dir
    except Exception:
        return None


def _find_binary(binary_dir: Optional[Path], name: str) -> str:
    """Find a binary by name, returning its full path or just the name as fallback."""
    if binary_dir is None:
        return name
    import platform as _platform
    suffix = ".exe" if _platform.system() == "Windows" else ""
    candidate = binary_dir / f"{name}{suffix}"
    if candidate.exists():
        return str(candidate)
    return name
