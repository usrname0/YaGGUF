"""
Streamlit GUI for GGUF Converter
"""

import streamlit as st
import sys
from pathlib import Path
import multiprocessing
import shutil

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
        render_split_merge_tab,
        render_llama_cpp_tab,
        render_info_tab,
        render_update_tab
    )
except ImportError:
    # Add parent directory for direct execution
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
        render_split_merge_tab,
        render_llama_cpp_tab,
        render_info_tab,
        render_update_tab
    )


def main() -> None:
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

    st.title("YaGGUF - Yet Another GGUF Converter")

    # Initialize converter with custom settings if specified
    if 'converter' not in st.session_state:
        config = st.session_state.get('config', load_config())
        custom_binaries = config.get("custom_binaries_folder", "") if config.get("use_custom_binaries", False) else None
        custom_repo = config.get("custom_llama_cpp_repo", "") if config.get("use_custom_conversion_script", False) else None
        st.session_state.converter = GGUFConverter(
            custom_binaries_folder=custom_binaries,
            custom_llama_cpp_repo=custom_repo
        )

    # Load config on first run
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

    if 'reset_count' not in st.session_state:
        st.session_state.reset_count = 0

    # Handle model path from downloader (before widget creation)
    if 'pending_model_path' in st.session_state:
        st.session_state.model_path_input = st.session_state.pending_model_path
        del st.session_state.pending_model_path

    # Handle reset to defaults (before widget creation)
    if 'pending_reset_defaults' in st.session_state:
        defaults = get_default_config()
        st.session_state.verbose_checkbox = defaults["verbose"]
        del st.session_state.pending_reset_defaults

    converter = st.session_state.converter
    config = st.session_state.config

    with st.sidebar:
        st.header("Settings")
        st.markdown("[YaGGUF (GitHub)](https://github.com/usrname0/YaGGUF)")

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
        verbose = st.checkbox(**verbose_kwargs)  # type: ignore[arg-type]
        st.markdown("---")
        st.markdown("**Performance:**")
        max_workers = multiprocessing.cpu_count()
        default_threads = max(1, max_workers - 1)
        st.text(f"Logical Processors: {max_workers}")

        def save_num_threads():
            config["num_threads"] = int(st.session_state.num_threads_input)
            save_config(config)

        num_threads = st.number_input(
            "Thread count",
            min_value=1,
            max_value=max_workers,
            value=int(config.get("num_threads") or default_threads),
            step=1,
            help=f"Number of threads for llama.cpp. Default = max - 1 to keep system responsive)",
            key="num_threads_input",
            on_change=save_num_threads
        )

        st.markdown("---")
        st.markdown("**Testing:**")
        if st.button("Reload UI", use_container_width=True, help="Reload the user interface via st.rerun()"):
            st.rerun()

        if st.button("Test Model Variants", use_container_width=True, help="Test all GGUF variants in the output directory interactively with llama-server"):
            import subprocess
            import platform
            script_path = Path(__file__).parent.parent / "tests" / "manual_variant_testing.py"

            # Get output directory from config
            output_dir = config.get("output_dir", "")

            # Build command arguments
            if output_dir and Path(output_dir).exists():
                # Run with output directory
                cmd_args = [sys.executable, str(script_path), str(output_dir)]
                toast_msg = f"Testing variants in: {output_dir}"
            elif output_dir:
                # Output dir set but doesn't exist
                st.toast(f"Output directory not found: {output_dir}")
                cmd_args = [sys.executable, str(script_path), "--help"]
                toast_msg = "Showing help (invalid output directory)"
            else:
                # No output dir set, show help
                st.toast("No output directory set. Showing help.")
                cmd_args = [sys.executable, str(script_path), "--help"]
                toast_msg = "Showing help (no output directory)"

            # Launch the script in a new terminal window
            system = platform.system()
            if system == "Windows":
                # Windows: use start to open new cmd window
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k"] + cmd_args,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            elif system == "Darwin":
                # macOS: use osascript to open new Terminal window
                cmd_str = " ".join(f'"{arg}"' for arg in cmd_args)
                subprocess.Popen([
                    "osascript", "-e",
                    f'tell app "Terminal" to do script "cd {Path.cwd()} && {cmd_str}"'
                ])
            else:
                # Linux: try various terminal emulators
                terminals = ["gnome-terminal", "konsole", "xterm"]
                for term in terminals:
                    if shutil.which(term):
                        if term == "gnome-terminal":
                            subprocess.Popen([term, "--"] + cmd_args)
                        else:
                            subprocess.Popen([term, "-e", " ".join(cmd_args)])
                        break

            st.toast(toast_msg)

        st.markdown("---")
        st.markdown("**Reset:**")
        if st.button("Reset to defaults", use_container_width=True, help="Reset all settings to default values"):
            st.session_state.config = reset_config()
            st.session_state.reset_count += 1
            if "download_just_completed" in st.session_state:
                st.session_state.download_just_completed = False
            st.session_state.model_path_input = ""
            st.session_state.output_dir_input = ""
            if "pending_model_path" in st.session_state:
                del st.session_state.pending_model_path
            keys_to_delete = [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(('trad_', 'k_', 'i_'))]
            for key in keys_to_delete:
                del st.session_state[key]
            if "iq_checkbox_states" in st.session_state:
                st.session_state.iq_checkbox_states = {}
            # Reset file handling and advanced quantization options
            st.session_state.file_mode_radio = "Single files"
            st.session_state.output_tensor_type_select = "Same as quant type (default)"
            st.session_state.token_embedding_type_select = "Same as quant type (default)"
            st.session_state.pure_quantization_checkbox = False
            # Reset split/merge options
            if "split_merge_operation_mode" in st.session_state:
                st.session_state.split_merge_operation_mode = "Split"
            if "split_merge_max_shard_size_input" in st.session_state:
                st.session_state.split_merge_max_shard_size_input = 2.0
            st.session_state.pending_reset_defaults = True
            st.rerun()

    # Main content - tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Convert & Quantize",
        "Imatrix Settings",
        "Imatrix Statistics",
        "HuggingFace Downloader",
        "Split/Merge Shards",
        "Info",
        "llama.cpp",
        "Update"
    ])

    with tab1:
        render_convert_tab(converter, config, verbose, num_threads)

    with tab2:
        render_imatrix_settings_tab(converter, config)

    with tab3:
        render_imatrix_stats_tab(converter, config)

    with tab4:
        render_downloader_tab(converter, config)

    with tab5:
        render_split_merge_tab(converter, config)

    with tab6:
        render_info_tab(converter, config)

    with tab7:
        render_llama_cpp_tab(converter, config)

    with tab8:
        render_update_tab(converter, config)


if __name__ == "__main__":
    main()
