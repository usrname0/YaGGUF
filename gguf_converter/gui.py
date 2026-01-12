"""
Streamlit GUI for GGUF Converter
"""

import streamlit as st
import sys
from pathlib import Path
import multiprocessing
import shutil


def get_python_executable():
    """
    Get the correct Python executable to use.
    Prefers venv Python if it exists, otherwise uses sys.executable.
    """
    import platform

    # Check for venv in project root
    project_root = Path(__file__).parent.parent

    # Platform-specific venv paths
    if platform.system() == "Windows":
        venv_python = project_root / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = project_root / "venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)

    # Fallback to sys.executable
    return sys.executable

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

        if st.button("Test Models", use_container_width=True, help="Test all GGUF variants in the output directory interactively with llama-server"):
            import subprocess
            import platform
            script_path = Path(__file__).parent.parent / "tests" / "manual_variant_testing.py"

            # Get output directory from config
            output_dir = config.get("output_dir", "")

            # Get correct Python executable (venv if available)
            python_exe = get_python_executable()

            # Build command arguments
            if output_dir and Path(output_dir).exists():
                # Run with output directory
                cmd_args = [python_exe, str(script_path), str(output_dir)]
                toast_msg = f"Testing variants in: {output_dir}"
            elif output_dir:
                # Output dir set but doesn't exist
                st.toast(f"Output directory not found: {output_dir}")
                cmd_args = [python_exe, str(script_path), "--help"]
                toast_msg = "Showing help (invalid output directory)"
            else:
                # No output dir set, show help
                st.toast("No output directory set. Showing help.")
                cmd_args = [python_exe, str(script_path), "--help"]
                toast_msg = "Showing help (no output directory)"

            # Launch the script in a new terminal window
            system = platform.system()

            # Get project root directory (needed for all platforms)
            project_root = Path(__file__).parent.parent

            if system == "Windows":
                # Windows: Create a bat file to avoid quoting issues
                import tempfile
                import os

                # Create a temporary batch file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as bat_file:
                    bat_path = bat_file.name
                    # Write commands to batch file
                    bat_file.write('@echo off\n')
                    bat_file.write(f'cd /d "{project_root}"\n')  # Change to project directory
                    # Quote each argument that might have spaces
                    cmd_line = ' '.join(f'"{arg}"' for arg in cmd_args)
                    bat_file.write(f'{cmd_line}\n')
                    bat_file.write('pause\n')

                # Launch the batch file in a new window
                subprocess.Popen(
                    ['cmd', '/c', 'start', 'Test Model Variants', bat_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            elif system == "Darwin":
                # macOS: use osascript to open new Terminal window
                cmd_str = " ".join(f'"{arg}"' for arg in cmd_args)
                subprocess.Popen([
                    "osascript", "-e",
                    f'tell app "Terminal" to do script "cd {project_root} && {cmd_str}"'
                ])
            else:
                # Linux: try various terminal emulators
                # x-terminal-emulator is Debian/Ubuntu's default terminal symlink
                # For multi-arg commands, wrap in sh -c
                cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd_args)
                terminals = [
                    ("x-terminal-emulator", ["-e", "sh", "-c", f"cd {project_root} && {cmd_str}; read -p 'Press Enter to close...'"]),
                    ("gnome-terminal", ["--", "sh", "-c", f"cd {project_root} && {cmd_str}; read -p 'Press Enter to close...'"]),
                    ("konsole", ["--", "sh", "-c", f"cd {project_root} && {cmd_str}; read -p 'Press Enter to close...'"]),
                    ("xfce4-terminal", ["-x", "sh", "-c", f"cd {project_root} && {cmd_str}; read -p 'Press Enter to close...'"]),
                    ("mate-terminal", ["-e", "sh", "-c", f"cd {project_root} && {cmd_str}; read -p 'Press Enter to close...'"]),
                    ("xterm", ["-e", "sh", "-c", f"cd {project_root} && {cmd_str}; read -p 'Press Enter to close...'"]),
                ]
                launched = False
                for term, term_args in terminals:
                    if shutil.which(term):
                        try:
                            subprocess.Popen([term] + term_args)
                            launched = True
                            break
                        except Exception as e:
                            continue

                if not launched:
                    st.error("No compatible terminal emulator found. Please install gnome-terminal, xterm, or another terminal.")
                    return

            st.toast(toast_msg)

        # Dev Tests button (only shown when dev_mode is enabled)
        if config.get("dev_mode", False):
            if st.button("Dev Tests", use_container_width=True, help="Run the full test suite (pytest) in a new terminal window"):
                import subprocess
                import platform
                import tempfile

                # Get correct Python executable (venv if available)
                python_exe = get_python_executable()

                # Launch in a new terminal window
                system = platform.system()
                project_root = Path(__file__).parent.parent

                if system == "Windows":
                    # Create a batch file that checks for pytest and installs if needed
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as bat_file:
                        bat_path = bat_file.name
                        bat_file.write('@echo off\n')
                        bat_file.write(f'cd /d "{project_root}"\n')
                        bat_file.write('echo Checking for pytest...\n')
                        bat_file.write(f'"{python_exe}" -m pytest --version >nul 2>&1\n')
                        bat_file.write('if errorlevel 1 (\n')
                        bat_file.write('    echo pytest is not installed.\n')
                        bat_file.write('    echo.\n')
                        bat_file.write('    set /p "install=Install test dependencies? (y/n): "\n')
                        bat_file.write('    if /i "%install%"=="y" (\n')
                        bat_file.write(f'        "{python_exe}" -m pip install -r tests\\requirements-dev.txt\n')
                        bat_file.write('        if errorlevel 1 (\n')
                        bat_file.write('            echo.\n')
                        bat_file.write('            echo Installation failed!\n')
                        bat_file.write('            pause\n')
                        bat_file.write('            exit /b 1\n')
                        bat_file.write('        )\n')
                        bat_file.write('    ) else (\n')
                        bat_file.write('        echo.\n')
                        bat_file.write('        echo Cancelled.\n')
                        bat_file.write('        pause\n')
                        bat_file.write('        exit /b 0\n')
                        bat_file.write('    )\n')
                        bat_file.write(')\n')
                        bat_file.write('echo.\n')
                        bat_file.write('echo Running tests...\n')
                        bat_file.write('echo.\n')
                        bat_file.write(f'"{python_exe}" -m pytest\n')
                        bat_file.write('echo.\n')
                        bat_file.write('echo Press any key to continue or just close this terminal...\n')
                        bat_file.write('pause > nul\n')

                    subprocess.Popen(
                        ['cmd', '/c', 'start', 'Dev Tests', bat_path],
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                elif system == "Darwin":
                    # macOS: use osascript to open new Terminal window
                    script = f'''cd {project_root}
echo "Checking for pytest..."
if ! "{python_exe}" -m pytest --version >/dev/null 2>&1; then
    echo "pytest is not installed."
    echo ""
    read -p "Install test dependencies? (y/n): " install
    if [ "$install" = "y" ] || [ "$install" = "Y" ]; then
        "{python_exe}" -m pip install -r tests/requirements-dev.txt
        if [ $? -ne 0 ]; then
            echo ""
            echo "Installation failed!"
            read -p "Press Enter to close..."
            exit 1
        fi
    else
        echo ""
        echo "Cancelled."
        read -p "Press Enter to close..."
        exit 0
    fi
fi
echo ""
echo "Running tests..."
echo ""
"{python_exe}" -m pytest
echo ""
read -p "Press Enter to close..."'''
                    subprocess.Popen([
                        "osascript", "-e",
                        f'tell app "Terminal" to do script "{script}"'
                    ])
                else:
                    # Linux: try various terminal emulators
                    # x-terminal-emulator is Debian/Ubuntu's default terminal symlink
                    # Build script that checks for pytest and installs if needed
                    script = f'''cd {project_root}
echo "Checking for pytest..."
if ! "{python_exe}" -m pytest --version >/dev/null 2>&1; then
    echo "pytest is not installed."
    echo ""
    echo -n "Install test dependencies? (y/n): "
    read install
    if [ "$install" = "y" ] || [ "$install" = "Y" ]; then
        "{python_exe}" -m pip install -r tests/requirements-dev.txt
        if [ $? -ne 0 ]; then
            echo ""
            echo "Installation failed!"
            echo ""
            echo "Press Enter to close..."
            read dummy
            exit 1
        fi
    else
        echo ""
        echo "Cancelled."
        echo ""
        echo "Press Enter to close..."
        read dummy
        exit 0
    fi
fi
echo ""
echo "Running tests..."
echo ""
"{python_exe}" -m pytest
echo ""
echo "Press Enter to close..."
read dummy
exec bash'''
                    terminals = [
                        ("x-terminal-emulator", ["-e", "bash", "-c", script]),
                        ("gnome-terminal", ["--", "bash", "-c", script]),
                        ("konsole", ["--", "bash", "-c", script]),
                        ("xfce4-terminal", ["-x", "bash", "-c", script]),
                        ("mate-terminal", ["-e", "bash", "-c", script]),
                        ("xterm", ["-e", "bash", "-c", script]),
                    ]
                    launched = False
                    for term, term_args in terminals:
                        if shutil.which(term):
                            try:
                                subprocess.Popen([term] + term_args)
                                launched = True
                                break
                            except Exception as e:
                                continue

                    if not launched:
                        st.error("No compatible terminal emulator found. Please install gnome-terminal, xterm, or another terminal.")
                        return

                st.toast("Opening dev tests terminal...")

        st.markdown("---")
        st.markdown("**Reset:**")
        if st.button("Reset to defaults", use_container_width=True, help="Reset all settings to default values"):
            st.session_state.config = reset_config()
            st.session_state.reset_count += 1
            if "download_just_completed" in st.session_state:
                st.session_state.download_just_completed = False
            st.session_state.model_path_input = ""
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
