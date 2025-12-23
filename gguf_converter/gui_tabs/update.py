"""
Update tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import subprocess
import platform
import io
import os
from contextlib import redirect_stdout

from ..gui_utils import (
    get_current_version, check_git_updates_available,
    run_and_stream_command, get_binary_version,
    display_binary_version_status, get_conversion_scripts_info,
    display_conversion_scripts_version_status
)


def render_update_tab(converter, config):
    """Render the Update tab"""
    st.header("Update")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Update YaGUFF")
        st.markdown("Check for the latest version of YaGUFF from GitHub.")
        st.markdown("[View YaGUFF on GitHub](https://github.com/usrname0/YaGUFF)")
        if st.button("Pull Latest YaGUFF Version"):
            # Check what version is available
            update_status = check_git_updates_available()

            if update_status["status"] == "updates_available" and update_status.get("latest_version"):
                # Pull and checkout the latest tag
                st.code("Fetching latest version...", language='bash')
                run_and_stream_command(["git", "fetch", "--tags"])
                run_and_stream_command(["git", "checkout", update_status["latest_version"]])
                st.toast(f"Updated to version {update_status['latest_version']}")
                st.rerun()
            elif update_status["status"] == "up_to_date":
                st.info("Already on the latest version")
            else:
                st.error("Could not check for updates. Please check your internet connection and git configuration.")
    with col2:
        st.subheader("YaGUFF Version Information")
        current_version = get_current_version()
        st.code(f"version: {current_version}", language=None)

        # Check if updates are available
        update_status = check_git_updates_available()
        if update_status["status"] == "updates_available":
            st.warning(update_status["message"])
        elif update_status["status"] == "up_to_date":
            st.info(update_status["message"])
        else:
            st.info(update_status["message"])

    st.markdown("---")

    # Update YaGUFF Binaries Section
    col_bin1, col_bin2 = st.columns(2)
    with col_bin1:
        st.subheader("Update YaGUFF Binaries")
        st.markdown("Force a re-download of the `llama.cpp` binaries. Choose **Recommended** for the tested version bundled with YaGUFF, or **Latest** for the newest llama.cpp release.")
        st.markdown("[View llama.cpp on GitHub](https://github.com/ggml-org/llama.cpp)")

        if st.button("Force Binary Update - Recommended Version"):
            output_container = st.empty()
            output_container.code("Starting binary update (Recommended version)...\nThis may take a moment.", language='bash')

            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    st.session_state.converter.binary_manager.download_binaries(force=True)
                    st.toast("Binaries updated successfully!")
                    success = True
                except Exception as e:
                    print(f"\n--- An error occurred ---\n{str(e)}")
                    st.toast(f"An error occurred during binary update: {e}")
                    success = False

            output = f.getvalue()
            output_container.code(output, language='bash')

            if success:
                st.rerun()

        if st.button("Force Binary Update - Latest Version"):
            output_container = st.empty()
            output_container.code("Fetching latest llama.cpp version...\nThis may take a moment.", language='bash')

            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    latest_version = st.session_state.converter.binary_manager.get_latest_version()
                    print(f"Latest version: {latest_version}")
                    st.session_state.converter.binary_manager.download_binaries(force=True, version=latest_version)
                    st.toast("Binaries updated successfully!")
                    success = True
                except Exception as e:
                    print(f"\n--- An error occurred ---\n{str(e)}")
                    st.toast(f"An error occurred during binary update: {e}")
                    success = False

            output = f.getvalue()
            output_container.code(output, language='bash')

            if success:
                st.rerun()
    with col_bin2:
        st.subheader("YaGUFF Binary Information")
        binary_info = get_binary_version(st.session_state.converter)
        if binary_info["status"] == "ok":
            st.success(binary_info['message'])
        elif binary_info["status"] == "missing":
            st.warning(binary_info['message'])
        else:
            st.error(binary_info['message'])

        display_binary_version_status(st.session_state.converter)

    st.markdown("---")

    # Update Conversion Scripts Section
    col_scripts1, col_scripts2 = st.columns(2)
    with col_scripts1:
        st.subheader("Update Conversion Scripts")
        st.markdown("Update the `llama.cpp` repository that contains the `convert_hf_to_gguf.py` script. Choose **Recommended** for the tested version matching YaGUFF binaries, or **Latest** for the newest conversion scripts.")
        st.markdown("[View llama.cpp on GitHub](https://github.com/ggml-org/llama.cpp)")

        if st.button("Force Conversion Scripts Update - Recommended Version"):
            output_container = st.empty()
            output_container.code("Updating conversion scripts (Recommended version)...\nThis may take a moment.", language='bash')

            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    result = st.session_state.converter.binary_manager.update_conversion_scripts(use_recommended=True)
                    if result['status'] in ['success', 'already_updated']:
                        st.toast("Conversion scripts updated successfully!")
                        success = True
                    else:
                        print(f"\n--- An error occurred ---\n{result['message']}")
                        st.toast(f"An error occurred: {result['message']}")
                        success = False
                except Exception as e:
                    print(f"\n--- An error occurred ---\n{str(e)}")
                    st.toast(f"An error occurred during update: {e}")
                    success = False

            output = f.getvalue()
            output_container.code(output, language='bash')

            if success:
                st.rerun()

        if st.button("Force Conversion Scripts Update - Latest Version"):
            output_container = st.empty()
            output_container.code("Updating conversion scripts to latest version...\nThis may take a moment.", language='bash')

            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    result = st.session_state.converter.binary_manager.update_conversion_scripts(use_recommended=False)
                    if result['status'] in ['success', 'already_updated']:
                        st.toast("Conversion scripts updated successfully!")
                        success = True
                    else:
                        print(f"\n--- An error occurred ---\n{result['message']}")
                        st.toast(f"An error occurred: {result['message']}")
                        success = False
                except Exception as e:
                    print(f"\n--- An error occurred ---\n{str(e)}")
                    st.toast(f"An error occurred during update: {e}")
                    success = False

            output = f.getvalue()
            output_container.code(output, language='bash')

            if success:
                st.rerun()

    with col_scripts2:
        st.subheader("Conversion Scripts Information")
        scripts_info = get_conversion_scripts_info(st.session_state.converter)

        if scripts_info["status"] == "ok":
            st.success(scripts_info['message'])
        elif scripts_info["status"] == "missing":
            st.warning(scripts_info['message'])
        else:
            st.error(scripts_info['message'])

        display_conversion_scripts_version_status(st.session_state.converter)

    st.markdown("---")

    st.subheader("Update Dependencies")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("Update PyTorch and Python dependencies from `requirements.txt`.")
        st.markdown("The GUI will close and restart automatically. All updates will run in the terminal.")

        if st.button("Update Dependencies & Restart"):
            # Get the current Streamlit port from the running URL
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(st.context.url)
                port = str(parsed_url.port) if parsed_url.port else "8501"
            except Exception:
                # Fallback to default port if we can't detect it
                port = "8501"

            # Determine platform-specific restart script
            is_windows = platform.system() == "Windows"
            if is_windows:
                restart_script = Path(__file__).parent.parent.parent / "scripts" / "update_and_restart.bat"
            else:
                restart_script = Path(__file__).parent.parent.parent / "scripts" / "update_and_restart.sh"

            if restart_script.exists():
                st.info("See terminal for update progress... Please wait.")

                # Give Streamlit time to send the message to browser before exiting
                import time
                time.sleep(0.5)

                # Start the update script with port parameter - output will show in original terminal
                if is_windows:
                    subprocess.Popen(
                        ["cmd", "/c", str(restart_script), port],
                        cwd=str(restart_script.parent)
                    )
                else:
                    subprocess.Popen(
                        [str(restart_script), port],
                        cwd=str(restart_script.parent)
                    )

                # Exit Streamlit - this releases the port
                os._exit(0)
            else:
                st.error(f"Restart script not found: {restart_script}")

    with col4:
        try:
            req_path = Path(__file__).parent.parent.parent / "requirements.txt"
            if req_path.exists():
                st.markdown("`requirements.txt`")
                st.code(req_path.read_text(), language='text')
            else:
                st.warning("`requirements.txt` not found.")
        except Exception as e:
            st.error(f"Error reading requirements.txt: {e}")
