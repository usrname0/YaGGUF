"""
HuggingFace Downloader tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import webbrowser

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, TKINTER_AVAILABLE
)


def render_downloader_tab(converter, config):
    """Render the HuggingFace Downloader tab"""
    st.header("HuggingFace Downloader")
    st.markdown("Download models from HuggingFace")
    st.markdown("[Browse models on HuggingFace](https://huggingface.co/models)")

    # Initialize session state for downloaded model path
    if "downloaded_model_path" not in st.session_state:
        st.session_state.downloaded_model_path = None

    # Repository ID with Check Repo button
    col_repo, col_repo_btn = st.columns([5, 1])
    with col_repo:
        repo_id = st.text_input(
            "Repository ID",
            value=config.get("repo_id", ""),
            placeholder="username/model-name"
        )
    with col_repo_btn:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input
        repo_id_populated = bool(repo_id and repo_id.strip())
        if st.button(
            "Check Repo",
            key="check_repo_download_btn",
            use_container_width=True,
            disabled=not repo_id_populated,
            help="Open HuggingFace repo in browser" if repo_id_populated else "Enter a repo ID first"
        ):
            if repo_id_populated:
                url = f"https://huggingface.co/{repo_id.strip()}"
                webbrowser.open(url)
                st.toast(f"Opened {url}")

    # Download directory with Select Folder and Open Folder buttons
    if TKINTER_AVAILABLE:
        col_download, col_download_browse, col_download_check = st.columns([4, 1, 1])
    else:
        col_download, col_download_check = st.columns([5, 1])
        col_download_browse = None  # Not used when tkinter unavailable

    with col_download:
        # Create dynamic label based on repository ID
        download_dir_label = "Download directory"
        if repo_id and repo_id.strip():
            # Extract model name from repo ID (e.g., "username/model-name" -> "model-name")
            model_name = repo_id.strip().split('/')[-1]
            # Only show the note if the download path doesn't already end with the model name
            current_path = config.get("download_dir", "")
            if current_path:
                current_path_name = Path(current_path.strip().strip('"').strip("'")).name
                if current_path_name != model_name:
                    download_dir_label = f"Download directory (/{model_name}/ folder will be created)"
            else:
                download_dir_label = f"Download directory (/{model_name}/ folder will be created)"

        download_dir = st.text_input(
            download_dir_label,
            value=config.get("download_dir", ""),
            placeholder="~/Models"
        )

    if TKINTER_AVAILABLE:
        with col_download_browse:  # type: ignore[union-attr]
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input
            if st.button(
                "Select Folder",
                key="browse_download_folder_btn",
                use_container_width=True,
                help="Select download directory"
            ):
                download_dir_check = strip_quotes(download_dir)
                initial_dir = download_dir_check if download_dir_check and Path(download_dir_check).exists() else None
                selected_folder = browse_folder(initial_dir)
                if selected_folder:
                    config["download_dir"] = selected_folder
                    save_config(config)
                    st.rerun()

    with col_download_check:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input
        download_dir_check = strip_quotes(download_dir)
        download_dir_exists = bool(download_dir_check and Path(download_dir_check).exists())
        if st.button(
            "Open Folder",
            key="check_download_folder_btn",
            use_container_width=True,
            disabled=not download_dir_exists,
            help="Open folder in file explorer" if download_dir_exists else "Path doesn't exist yet"
        ):
            if download_dir_exists:
                try:
                    open_folder(download_dir_check)
                    st.toast("Opened folder")
                except Exception as e:
                    st.toast(f"Could not open folder: {e}")

    if st.button("Download", use_container_width=True):
        # Strip quotes from paths
        repo_id_clean = strip_quotes(repo_id)
        download_dir_clean = strip_quotes(download_dir)

        if not repo_id_clean:
            st.error("Please provide a repository ID")
        elif not download_dir_clean:
            st.error("Please provide a download directory")
        else:
            # Save current settings before downloading
            config["repo_id"] = repo_id_clean
            config["download_dir"] = download_dir_clean
            save_config(config)

            try:
                with st.spinner(f"Downloading {repo_id_clean}..."):
                    model_path = converter.download_model(repo_id_clean, download_dir_clean)

                # Store in session state and mark as just completed
                st.session_state.downloaded_model_path = str(model_path)
                st.session_state.download_just_completed = True
                st.rerun()

            except Exception as e:
                # Show in Streamlit UI
                st.error(f"Error: {e}")

                # ALSO print to terminal so user sees it
                print(f"\nError: {e}", flush=True)
                import traceback
                traceback.print_exc()

    # Show downloaded model path if one exists (persistent)
    if st.session_state.get("downloaded_model_path"):
        st.markdown("---")
        st.subheader("Downloaded Model")

        col_path, col_set_path = st.columns([5, 1])
        with col_path:
            st.code(st.session_state.downloaded_model_path, language=None)
        with col_set_path:
            if st.button("Set as model path", key="set_model_path_btn", use_container_width=True, help="Set this as the model path in Convert & Quantize tab"):
                path_to_set = st.session_state.downloaded_model_path
                config["model_path"] = path_to_set
                save_config(config)
                # Set pending flag - will be applied before widget creation on next run
                st.session_state.pending_model_path = path_to_set
                # Set success flag for this action
                st.session_state.model_path_set = True
                st.rerun()

    # Show temporary success messages
    if st.session_state.get("download_just_completed", False):
        st.success(f"Model downloaded successfully!")
        st.session_state.download_just_completed = False

    if st.session_state.get("model_path_set", False):
        st.success("Model path set in Convert & Quantize tab")
        st.session_state.model_path_set = False
