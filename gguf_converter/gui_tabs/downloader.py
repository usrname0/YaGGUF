"""
HuggingFace Downloader tab for GGUF Converter GUI
"""

import streamlit as st
from pathlib import Path
import webbrowser
from typing import Dict, Any, TYPE_CHECKING
from huggingface_hub import HfApi, login as hf_login, get_token, logout as hf_logout
from huggingface_hub.errors import GatedRepoError
from colorama import Style
from ..theme import THEME as theme

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, TKINTER_AVAILABLE, extract_repo_id_from_url,
    get_platform_path
)

if TYPE_CHECKING:
    from ..converter import GGUFConverter


def render_downloader_tab(converter: "GGUFConverter", config: Dict[str, Any]) -> None:
    """Render the HuggingFace Downloader tab"""
    st.header("HuggingFace Downloader")
    st.markdown("Download models from HuggingFace")
    st.markdown("[Browse models on HuggingFace](https://huggingface.co/models)")

    # Initialize session state for downloaded model path
    if "downloaded_model_path" not in st.session_state:
        st.session_state.downloaded_model_path = None

    # Initialize session state for repo size data
    if "repo_size_data" not in st.session_state:
        st.session_state.repo_size_data = None

    # Initialize session state for gated repo handling
    if "show_hf_login" not in st.session_state:
        st.session_state.show_hf_login = False
    if "gated_repo_id" not in st.session_state:
        st.session_state.gated_repo_id = None
    if "hf_auth_message" not in st.session_state:
        st.session_state.hf_auth_message = None
    if "download_error" not in st.session_state:
        st.session_state.download_error = None

    # Repository ID with View Repo button
    col_repo, col_repo_view = st.columns([5, 1])
    with col_repo:
        repo_id = st.text_input(
            "Repository ID",
            value=config.get("repo_id", ""),
            placeholder="username/model-name"
        )

    # Extract repo ID from URL if user pasted a full HuggingFace URL
    if repo_id and ("huggingface.co/" in repo_id or "hf.co/" in repo_id):
        extracted_repo_id = extract_repo_id_from_url(repo_id)
        if extracted_repo_id and extracted_repo_id != repo_id:
            # Update config with extracted repo ID
            config["repo_id"] = extracted_repo_id
            save_config(config)
            st.toast(f"Extracted repo ID: {extracted_repo_id}")
            st.rerun()

    # Save repo_id to config if it changed
    if repo_id != config.get("repo_id", ""):
        config["repo_id"] = repo_id
        save_config(config)

    repo_id_populated = bool(repo_id and repo_id.strip())

    # Auto-fetch repo size when repo ID changes
    if repo_id_populated:
        # Check if we need to fetch size for this repo
        if not st.session_state.repo_size_data or st.session_state.repo_size_data.get("repo_id") != repo_id.strip():
            try:
                api = HfApi()
                repo_info = api.repo_info(repo_id=repo_id.strip(), files_metadata=True, timeout=10.0)

                # Calculate total size from all files
                total_size_bytes = 0
                if hasattr(repo_info, 'siblings') and repo_info.siblings:
                    for file in repo_info.siblings:
                        if hasattr(file, 'size') and file.size:
                            total_size_bytes += file.size

                if total_size_bytes > 0:
                    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
                    st.session_state.repo_size_data = {
                        "repo_id": repo_id.strip(),
                        "size_gb": total_size_gb
                    }
                else:
                    st.session_state.repo_size_data = None
            except Exception:
                # Silently fail on auto-fetch - user can click Check Size button if needed
                st.session_state.repo_size_data = None
    else:
        # Clear repo size data if repo ID is empty
        st.session_state.repo_size_data = None

    with col_repo_view:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input
        if st.button(
            "View Repo",
            key="view_repo_download_btn",
            use_container_width=True,
            disabled=not repo_id_populated,
            help="Open HuggingFace repo in browser" if repo_id_populated else "Enter a repo ID first"
        ):
            if repo_id_populated:
                url = f"https://huggingface.co/{repo_id.strip()}"
                webbrowser.open(url)
                st.toast(f"Opened {url}")

    # Show repo size info
    col_info = st.columns([5, 1])[0]
    with col_info:
        if st.session_state.repo_size_data:
            repo_size_gb = st.session_state.repo_size_data["size_gb"]
            st.info(f"Repository size: {repo_size_gb:.2f} GB")
        else:
            st.code("Enter Repository ID to see repository size", language=None)

    # Download directory with Select Folder button
    if TKINTER_AVAILABLE:
        col_download, col_download_browse = st.columns([5, 1])
    else:
        col_download = st.columns([6])[0]
        col_download_browse = None

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
                    subfolder_hint = get_platform_path(f"\\{model_name}\\", f"/{model_name}/")
                    download_dir_label = f"Download directory ({subfolder_hint} subfolder will be created)"
            else:
                subfolder_hint = get_platform_path(f"\\{model_name}\\", f"/{model_name}/")
                download_dir_label = f"Download directory ({subfolder_hint} subfolder will be created)"

        download_dir = st.text_input(
            download_dir_label,
            value=config.get("download_dir", ""),
            placeholder=get_platform_path("C:\\Models", "/home/user/Models")
        )

    # Save download_dir to config if it changed
    if download_dir != config.get("download_dir", ""):
        config["download_dir"] = download_dir
        save_config(config)

    if TKINTER_AVAILABLE and col_download_browse:
        with col_download_browse:
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

    # Show download directory info or placeholder with Open Folder button
    col_dir_info1, col_dir_info2 = st.columns([5, 1])

    with col_dir_info1:
        # Use current value from text input, not config
        download_dir_clean = strip_quotes(download_dir)
        if download_dir_clean and Path(download_dir_clean).exists():
            import shutil

            # Calculate full download path including model subfolder
            full_path = Path(download_dir_clean)
            if repo_id and repo_id.strip():
                model_name = repo_id.strip().split('/')[-1]
                full_path = full_path / model_name

            stat = shutil.disk_usage(download_dir_clean)
            free_gb = stat.free / (1024 * 1024 * 1024)

            # Check if we have enough space for the repository
            insufficient_space = False
            if st.session_state.repo_size_data:
                repo_size_gb = st.session_state.repo_size_data["size_gb"]
                buffer_gb = 0.5  # 500MB buffer
                required_gb = repo_size_gb + buffer_gb
                if free_gb < required_gb:
                    insufficient_space = True

            if insufficient_space:
                st.warning(f"{full_path}\n\nInsufficient free space: {free_gb:.2f} GB")
            else:
                st.info(f"{full_path}\n\nFree space: {free_gb:.2f} GB")
        else:
            st.code("Select a download directory.\n\nFree space: unknown", language=None)

    with col_dir_info2:
        download_dir_check = strip_quotes(download_dir)
        # Calculate full path with model subfolder if repo ID exists
        full_folder_path = download_dir_check
        if download_dir_check and repo_id and repo_id.strip():
            model_name = repo_id.strip().split('/')[-1]
            full_folder_path = str(Path(download_dir_check) / model_name)

        # Check if the full path exists, otherwise fall back to base download dir
        folder_to_open = full_folder_path if Path(full_folder_path).exists() else download_dir_check
        folder_exists = bool(folder_to_open and Path(folder_to_open).exists())

        if st.button(
            "Open Folder",
            key="check_download_folder_btn",
            use_container_width=True,
            disabled=not folder_exists,
            help="Open folder in file explorer" if folder_exists else "Path doesn't exist yet"
        ):
            if folder_exists:
                try:
                    open_folder(folder_to_open)
                    st.toast("Opened folder")
                except Exception as e:
                    st.toast(f"Could not open folder: {e}")

    # HuggingFace Authorization section (always available, auto-expands on auth errors)
    gated_repo = st.session_state.get("gated_repo_id", repo_id if repo_id else "")
    repo_url = f"https://huggingface.co/{gated_repo}" if gated_repo else "https://huggingface.co"

    col_auth = st.columns([5, 1])[0]
    with col_auth:
        with st.expander("HuggingFace Authorization", expanded=st.session_state.get("show_hf_login", False)):
            # Check if user already has a token
            existing_token = get_token()

            if existing_token:
                # Show temporary login success message
                if st.session_state.get("hf_just_logged_in", False):
                    col_msg = st.columns([5, 1])[0]
                    with col_msg:
                        st.info("You are logged in to HuggingFace")

                # User has a token but may need to accept terms for gated models
                if st.session_state.get("show_hf_login", False) and gated_repo:
                    col_msg = st.columns([5, 1])[0]
                    with col_msg:
                        st.warning(
                            f"The model **{gated_repo}** requires you to accept its terms.\n\n"
                            f"Visit [{gated_repo}]({repo_url}) and click **'Agree and access repository'**, then try downloading again."
                        )

                col_token, col_buttons = st.columns([5, 1])
                with col_token:
                    st.text_input(
                        "HuggingFace Token",
                        key="hf_token_saved",
                        value="",
                        placeholder="Token detected",
                        disabled=True,
                        help="Your token is saved on this system"
                    )

                with col_buttons:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input
                    if st.button("Delete Token", key="hf_logout_btn", use_container_width=True):
                        hf_logout()
                        # Keep expander open to show login UI
                        st.session_state.show_hf_login = True
                        st.session_state.gated_repo_id = None
                        st.session_state.hf_auth_message = None
                        print(f"{theme['info']}Logged out from HuggingFace{Style.RESET_ALL}")
                        st.rerun()
            else:
                # User needs to login
                col_msg = st.columns([5, 1])[0]
                with col_msg:
                    if st.session_state.get("show_hf_login", False) and gated_repo:
                        st.warning(
                            f"The model **{gated_repo}** requires authorization.\n\n"
                            f"1. Visit [{gated_repo}]({repo_url}) and click **'Agree and access repository'**\n"
                            f"2. Get a token from [HuggingFace Settings](https://huggingface.co/settings/tokens) (type: **Read**)\n"
                            f"3. Paste your token below (saves automatically when you press Enter)"
                        )
                    else:
                        st.info(
                            "Login to access gated models (like Llama, Gemma, etc.)\n\n"
                            "Get a token from [HuggingFace Settings](https://huggingface.co/settings/tokens) (type: **Read**)\n\n"
                        )

                def save_hf_token():
                    """Callback to save HuggingFace token when input changes"""
                    token = st.session_state.get("hf_token_input", "").strip()
                    if token:
                        try:
                            hf_login(token=token)
                            st.session_state.show_hf_login = True
                            st.session_state.gated_repo_id = None
                            st.session_state.hf_auth_message = None
                            st.session_state.hf_just_logged_in = True
                            print(f"{theme['success']}HuggingFace token saved{Style.RESET_ALL}")
                        except Exception as e:
                            st.session_state.hf_auth_message = ("error", f"Failed to save token: {e}")
                            print(f"{theme['error']}HuggingFace token save failed: {e}{Style.RESET_ALL}")

                col_token, col_buttons = st.columns([5, 1])
                with col_token:
                    hf_token = st.text_input(
                        "HuggingFace Token",
                        key="hf_token_input",
                        type="password",
                        autocomplete="off",
                        placeholder="hf_...",
                        help="Token will be saved automatically",
                        on_change=save_hf_token
                    )

                with col_buttons:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with input
                    if st.button("Save Token", key="hf_login_btn", use_container_width=True):
                        save_hf_token()
                        st.rerun()

            # Display error messages only
            if st.session_state.hf_auth_message:
                msg_type, msg_text = st.session_state.hf_auth_message
                if msg_type == "error":
                    st.error(msg_text)

    if st.button("Download", use_container_width=True):
        # Clear any previous download info and auth UI
        st.session_state.downloaded_model_path = None
        st.session_state.show_hf_login = False
        st.session_state.gated_repo_id = None
        st.session_state.hf_auth_message = None
        st.session_state.download_error = None

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

                # Print success to terminal
                print(f"\n{theme['success']}Download complete: {model_path}{Style.RESET_ALL}\n")

                # Store in session state and mark as just completed
                st.session_state.downloaded_model_path = str(model_path)
                st.session_state.download_just_completed = True
                st.rerun()

            except GatedRepoError as e:
                # Handle gated repository - show login UI and error below Download button
                st.session_state.show_hf_login = True
                st.session_state.gated_repo_id = repo_id_clean
                st.session_state.hf_auth_message = None
                st.session_state.download_error = "This model requires HuggingFace authorization. See the HuggingFace Authorization section above."
                print(f"\n{theme['error']}GatedRepoError: {e}{Style.RESET_ALL}\n")
                st.rerun()

            except Exception as e:
                # Show in Streamlit UI
                st.error(f"Error: {e}")

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

    # Show download error if one exists
    if st.session_state.get("download_error"):
        st.error(st.session_state.download_error)

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
        col_msg = st.columns([5, 1])[0]
        with col_msg:
            st.success(f"Model ready!")
        st.session_state.download_just_completed = False

    if st.session_state.get("model_path_set", False):
        col_msg = st.columns([5, 1])[0]
        with col_msg:
            st.success("Model path set in Convert & Quantize tab")
        st.session_state.model_path_set = False
