"""
Split/Merge Shards Tab

Provides functionality to split and merge sharded files (safetensors or GGUF).
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TypedDict, cast
import re
import sys
import subprocess
from collections import defaultdict
from datetime import datetime
from colorama import Style
import shutil

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, path_input_columns, get_platform_path,
    detect_all_model_files
)
from ..theme import THEME


class ShardInfo(TypedDict):
    total_expected: int
    shards_found: List[int]
    files: List[Path]
    complete: bool
    output_filename: str
    error: Optional[str]
    missing_shards: Optional[List[int]]


def analyze_shards(directory: Path, extension: str) -> Dict[str, ShardInfo]:
    """
    Analyze sharded files in a directory and group them by base model name.

    Args:
        directory: Directory to scan for shard files
        extension: File extension to look for (e.g., 'gguf' or 'safetensors')

    Returns:
        Dictionary mapping base names to shard information.
    """
    # Pattern: {base_name}-{shard_num}-of-{total_shards}.{extension}
    # Example: Qwen3-VL-4B-Instruct_F16-00001-of-00009.gguf
    pattern = re.compile(r'^(.+)-(\d+)-of-(\d+)\.' + re.escape(extension) + r'$')

    # Use a standard dict since we need specific TypedDict structure that defaultdict makes hard to type
    models: Dict[str, ShardInfo] = {}

    # Find all matching files
    for file_path in directory.glob(f"*-*-of-*.{extension}"):
        match = pattern.match(file_path.name)
        if match:
            base_name = match.group(1)
            shard_num = int(match.group(2))
            total_shards = int(match.group(3))

            # Skip files with only 1 shard (e.g., -00001-of-00001)
            if total_shards == 1:
                continue

            # Initialize model info if not present
            if base_name not in models:
                models[base_name] = {
                    'total_expected': total_shards,
                    'shards_found': [],
                    'files': [],
                    'complete': False,
                    'output_filename': f"{base_name}.{extension}",
                    'error': None,
                    'missing_shards': None
                }
            
            info = models[base_name]

            # Check for consistency
            if info['total_expected'] != total_shards:
                info['error'] = f"Inconsistent shard counts found"

            info['shards_found'].append(shard_num)
            info['files'].append(file_path)

    # Check completeness for each model
    for base_name, info in models.items():
        if not info.get('error'):
            total = int(info['total_expected']) # Ensure int for range
            expected_shards = set(range(1, total + 1))
            found_shards = set(info['shards_found'])
            info['complete'] = expected_shards == found_shards
            info['missing_shards'] = sorted(list(expected_shards - found_shards))

            # Sort shards_found list for consistent ordering across platforms
            info['shards_found'] = sorted(info['shards_found'])

            # Sort files by shard number for proper merging
            # We know pattern matches because these files were added based on the match
            def get_shard_num(p: Path) -> int:
                m = pattern.match(p.name)
                return int(m.group(2)) if m else 0

            info['files'] = sorted(info['files'], key=get_shard_num)

    return models


def render_split_merge_tab(converter, config: Dict[str, Any]):
    """
    Render the split/merge shards tab.

    Args:
        converter: GGUFConverter instance
        config: Configuration dictionary
    """
    st.header("Split/Merge Sharded Files")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")

        # Model path with Select Folder and Open Folder buttons
        cols, has_browse = path_input_columns()

        with cols[0]:
            input_dir = st.text_input(
                "Model path",
                value=config.get("split_merge_input_dir", ""),
                placeholder=get_platform_path("C:\\Models\\my-model", "/home/user/Models/my-model"),
                help="Directory containing model files."
            )

        # Save to config when changed
        if input_dir != config.get("split_merge_input_dir", ""):
            config["split_merge_input_dir"] = input_dir
            save_config(config)

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Select Folder",
                    key="browse_split_merge_model_folder_btn",
                    use_container_width=True,
                    help="Select model directory"
                ):
                    input_dir_clean = strip_quotes(input_dir)
                    initial_dir = input_dir_clean if input_dir_clean and Path(input_dir_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["split_merge_input_dir"] = selected_folder
                        save_config(config)
                        st.rerun()

        with cols[-1]:  # Last column is always the check button
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            input_dir_clean = strip_quotes(input_dir)
            input_dir_exists = bool(input_dir_clean and Path(input_dir_clean).exists())
            if st.button(
                "Open Folder",
                key="check_split_merge_model_folder_btn",
                use_container_width=True,
                disabled=not input_dir_exists,
                help="Open folder in file explorer" if input_dir_exists else "Path doesn't exist yet"
            ):
                if input_dir_exists:
                    try:
                        open_folder(input_dir_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

        # Detect available model files
        model_files_options = []
        model_files_map = {}  # Maps display name to file info

        input_dir_clean = strip_quotes(input_dir)
        input_dir_valid = bool(input_dir_clean and Path(input_dir_clean).exists())

        if not input_dir_valid:
            model_files_options = ["Model path invalid - provide valid path above"]
            dropdown_disabled = True
        elif not Path(input_dir_clean).is_dir():
            model_files_options = ["Model path must be a directory"]
            dropdown_disabled = True
        else:
            input_path = Path(input_dir_clean)

            # Detect all model files using shared utility
            detected = detect_all_model_files(input_path)

            # Build options sorted by name
            for key in sorted(detected.keys()):
                info = detected[key]
                display_name = info['display_name']
                model_files_options.append(display_name)
                model_files_map[display_name] = info

            # If no files found, show disabled message
            if not model_files_options:
                model_files_options = ["No files found"]
                dropdown_disabled = True
            else:
                dropdown_disabled = False

        # Determine default selection
        default_index = 0
        saved_selection = config.get("split_merge_selected_file", "")

        if saved_selection and saved_selection in model_files_options:
            default_index = model_files_options.index(saved_selection)

        # Model files dropdown with same column structure as path inputs
        cols, has_browse = path_input_columns()

        # Create dynamic label showing count when files are detected
        if dropdown_disabled:
            model_files_label = "Model files"
        else:
            file_count = len(model_files_options)
            model_files_label = f"Model files: {file_count} detected"

        with cols[0]:
            selected_file = st.selectbox(
                model_files_label,
                options=model_files_options,
                index=default_index,
                disabled=dropdown_disabled,
                help="Select a model file to split or merge",
                key="split_merge_file_selection"
            )

            # Auto-sync selection with config (only save the selection string, not the file info)
            if selected_file in model_files_map:
                if config.get("split_merge_selected_file") != selected_file:
                    config["split_merge_selected_file"] = selected_file
                    save_config(config)
            else:
                # Not a valid file - clear settings
                if config.get("split_merge_selected_file"):
                    config["split_merge_selected_file"] = None
                    save_config(config)

        # Refresh button in middle column (if tkinter available) or last column
        button_col = cols[1] if has_browse else cols[-1]
        with button_col:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
            if st.button("Refresh File List", key="refresh_split_merge_model_list", use_container_width=True):
                st.rerun()

        # Output directory with Select Folder and Open Folder buttons
        cols, has_browse = path_input_columns()

        with cols[0]:
            output_dir = st.text_input(
                "Output directory",
                value=config.get("split_merge_output_dir", ""),
                placeholder=get_platform_path("C:\\Models\\output", "/home/user/Models/output"),
                help="Directory where the output file(s) will be saved"
            )

        # Save to config when changed
        if output_dir != config.get("split_merge_output_dir", ""):
            config["split_merge_output_dir"] = output_dir
            save_config(config)

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Select Folder",
                    key="browse_split_merge_output_folder_btn",
                    use_container_width=True,
                    help="Select output directory"
                ):
                    output_dir_clean = strip_quotes(output_dir)
                    initial_dir = output_dir_clean if output_dir_clean and Path(output_dir_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["split_merge_output_dir"] = selected_folder
                        save_config(config)
                        st.rerun()

        with cols[-1]:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            output_dir_clean = strip_quotes(output_dir)
            output_dir_exists = bool(output_dir_clean and Path(output_dir_clean).exists())
            if st.button(
                "Open Folder",
                key="check_split_merge_output_folder_btn",
                use_container_width=True,
                disabled=not output_dir_exists,
                help="Open output folder in file explorer" if output_dir_exists else "Output folder doesn't exist yet"
            ):
                if output_dir_exists:
                    try:
                        open_folder(output_dir_clean)
                        st.toast("Opened folder")
                    except Exception as e:
                        st.toast(f"Could not open folder: {e}")

        # Show info about selected file at the bottom
        selected_file_info = None
        if selected_file in model_files_map:
            selected_file_info = model_files_map[selected_file]

            # Build detailed info
            info_lines = []

            # Add model path and filename
            info_lines.append(f"**Model path:** `{input_dir_clean}`")

            if selected_file_info['type'] == 'single':
                info_lines.append(f"**Filename:** `{selected_file_info['primary_file'].name}`")
                info_lines.append(f"**Type:** Single {selected_file_info['extension'].upper()} file")
                info_lines.append(f"**Size:** {selected_file_info['total_size_gb']:.2f} GB")
            else:
                # For sharded files, show the base name pattern
                base_name = re.sub(r'-\d+-of-\d+$', '', selected_file_info['primary_file'].stem)
                info_lines.append(f"**Filename:** `{base_name}-#####-of-#####.{selected_file_info['extension']}`")
                info_lines.append(f"**Type:** {selected_file_info['shard_count']} {selected_file_info['extension'].upper()} shards")
                info_lines.append(f"**Total size:** {selected_file_info['total_size_gb']:.2f} GB")

                # Calculate average shard size (excluding the last/odd shard)
                shard_sizes = [f.stat().st_size / (1024**3) for f in selected_file_info['files']]
                if len(shard_sizes) > 1:
                    # Average of all but the last shard
                    avg_size = sum(shard_sizes[:-1]) / len(shard_sizes[:-1])
                    last_size = shard_sizes[-1]
                    info_lines.append(f"**Avg shard size:** {avg_size:.2f} GB (excluding last shard: {last_size:.2f} GB)")
                else:
                    info_lines.append(f"**Shard size:** {shard_sizes[0]:.2f} GB")

            st.info("\n\n".join(info_lines))

    with col2:
        st.subheader("Options")

        # Operation mode selection - filtered based on file type
        def save_operation_mode():
            if "split_merge_operation_mode" in st.session_state:
                config["split_merge_operation_mode"] = st.session_state.split_merge_operation_mode
                save_config(config)

        # Determine available operations based on selected file
        if selected_file_info:
            if selected_file_info['type'] == 'single':
                # Single files can only be split
                available_operations = ["Split"]
            else:
                # Split files can be merged or resplit
                available_operations = ["Merge", "Resplit"]
        else:
            # No file selected, show all options
            available_operations = ["Split", "Merge", "Resplit"]

        # Get saved operation mode, default to first available option
        saved_mode = config.get("split_merge_operation_mode", available_operations[0])

        # If saved mode is not in available operations, use first available
        if saved_mode not in available_operations:
            default_index = 0
        else:
            default_index = available_operations.index(saved_mode)

        # Build dynamic label based on selected file
        if selected_file_info:
            if selected_file_info['type'] == 'single':
                operation_label = f"Operation: Single {selected_file_info['extension'].upper()} file"
            else:
                operation_label = f"Operation: {selected_file_info['shard_count']} {selected_file_info['extension'].upper()} shards"
        else:
            operation_label = "Operation"

        operation_mode = st.radio(
            operation_label,
            options=available_operations,
            index=default_index,
            horizontal=True,
            key="split_merge_operation_mode",
            on_change=save_operation_mode,
            help="Split: Split a single file into shards | Merge: Merge shards into a single file | Resplit: Merge then split with different shard size"
        )

        # Max shard size (disabled for merge, enabled for split/resplit)
        shard_size_disabled = (operation_mode == "Merge")

        def save_max_shard_size():
            if "split_merge_max_shard_size_input" in st.session_state:
                config["split_merge_max_shard_size_gb"] = st.session_state.split_merge_max_shard_size_input
                save_config(config)

        max_shard_size_gb = st.number_input(
            "Max size per shard (GB)",
            min_value=0.1,
            value=config.get("split_merge_max_shard_size_gb", 2.0),
            step=0.1,
            format="%.1f",
            help="Maximum size per shard file in GB. The tool will create as many shards as needed to stay under this limit." if not shard_size_disabled else "Disabled for merge operation.",
            key="split_merge_max_shard_size_input",
            on_change=save_max_shard_size,
            disabled=shard_size_disabled
        )

        # Check if input and output directories are the same
        same_directory = False
        if input_dir and output_dir:
            input_dir_clean = strip_quotes(input_dir)
            output_dir_clean = strip_quotes(output_dir)
            if input_dir_clean and output_dir_clean:
                input_path = Path(input_dir_clean)
                output_path = Path(output_dir_clean)
                if input_path.exists() and output_path.exists():
                    same_directory = input_path.resolve() == output_path.resolve()

        # Copy auxiliary files checkbox (only enabled when directories are different)
        has_file_selected = selected_file_info is not None
        enable_copy_aux = has_file_selected and not same_directory

        def save_copy_aux_files():
            widget_key = f"split_merge_copy_aux_files_{has_file_selected}_{same_directory}"
            if widget_key in st.session_state:
                config["split_merge_copy_aux_files"] = st.session_state[widget_key]
                save_config(config)

        if enable_copy_aux:
            help_text = "Copy non-model files from input to output directory (config.json, tokenizer files, imatrix files, mmproj files, etc.). Includes: .json, .txt, .md, .proto, .model, .py, .yaml, .yml, .jinja, .spm, .toml, .msgpack, .imatrix"
            checkbox_value = config.get("split_merge_copy_aux_files", False)
        else:
            help_text = "Disabled when input and output directories are the same, or when no file is selected."
            checkbox_value = False

        copy_aux_files = st.checkbox(
            "Copy auxiliary files (config, tokenizer, imatrix, mmproj, etc.)",
            value=checkbox_value,
            key=f"split_merge_copy_aux_files_{has_file_selected}_{same_directory}",
            on_change=save_copy_aux_files,
            disabled=not enable_copy_aux,
            help=help_text
        )

        if same_directory:
            st.warning("**Warning:** Input and output directories are the same. Files will be modified in place. Make sure you have backups!")

        # Check for existing files that would be overwritten
        files_to_delete = []
        aux_files_to_overwrite = []
        if selected_file_info and output_dir:
            output_dir_clean = strip_quotes(output_dir)
            if output_dir_clean and Path(output_dir_clean).exists():
                output_dir_path = Path(output_dir_clean)

                if operation_mode == "Merge":
                    # Check for single merged file
                    base_name = selected_file_info['primary_file'].stem
                    base_name = re.sub(r'-\d+-of-\d+$', '', base_name)
                    output_file = output_dir_path / f"{base_name}.{selected_file_info['extension']}"
                    if output_file.exists():
                        files_to_delete.append(output_file)

                elif operation_mode in ["Split", "Resplit"]:
                    # Check for sharded files with same base name
                    if selected_file_info['type'] == 'single':
                        base_name = selected_file_info['primary_file'].stem
                    else:
                        base_name = selected_file_info['primary_file'].stem
                        base_name = re.sub(r'-\d+-of-\d+$', '', base_name)

                    # Find all existing shards with this base name
                    pattern = f"{base_name}-*-of-*.{selected_file_info['extension']}"
                    existing_shards = list(output_dir_path.glob(pattern))
                    files_to_delete.extend(existing_shards)

                # Check for auxiliary files that will be overwritten (if copying is enabled)
                if copy_aux_files and not same_directory and input_dir:
                    input_dir_clean = strip_quotes(input_dir)
                    if input_dir_clean and Path(input_dir_clean).exists():
                        input_dir_path = Path(input_dir_clean)
                        auxiliary_extensions = {
                            '.json', '.txt', '.md', '.proto', '.model', '.py',
                            '.yaml', '.yml', '.jinja', '.spm', '.toml', '.msgpack', '.imatrix',
                        }

                        for file_path in input_dir_path.iterdir():
                            if file_path.is_file():
                                suffix_lower = file_path.suffix.lower()
                                # Check auxiliary extensions
                                if suffix_lower in auxiliary_extensions:
                                    dest_path = output_dir_path / file_path.name
                                    if dest_path.exists():
                                        aux_files_to_overwrite.append(dest_path)
                                # Also check for mmproj files
                                elif suffix_lower == '.gguf' and 'mmproj' in file_path.name.lower():
                                    dest_path = output_dir_path / file_path.name
                                    if dest_path.exists():
                                        aux_files_to_overwrite.append(dest_path)

        if files_to_delete:
            file_list = "\n".join([f"  - `{f.name}`" for f in files_to_delete])
            st.warning(f"**Warning:** The following {len(files_to_delete)} file(s) will be deleted before creating new files:\n\n{file_list}")

        if aux_files_to_overwrite:
            aux_file_list = "\n".join([f"  - `{f.name}`" for f in aux_files_to_overwrite])
            st.warning(f"**Warning:** The following {len(aux_files_to_overwrite)} auxiliary file(s) will be overwritten:\n\n{aux_file_list}")

        st.markdown("---")

        # Action button - changes based on operation mode
        if operation_mode == "Merge":
            # Merge button (only enabled if split files are selected)
            button_enabled = selected_file_info and selected_file_info['type'] == 'split'
            button_help = "Select split files to merge" if not button_enabled else None

            if st.button("Merge Shards", type="primary", use_container_width=True, disabled=not button_enabled, help=button_help):
                if not input_dir:
                    st.error("Please specify an input directory")
                elif not output_dir:
                    st.error("Please specify an output directory")
                else:
                    # Type assertion: button is only enabled when selected_file_info is not None
                    assert selected_file_info is not None

                    output_dir_path = Path(strip_quotes(output_dir))

                    if not output_dir_path.exists():
                        st.error(f"Output directory does not exist: {output_dir}")
                    else:
                        # Determine output filename (remove shard pattern)
                        base_name = selected_file_info['primary_file'].stem
                        # Remove -00001-of-00003 pattern
                        base_name = re.sub(r'-\d+-of-\d+$', '', base_name)
                        output_filename = f"{base_name}.{selected_file_info['extension']}"
                        output_path = output_dir_path / output_filename

                        # Delete existing output file if it exists
                        if output_path.exists():
                            print(f"{THEME['warning']}Deleting existing file: {output_path.name}{Style.RESET_ALL}")
                            output_path.unlink()

                        # Print header banner
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        banner_line = "=" * 80
                        print(f"\n{THEME['info']}{banner_line}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{'MERGE SHARDS'.center(80)}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{timestamp.center(80)}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{banner_line}{Style.RESET_ALL}\n")

                        try:
                            with st.spinner(f"Merging {selected_file_info['extension'].upper()} shards..."):
                                if selected_file_info['extension'] == 'gguf':
                                    merge_gguf_shards(selected_file_info['files'], output_path)
                                else:
                                    merge_safetensors_shards(selected_file_info['files'], output_path)

                            file_size = output_path.stat().st_size / (1024**3)
                            st.success(f"Successfully merged shards!")
                            st.write(f"`{output_path}` ({file_size:.2f} GB)")

                            # Copy auxiliary files if requested
                            if copy_aux_files and not same_directory:
                                input_path = Path(strip_quotes(input_dir))
                                copied_files = copy_auxiliary_files(input_path, output_dir_path)
                                if copied_files:
                                    st.write(f"+ {len(copied_files)} auxiliary file(s)")

                        except Exception as e:
                            st.error(f"Failed to merge: {e}")
                            import traceback
                            st.exception(e)

        elif operation_mode == "Split":
            # Split button (only enabled if single file is selected)
            button_enabled = (selected_file_info and
                            selected_file_info['type'] == 'single')

            if not selected_file_info:
                button_help = "Select a file to split"
            elif selected_file_info['type'] != 'single':
                button_help = "Select a single file (not shards) to split"
            else:
                button_help = None

            if st.button("Split File", type="primary", use_container_width=True, disabled=not button_enabled, help=button_help):
                if not input_dir:
                    st.error("Please specify an input directory")
                elif not output_dir:
                    st.error("Please specify an output directory")
                else:
                    # Type assertion: button is only enabled when selected_file_info is not None
                    assert selected_file_info is not None

                    output_dir_path = Path(strip_quotes(output_dir))

                    if not output_dir_path.exists():
                        st.error(f"Output directory does not exist: {output_dir}")
                    else:
                        input_file = selected_file_info['primary_file']
                        base_name = input_file.stem

                        # Delete existing shards with same base name
                        pattern = f"{base_name}-*-of-*.{selected_file_info['extension']}"
                        existing_shards = list(output_dir_path.glob(pattern))
                        if existing_shards:
                            print(f"{THEME['warning']}Deleting {len(existing_shards)} existing shard(s):{Style.RESET_ALL}")
                            for shard in existing_shards:
                                print(f"{THEME['warning']}  - {shard.name}{Style.RESET_ALL}")
                                shard.unlink()

                        # Convert GB to MB (same as convert tab)
                        split_size_mb = f"{round(max_shard_size_gb * 1000)}M"

                        # Print header banner
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        banner_line = "=" * 80
                        print(f"\n{THEME['info']}{banner_line}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{'SPLIT FILE'.center(80)}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{timestamp.center(80)}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{banner_line}{Style.RESET_ALL}\n")

                        try:
                            with st.spinner(f"Splitting {selected_file_info['extension'].upper()} file..."):
                                if selected_file_info['extension'] == 'gguf':
                                    output_files = split_gguf_file(input_file, output_dir_path, split_size_mb)
                                else:
                                    output_files = split_safetensors_file(input_file, output_dir_path, max_shard_size_gb)

                            st.success(f"Successfully split into {len(output_files)} shards!")
                            for output_path in output_files:
                                file_size = output_path.stat().st_size / (1024**3)
                                st.write(f"`{output_path.name}` ({file_size:.2f} GB)")

                            # Copy auxiliary files if requested
                            if copy_aux_files and not same_directory:
                                input_path = Path(strip_quotes(input_dir))
                                copied_files = copy_auxiliary_files(input_path, output_dir_path)
                                if copied_files:
                                    st.write(f"+ {len(copied_files)} auxiliary file(s)")

                        except Exception as e:
                            st.error(f"Failed to split: {e}")
                            import traceback
                            st.exception(e)

        else:  # Resplit
            # Resplit button (only enabled if split files are selected)
            button_enabled = (selected_file_info and
                            selected_file_info['type'] == 'split')

            if not selected_file_info:
                button_help = "Select sharded files to resplit"
            elif selected_file_info['type'] != 'split':
                button_help = "Select split files (shards) to resplit"
            else:
                button_help = None

            if st.button("Resplit Shards", type="primary", use_container_width=True, disabled=not button_enabled, help=button_help):
                if not input_dir:
                    st.error("Please specify an input directory")
                elif not output_dir:
                    st.error("Please specify an output directory")
                else:
                    # Type assertion: button is only enabled when selected_file_info is not None
                    assert selected_file_info is not None

                    output_dir_path = Path(strip_quotes(output_dir))

                    if not output_dir_path.exists():
                        st.error(f"Output directory does not exist: {output_dir}")
                    else:
                        # Convert GB to MB (same as convert tab)
                        split_size_mb = f"{round(max_shard_size_gb * 1000)}M"

                        # Print header banner
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        banner_line = "=" * 80
                        print(f"\n{THEME['info']}{banner_line}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{'RESPLIT SHARDS'.center(80)}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{timestamp.center(80)}{Style.RESET_ALL}")
                        print(f"{THEME['info']}{banner_line}{Style.RESET_ALL}\n")

                        try:
                            with st.spinner(f"Resplitting {selected_file_info['extension'].upper()} shards..."):
                                # Deletion happens inside resplit function after merge but before split
                                if selected_file_info['extension'] == 'gguf':
                                    output_files = resplit_gguf_shards(
                                        selected_file_info['files'],
                                        output_dir_path,
                                        split_size_mb
                                    )
                                else:
                                    output_files = resplit_safetensors_shards(
                                        selected_file_info['files'],
                                        output_dir_path,
                                        max_shard_size_gb
                                    )

                            st.success(f"Successfully resplit into {len(output_files)} shards!")
                            for output_path in output_files:
                                file_size = output_path.stat().st_size / (1024**3)
                                st.write(f"`{output_path.name}` ({file_size:.2f} GB)")

                            # Copy auxiliary files if requested
                            if copy_aux_files and not same_directory:
                                input_path = Path(strip_quotes(input_dir))
                                copied_files = copy_auxiliary_files(input_path, output_dir_path)
                                if copied_files:
                                    st.write(f"+ {len(copied_files)} auxiliary file(s)")

                        except Exception as e:
                            st.error(f"Failed to resplit: {e}")
                            import traceback
                            st.exception(e)


def copy_auxiliary_files(input_dir: Path, output_dir: Path) -> List[Path]:
    """
    Copy auxiliary files from input to output directory.
    Includes config files, tokenizer files, and mmproj files for vision models.
    Excludes main model weight files.

    Args:
        input_dir: Source directory containing auxiliary files
        output_dir: Destination directory

    Returns:
        List of copied file paths
    """
    # Extensions to copy (auxiliary files)
    auxiliary_extensions = {
        '.json',
        '.txt',
        '.md',
        '.proto',
        '.model',
        '.py',
        '.yaml',
        '.yml',
        '.jinja',
        '.spm',
        '.toml',
        '.msgpack',
        '.imatrix',
    }

    # Extensions to exclude (model weight files)
    # Note: .gguf files with 'mmproj' in the name are copied (vision model projectors)
    exclude_extensions = {
        '.safetensors',
        '.bin',  # PyTorch .bin files
        '.pth',
        '.pt',
    }

    copied_files = []

    # Iterate through all files in input directory
    for file_path in input_dir.iterdir():
        if file_path.is_file():
            suffix_lower = file_path.suffix.lower()

            # Copy files with auxiliary extensions
            if suffix_lower in auxiliary_extensions:
                dest_path = output_dir / file_path.name
                print(f"{THEME['info']}Copying {file_path.name}...{Style.RESET_ALL}")
                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)

            # Also copy mmproj files (vision model projectors)
            # These are .gguf files but should be copied as auxiliary files
            elif suffix_lower == '.gguf' and 'mmproj' in file_path.name.lower():
                dest_path = output_dir / file_path.name
                print(f"{THEME['info']}Copying {file_path.name} (vision projector)...{Style.RESET_ALL}")
                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)

    if copied_files:
        print(f"{THEME['success']}Copied {len(copied_files)} auxiliary file(s){Style.RESET_ALL}\n")
    else:
        print(f"{THEME['info']}No auxiliary files found to copy{Style.RESET_ALL}\n")

    return copied_files


def resplit_gguf_shards(shard_files: List[Path], output_dir: Path, split_size: str) -> List[Path]:
    """
    Resplit GGUF shards by merging them and splitting with a new shard size.

    Args:
        shard_files: List of GGUF shard file paths (already sorted)
        output_dir: Directory where new split files will be saved
        split_size: Size limit per shard (e.g., "2000M" for 2GB)

    Returns:
        List of newly created shard files
    """
    print(f"{THEME['info']}Resplitting {len(shard_files)} shard(s) with new size {split_size}:{Style.RESET_ALL}\n")

    # Extract base name from first shard (remove -00001-of-00003 pattern)
    base_name = re.sub(r'-\d+-of-\d+$', '', shard_files[0].stem)

    # Create temp file in output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_merged_file = output_dir / f"{base_name}_temp_{timestamp}.gguf"

    try:
        # Step 1: Merge shards into temporary file
        print(f"{THEME['info']}Step 1: Merging shards into temporary file...{Style.RESET_ALL}")
        print(f"{THEME['info']}Temp file: {temp_merged_file.name}{Style.RESET_ALL}")
        merge_gguf_shards(shard_files, temp_merged_file)

        # Step 1.5: Delete existing shards in output directory (now safe since we have the merged temp file)
        pattern = f"{base_name}-*-of-*.gguf"
        existing_shards = list(output_dir.glob(pattern))
        if existing_shards:
            print(f"{THEME['warning']}Deleting {len(existing_shards)} existing shard(s):{Style.RESET_ALL}")
            for shard in existing_shards:
                print(f"{THEME['warning']}  - {shard.name}{Style.RESET_ALL}")
                shard.unlink()

        # Step 2: Split the merged file with new size (use original base name, not temp name)
        print(f"{THEME['info']}Step 2: Splitting with new shard size...{Style.RESET_ALL}")
        output_files = split_gguf_file(temp_merged_file, output_dir, split_size, output_base_name=base_name)

        # Step 3: Clean up temp file
        print(f"{THEME['info']}Cleaning up temporary file...{Style.RESET_ALL}")
        temp_merged_file.unlink()

        print(f"{THEME['success']}Resplit complete!{Style.RESET_ALL}\n")
        return output_files

    except Exception as e:
        # Clean up temp file on error if it exists
        if temp_merged_file.exists():
            print(f"{THEME['warning']}Cleaning up temporary file after error...{Style.RESET_ALL}")
            temp_merged_file.unlink()
        raise


def split_gguf_file(input_file: Path, output_dir: Path, split_size: str, output_base_name: Optional[str] = None) -> List[Path]:
    """
    Split a GGUF file into shards using llama-gguf-split.

    Args:
        input_file: Path to the GGUF file to split
        output_dir: Directory where split files will be saved
        split_size: Size limit per shard (e.g., "2000M" for 2GB)
        output_base_name: Optional base name for output files (defaults to input file stem)

    Returns:
        List of created shard files
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Print file info to terminal
    print(f"{THEME['info']}Splitting {input_file.name} into shards of max {split_size} each:{Style.RESET_ALL}\n")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find llama-gguf-split executable
    bin_dir = Path(__file__).parent.parent.parent / "bin"
    gguf_split_exe = bin_dir / "llama-gguf-split.exe" if sys.platform == "win32" else bin_dir / "llama-gguf-split"

    if not gguf_split_exe.exists():
        raise FileNotFoundError(f"llama-gguf-split not found at {gguf_split_exe}")

    # Output files will be in the output directory
    # llama-gguf-split creates files like: basename-00001-of-00003.gguf
    # Use provided base name or default to input file stem
    base_name = output_base_name if output_base_name else input_file.stem
    output_base = output_dir / base_name

    # Run llama-gguf-split --split
    cmd = [str(gguf_split_exe), "--split", "--split-max-size", split_size, str(input_file), str(output_base)]

    print(f"{THEME['highlight']}Running: {' '.join(cmd)}{Style.RESET_ALL}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"llama-gguf-split failed:\n{result.stderr}")

    if result.stdout:
        # Print tool output to terminal
        for line in result.stdout.strip().split('\n'):
            print(f"{THEME['metadata']}{line}{Style.RESET_ALL}")

    # Find the created shard files
    pattern = f"{base_name}-*-of-*.gguf"
    output_files = sorted(output_dir.glob(pattern))

    if not output_files:
        raise RuntimeError(f"No output files found matching pattern: {pattern}")

    print(f"{THEME['success']}Successfully split into {len(output_files)} shard(s){Style.RESET_ALL}\n")

    return output_files


def merge_gguf_shards(shard_files: List[Path], output_file: Path):
    """
    Merge GGUF sharded files into a single file using llama-gguf-split.

    Args:
        shard_files: List of GGUF shard file paths (already sorted)
        output_file: Output path for merged file
    """
    if not shard_files:
        raise ValueError("No GGUF shard files provided")

    # Print model info to terminal
    print(f"{THEME['info']}Merging {output_file.name} from {len(shard_files)} shard(s):{Style.RESET_ALL}")
    for shard in shard_files:
        print(f"{THEME['metadata']}  {shard.name}{Style.RESET_ALL}")
    print()

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Delete existing file if it exists (llama-gguf-split won't overwrite)
    if output_file.exists():
        print(f"{THEME['warning']}Deleting existing file: {output_file.name}{Style.RESET_ALL}")
        output_file.unlink()

    # Find llama-gguf-split executable
    bin_dir = Path(__file__).parent.parent.parent / "bin"
    gguf_split_exe = bin_dir / "llama-gguf-split.exe" if sys.platform == "win32" else bin_dir / "llama-gguf-split"

    if not gguf_split_exe.exists():
        raise FileNotFoundError(f"llama-gguf-split not found at {gguf_split_exe}")

    # Use the first shard as input (llama-gguf-split will find the others)
    first_shard = shard_files[0]

    # Run llama-gguf-split --merge
    cmd = [str(gguf_split_exe), "--merge", str(first_shard), str(output_file)]

    print(f"{THEME['highlight']}Running: {' '.join(cmd)}{Style.RESET_ALL}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"llama-gguf-split failed:\n{result.stderr}")

    if result.stdout:
        # Print tool output to terminal
        for line in result.stdout.strip().split('\n'):
            print(f"{THEME['metadata']}{line}{Style.RESET_ALL}")

    print(f"{THEME['success']}Successfully merged: {output_file.name}{Style.RESET_ALL}\n")


def merge_safetensors_shards(shard_files: List[Path], output_file: Path):
    """
    Merge safetensors sharded files into a single file.

    Args:
        shard_files: List of safetensors shard file paths (already sorted)
        output_file: Output path for merged file
    """
    try:
        from safetensors.torch import load_file, save_file
        import torch
    except ImportError:
        raise ImportError(
            "safetensors and torch are required to merge safetensors files.\n"
            "Install them with: pip install safetensors torch"
        )

    if not shard_files:
        raise ValueError("No safetensors shard files provided")

    # Print model info to terminal
    print(f"{THEME['info']}Merging {output_file.name} from {len(shard_files)} shard(s):{Style.RESET_ALL}")
    for shard in shard_files:
        print(f"{THEME['metadata']}  {shard.name}{Style.RESET_ALL}")
    print()

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load and merge all shards
    merged_tensors = {}

    for shard in shard_files:
        print(f"{THEME['info']}Loading {shard.name}...{Style.RESET_ALL}")
        tensors = load_file(str(shard))
        merged_tensors.update(tensors)

    print(f"{THEME['info']}Saving merged file...{Style.RESET_ALL}")
    save_file(merged_tensors, str(output_file))

    print(f"{THEME['success']}Successfully merged: {output_file.name}{Style.RESET_ALL}\n")


def split_safetensors_file(input_file: Path, output_dir: Path, max_shard_size_gb: float) -> List[Path]:
    """
    Split a safetensors file into shards.

    Args:
        input_file: Path to the safetensors file to split
        output_dir: Directory where split files will be saved
        max_shard_size_gb: Maximum size per shard in GB

    Returns:
        List of created shard files
    """
    try:
        from safetensors.torch import load_file, save_file
        import torch
    except ImportError:
        raise ImportError(
            "safetensors and torch are required to split safetensors files.\n"
            "Install them with: pip install safetensors torch"
        )

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"{THEME['info']}Splitting {input_file.name} into shards of max {max_shard_size_gb:.2f} GB each:{Style.RESET_ALL}\n")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tensors
    print(f"{THEME['info']}Loading tensors...{Style.RESET_ALL}")
    tensors = load_file(str(input_file))

    # Calculate tensor sizes
    max_shard_size_bytes = max_shard_size_gb * 1024**3
    tensor_sizes = {}
    for name, tensor in tensors.items():
        tensor_sizes[name] = tensor.nbytes

    # Group tensors into shards
    shards = []
    current_shard = {}
    current_size = 0

    for name in sorted(tensor_sizes.keys()):  # Sort for deterministic order
        tensor_size = tensor_sizes[name]

        # If adding this tensor would exceed the limit, start a new shard
        if current_shard and current_size + tensor_size > max_shard_size_bytes:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[name] = tensors[name]
        current_size += tensor_size

    # Add the last shard if not empty
    if current_shard:
        shards.append(current_shard)

    # Write shards
    base_name = input_file.stem
    output_files = []

    for i, shard_tensors in enumerate(shards, start=1):
        shard_filename = f"{base_name}-{i:05d}-of-{len(shards):05d}.safetensors"
        shard_path = output_dir / shard_filename

        print(f"{THEME['info']}Writing shard {i}/{len(shards)}: {shard_filename}{Style.RESET_ALL}")
        save_file(shard_tensors, str(shard_path))
        output_files.append(shard_path)

    print(f"{THEME['success']}Successfully split into {len(output_files)} shard(s){Style.RESET_ALL}\n")

    return output_files


def resplit_safetensors_shards(shard_files: List[Path], output_dir: Path, max_shard_size_gb: float) -> List[Path]:
    """
    Resplit safetensors shards by merging them and splitting with a new shard size.

    Args:
        shard_files: List of safetensors shard file paths (already sorted)
        output_dir: Directory where new split files will be saved
        max_shard_size_gb: Maximum size per shard in GB

    Returns:
        List of newly created shard files
    """
    print(f"{THEME['info']}Resplitting {len(shard_files)} shard(s) with new size {max_shard_size_gb:.2f} GB:{Style.RESET_ALL}\n")

    # Extract base name from first shard (remove -00001-of-00003 pattern)
    base_name = re.sub(r'-\d+-of-\d+$', '', shard_files[0].stem)

    # Create temp file in output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_merged_file = output_dir / f"{base_name}_temp_{timestamp}.safetensors"

    try:
        # Step 1: Merge shards into temporary file
        print(f"{THEME['info']}Step 1: Merging shards into temporary file...{Style.RESET_ALL}")
        print(f"{THEME['info']}Temp file: {temp_merged_file.name}{Style.RESET_ALL}")
        merge_safetensors_shards(shard_files, temp_merged_file)

        # Step 1.5: Delete existing shards in output directory (now safe since we have the merged temp file)
        pattern = f"{base_name}-*-of-*.safetensors"
        existing_shards = list(output_dir.glob(pattern))
        if existing_shards:
            print(f"{THEME['warning']}Deleting {len(existing_shards)} existing shard(s):{Style.RESET_ALL}")
            for shard in existing_shards:
                print(f"{THEME['warning']}  - {shard.name}{Style.RESET_ALL}")
                shard.unlink()

        # Step 2: Split the merged file with new size
        print(f"{THEME['info']}Step 2: Splitting with new shard size...{Style.RESET_ALL}")
        output_files = split_safetensors_file(temp_merged_file, output_dir, max_shard_size_gb)

        # Step 3: Clean up temp file
        print(f"{THEME['info']}Cleaning up temporary file...{Style.RESET_ALL}")
        temp_merged_file.unlink()

        print(f"{THEME['success']}Resplit complete!{Style.RESET_ALL}\n")
        return output_files

    except Exception as e:
        # Clean up temp file on error if it exists
        if temp_merged_file.exists():
            print(f"{THEME['warning']}Cleaning up temporary file after error...{Style.RESET_ALL}")
            temp_merged_file.unlink()
        raise
