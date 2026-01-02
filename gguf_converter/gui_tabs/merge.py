"""
Merge Shards Tab

Provides functionality to merge sharded files (safetensors or GGUF) into a single file.
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import re
from collections import defaultdict

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, path_input_columns, get_platform_path
)


def analyze_shards(directory: Path, extension: str) -> Dict[str, Dict[str, Any]]:
    """
    Analyze sharded files in a directory and group them by base model name.

    Args:
        directory: Directory to scan for shard files
        extension: File extension to look for (e.g., 'gguf' or 'safetensors')

    Returns:
        Dictionary mapping base names to shard information:
        {
            'base_name': {
                'total_expected': int,
                'shards_found': [shard_numbers],
                'files': [Path objects],
                'complete': bool,
                'output_filename': str
            }
        }
    """
    # Pattern: {base_name}-{shard_num}-of-{total_shards}.{extension}
    # Example: Qwen3-VL-4B-Instruct_F16-00001-of-00009.gguf
    pattern = re.compile(r'^(.+)-(\d+)-of-(\d+)\.' + re.escape(extension) + r'$')

    models = defaultdict(lambda: {
        'total_expected': 0,
        'shards_found': [],
        'files': [],
        'complete': False,
        'output_filename': ''
    })

    # Find all matching files
    for file_path in directory.glob(f"*-*-of-*.{extension}"):
        match = pattern.match(file_path.name)
        if match:
            base_name = match.group(1)
            shard_num = int(match.group(2))
            total_shards = int(match.group(3))

            # Initialize or update model info
            if models[base_name]['total_expected'] == 0:
                models[base_name]['total_expected'] = total_shards
                models[base_name]['output_filename'] = f"{base_name}.{extension}"
            elif models[base_name]['total_expected'] != total_shards:
                # Inconsistent total counts - this is an error
                models[base_name]['error'] = f"Inconsistent shard counts found"

            models[base_name]['shards_found'].append(shard_num)
            models[base_name]['files'].append(file_path)

    # Check completeness for each model
    for base_name, info in models.items():
        if 'error' not in info:
            expected_shards = set(range(1, info['total_expected'] + 1))
            found_shards = set(info['shards_found'])
            info['complete'] = expected_shards == found_shards
            info['missing_shards'] = sorted(expected_shards - found_shards)

            # Sort files by shard number for proper merging
            info['files'] = sorted(info['files'], key=lambda p: int(pattern.match(p.name).group(2)))

    return dict(models)


def render_merge_tab(converter, config: Dict[str, Any]):
    """
    Render the merge shards tab.

    Args:
        converter: GGUFConverter instance
        config: Configuration dictionary
    """
    st.header("Merge Sharded Files")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")

        # Input directory with Select Folder and Open Folder buttons
        cols, has_browse = path_input_columns()

        with cols[0]:
            input_dir = st.text_input(
                "Input directory",
                value=config.get("merge_input_dir", ""),
                placeholder=get_platform_path("C:\\Models\\sharded-model", "/home/user/Models/sharded-model"),
                help="Directory containing sharded model files (e.g., model-00001-of-00002.gguf or model-00001-of-00002.safetensors)"
            )

        # Save to config when changed
        if input_dir != config.get("merge_input_dir", ""):
            config["merge_input_dir"] = input_dir
            save_config(config)

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Select Folder",
                    key="browse_merge_input_folder_btn",
                    use_container_width=True,
                    help="Select input directory"
                ):
                    input_dir_clean = strip_quotes(input_dir)
                    initial_dir = input_dir_clean if input_dir_clean and Path(input_dir_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["merge_input_dir"] = selected_folder
                        save_config(config)
                        st.rerun()

        with cols[-1]:  # Last column is always the check button
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            input_dir_clean = strip_quotes(input_dir)
            input_dir_exists = bool(input_dir_clean and Path(input_dir_clean).exists())
            if st.button(
                "Open Folder",
                key="check_merge_input_folder_btn",
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

        # Auto-detect file type and show what was found
        file_type = None
        models_info = {}
        total_shards = 0

        if input_dir:
            input_path = Path(strip_quotes(input_dir))
            if input_path.exists() and input_path.is_dir():
                # Analyze GGUF shards
                gguf_models = analyze_shards(input_path, "gguf")
                # Analyze safetensors shards
                safetensors_models = analyze_shards(input_path, "safetensors")

                if gguf_models:
                    file_type = "GGUF"
                    models_info = gguf_models
                    total_shards = sum(len(info['shards_found']) for info in gguf_models.values())
                elif safetensors_models:
                    file_type = "Safetensors"
                    models_info = safetensors_models
                    total_shards = sum(len(info['shards_found']) for info in safetensors_models.values())

                # Display detailed information
                if models_info:
                    complete_models = [name for name, info in models_info.items() if info.get('complete', False)]
                    incomplete_models = [name for name, info in models_info.items() if not info.get('complete', False)]

                    info_lines = [f"**Total shards found:** {total_shards}"]
                    info_lines.append(f"**Unique models detected:** {len(models_info)}")

                    if complete_models:
                        info_lines.append(f"**Complete models:** {len(complete_models)}")
                    if incomplete_models:
                        info_lines.append(f"**Incomplete models:** {len(incomplete_models)}")

                    st.info("\n\n".join(info_lines))

                    # Show details for each model
                    for base_name, info in sorted(models_info.items()):
                        if info.get('complete', False):
                            status = "Complete"
                            status_prefix = "[OK]"
                        else:
                            status = "Incomplete"
                            status_prefix = "[!]"

                        with st.expander(f"{status_prefix} {base_name} - {status} ({len(info['shards_found'])}/{info['total_expected']} shards)"):
                            st.write(f"**Expected shards:** {info['total_expected']}")
                            st.write(f"**Found shards:** {len(info['shards_found'])}")

                            if 'error' in info:
                                st.error(info['error'])
                            elif not info['complete']:
                                missing = info.get('missing_shards', [])
                                if missing:
                                    missing_str = ", ".join(str(s) for s in missing)
                                    st.warning(f"Missing shard(s): {missing_str}")
                            else:
                                st.success(f"All shards present. Ready to merge into: `{info['output_filename']}`")

                            # List all found files
                            if st.checkbox(f"Show shard files", key=f"show_files_{base_name}"):
                                for file_path in info['files']:
                                    st.text(f"  {file_path.name}")
                else:
                    st.warning("No sharded files found in directory (looking for *-*-of-*.gguf or *-*-of-*.safetensors)")

    with col2:
        st.subheader("Output")

        # Output directory with Select Folder and Open Folder buttons
        cols, has_browse = path_input_columns()

        with cols[0]:
            output_dir = st.text_input(
                "Output directory",
                value=config.get("merge_output_dir", ""),
                placeholder=get_platform_path("C:\\Models\\output", "/home/user/Models/output"),
                help="Directory where the merged file will be saved"
            )

        # Save to config when changed
        if output_dir != config.get("merge_output_dir", ""):
            config["merge_output_dir"] = output_dir
            save_config(config)

        if has_browse:
            with cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # Align with input
                if st.button(
                    "Select Folder",
                    key="browse_merge_output_folder_btn",
                    use_container_width=True,
                    help="Select output directory"
                ):
                    output_dir_clean = strip_quotes(output_dir)
                    initial_dir = output_dir_clean if output_dir_clean and Path(output_dir_clean).exists() else None
                    selected_folder = browse_folder(initial_dir)
                    if selected_folder:
                        config["merge_output_dir"] = selected_folder
                        save_config(config)
                        st.rerun()

        with cols[-1]:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with input
            output_dir_clean = strip_quotes(output_dir)
            output_dir_exists = bool(output_dir_clean and Path(output_dir_clean).exists())
            if st.button(
                "Open Folder",
                key="check_merge_output_folder_btn",
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

        # Model selection if multiple models detected
        selected_model = None
        output_filename = None

        if models_info:
            complete_models = {name: info for name, info in models_info.items() if info.get('complete', False)}

            if len(complete_models) == 1:
                # Only one complete model, auto-select it
                selected_model = list(complete_models.keys())[0]
                output_filename = complete_models[selected_model]['output_filename']
                st.info(f"Output file will be named: `{output_filename}`")
            elif len(complete_models) > 1:
                # Multiple complete models, let user choose
                st.subheader("Model Selection")
                model_options = [f"{name} ({info['total_expected']} shards)" for name, info in sorted(complete_models.items())]
                model_names = list(sorted(complete_models.keys()))

                selected_index = st.selectbox(
                    "Select model to merge",
                    range(len(model_options)),
                    format_func=lambda i: model_options[i],
                    key="model_selection"
                )

                if selected_index is not None:
                    selected_model = model_names[selected_index]
                    output_filename = complete_models[selected_model]['output_filename']
                    st.info(f"Output file will be named: `{output_filename}`")
            elif models_info:
                # Only incomplete models found
                st.warning("No complete models found. All models are missing one or more shards.")

        st.markdown("---")

        # Merge button
        if st.button("Merge Shards", type="primary", use_container_width=True):
            if not input_dir:
                st.error("Please specify an input directory")
                return

            if not output_dir:
                st.error("Please specify an output directory")
                return

            if not file_type:
                st.error("No sharded files detected in the input directory")
                return

            if not selected_model:
                st.error("No complete model selected for merging")
                return

            if not output_filename:
                st.error("Could not determine output filename from shards")
                return

            output_dir_path = Path(strip_quotes(output_dir))
            output_path = output_dir_path / output_filename

            if not output_dir_path.exists():
                st.error(f"Output directory does not exist: {output_dir}")
                return

            # Get the shard files for the selected model
            complete_models = {name: info for name, info in models_info.items() if info.get('complete', False)}
            model_info = complete_models[selected_model]
            shard_files = model_info['files']

            try:
                with st.spinner(f"Merging {file_type} shards for {selected_model}..."):
                    if file_type == "GGUF":
                        merge_gguf_shards(shard_files, output_path)
                    else:
                        merge_safetensors_shards(shard_files, output_path)

                st.success(f"Successfully merged shards to: {output_path}")

                # Show file info
                if output_path.exists():
                    file_size = output_path.stat().st_size / (1024**3)
                    st.info(f"Output file size: {file_size:.2f} GB")
                    st.code(str(output_path), language=None)

            except Exception as e:
                st.error(f"Error merging shards: {e}")
                import traceback
                st.exception(e)


def merge_gguf_shards(shard_files: List[Path], output_file: Path):
    """
    Merge GGUF sharded files into a single file.

    Args:
        shard_files: List of GGUF shard file paths (already sorted)
        output_file: Output path for merged file
    """
    if not shard_files:
        raise ValueError("No GGUF shard files provided")

    st.write(f"Merging {len(shard_files)} shard file(s):")
    for shard in shard_files:
        st.write(f"  - {shard.name}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Merge by concatenating files
    with open(output_file, 'wb') as outfile:
        for shard in shard_files:
            with open(shard, 'rb') as infile:
                # Copy in chunks to handle large files
                chunk_size = 1024 * 1024 * 100  # 100MB chunks
                while True:
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    outfile.write(chunk)


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

    st.write(f"Merging {len(shard_files)} shard file(s):")
    for shard in shard_files:
        st.write(f"  - {shard.name}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load and merge all shards
    merged_tensors = {}

    for shard in shard_files:
        st.write(f"Loading {shard.name}...")
        tensors = load_file(str(shard))
        merged_tensors.update(tensors)

    st.write(f"Saving merged file to {output_file.name}...")
    save_file(merged_tensors, str(output_file))
