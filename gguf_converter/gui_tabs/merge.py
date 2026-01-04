"""
Merge Shards Tab

Provides functionality to merge sharded files (safetensors or GGUF) into a single file.
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import re
import sys
import subprocess
from collections import defaultdict
from datetime import datetime
from colorama import Style

from ..gui_utils import (
    strip_quotes, open_folder, browse_folder,
    save_config, path_input_columns, get_platform_path
)
from ..theme import THEME as theme


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

            # Skip files with only 1 shard (e.g., -00001-of-00001)
            if total_shards == 1:
                continue

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
                    complete_models_list = [name for name, info in models_info.items() if info.get('complete', False)]
                    incomplete_models = [name for name, info in models_info.items() if not info.get('complete', False)]

                    info_lines = [f"**Total shards found:** {total_shards}"]
                    info_lines.append(f"**Unique models detected:** {len(models_info)}")

                    if complete_models_list:
                        info_lines.append(f"**Complete models:** {len(complete_models_list)}")
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
                            if 'error' in info:
                                st.error(info['error'])
                            elif not info['complete']:
                                st.warning(f"{len(info['shards_found'])}/{info['total_expected']} shards present. Will not merge.")
                            else:
                                st.success(f"{len(info['shards_found'])}/{info['total_expected']} shards present. Ready to merge into: `{info['output_filename']}`")

                            # List all found files
                            for file_path in info['files']:
                                st.text(f"  {file_path.name}")
                else:
                    st.warning("No sharded files found in directory (looking for xxxxx-of-xxxxx.gguf or .safetensors)")

    with col2:
        st.subheader("Model Selection")

        # Model selection if multiple models detected
        selected_model = None
        selected_models = []
        output_filename = None

        if models_info:
            complete_models = {name: info for name, info in models_info.items() if info.get('complete', False)}

            if len(complete_models) == 1:
                # Only one complete model, auto-select it
                selected_model = list(complete_models.keys())[0]
                selected_models = [selected_model]
                output_filename = complete_models[selected_model]['output_filename']
                st.info(f"Output file will be named: `{output_filename}`")
            elif len(complete_models) > 1:
                # Multiple complete models, let user choose
                # Add "All complete models" option as first choice
                model_options = [f"All complete models ({len(complete_models)} models)"]
                model_options.extend([f"{name} ({info['total_expected']} shards)" for name, info in sorted(complete_models.items())])
                model_names = [None] + list(sorted(complete_models.keys()))  # None for "all models"

                selected_index = st.selectbox(
                    "Select model to merge",
                    range(len(model_options)),
                    format_func=lambda i: model_options[i],
                    key="model_selection"
                )

                if selected_index == 0:
                    # All models selected
                    selected_model = None
                    selected_models = list(sorted(complete_models.keys()))
                    model_list = "\n".join([f"  - `{complete_models[name]['output_filename']}`" for name in selected_models])
                    st.info(f"Will merge all {len(selected_models)} complete models:\n\n{model_list}")
                elif selected_index is not None:
                    selected_model = model_names[selected_index]
                    selected_models = [selected_model]
                    output_filename = complete_models[selected_model]['output_filename']
                    st.info(f"Output file will be named: `{output_filename}`")
            else:
                # Only incomplete models found
                st.selectbox(
                    "Select model to merge",
                    options=["No complete models available"],
                    disabled=True,
                    key="model_selection_disabled"
                )
                st.warning("No complete models found. All models are missing one or more shards.")
        else:
            # No models found at all
            st.selectbox(
                "Select model to merge",
                options=["No models detected"],
                disabled=True,
                key="model_selection_empty"
            )

        # Check for existing files that would be overwritten
        existing_files = []
        if selected_models and output_dir:
            output_dir_path = Path(strip_quotes(output_dir))
            if output_dir_path.exists():
                complete_models = {name: info for name, info in models_info.items() if info.get('complete', False)}
                for model_name in selected_models:
                    model_info = complete_models[model_name]
                    output_filename = model_info['output_filename']
                    output_path = output_dir_path / output_filename
                    if output_path.exists():
                        existing_files.append(output_filename)

        if existing_files:
            file_list = "\n".join([f"  - `{filename}`" for filename in existing_files])
            st.warning(f"**Warning:** The following {len(existing_files)} file(s) already exist and will be overwritten:\n\n{file_list}")

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

            if not selected_models:
                st.error("No complete model selected for merging")
                return

            output_dir_path = Path(strip_quotes(output_dir))

            if not output_dir_path.exists():
                st.error(f"Output directory does not exist: {output_dir}")
                return

            # Get the shard files for the selected model(s)
            complete_models = {name: info for name, info in models_info.items() if info.get('complete', False)}

            # Print header banner once at the start
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            banner_line = "=" * 80
            print(f"\n{theme['info']}{banner_line}{Style.RESET_ALL}")
            print(f"{theme['info']}{'MERGE SHARDS'.center(80)}{Style.RESET_ALL}")
            print(f"{theme['info']}{timestamp.center(80)}{Style.RESET_ALL}")
            print(f"{theme['info']}{banner_line}{Style.RESET_ALL}\n")

            merged_files = []
            errors = []

            for model_name in selected_models:
                model_info = complete_models[model_name]
                shard_files = model_info['files']
                output_filename = model_info['output_filename']
                output_path = output_dir_path / output_filename

                try:
                    with st.spinner(f"Merging {file_type} shards for {model_name}..."):
                        if file_type == "GGUF":
                            merge_gguf_shards(shard_files, output_path)
                        else:
                            merge_safetensors_shards(shard_files, output_path)

                    merged_files.append(output_path)

                except Exception as e:
                    errors.append((model_name, e))

            # Show results
            if merged_files:
                st.success(f"Successfully merged {len(merged_files)} model(s)")
                for output_path in merged_files:
                    file_size = output_path.stat().st_size / (1024**3)
                    st.write(f"`{output_path}` ({file_size:.2f} GB)")

            if errors:
                st.error(f"Failed to merge {len(errors)} model(s)")
                for model_name, error in errors:
                    st.error(f"**{model_name}:** {error}")
                    import traceback
                    st.exception(error)


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
    print(f"{theme['info']}Merging {output_file.name} from {len(shard_files)} shard(s):{Style.RESET_ALL}")
    for shard in shard_files:
        print(f"{theme['metadata']}  {shard.name}{Style.RESET_ALL}")
    print()

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Delete existing file if it exists (llama-gguf-split won't overwrite)
    if output_file.exists():
        print(f"{theme['warning']}Deleting existing file: {output_file.name}{Style.RESET_ALL}")
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

    print(f"{theme['highlight']}Running: {' '.join(cmd)}{Style.RESET_ALL}\n")

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
            print(f"{theme['metadata']}{line}{Style.RESET_ALL}")

    print(f"{theme['success']}Successfully merged: {output_file.name}{Style.RESET_ALL}\n")


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
    print(f"{theme['info']}Merging {output_file.name} from {len(shard_files)} shard(s):{Style.RESET_ALL}")
    for shard in shard_files:
        print(f"{theme['metadata']}  {shard.name}{Style.RESET_ALL}")
    print()

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load and merge all shards
    merged_tensors = {}

    for shard in shard_files:
        print(f"{theme['info']}Loading {shard.name}...{Style.RESET_ALL}")
        tensors = load_file(str(shard))
        merged_tensors.update(tensors)

    print(f"{theme['info']}Saving merged file...{Style.RESET_ALL}")
    save_file(merged_tensors, str(output_file))

    print(f"{theme['success']}Successfully merged: {output_file.name}{Style.RESET_ALL}\n")
