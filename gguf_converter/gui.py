"""
Streamlit GUI for GGUF Converter
"""

import streamlit as st
from pathlib import Path
import sys
import json

# Handle both direct execution and module import
try:
    from .converter import GGUFConverter
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gguf_converter.converter import GGUFConverter


# Config file location
CONFIG_FILE = Path.home() / ".gguf_converter_config.json"


def get_default_config():
    """Get default configuration"""
    return {
        # Sidebar settings
        "verbose": False,
        "use_imatrix": False,
        "keep_imatrix": True,
        "nthreads": None,  # None = auto-detect

        # Imatrix generation settings
        "imatrix_ctx_size": 512,
        "imatrix_chunks": None,  # None = all chunks
        "imatrix_collect_output_weight": False,

        # Convert & Quantize tab
        "hf_repo": "",
        "model_path": "",
        "output_dir": "",
        "intermediate_type": "f16",
        "keep_intermediate": False,

        # Quantization types - all stored in other_quants dict
        "other_quants": {
            "Q4_K_M": True,  # Default to Q4_K_M
        },

        # Download tab
        "repo_id": "",
        "download_dir": ""
    }


def load_config():
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)

            # Merge with defaults to handle new settings
            config = get_default_config()
            config.update(saved_config)
            return config
        except Exception as e:
            print(f"Warning: Could not load config: {e}", flush=True)
            return get_default_config()
    return get_default_config()


def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}", flush=True)


def reset_config():
    """Reset configuration to defaults"""
    config = get_default_config()
    save_config(config)
    return config


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="GGUF Converter",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("Yet Another GGUF Converter")
    st.markdown("*Because there are simultaneously too many and not enough GGUF converters*")

    # Initialize converter
    if 'converter' not in st.session_state:
        st.session_state.converter = GGUFConverter()

    # Load config on first run
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

    converter = st.session_state.converter
    config = st.session_state.config

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("---")
        verbose = st.checkbox("Verbose output", value=config.get("verbose", False))

        st.markdown("---")
        st.markdown("**Performance:**")
        import multiprocessing
        max_workers = multiprocessing.cpu_count()

        nthreads = st.number_input(
            "Thread count",
            min_value=1,
            max_value=max_workers,
            value=int(config.get("nthreads") or max_workers),
            step=1,
            help=f"Number of threads for llama.cpp (CPU cores: {max_workers})"
        )

        st.markdown("---")
        st.markdown("**Imatrix Generation:**")
        imatrix_ctx_size = st.number_input(
            "Context size",
            min_value=128,
            max_value=8192,
            value=int(config.get("imatrix_ctx_size", 512)),
            step=128,
            help="Context window size for processing calibration data. Larger = more context but more memory. Default: 512"
        )

        imatrix_chunks = st.number_input(
            "Chunks to process",
            min_value=0,
            max_value=10000,
            value=int(config.get("imatrix_chunks") or 0),
            step=10,
            help="Number of chunks to process from calibration file. 0 = process all. More chunks = better coverage but slower."
        )

        imatrix_collect_output = st.checkbox(
            "Collect output weight",
            value=config.get("imatrix_collect_output_weight", False),
            help="Collect importance matrix for output.weight tensor. May improve quality for some quantization types."
        )

        # Save settings button
        st.markdown("---")
        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button("💾 Save", use_container_width=True, help="Save current settings"):
                # Update config with current values
                config["verbose"] = verbose
                config["nthreads"] = int(nthreads)
                config["imatrix_ctx_size"] = int(imatrix_ctx_size)
                config["imatrix_chunks"] = int(imatrix_chunks) if imatrix_chunks > 0 else None
                config["imatrix_collect_output_weight"] = imatrix_collect_output
                save_config(config)
                st.success("Settings saved!")

        with col_reset:
            if st.button("🔄 Reset", use_container_width=True, help="Reset to default settings"):
                st.session_state.config = reset_config()
                st.rerun()

    # Main content
    tab1, tab2, tab3 = st.tabs(["🔄 Convert & Quantize", "📥 Download Only", "ℹ️ Info"])

    with tab1:
        st.header("Convert and Quantize Model")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")

            hf_repo = st.text_input(
                "HuggingFace Repo ID (optional)",
                value=config.get("hf_repo", ""),
                placeholder="username/model-name",
                help="Optional: Download model from HuggingFace. If provided, model will be downloaded to Model Path below."
            )

            model_path = st.text_input(
                "Model path",
                value=config.get("model_path", ""),
                placeholder="E:/Models/my-model",
                help="Local model directory. If HuggingFace repo provided above, model will be downloaded here first."
            )

            output_dir = st.text_input(
                "Output directory",
                value=config.get("output_dir", ""),
                placeholder="E:/Models/converted",
                help="Where to save the converted files"
            )

            st.markdown("**Conversion Options:**")
            intermediate_type = st.selectbox(
                "Intermediate format",
                ["f16", "f32"],
                index=0 if config.get("intermediate_type", "f16") == "f16" else 1,
                help="Format used before quantization (f32=better but bigger, f16=smaller but ok)"
            )

            keep_intermediate = st.checkbox(
                "Keep intermediate file",
                value=config.get("keep_intermediate", False),
                help="Keep the f16/f32 file after quantization"
            )

            st.markdown("**Importance Matrix:**")
            use_imatrix = st.checkbox(
                "Generate importance matrix",
                value=config.get("use_imatrix", False),
                help="Generate importance matrix for better low-bit quantization (IQ2, IQ3).  \nRequired for best IQ2/IQ3 quality. Will be generated and saved in output directory."
            )

            keep_imatrix = st.checkbox(
                "Keep and reuse importance matrix",
                value=config.get("keep_imatrix", True),
                help="Keep the .imatrix file after quantization for reuse. If unchecked, it will be deleted after quantization."
            )

        with col2:
            st.subheader("Quantization Types")
            st.markdown("Select one or more quantization types:")

            # Traditional Quants
            st.markdown("**Traditional Quants:**")
            trad_cols = st.columns(5)
            trad_quants = {
                "Q4_0": "4-bit legacy",
                "Q4_1": "4-bit legacy improved",
                "Q5_0": "5-bit legacy",
                "Q5_1": "5-bit legacy improved",
                "Q8_0": "8-bit (highest quality)",
            }
            trad_checkboxes = {}
            for idx, (qtype, tooltip) in enumerate(trad_quants.items()):
                with trad_cols[idx]:
                    trad_checkboxes[qtype] = st.checkbox(
                        qtype,
                        value=config.get("other_quants", {}).get(qtype, qtype == "Q8_0" if qtype == "Q8_0" else False),
                        help=tooltip,
                        key=f"trad_{qtype}"
                    )

            # K Quants
            st.markdown("**K Quants (Recommended):**")
            k_quants = {
                "Q2_K": "2-bit K",
                "Q2_K_S": "2-bit K small",
                "Q3_K_S": "3-bit K small",
                "Q3_K_M": "3-bit K medium",
                "Q3_K_L": "3-bit K large",
                "Q4_K_S": "4-bit K small",
                "Q4_K_M": "4-bit K medium (best balance)",
                "Q5_K_S": "5-bit K small",
                "Q5_K_M": "5-bit K medium",
                "Q6_K": "6-bit K (very high quality)",
            }
            k_checkboxes = {}
            k_cols = st.columns(3)
            for idx, (qtype, tooltip) in enumerate(k_quants.items()):
                with k_cols[idx % 3]:
                    default_val = config.get("other_quants", {}).get(qtype, qtype == "Q4_K_M")
                    k_checkboxes[qtype] = st.checkbox(
                        qtype,
                        value=default_val,
                        help=tooltip,
                        key=f"k_{qtype}"
                    )

            # I Quants
            st.markdown("**I Quants (Importance Matrix Recommended):**")
            i_quants = {
                "IQ1_S": "1-bit IQ small",
                "IQ1_M": "1-bit IQ medium",
                "IQ2_XXS": "2-bit IQ extra-extra-small",
                "IQ2_XS": "2-bit IQ extra-small",
                "IQ2_S": "2-bit IQ small",
                "IQ2_M": "2-bit IQ medium",
                "IQ3_XXS": "3-bit IQ extra-extra-small",
                "IQ3_XS": "3-bit IQ extra-small",
                "IQ3_S": "3.4-bit IQ small",
                "IQ3_M": "3-bit IQ medium",
                "IQ4_XS": "4-bit IQ extra-small",
                "IQ4_NL": "4-bit IQ non-linear",
            }
            i_checkboxes = {}
            i_cols = st.columns(3)
            for idx, (qtype, tooltip) in enumerate(i_quants.items()):
                with i_cols[idx % 3]:
                    i_checkboxes[qtype] = st.checkbox(
                        qtype,
                        value=config.get("other_quants", {}).get(qtype, False),
                        help=tooltip,
                        key=f"i_{qtype}"
                    )

        # Collect selected quantization types
        selected_quants = []
        for qtype, checked in trad_checkboxes.items():
            if checked:
                selected_quants.append(qtype)
        for qtype, checked in k_checkboxes.items():
            if checked:
                selected_quants.append(qtype)
        for qtype, checked in i_checkboxes.items():
            if checked:
                selected_quants.append(qtype)

        # Convert button
        st.markdown("---")
        if st.button("🚀 Start Conversion", type="primary", use_container_width=True):
            if not model_path:
                st.error("Please provide a model path")
            elif not output_dir:
                st.error("Please provide an output directory")
            elif not selected_quants:
                st.error("Please select at least one quantization type")
            else:
                # Update config with current values
                config["verbose"] = verbose
                config["nthreads"] = nthreads
                config["use_imatrix"] = use_imatrix
                config["keep_imatrix"] = keep_imatrix
                config["hf_repo"] = hf_repo
                config["model_path"] = model_path
                config["output_dir"] = output_dir
                config["intermediate_type"] = intermediate_type
                config["keep_intermediate"] = keep_intermediate

                # Save all quantization selections in other_quants
                all_quant_selections = {}
                all_quant_selections.update(trad_checkboxes)
                all_quant_selections.update(k_checkboxes)
                all_quant_selections.update(i_checkboxes)
                config["other_quants"] = all_quant_selections

                save_config(config)

                try:
                    # Download from HuggingFace if repo is provided
                    actual_model_path = model_path
                    if hf_repo and hf_repo.strip():
                        with st.spinner(f"Downloading {hf_repo} from HuggingFace..."):
                            from pathlib import Path
                            download_path = converter.download_model(
                                repo_id=hf_repo.strip(),
                                output_dir=Path(model_path)
                            )
                            actual_model_path = str(download_path)
                            st.info(f"Downloaded to: {actual_model_path}")

                    with st.spinner("Converting and quantizing... This may take a while."):
                        output_files = converter.convert_and_quantize(
                            model_path=actual_model_path,
                            output_dir=output_dir,
                            quantization_types=selected_quants,
                            intermediate_type=intermediate_type,
                            keep_intermediate=keep_intermediate,
                            verbose=verbose,
                            generate_imatrix=use_imatrix,
                            keep_imatrix=keep_imatrix,
                            nthreads=nthreads,
                            imatrix_ctx_size=imatrix_ctx_size,
                            imatrix_chunks=imatrix_chunks if imatrix_chunks > 0 else None,
                            imatrix_collect_output=imatrix_collect_output
                        )

                    st.success(f"✅ Successfully created {len(output_files)} files!")

                    st.subheader("Created Files")
                    for file_path in output_files:
                        file_size = file_path.stat().st_size / (1024**3)  # GB
                        st.write(f"✓ `{file_path.name}` ({file_size:.2f} GB)")
                        st.code(str(file_path), language=None)

                except Exception as e:
                    # Show in Streamlit UI
                    st.error(f"Error: {e}")
                    if verbose:
                        st.exception(e)

                    # ALSO print to terminal so user sees it
                    print(f"\nError: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    with tab2:
        st.header("Download from HuggingFace")
        st.markdown("Download a model without converting")

        repo_id = st.text_input(
            "Repository ID",
            value=config.get("repo_id", ""),
            placeholder="username/model-name"
        )

        download_dir = st.text_input(
            "Download directory",
            value=config.get("download_dir", ""),
            placeholder="E:/Models/downloads"
        )

        if st.button("📥 Download", use_container_width=True):
            if not repo_id:
                st.error("Please provide a repository ID")
            elif not download_dir:
                st.error("Please provide a download directory")
            else:
                # Save current settings before downloading
                config["repo_id"] = repo_id
                config["download_dir"] = download_dir
                save_config(config)

                try:
                    with st.spinner(f"Downloading {repo_id}..."):
                        model_path = converter.download_model(repo_id, download_dir)

                    st.success(f"✅ Downloaded to: {model_path}")
                    st.code(str(model_path), language=None)

                except Exception as e:
                    # Show in Streamlit UI
                    st.error(f"Error: {e}")

                    # ALSO print to terminal so user sees it
                    print(f"\nError: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    with tab3:
        st.header("About")
        st.markdown(f"""
        ### Yet Another GGUF Converter

        A user-friendly GGUF converter that shields you from llama.cpp complexity.
        No manual compilation or terminal commands required!

        **Features:**
        - Convert HuggingFace models to GGUF
        - Quantize to multiple formats at once using llama.cpp
        - Auto-downloads pre-compiled binaries (no compilation needed!)
        - Cross-platform (Windows, Mac, Linux)
        - Persistent settings - remembers your preferences
        - All llama.cpp quantization types supported

        **Settings:**
        - Your settings are automatically saved when you start a conversion or click "Save" in the sidebar
        - Settings are stored in: `{CONFIG_FILE}`
        - Use the "Reset" button in the sidebar to restore default settings

        **Quantization Types (via llama.cpp):**

        | Type | Size | Quality | Use Case |
        |------|------|---------|----------|
        | Q4_K_M | Small | Good | Recommended for most users |
        | Q5_K_M | Medium | Better | Higher quality needed |
        | Q6_K | Large | Very Good | Very high quality |
        | Q8_0 | Largest | Excellent | Near-original quality |
        | IQ3_XXS | Tiny | Fair | 3-bit compression (use imatrix) |
        | IQ3_S | Tiny+ | Fair+ | 3.4-bit compression |
        | IQ4_NL | Small- | Good | 4-bit non-linear |
        | Q4_0, Q5_0 | Various | Various | Legacy formats |

        Quantization is powered by llama.cpp - battle-tested and fully compatible!

        **Requirements:**
        - Python 3.8+
        - huggingface-hub (for downloading models)
        - streamlit (for GUI)
        - llama.cpp binaries (auto-downloaded)

        **Command Line Usage:**
        ```bash
        # Convert and quantize
        python -m gguf_converter /path/to/model output/ -q Q4_K_M Q5_K_M

        # Download from HuggingFace
        python -m gguf_converter username/model-name output/ -q Q4_K_M

        # List available types
        python -m gguf_converter --list-types
        ```
        """)


if __name__ == "__main__":
    main()
