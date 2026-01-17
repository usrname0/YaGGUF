"""
Info tab for GGUF Converter GUI
"""

import streamlit as st
from typing import Dict, Any, TYPE_CHECKING

from ..gui_utils import CONFIG_FILE, HF_TOKEN_PATH, save_config

if TYPE_CHECKING:
    from ..converter import GGUFConverter


def render_info_tab(converter: "GGUFConverter", config: Dict[str, Any]) -> None:
    """Render the Info tab"""
    st.header("About")
    st.markdown("""
    ### YaGGUF - Yet Another GGUF Converter

    A user-friendly GGUF converter that sits on top of llama.cpp.  
    Intended for anyone who prefers a GUI or wants a quant without going down the whole rabbithole.

    **Features:**
    - **Convert & Quantize** - HuggingFace models to GGUF with multiple quantization formats at once
    - **All quantization types** - Full support for llama.cpp quantization types
    - **Vision/Multimodal models** - Automatic detection and two-step conversion (text model + vision projector)
    - **Sentence-transformers** - Auto-detect and include dense modules for embedding models
    - **Model quirks detection** - Handles Mistral format, pre-quantized models, and architecture-specific flags
    - **Split files mode** - Generate split shards for intermediates and quants
    - **Custom intermediates** - Use existing GGUF files as intermediates for quantization
    - **Enhanced dtype detection** - Detects model precision (BF16, F16, etc.) from configs and safetensors headers
    - **Importance Matrix** - Generate or reuse imatrix files for better low-bit quantization (IQ2, IQ3)
    - **Imatrix Statistics** - Analyze importance matrix files to view statistics
    - **HuggingFace Downloader** - Download models and their supporting files
    - **Split/Merge Shards** - Split and merge GGUF and safetensors files with custom shard sizes
    - **Cross-platform** - Windows & Linux support (Mac should work too, but untested)
    - **Auto-downloads binaries** - Pre-compiled llama.cpp binaries (CPU)
    - **Customization** - Can use other llama.cpp binaries if desired
    - **Persistent settings** - Automatically saves your preferences

    **Tabs:**
    1. **Convert & Quantize** - Main conversion interface with imatrix options
    2. **Imatrix Settings** - Configure calibration data and processing settings
    3. **Imatrix Statistics** - Analyze existing imatrix files
    4. **HuggingFace Downloader** - Download models from HuggingFace
    5. **Split/Merge Shards** - Split and merge GGUF and safetensors files
    6. **Info** - This tab
    7. **llama.cpp** - Customize llama.cpp setup
    8. **Update** - Update YaGGUF, llama.cpp and dependencies

    **Settings:**
    - Your settings are automatically saved as you change them
    - Use "Reset to defaults" in the sidebar to restore default settings
    
    **Quantization Types (via llama.cpp):**

    | Type | Size | Quality | Category | Imatrix | Notes |
    |------|------|---------|----------|---------|-------|
    | **F32** | Largest | Original | Unquantized | - | Full 32-bit precision |
    | **F16** | Large | Near-original | Unquantized | - | Half precision |
    | **BF16** | Large | Near-original | Unquantized | - | Brain float 16-bit |
    | **Q8_0** | Very Large | Excellent | Legacy | - | Near-original quality |
    | Q5_1, Q5_0 | Medium | Good | Legacy | - | Legacy 5-bit |
    | Q4_1, Q4_0 | Small | Fair | Legacy | - | Legacy 4-bit |
    | **Q6_K** | Large | Very High | K-Quant | Suggested | Near-F16 quality |
    | **Q5_K_M** | Medium | Better | K-Quant | Suggested | Higher quality |
    | Q5_K_S | Medium | Better | K-Quant | Suggested | 5-bit K small |
    | **Q4_K_M** | Small | Good | K-Quant | Suggested | 4-bit K medium |
    | Q4_K_S | Small | Good | K-Quant | Suggested | 4-bit K small |
    | Q3_K_L | Very Small | Fair | K-Quant | Recommended | 3-bit K large |
    | Q3_K_M | Very Small | Fair | K-Quant | Recommended | 3-bit K medium |
    | Q3_K_S | Very Small | Fair | K-Quant | Recommended | 3-bit K small |
    | Q2_K | Tiny | Minimal | K-Quant | Recommended | 2-bit K |
    | Q2_K_S | Tiny | Minimal | K-Quant | Recommended | 2-bit K small |
    | **IQ4_NL** | Small | Good | I-Quant | Recommended | 4-bit non-linear |
    | IQ4_XS | Small | Good | I-Quant | Recommended | 4-bit extra-small |
    | IQ3_M | Very Small | Fair | I-Quant | Recommended | 3-bit medium |
    | IQ3_S | Very Small | Fair+ | I-Quant | Recommended | 3.4-bit small |
    | IQ3_XS | Very Small | Fair | I-Quant | Required | 3-bit extra-small |
    | IQ3_XXS | Very Small | Fair | I-Quant | Required | 3-bit extra-extra-small |
    | IQ2_M | Tiny | Minimal | I-Quant | Required | 2-bit medium |
    | IQ2_S | Tiny | Minimal | I-Quant | Required | 2-bit small |
    | IQ2_XS | Tiny | Minimal | I-Quant | Required | 2-bit extra-small |
    | IQ2_XXS | Tiny | Minimal | I-Quant | Required | 2-bit extra-extra-small |
    | IQ1_M | Extreme | Poor | I-Quant | Required | 1-bit medium |
    | IQ1_S | Extreme | Poor | I-Quant | Required | 1-bit small |

    **Quick Guide:**
    - Bigger is better (more precision)
    - For best quality use **F16** or **Q8_0**
    - For decent quality use **Q6_K** or **Q5_K_M**
    - Medium quality... Use **Q4_K_M**
    - For smallest size use IQ3_M or IQ2_M with importance matrix
    
    Quantization is done by [llama.cpp](https://github.com/ggml-org/llama.cpp).
                
    ---
    """)

    # Dev Options expander (constrained width)
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.expander("Dev Options"):

            def save_dev_mode():
                config["dev_mode"] = st.session_state.dev_mode_checkbox
                save_config(config)

            dev_mode = st.checkbox(
                "Enable developer mode",
                value=config.get("dev_mode", False),
                help="Show developer tools in the sidebar",
                key="dev_mode_checkbox",
                on_change=save_dev_mode
            )

            st.markdown(f"""
            Enables extra tools that are unsupported but might be useful for advanced users:
            - **Test Models** - Test all GGUF variants in the output directory interactively with llama-server
                - Launches a new terminal window, hit enter in the terminal to load the next model
                - Custom binary with gpu enabled is recommended but not required
            - **Dev Tests** - Run the full test suite (pytest) in a new terminal window
                - First checksum test run will be "SKIPPED" because there's no checksum to compare against yet
            ---
            **Information:**
            - Settings stored in: `{CONFIG_FILE}`
            - HuggingFace token stored in: `{HF_TOKEN_PATH}` (managed by huggingface_hub)
            """)

