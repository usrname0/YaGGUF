"""
Info tab for GGUF Converter GUI
"""

import streamlit as st
from typing import Dict, Any, TYPE_CHECKING

from ..gui_utils import CONFIG_FILE, HF_TOKEN_PATH

if TYPE_CHECKING:
    from ..converter import GGUFConverter


def render_info_tab(converter: "GGUFConverter", config: Dict[str, Any]) -> None:
    """Render the Info tab"""
    st.header("About")
    st.markdown(f"""
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
    - Settings are stored in: `{CONFIG_FILE}`
    - HuggingFace token stored in: `{HF_TOKEN_PATH}` (managed by huggingface_hub)
    - Use "Reset to defaults" in the sidebar to restore default settings

    **Testing (Sidebar):**
    - **Test Models** - Test GGUF models in your output directory with llama-server
    - **Dev Tests** - Run the full pytest suite (downloads test model, runs all tests)

    **Quantization Types (via llama.cpp):**

    | Type | Size | Quality | Category | Notes |
    |------|------|---------|----------|-------|
    | **F32** | Largest | Original | Unquantized | Full 32-bit precision |
    | **F16** | Large | Near-original | Unquantized | Half precision |
    | **BF16** | Large | Near-original | Unquantized | Brain float 16-bit |
    | **Q8_0** | Very Large | Excellent | Legacy | Near-original quality |
    | Q5_1, Q5_0 | Medium | Good | Legacy | Legacy 5-bit |
    | Q4_1, Q4_0 | Small | Fair | Legacy | Legacy 4-bit |
    | **Q6_K** | Large | Very High | K-Quant | Near-F16 quality |
    | **Q5_K_M** | Medium | Better | K-Quant | Higher quality |
    | Q5_K_S | Medium | Better | K-Quant | 5-bit K small |
    | **Q4_K_M** | Small | Good | K-Quant | 4-bit K medium |
    | Q4_K_S | Small | Good | K-Quant | 4-bit K small |
    | Q3_K_L | Very Small | Fair | K-Quant | 3-bit K large |
    | Q3_K_M | Very Small | Fair | K-Quant | 3-bit K medium |
    | Q3_K_S | Very Small | Fair | K-Quant | 3-bit K small |
    | Q2_K | Tiny | Minimal | K-Quant | 2-bit K |
    | Q2_K_S | Tiny | Minimal | K-Quant | 2-bit K small |
    | **IQ4_NL** | Small | Good | I-Quant | 4-bit non-linear (use imatrix) |
    | IQ4_XS | Small | Good | I-Quant | 4-bit extra-small (use imatrix) |
    | IQ3_M | Very Small | Fair | I-Quant | 3-bit medium (use imatrix) |
    | IQ3_S | Very Small | Fair+ | I-Quant | 3.4-bit (use imatrix) |
    | IQ3_XS | Very Small | Fair | I-Quant | 3-bit extra-small (use imatrix) |
    | IQ3_XXS | Very Small | Fair | I-Quant | 3-bit extra-extra-small (use imatrix) |
    | IQ2_M | Tiny | Minimal | I-Quant | 2-bit medium (use imatrix) |
    | IQ2_S | Tiny | Minimal | I-Quant | 2-bit small (use imatrix) |
    | IQ2_XS | Tiny | Minimal | I-Quant | 2-bit extra-small (use imatrix) |
    | IQ2_XXS | Tiny | Minimal | I-Quant | 2-bit extra-extra-small (use imatrix) |
    | IQ1_M | Extreme | Poor | I-Quant | 1-bit medium (use imatrix) |
    | IQ1_S | Extreme | Poor | I-Quant | 1-bit small (use imatrix) |

    **Quick Guide:**
    - Bigger is better (more precision)
    - For best quality use **F16** or **Q8_0**
    - For decent quality use **Q6_K** or **Q5_K_M**
    - Medium quality... Use **Q4_K_M**
    - For smallest size use IQ3_M or IQ2_M with importance matrix
    
    Quantization is done by [llama.cpp](https://github.com/ggml-org/llama.cpp).

    """)
