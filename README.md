# Yet Another GGUF Convertor

There are simultaneously too many and not enough GGUF Converters in the world. I can fix this.

## Features

- **llama.cpp under the hood** - So you can trust that part
- **Convert** - safetensors and PyTorch models to GGUF format
- **Quantize** - to multiple formats at once
- **Cross-platform** - scripts install on Windows, Mac, and Linux 
- **No compiling** - installs llama.cpp binaries
- **No mess** - it all lives in one folder/venv

## Quantization Types

All quantization types from llama.cpp are supported. Choose based on your size/quality tradeoff:

### Full Precision (No Quantization)
| Type | Size | Quality | Notes |
|------|------|---------|-------|
| F32 | Largest | Original | Full 32-bit precision |
| F16 | Large | Near-original | Half precision, minimal quality loss |
| BF16 | Large | Near-original | Brain float 16-bit |

### Recommended K-Quants (Best Quality/Size Balance)
| Type | Size | Quality | Best For |
|------|------|---------|----------|
| Q4_K_M | Small | Good | **Default recommendation** - best balance |
| Q5_K_M | Medium | Better | Higher quality with acceptable size |
| Q6_K | Large | Very High | Near-F16 quality, larger size |
| Q8_0 | Very Large | Excellent | Near-original quality |
| Q3_K_M | Very Small | Fair | Aggressive compression |
| Q2_K | Tiny | Minimal | Maximum compression |

### Traditional Quants (Legacy)
| Type | Size | Quality | Notes |
|------|------|---------|-------|
| Q4_0 | Small | Fair | Legacy 4-bit |
| Q4_1 | Small | Fair | Legacy 4-bit improved |
| Q5_0 | Medium | Good | Legacy 5-bit |
| Q5_1 | Medium | Good | Legacy 5-bit improved |

### I-Quants (Importance Matrix - Advanced)
Requires generating an importance matrix for best results. Better quality at ultra-low bit rates.

| Type | Size | Quality | Notes |
|------|------|---------|-------|
| IQ4_NL | Small | Good | 4-bit non-linear |
| IQ3_M | Very Small | Fair | 3-bit medium |
| IQ3_S | Very Small | Fair+ | 3.4-bit compression |
| IQ2_M | Tiny | Minimal | 2-bit medium |
| IQ2_S | Tiny | Minimal | 2-bit small |
| IQ1_M | Extreme | Poor | 1-bit, experimental |

**Quick Guide:**
- Just starting? Use **Q4_K_M**
- Want better quality? Use **Q5_K_M** or **Q6_K**
- Need original quality? Use **Q8_0** or **F16**
- Want smallest size? Use **IQ3_M** or **IQ2_M** with importance matrix

## Installation

```bash
# Clone the repository
git clone https://github.com/usrname0/Yet_Another_GGUF_Converter.git
cd Yet_Another_GGUF_Converter
# Windows (run_gui.bat automatically runs a setup script if no venv detected)
run_gui.bat
# Linux / Mac (run_gui.sh automatically runs a setup script if no venv detected)
run_gui.sh
```

## Usage

### GUI (Recommended for normals)

**Windows**
double-click run_gui.bat

**Linux / Mac**
double-click run_gui.sh

**or type this:**
```bash
streamlit run gguf_converter/gui.py
```
Then open your browser to `http://localhost:8501`

### CLI (Recommended for nerds)

```bash
# Convert a local model and quantize to Q4_K_M
python -m gguf_converter /path/to/model output/ -q Q4_K_M

# Download from HuggingFace and create multiple quants
python -m gguf_converter username/model-name output/ -q Q4_K_M Q5_K_M Q8_0

# Just convert without quantization
python -m gguf_converter /path/to/model output/ --no-quantize

# List all available quantization types
python -m gguf_converter --list-types

# Verbose output for debugging
python -m gguf_converter /path/to/model output/ -q Q4_K_M --verbose
```

### More CLI Examples

```bash
# Basic usage - convert and quantize
python -m gguf_converter ~/models/llama-2-7b output/ -q Q4_K_M

# Multiple quantization types at once
python -m gguf_converter ~/models/mistral-7b output/ -q Q4_K_M Q5_K_M Q8_0

# Download from HuggingFace
python -m gguf_converter meta-llama/Llama-2-7b-hf output/ -q Q4_K_M

# Keep intermediate f16 file (useful for making multiple quants later)
python -m gguf_converter ~/models/model output/ -q Q4_K_M --keep-intermediate

# Use f32 intermediate (higher quality, larger size)
python -m gguf_converter ~/models/model output/ -q Q4_K_M --intermediate f32

# Convert only, no quantization
python -m gguf_converter ~/models/model output/ --no-quantize --output-type f16
```

## Requirements

- Python 3.8 or higher
- 8GB+ RAM (more for larger models)
- Disk space for models and outputs

## Troubleshooting

### Out of Memory

For large models, ensure you have enough RAM. You can also:
- Use swap space
- Close other applications
- Try a smaller model first

### Conversion is slow

This is normal for large models. The process is CPU-intensive.
- Use fewer quantization types
- Enable `--verbose` to see progress
- Adjust parallel quantization settings
- Be patient - it will finish!

## License

MIT License - see LICENSE file for details

## Credits

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format and conversion/quantization tools
- [HuggingFace](https://huggingface.co/) - Model hosting and transformers library
