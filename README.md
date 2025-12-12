# YaGUFF - Yet Another GGUF Converter

There are simultaneously too many and not enough GGUF converters in the world. I can fix this.

## Features

- **llama.cpp under the hood** - So you can trust that part
- **Convert** - safetensors and PyTorch models to GGUF format
- **Quantize** - to multiple formats at once
- **Cross-platform** - scripts install on Windows, Mac, and Linux 
- **No compiling** - installs llama.cpp binaries
- **No mess** - it all lives in one folder/venv

## Quantization Types

All quantization types from llama.cpp are supported. Choose based on your size/quality tradeoff:

| Type | Size | Quality | Category | Notes |
|------|------|---------|----------|-------|
| **F32** | Largest | Original | Unquantized | Full 32-bit precision |
| **F16** | Large | Near-original | Unquantized | Half precision, minimal quality loss |
| BF16 | Large | Near-original | Unquantized | Brain float 16-bit |
| Q8_0 | Very Large | Excellent | Legacy | Near-original quality, 8-bit |
| Q5_1 | Medium | Good | Legacy | Legacy 5-bit improved |
| Q5_0 | Medium | Good | Legacy | Legacy 5-bit |
| Q4_1 | Small | Fair | Legacy | Legacy 4-bit improved |
| Q4_0 | Small | Fair | Legacy | Legacy 4-bit |
| **Q6_K** | Large | Very High | K-Quant | Near-F16 quality, larger size |
| **Q5_K_M** | Medium | Better | K-Quant | Higher quality with acceptable size |
| Q5_K_S | Medium | Better | K-Quant | 5-bit K small |
| **Q4_K_M** | Small | Good | K-Quant | **Default recommendation** - best balance |
| Q4_K_S | Small | Good | K-Quant | 4-bit K small |
| Q3_K_M | Very Small | Fair | K-Quant | Aggressive compression |
| Q3_K_L | Very Small | Fair | K-Quant | 3-bit K large |
| Q3_K_S | Very Small | Fair | K-Quant | 3-bit K small |
| Q2_K | Tiny | Minimal | K-Quant | Maximum compression |
| Q2_K_S | Tiny | Minimal | K-Quant | 2-bit K small |
| IQ4_NL | Small | Good | I-Quant | 4-bit non-linear (use imatrix) |
| IQ4_XS | Small | Good | I-Quant | 4-bit extra-small (use imatrix) |
| IQ3_M | Very Small | Fair | I-Quant | 3-bit medium (use imatrix) |
| IQ3_S | Very Small | Fair+ | I-Quant | 3.4-bit compression (use imatrix) |
| IQ3_XS | Very Small | Fair | I-Quant | 3-bit extra-small (use imatrix) |
| IQ3_XXS | Very Small | Fair | I-Quant | 3-bit extra-extra-small (use imatrix) |
| IQ2_M | Tiny | Minimal | I-Quant | 2-bit medium (use imatrix) |
| IQ2_S | Tiny | Minimal | I-Quant | 2-bit small (use imatrix) |
| IQ2_XS | Tiny | Minimal | I-Quant | 2-bit extra-small (use imatrix) |
| IQ2_XXS | Tiny | Minimal | I-Quant | 2-bit extra-extra-small (use imatrix) |
| IQ1_M | Extreme | Poor | I-Quant | 1-bit medium, experimental (use imatrix) |
| IQ1_S | Extreme | Poor | I-Quant | 1-bit small, experimental (use imatrix) |


**Quick Guide:**
- Just starting? Use **Q4_K_M**
- Want better quality? Use **Q5_K_M** or **Q6_K**
- Need omega deluxe quality? Use **Q8_0** or **F16**
- Want smallest size? Use IQ3_M or IQ2_M with importance matrix

## Installation

```bash
# Clone the repository
    git clone https://github.com/usrname0/YaGUFF.git
    cd YaGUFF# Run the launcher script for your platform (runs a setup script if no venv detected):
- Windows: `run_gui.bat`
- Linux/Mac: `./run_gui.sh`
```

## Usage

### GUI (Recommended)

**Windows:**
- Double-click `run_gui.bat`

**Linux/Mac:**
```bash
./run_gui.sh
```

The GUI will automatically open in your browser at `http://localhost:8501`

### CLI

A command-line interface is available for automation and scripting:

```bash
# Convert and quantize
python -m gguf_converter /path/to/model output/ -q Q4_K_M Q5_K_M

# List available quantization types
python -m gguf_converter --list-types
```

For advanced CLI usage, most users should use [llama.cpp](https://github.com/ggerganov/llama.cpp) directly.

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
