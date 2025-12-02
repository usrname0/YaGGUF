# Yet Another GGUF Converter

There are simultaneously too many and not enough GGUF Converters in the world. I can fix this.

## Features

- **All Python** - No extra binaries or compiling
- **Convert** safetensors and PyTorch models to GGUF format
- **Quantize** to multiple formats at once
- **Cross-platform** - Works on Windows, Mac, and Linux
- **Two interfaces** - CLI for automation, GUI for ease of use
- **Fast** - Parallel quantization makes up for python and then some

## Quantization Types
Pretend there's a list here.

## Installation

```bash
# Clone the repository
git clone https://github.com/usrname0/Yet_Another_GGUF_Converter.git
cd Yet_Another_GGUF_Converter

# Install dependencies
pip install -r requirements.txt
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
Then open your browser to http://localhost:8501

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

### Python Dependencies

All installed automatically with `pip install -r requirements.txt`:

- `huggingface-hub` - Model downloading
- `numpy` - Numerical operations
- `gguf` - GGUF file format support
- `torch` - Model conversion (used by llama.cpp convert script)
- `transformers` - Model loading (used by llama.cpp convert script)
- `streamlit` (optional) - GUI


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

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format and tools
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings
- [HuggingFace](https://huggingface.co/) - Model hosting and transformers library