## Project: Yet Another GGUF Converter

### Essential Project Rules
1. Research things before guessing
2. Use docstrings
3. No emojis or unicode characters anywhere
4. Discuss big changes before going ahead
5. We are in a Windows environment but target all platforms

### Goal
User-friendly GGUF converter that shields users from llama.cpp complexity - no manual compilation or terminal commands required!

### Core Components

**Converter** (`gguf_converter/converter.py`):
- Wrapper around llama.cpp for model conversion and quantization
- Handles llama.cpp binary management (download/bundling)
- All quantization types supported by llama.cpp
- Clear error messages and progress tracking

**CLI** (`gguf_converter/cli.py`):
- Command-line interface for automation
- Batch quantization (multiple types at once)
- Auto-runs setup scripts if needed

**GUI** (`gguf_converter/gui.py`):
- Streamlit web interface
- Persistent settings
- Auto-runs setup on first launch

**Setup Scripts**:
- `setup_windows.bat` / `setup_linux.sh` - One-click setup
- `run_gui.bat` / `run_gui.sh` - Launch GUI (auto-setup if needed)
- Downloads/installs llama.cpp binaries automatically

### Key Design Decisions

1. **llama.cpp wrapper**: Use battle-tested quantization, avoid reimplementation
2. **Noob-friendly**: Auto-setup, clear error messages, persistent settings
3. **Binary management**: Auto-download pre-compiled llama.cpp on first run
4. **Full compatibility**: All llama.cpp quantization types and features
5. **Cross-platform**: Windows, Mac, Linux with platform-specific binaries

