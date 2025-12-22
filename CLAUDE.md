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

**GUI** (`gguf_converter/gui.py`):
- Streamlit web interface for all user interactions
- Persistent settings in `~/.gguf_converter_config.json`
- Auto-runs setup on first launch
- Multi-tab interface for conversion, imatrix, downloads, llama.cpp binaries, and updates
- Auto-restart functionality for dependency updates

**Binary Manager** (`gguf_converter/binary_manager.py`):
- Downloads pre-compiled llama.cpp binaries from GitHub releases
- Version specified by `LLAMA_CPP_VERSION` constant in binary_manager.py
- Platform detection and architecture support (x64, arm64)
- Custom binary path support for advanced users

**Scripts** (`scripts/`):
- `setup_windows.bat` / `setup_linux.sh` - One-click setup (venv + dependencies + PyTorch CPU)
- `check_binaries.py` - Verify and download llama.cpp binaries if missing
- `update_and_restart.bat` / `update_and_restart.sh` - Update dependencies with auto-restart
- `setup.py` - pip installation configuration (reads version from `__init__.py`)

**Launchers** (project root):
- `run_gui.bat` / `run_gui.sh` - Launch GUI with auto-setup if needed

### Version Management

**Single Source of Truth:**
- Project version stored in `gguf_converter/__init__.py` as `__version__`
- `setup.py` reads version dynamically from `__init__.py`
- GUI displays version from `__init__.py`
- No separate VERSION files

**Binary Versioning:**
- llama.cpp version hardcoded in `binary_manager.py` as `LLAMA_CPP_VERSION`
- No separate binary version tracking files
- Update by changing constant and users update via GUI

### Update System

**Application Updates:**
- GUI Update tab has "Check for Updates" button (runs `git pull`)
- Users pull directly from GitHub repository

**Dependency Updates:**
- GUI "Update Dependencies & Restart" button
- Triggers auto-restart script to avoid file locking issues
- Updates all packages in requirements.txt including Streamlit
- PyTorch note in requirements.txt (installed separately via setup scripts)

**Binary Updates:**
- GUI llama.cpp tab has "Force Binary Update" button
- Downloads latest version specified in `LLAMA_CPP_VERSION`
- Custom binary paths supported for advanced users

### Key Design Decisions

1. **llama.cpp wrapper**: Use battle-tested quantization, avoid reimplementation
2. **Noob-friendly**: Auto-setup, auto-restart updates, clear error messages, persistent settings
3. **Binary management**: Auto-download pre-compiled llama.cpp on first run
4. **Full compatibility**: All llama.cpp quantization types and features
5. **Cross-platform**: Windows, Mac, Linux with platform-specific binaries
6. **Maintenance simplicity**: Single version source, consolidated scripts folder

