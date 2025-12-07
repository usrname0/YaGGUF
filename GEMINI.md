# Yet Another GGUF GUI

### Essential Project Rules
1. Research things before guessing
2. Use docstrings
3. No emojis or unicode characters in production code
4. Discuss big changes before going ahead
5. We are in a Windows environment but target all platforms

## Project Overview

"Yet Another GGUF GUI" is a Python-based application designed to simplify the process of converting and quantizing pre-trained AI models (specifically HuggingFace `safetensors` and PyTorch models) into the GGUF format, leveraging the powerful `llama.cpp` toolchain. It aims to provide a user-friendly experience for both command-line users and those who prefer a graphical interface, abstracting away the complexities of `llama.cpp` compilation and binary management.

**Key Features:**

*   **Model Conversion:** Converts `safetensors` and PyTorch models to GGUF format.
*   **Quantization:** Supports a wide range of `llama.cpp` quantization types (e.g., Q4_K_M, Q5_K_M, Q8_0, various IQ and K quants).
*   **HuggingFace Integration:** Seamlessly downloads models directly from HuggingFace repositories.
*   **Automatic Binary Management:** Downloads pre-compiled `llama.cpp` binaries (`llama-quantize`, `llama-imatrix`) for Windows, Linux, and macOS, eliminating the need for manual compilation.
*   **Importance Matrix Generation:** Can generate importance matrices for improved low-bit quantization quality.
*   **Cross-platform:** Works on Windows, Linux, and macOS.
*   **User-friendly Interfaces:** Provides both a rich Streamlit-based GUI and a flexible command-line interface (CLI).
*   **Persistent Settings:** The GUI saves user preferences across sessions in `~/.gguf_converter_config.json`.

**Core Technologies:**

*   **Python 3.8+**
*   **`llama.cpp`:** Underpins all conversion and quantization processes.
*   **`huggingface-hub`:** For model downloads from HuggingFace.
*   **`streamlit`:** Powers the graphical user interface.
*   **`numpy`, `gguf`, `torch`, `transformers`, `sentencepiece`:** Core dependencies for model handling and conversion.

## Building and Running

This project is primarily a Python application. It does not require traditional "building" in the C++ sense, as it downloads pre-compiled `llama.cpp` binaries.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/usrname0/Yet_Another_GGUF_Converter.git
    cd Yet_Another_GGUF_Converter
    ```
2.  **Run setup script (Windows/Linux/Mac):**
    *   **Windows:** Double-click `run_gui.bat` or execute it from the command line. This script will automatically set up a Python virtual environment and install dependencies.
    *   **Linux / Mac:** Execute `run_gui.sh` from the terminal. This script will also automatically set up a Python virtual environment and install dependencies.

    Alternatively, you can manually set up the environment:
    ```bash
    python -m venv venv
    ./venv/Scripts/activate  # Windows
    source venv/bin/activate # Linux/Mac
    pip install -r requirements.txt
    pip install ".[gui]" # Install GUI dependencies
    ```

### Running the Application

#### GUI (Recommended for most users)

*   **Windows:** Double-click `run_gui.bat`.
*   **Linux / Mac:** Execute `run_gui.sh`.
*   **Manual (after activating venv):**
    ```bash
    streamlit run gguf_converter/gui.py
    ```
    Then open your web browser to `http://localhost:8501`.

#### CLI (For advanced users and scripting)

After activating your virtual environment:

*   **Convert a local model and quantize to Q4\_K\_M:**
    ```bash
    python -m gguf_converter /path/to/model output/ -q Q4_K_M
    ```
*   **Download from HuggingFace and create multiple quants:**
    ```bash
    python -m gguf_converter username/model-name output/ -q Q4_K_M Q5_K_M Q8_0
    ```
*   **Just convert without quantization:**
    ```bash
    python -m gguf_converter /path/to/model output/ --no-quantize
    ```
*   **List all available quantization types:**
    ```bash
    python -m gguf_converter --list-types
    ```
*   **Full CLI options:** Refer to `python -m gguf_converter --help` for all arguments.

## Development Conventions

*   **Python Typing:** The codebase utilizes type hints for improved readability and maintainability.
*   **Dependency Management:** `requirements.txt` lists core dependencies, with `setup/setup.py` handling package metadata and entry points. `streamlit` is an optional `extra_require` for the GUI.
*   **`llama.cpp` Integration:** Interaction with `llama.cpp` tools (`convert_hf_to_gguf.py`, `llama-quantize`, `llama-imatrix`) is managed via `subprocess` calls. The `BinaryManager` ensures these tools are available.
*   **Configuration:** The GUI stores its settings in a JSON file at `~/.gguf_converter_config.json`.
*   **Structure:**
    *   `gguf_converter/`: Main Python package containing application logic.
        *   `cli.py`: Command-line interface entry point.
        *   `gui.py`: Streamlit GUI application.
        *   `converter.py`: Core conversion, quantization, and imatrix generation logic.
        *   `binary_manager.py`: Handles `llama.cpp` binary download and management.
        *   `quantization/`: Contains definitions for various quantization types and related logic.
    *   `llama.cpp/`: Cloned `llama.cpp` repository (if not already present) for scripts like `convert_hf_to_gguf.py`.
    *   `bin/`: Directory where downloaded `llama.cpp` executables are stored.
    *   `calibration_data/`: Directory containing calibration text files for importance matrix generation.
        *   `_default.txt`: Default calibration data file.
    *   `run_gui.bat`, `run_gui.sh`: Platform-specific scripts for easy setup and GUI launch.
