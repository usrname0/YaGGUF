"""
Setup script for Yet Another GGUF Converter
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
init_file = Path(__file__).parent.parent / "gguf_converter" / "__init__.py"
version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_file.read_text(), re.MULTILINE)
version = version_match.group(1) if version_match else "0.0.0"

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="gguf-converter",
    version=version,
    author="usrname0",
    description="Pure Python GGUF converter - easy installation, easy use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usrname0/YaGGUF",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        # GUI
        "streamlit>=1.40.0",
        # Core dependencies
        "huggingface-hub>=0.36.0",
        "colorama>=0.4.6",
        # Model conversion (llama.cpp convert_hf_to_gguf.py requirements)
        "transformers>=4.50.0",
        "sentencepiece>=0.2.0",
        "numpy>=2.0.0",
        "gguf>=0.15.0",
        "tokenizers>=0.21.0",
        "mistral-common>=1.8.0",
        # Note: PyTorch is installed separately by setup scripts (CPU-only version)
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-mock>=3.10.0"],
    },
    entry_points={
        "console_scripts": [
            "yagguf=gguf_converter.gui:main",
        ],
    },
)
