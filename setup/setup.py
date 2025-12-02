"""
Setup script for Yet Another GGUF Converter
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="gguf-converter",
    version="0.1.0",
    author="usrname0",
    description="Pure Python GGUF converter - easy installation, easy use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usrname0/Yet_Another_GGUF_Converter",
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
        "huggingface-hub>=0.20.0",
        "numpy>=1.24.0",
        "gguf>=0.1.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "gui": ["streamlit>=1.30.0"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "gguf-converter=gguf_converter.cli:main",
        ],
    },
)
