"""
Standalone script to download llama.cpp binaries
Can be run as: python -m gguf_converter.download_binaries
"""

import sys
import argparse
from pathlib import Path
from .binary_manager import BinaryManager


def main() -> int:
    """
    Download llama.cpp binaries
    """
    parser = argparse.ArgumentParser(
        description="Download llama.cpp binaries for GGUF conversion"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if binaries exist"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Don't fallback to system PATH if download fails"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("GGUF Converter - Binary Download")
    print("=" * 50)
    print()

    try:
        manager = BinaryManager()

        # Ensure binaries are available
        if manager.ensure_binaries(fallback_to_system=not args.no_fallback):
            print()
            print("=" * 50)
            print("Binaries ready!")
            print("=" * 50)
            print()
            print(f"llama-quantize: {manager.get_quantize_path()}")
            print(f"llama-imatrix:  {manager.get_imatrix_path()}")
            print()
            return 0
        else:
            print()
            print("=" * 50)
            print("ERROR: Failed to get binaries")
            print("=" * 50)
            print()
            print("Could not download or find llama.cpp binaries.")
            print("Please check your internet connection or install llama.cpp manually.")
            print()
            return 1

    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        return 1
    except Exception as e:
        print()
        print("=" * 50)
        print("ERROR")
        print("=" * 50)
        print()
        print(f"An error occurred: {e}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
