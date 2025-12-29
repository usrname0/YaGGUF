"""
Standalone script to download llama.cpp binaries
Can be run as: python -m gguf_converter.download_binaries
"""

import sys
import argparse
from pathlib import Path
from .binary_manager import BinaryManager
from .theme import THEME as theme
from colorama import init as colorama_init, Style

# Initialize colorama for cross-platform color support
colorama_init(autoreset=True)


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

    print(f"{theme['highlight']}{'=' * 50}{Style.RESET_ALL}")
    print(f"{theme['highlight']}GGUF Converter - Binary Download{Style.RESET_ALL}")
    print(f"{theme['highlight']}{'=' * 50}{Style.RESET_ALL}")
    print()

    try:
        manager = BinaryManager()

        # Ensure binaries are available
        if manager.ensure_binaries(fallback_to_system=not args.no_fallback):
            print()
            print(f"{theme['success']}{'=' * 50}{Style.RESET_ALL}")
            print(f"{theme['success']}Binaries ready!{Style.RESET_ALL}")
            print(f"{theme['success']}{'=' * 50}{Style.RESET_ALL}")
            print()
            print(f"{theme['info']}llama-quantize: {theme['highlight']}{manager.get_quantize_path()}{Style.RESET_ALL}")
            print(f"{theme['info']}llama-imatrix:  {theme['highlight']}{manager.get_imatrix_path()}{Style.RESET_ALL}")
            print()
            return 0
        else:
            print()
            print(f"{theme['error']}{'=' * 50}{Style.RESET_ALL}")
            print(f"{theme['error']}ERROR: Failed to get binaries{Style.RESET_ALL}")
            print(f"{theme['error']}{'=' * 50}{Style.RESET_ALL}")
            print()
            print(f"{theme['error']}Could not download or find llama.cpp binaries.{Style.RESET_ALL}")
            print(f"{theme['error']}Please check your internet connection or install llama.cpp manually.{Style.RESET_ALL}")
            print()
            return 1

    except KeyboardInterrupt:
        print(f"\n\n{theme['warning']}Download cancelled by user{Style.RESET_ALL}")
        return 1
    except Exception as e:
        print()
        print(f"{theme['error']}{'=' * 50}{Style.RESET_ALL}")
        print(f"{theme['error']}ERROR{Style.RESET_ALL}")
        print(f"{theme['error']}{'=' * 50}{Style.RESET_ALL}")
        print()
        print(f"{theme['error']}An error occurred: {e}{Style.RESET_ALL}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
