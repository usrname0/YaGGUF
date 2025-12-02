#!/usr/bin/env python3
"""
Command-line interface for GGUF Converter
"""

import argparse
import sys
from pathlib import Path
from typing import List

from .converter import GGUFConverter


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Yet Another GGUF Converter - Easy GGUF conversion and quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert and quantize a local model (parallel mode is default)
  python -m gguf_converter /path/to/model output/ -q Q4_K_M Q5_K_M

  # Download from HuggingFace and convert
  python -m gguf_converter username/model-name output/ -q Q4_K_M

  # Use serial mode if needed (parallel is 7-8x faster and default)
  python -m gguf_converter /path/to/model output/ -q Q4_K_M --serial

  # Specify number of workers (default is CPU cores - 1)
  python -m gguf_converter /path/to/model output/ -q Q4_K_M --workers 4

  # Just convert without quantization
  python -m gguf_converter /path/to/model output/ --no-quantize

  # List available quantization types
  python -m gguf_converter --list-types
        """
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="Path to model directory or HuggingFace repo ID (e.g., username/model-name)"
    )

    parser.add_argument(
        "output",
        nargs="?",
        help="Output directory for converted files"
    )

    parser.add_argument(
        "-q", "--quantize",
        nargs="+",
        default=["Q4_K_M"],
        help="Quantization types to create (default: Q4_K_M). Use multiple types like: -q Q4_K_M Q5_K_M Q8_0"
    )

    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Only convert to GGUF, skip quantization"
    )

    parser.add_argument(
        "--intermediate",
        choices=["f16", "f32"],
        default="f16",
        help="Intermediate format for conversion (default: f16)"
    )

    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the intermediate f16/f32 GGUF file"
    )

    parser.add_argument(
        "--output-type",
        choices=["f16", "f32", "bf16", "q8_0", "auto"],
        help="Output type if --no-quantize is used (default: f16)"
    )

    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="Only extract vocabulary"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--serial",
        action="store_true",
        help="Disable parallel processing (parallel is default, 7-8x faster)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel mode (default: CPU cores - 1)"
    )

    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List all available quantization types and exit"
    )

    args = parser.parse_args()

    # Handle --list-types
    if args.list_types:
        print("Available quantization types:")
        print()
        converter = GGUFConverter()
        for qtype in sorted(converter.QUANTIZATION_TYPES.keys()):
            print(f"  {qtype}")
        print()
        print("Recommended types:")
        print("  Q4_K_M  - Good balance of size and quality (recommended)")
        print("  Q5_K_M  - Higher quality, larger size")
        print("  Q8_0    - Very high quality, close to f16")
        print("  IQ4_XS  - Smallest size with decent quality")
        return 0

    # Validate required arguments
    if not args.model or not args.output:
        parser.print_help()
        print("\nError: model and output arguments are required", file=sys.stderr)
        return 1

    try:
        converter = GGUFConverter()

        if args.no_quantize:
            # Just convert, no quantization
            output_type = args.output_type or "f16"
            output_path = Path(args.output) / f"{Path(args.model).name}_{output_type}.gguf"

            converter.convert_to_gguf(
                model_path=args.model,
                output_path=output_path,
                output_type=output_type,
                vocab_only=args.vocab_only,
                verbose=args.verbose
            )

            print(f"\nSuccess! Created: {output_path}")

        else:
            # Convert and quantize
            output_files = converter.convert_and_quantize(
                model_path=args.model,
                output_dir=args.output,
                quantization_types=args.quantize,
                intermediate_type=args.intermediate,
                keep_intermediate=args.keep_intermediate,
                verbose=args.verbose,
                parallel=not args.serial,
                num_workers=args.workers
            )

            print(f"\nSuccess! Created {len(output_files)} files:")
            for file_path in output_files:
                file_size = file_path.stat().st_size / (1024**3)  # GB
                print(f"  - {file_path.name} ({file_size:.2f} GB)")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr, flush=True)
        # Always show traceback for debugging (not just in verbose mode)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
