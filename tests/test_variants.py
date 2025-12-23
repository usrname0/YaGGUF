#!/usr/bin/env python3
"""
Test all quantized GGUF variants with llama-server
Loads each variant in llama-server, waits for manual testing, then moves to next
python test_variants.py E:\\LLM\\lmstudio-community\\Qwen3-VL-4B-Instruct-GGUF --exclude f16
"""

import subprocess
import sys
from pathlib import Path
import time
import shutil
import socket
import webbrowser
import argparse


def test_gguf_file(gguf_path: Path, server_path: Path, mmproj_path: Path = None, port: int = 8080) -> None:
    """
    Load a GGUF file in llama-server and wait for user testing

    Args:
        gguf_path: Path to GGUF file to test
        server_path: Path to llama-server executable
        mmproj_path: Path to mmproj file for multimodal models (optional)
        port: Port for the server (default 8080)
    """
    print(f"\n{'='*70}")
    print(f"Testing: {gguf_path.name}")
    print(f"Size: {gguf_path.stat().st_size / (1024**3):.2f} GB")
    print(f"{'='*70}")

    # Build llama-server command
    cmd = [
        str(server_path),
        "-m", str(gguf_path),
        "--port", str(port),
        "--host", "127.0.0.1",
    ]

    # Add mmproj if available (for vision-language models)
    if mmproj_path and mmproj_path.exists():
        cmd.extend(["--mmproj", str(mmproj_path)])
        print(f"Using mmproj: {mmproj_path.name}")

    print(f"\nStarting llama-server...")
    print(f"Command: {' '.join(cmd)}")
    print(f"\n--- Server Output ---")

    try:
        # Start the server process - let output go to console
        process = subprocess.Popen(cmd)

        # Wait for server to be ready
        print(f"\nWaiting for server to start (this may take 10-30 seconds)...")
        max_wait = 60  # Max 60 seconds
        waited = 0

        while waited < max_wait:
            # Check if process crashed
            if process.poll() is not None:
                print(f"\nServer failed to start (exit code: {process.returncode})")
                return

            # Try to connect to the server to see if it's ready
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                if result == 0:
                    # Server is accepting connections!
                    break
            except:
                pass

            time.sleep(1)
            waited += 1

        if waited >= max_wait:
            print(f"\nServer did not start within {max_wait} seconds")
            process.terminate()
            return

        print(f"\n{'='*70}")
        print(f"Server is ready!")
        print(f"\nServer URL: http://127.0.0.1:{port}")
        print(f"\nOpening browser automatically...")
        print(f"Press ENTER when you're done testing to move to the next model...")
        print(f"{'='*70}")

        # Open browser automatically
        server_url = f"http://127.0.0.1:{port}"
        webbrowser.open(server_url)

        # Wait for user input
        input()

        # Terminate the server
        print(f"\nStopping server...")
        process.terminate()

        # Give it a moment to shut down gracefully
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            process.kill()
            process.wait()

        print(f"Server stopped")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user. Stopping server...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        if 'process' in locals():
            try:
                process.terminate()
            except:
                pass


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="GGUF Variant Interactive Test Suite",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""\
Examples:
  python test_variants.py output/my-model
  python test_variants.py output/my-model --exclude F16 --exclude Q4_0
  python test_variants.py output/my-model --exclude "*_K_S"
"""
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing GGUF model files"
    )
    parser.add_argument(
        "--server-path",
        type=Path,
        help="Path to llama-server executable (optional, will be auto-detected)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the server (default: 8080)"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        action="append",
        default=[],
        help="Exclude models by substring (e.g., 'F16', 'Q4_0') or glob pattern (e.g., '*_K_S').\nCan be used multiple times."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GGUF Variant Interactive Test Suite")
    print("=" * 70)

    model_dir = args.model_dir
    port = args.port
    server_path = args.server_path
    exclude_patterns = args.exclude

    if not model_dir.exists() or not model_dir.is_dir():
        print(f"\nError: Directory not found or is not a directory: {model_dir}")
        return 1

    # Find llama-server executable if not provided
    if not server_path:
        # Try to find on PATH first
        for exe_name in ["llama-server", "llama-server.exe"]:
            found = shutil.which(exe_name)
            if found:
                server_path = Path(found)
                print(f"\nFound {exe_name} on PATH")
                break

        # If not on PATH, try common local locations
        if not server_path:
            possible_paths = [
                Path("llama.cpp/build/bin/llama-server.exe"),
                Path("llama.cpp/build/bin/llama-server"),
                Path("llama.cpp/build/bin/Release/llama-server.exe"),
                Path("llama.cpp/llama-server.exe"),
                Path("llama.cpp/llama-server"),
            ]
            for path in possible_paths:
                if path.exists():
                    server_path = path
                    break

    if not server_path or not server_path.exists():
        print("\nError: Could not find or verify llama-server executable.")
        print("Please either:")
        print("  1. Add llama-server to your PATH, or")
        print("  2. Specify the path using --server-path")
        return 1

    print(f"\nModel directory: {model_dir}")
    print(f"llama-server path: {server_path}")
    print(f"Port: {port}")
    if exclude_patterns:
        print(f"Excluding models matching: {', '.join(exclude_patterns)}")

    # Find all GGUF files
    all_gguf_files = sorted(model_dir.glob("*.gguf"))

    # Separate mmproj files from model files
    mmproj_files = [f for f in all_gguf_files if "mmproj" in f.name.lower()]
    gguf_files = [f for f in all_gguf_files if "mmproj" not in f.name.lower()]

    # Filter out excluded files
    if exclude_patterns:
        print("") # Add a newline for cleaner output
        included_files = []
        for f in gguf_files:
            is_excluded = False
            for pattern in exclude_patterns:
                # If pattern contains a wildcard, use glob matching. Otherwise, use case-insensitive string containment.
                has_wildcard = any(c in pattern for c in '*?[]')
                if has_wildcard:
                    if f.match(pattern):
                        print(f"Excluding '{f.name}' (matches glob pattern: '{pattern}')")
                        is_excluded = True
                        break
                else:
                    if pattern.lower() in f.name.lower():
                        print(f"Excluding '{f.name}' (contains string: '{pattern}')")
                        is_excluded = True
                        break
            if not is_excluded:
                included_files.append(f)
        gguf_files = included_files


    if not gguf_files:
        print(f"\nError: No .gguf model files found (or all were excluded) in {model_dir}")
        if mmproj_files:
            print(f"(Found {len(mmproj_files)} mmproj file(s), but no model files)")
        return 1

    # Find the mmproj file if it exists (for vision-language models)
    mmproj_path = mmproj_files[0] if mmproj_files else None
    if mmproj_path:
        print(f"\nFound mmproj file: {mmproj_path.name}")
        print(f"This will be used for multimodal (vision) capabilities")

    print(f"\nFound {len(gguf_files)} GGUF model file(s) to test")
    print(f"\nEach model will be loaded in llama-server.")
    print(f"Test the model in your browser, then press ENTER to continue.\n")

    # Test each file
    try:
        for i, gguf_file in enumerate(gguf_files, 1):
            print(f"\n[{i}/{len(gguf_files)}]", end=" ")
            test_gguf_file(gguf_file, server_path, mmproj_path, port)
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        return 1

    # Done
    print("\n" + "=" * 70)
    print("All models tested!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
