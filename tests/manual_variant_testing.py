#!/usr/bin/env python3
"""
Test all quantized GGUF variants with llama-server
Loads each variant in llama-server, waits for manual testing, then moves to next
python manual_variant_testing.py E:\\LLM\\lmstudio-community\\Qwen3-VL-4B-Instruct-GGUF --exclude f16
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional
import time
import shutil
import socket
import webbrowser
import argparse
import platform
import threading
import queue

# Add parent directory to path to import YaGGUF modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from gguf_converter.theme import THEME as theme
from colorama import Style


def print_success_message():
    """Print success message and instructions"""
    print(f"\n{theme['success']}{'='*70}{Style.RESET_ALL}")
    print(f"{theme['info']}Instructions for next model:{Style.RESET_ALL}")
    print(f"{theme['info']}  1. Close the browser tab/window{Style.RESET_ALL}")
    print(f"{theme['info']}  2. Return to this terminal{Style.RESET_ALL}")
    print(f"{theme['info']}  3. Press ENTER below to load the next variant{Style.RESET_ALL}")
    print(f"\n{theme['success']}>>> Press ENTER to load the next model >>> {Style.RESET_ALL}")
    print(f"{theme['success']}{'='*70}{Style.RESET_ALL}")


def monitor_server_output(pipe, output_queue, ready_event, completion_event=None):
    """
    Monitor server output in a separate thread, print to console, and detect when ready

    Args:
        pipe: stdout/stderr pipe from subprocess
        output_queue: Queue to signal completion
        ready_event: Event to set when "all slots are idle" is detected
        completion_event: Event to set when a chat completion is successfully finished
    """
    try:
        for line in iter(pipe.readline, b''):
            if line:
                try:
                    decoded = line.decode('utf-8', errors='replace')
                    print(decoded, end='', flush=True)

                    # Look for the signal that server is fully initialized
                    if 'all slots are idle' in decoded:
                        ready_event.set()
                    
                    # Look for successful chat completion (POST /v1/chat/completions ... 200)
                    if completion_event and 'POST /v1/chat/completions' in decoded and ' 200' in decoded:
                        print_success_message()
                        completion_event.set()

                except Exception:
                    pass
    except Exception:
        pass
    finally:
        output_queue.put(None)


def test_gguf_file(gguf_path: Path, server_path: Path, mmproj_path: Optional[Path] = None, port: int = 8080) -> None:
    """
    Load a GGUF file in llama-server and wait for user testing

    Args:
        gguf_path: Path to GGUF file to test
        server_path: Path to llama-server executable
        mmproj_path: Path to mmproj file for multimodal models (optional)
        port: Port for the server (default 8080)
    """
    print(f"\n{theme['info']}{'='*70}{Style.RESET_ALL}")
    print(f"{theme['info']}Testing: {gguf_path.name}{Style.RESET_ALL}")
    print(f"{theme['metadata']}Size: {gguf_path.stat().st_size / (1024**3):.2f} GB{Style.RESET_ALL}")
    print(f"{theme['info']}{'='*70}{Style.RESET_ALL}")

    # Build llama-server command
    cmd = [
        str(server_path),
        "-m", str(gguf_path),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-c", "4096",  # Context size
        "-ngl", "99",  # GPU layers (will use what's available)
    ]

    # Add mmproj if available (for vision-language models)
    if mmproj_path and mmproj_path.exists():
        cmd.extend(["--mmproj", str(mmproj_path)])
        print(f"{theme['metadata']}Using mmproj: {mmproj_path.name}{Style.RESET_ALL}")

    print(f"\n{theme['success']}Starting llama-server...{Style.RESET_ALL}")
    print(f"{theme['highlight']}Command: {' '.join(cmd)}{Style.RESET_ALL}")
    print(f"\n{theme['info']}--- Server Output ---{Style.RESET_ALL}")

    try:
        # Start the server process - capture output to monitor it
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            bufsize=1
        )

        # Create event to signal when server is fully ready
        ready_event = threading.Event()
        completion_event = threading.Event()
        output_queue = queue.Queue()

        # Start thread to monitor server output
        monitor_thread = threading.Thread(
            target=monitor_server_output,
            args=(process.stdout, output_queue, ready_event, completion_event),
            daemon=True
        )
        monitor_thread.start()

        # Wait for server to be ready (accepting connections)
        print(f"\n{theme['info']}Waiting for server to start (this may take 10-30 seconds)...{Style.RESET_ALL}")
        max_wait = 60  # Max 60 seconds
        waited = 0

        while waited < max_wait:
            # Check if process crashed
            if process.poll() is not None:
                print(f"\n{theme['error']}Server failed to start (exit code: {process.returncode}){Style.RESET_ALL}")
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
            print(f"\n{theme['error']}Server did not start within {max_wait} seconds{Style.RESET_ALL}")
            process.terminate()
            return

        print(f"\n{theme['info']}{'='*70}{Style.RESET_ALL}")
        print(f"{theme['success']}SERVER IS READY!{Style.RESET_ALL}")
        print(f"{theme['info']}Server URL: http://127.0.0.1:{port}{Style.RESET_ALL}")
        print(f"{theme['info']}{'='*70}{Style.RESET_ALL}")

        # Wait for server to finish initialization logging
        # Look for "all slots are idle" in the output
        print(f"\n{theme['info']}Waiting for server to finish initialization...{Style.RESET_ALL}")
        server_url = f"http://127.0.0.1:{port}"

        if not ready_event.wait(timeout=30):
            # Timeout after 30 seconds, but continue anyway
            print(f"{theme['warning']}(Initialization detection timed out, but server should be ready){Style.RESET_ALL}")
            webbrowser.open(server_url)
            time.sleep(3)
        else:
            # Server is initialized, open browser
            # This will trigger GET requests that appear in logs
            webbrowser.open(server_url)

            # Wait for browser to make its initial requests (GET /, /props, /v1/models)
            time.sleep(3)
            
        # Wait for completion request in browser
        print(f"\n{theme['success']}Waiting for you to test the model in the browser...{Style.RESET_ALL}")
        while not completion_event.is_set():
            if process.poll() is not None:
                print(f"\n{theme['error']}Server stopped unexpectedly.{Style.RESET_ALL}")
                return
            time.sleep(0.5)

        # Wait for user input (prompt printed by monitor thread)
        try:
            input()
            print()  # Blank line after input
        except EOFError:
            # Handle Ctrl+D gracefully
            print(f"\n{theme['info']}Received EOF, moving to next model...{Style.RESET_ALL}")
            print()

        # Terminate the server
        print(f"\n{theme['info']}Stopping server...{Style.RESET_ALL}")
        process.terminate()

        # Give it a moment to shut down gracefully
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            process.kill()
            process.wait()

        print(f"{theme['success']}Server stopped{Style.RESET_ALL}")

    except KeyboardInterrupt:
        print(f"\n\n{theme['warning']}Interrupted by user. Stopping server...{Style.RESET_ALL}")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        raise
    except Exception as e:
        print(f"{theme['error']}ERROR: {e}{Style.RESET_ALL}")
        if 'process' in locals():
            try:
                process.terminate()
            except:
                pass


def get_yagguf_server_path() -> Optional[Path]:
    """
    Get llama-server path from YaGGUF configuration

    Returns:
        Path to llama-server executable, or None if not found
    """
    # Try to load YaGGUF config
    try:
        # Add parent directory to path to import YaGGUF modules
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))

        from gguf_converter.gui_utils import load_config
        from gguf_converter.llama_cpp_manager import LlamaCppManager

        config = load_config()

        # Check if using custom binaries
        custom_binaries = None
        custom_repo = None

        if config.get("use_custom_binaries", False):
            custom_binaries = config.get("custom_binaries_folder", "")
            if custom_binaries:
                print(f"{theme['info']}Found custom binaries config: {custom_binaries}{Style.RESET_ALL}")

        if config.get("use_custom_conversion_script", False):
            custom_repo = config.get("custom_llama_cpp_repo", "")

        # Initialize LlamaCppManager with custom settings
        manager = LlamaCppManager(
            custom_binaries_folder=custom_binaries if custom_binaries else None
        )

        # Get llama-server path
        try:
            server_path = manager.get_server_path()
        except RuntimeError:
            server_path = manager.bin_dir / "llama-server.exe" # Fallback to default check

        if server_path.exists():
            return server_path
        else:
            return None

    except Exception as e:
        print(f"{theme['warning']}Warning: Could not load YaGGUF configuration: {e}{Style.RESET_ALL}")
        return None


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="GGUF Variant Interactive Test Suite - Interactively test all quantized variants with llama-server",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""\
Examples:
  python manual_variant_testing.py E:\\LLM\\my-model
  python manual_variant_testing.py E:\\LLM\\my-model --exclude F16 --exclude Q4_0
  python manual_variant_testing.py E:\\LLM\\my-model --exclude "*_K_S"
  python manual_variant_testing.py E:\\LLM\\my-model --port 9090

This script will:
  1. Find all .gguf files in the specified directory
  2. Load each variant in llama-server (using YaGGUF's configured server)
  3. Open the server in your browser for testing
  4. Wait for you to press ENTER to move to the next variant
  5. Automatically stop the server and start the next one
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

    print(f"{theme['info']}{'=' * 70}{Style.RESET_ALL}")
    print(f"{theme['success']}GGUF Variant Interactive Test Suite{Style.RESET_ALL}")
    print(f"{theme['info']}{'=' * 70}{Style.RESET_ALL}")

    model_dir = args.model_dir
    port = args.port
    server_path = args.server_path
    exclude_patterns = args.exclude

    if not model_dir.exists() or not model_dir.is_dir():
        print(f"\n{theme['error']}Error: Directory not found or is not a directory: {model_dir}{Style.RESET_ALL}")
        return 1

    # Find llama-server executable if not provided
    if not server_path:
        # Try YaGGUF configuration first
        yagguf_server = get_yagguf_server_path()
        if yagguf_server:
            server_path = yagguf_server
        else:
            # Try to find on PATH
            for exe_name in ["llama-server", "llama-server.exe"]:
                found = shutil.which(exe_name)
                if found:
                    server_path = Path(found)
                    print(f"\n{theme['success']}Found {exe_name} on PATH{Style.RESET_ALL}")
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
        print(f"\n{theme['error']}Error: Could not find or verify llama-server executable.{Style.RESET_ALL}")
        print(f"{theme['info']}Please either:{Style.RESET_ALL}")
        print(f"{theme['info']}  1. Add llama-server to your PATH, or{Style.RESET_ALL}")
        print(f"{theme['info']}  2. Specify the path using --server-path{Style.RESET_ALL}")
        return 1

    # Get server version
    try:
        result = subprocess.run(
            [str(server_path), "--version"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=5
        )
        if result.returncode == 0:
            output = result.stderr if result.stderr else result.stdout
            # Look for version line
            version_line = None
            for line in output.split('\n'):
                if line.startswith('version:'):
                    version_line = line.strip()
                    break
            # Extract version number
            if version_line:
                import re
                match = re.search(r'\b(b?\d{4,5})\b', version_line)
                if match:
                    version = match.group(1)
                    if not version.startswith('b'):
                        version = f'b{version}'
                else:
                    version = 'unknown'
            else:
                version = 'unknown'
        else:
            version = 'unknown'
    except Exception:
        version = 'unknown'

    print(f"\n{theme['success']}llama-server: {version} ({server_path.parent}){Style.RESET_ALL}")
    print(f"{theme['info']}Model directory: {model_dir}{Style.RESET_ALL}")
    print(f"{theme['info']}Port: {port}{Style.RESET_ALL}")
    if exclude_patterns:
        print(f"{theme['info']}Excluding models matching: {', '.join(exclude_patterns)}{Style.RESET_ALL}")

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
                        print(f"{theme['warning']}Excluding '{f.name}' (matches glob pattern: '{pattern}'){Style.RESET_ALL}")
                        is_excluded = True
                        break
                else:
                    if pattern.lower() in f.name.lower():
                        print(f"{theme['warning']}Excluding '{f.name}' (contains string: '{pattern}'){Style.RESET_ALL}")
                        is_excluded = True
                        break
            if not is_excluded:
                included_files.append(f)
        gguf_files = included_files


    if not gguf_files:
        print(f"\n{theme['error']}Error: No .gguf model files found (or all were excluded) in {model_dir}{Style.RESET_ALL}")
        if mmproj_files:
            print(f"{theme['info']}(Found {len(mmproj_files)} mmproj file(s), but no model files){Style.RESET_ALL}")
        return 1

    # Find the mmproj file if it exists (for vision-language models)
    mmproj_path = mmproj_files[0] if mmproj_files else None
    if mmproj_path:
        print(f"\n{theme['success']}Found mmproj file: {mmproj_path.name}{Style.RESET_ALL}")
        print(f"{theme['info']}This will be used for multimodal (vision) capabilities{Style.RESET_ALL}")

    print(f"\n{theme['success']}Found {len(gguf_files)} GGUF model file(s) to test{Style.RESET_ALL}")
    print(f"\n{theme['info']}How this works:{Style.RESET_ALL}")
    print(f"{theme['info']}  1. Each model will be loaded in llama-server{Style.RESET_ALL}")
    print(f"{theme['info']}  2. Your browser will open automatically to the chat interface{Style.RESET_ALL}")
    print(f"{theme['info']}  3. Test the model (try different prompts, check quality, speed, etc.){Style.RESET_ALL}")
    print(f"{theme['info']}  4. When done, return to this terminal and press ENTER{Style.RESET_ALL}")
    print(f"{theme['info']}  5. The server will stop and the next variant will load{Style.RESET_ALL}")

    # Test each file
    try:
        for i, gguf_file in enumerate(gguf_files, 1):
            print(f"\n{theme['info']}[{i}/{len(gguf_files)}]{Style.RESET_ALL}", end=" ")
            test_gguf_file(gguf_file, server_path, mmproj_path, port)
    except KeyboardInterrupt:
        print(f"\n\n{theme['warning']}Testing interrupted by user.{Style.RESET_ALL}")
        return 1

    # Done
    print(f"\n{theme['info']}{'=' * 70}{Style.RESET_ALL}")
    print(f"{theme['success']}All models tested!{Style.RESET_ALL}")
    print(f"{theme['info']}{'=' * 70}{Style.RESET_ALL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
