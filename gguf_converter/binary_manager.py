"""
Binary manager for downloading and managing llama.cpp executables
"""

import os
import platform
import sys
import zipfile
import tarfile
import shutil
import json
import stat
import socket
from pathlib import Path
from typing import Optional, Dict
from urllib.request import urlretrieve, urlopen


def remove_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree to handle read-only files on Windows

    Args:
        func: Function that raised the exception
        path: Path to the file
        excinfo: Exception information
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


class BinaryManager:
    """
    Manages llama.cpp binary downloads and locations
    """

    LLAMA_CPP_VERSION = "b7558"
    RELEASE_URL_TEMPLATE = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{filename}"

    def __init__(self, bin_dir: Optional[Path] = None, custom_binaries_folder: Optional[str] = None):
        """
        Initialize binary manager

        Args:
            bin_dir: Directory to store binaries (default: ./bin/)
            custom_binaries_folder: Optional path to folder containing custom llama.cpp binaries.
                                   If empty string, will use system PATH.
                                   If None, will use auto-downloaded binaries.
        """
        if bin_dir is None:
            project_root = Path(__file__).parent.parent
            bin_dir = project_root / "bin"

        self.bin_dir = Path(bin_dir)
        self.bin_dir.mkdir(parents=True, exist_ok=True)

        self.custom_binaries_folder = custom_binaries_folder
        self.platform_info = self._detect_platform()

    def _detect_platform(self) -> Dict[str, str]:
        """
        Detect current platform and architecture

        Returns:
            Dict with 'os', 'arch', and 'filename' keys
        """
        system = platform.system().lower()
        machine = platform.machine().lower()

        if machine in ('x86_64', 'amd64', 'x64'):
            arch = 'x64'
        elif machine in ('arm64', 'aarch64'):
            arch = 'arm64'
        else:
            raise RuntimeError(f"Unsupported architecture: {machine}")

        if system == 'windows':
            os_name = 'win'
            build_type = 'cpu'
            variant = arch
            ext = 'zip'
            filename = f"llama-{self.LLAMA_CPP_VERSION}-bin-{os_name}-{build_type}-{arch}.{ext}"
        elif system == 'linux':
            os_name = 'ubuntu'
            variant = arch
            ext = 'tar.gz'
            filename = f"llama-{self.LLAMA_CPP_VERSION}-bin-{os_name}-{variant}.{ext}"
        elif system == 'darwin':
            os_name = 'macos'
            variant = arch
            ext = 'zip'
            filename = f"llama-{self.LLAMA_CPP_VERSION}-bin-{os_name}-{variant}.{ext}"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        return {
            'os': os_name,
            'arch': arch,
            'variant': variant,
            'filename': filename,
            'ext': ext
        }

    def _progress_hook(self, block_num, block_size, total_size):
        """
        Progress callback for urlretrieve
        """
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100.0 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

    def _check_disk_space(self, required_bytes: int, path: Optional[Path] = None) -> bool:
        """
        Check if there is enough disk space available

        Args:
            required_bytes: Minimum bytes required
            path: Path to check (default: bin_dir)

        Returns:
            True if enough space available, False otherwise
        """
        check_path = path if path else self.bin_dir
        try:
            stat = shutil.disk_usage(check_path)
            available_mb = stat.free / (1024 * 1024)
            required_mb = required_bytes / (1024 * 1024)

            if stat.free < required_bytes:
                print(f"ERROR: Insufficient disk space")
                print(f"  Required: {required_mb:.1f} MB")
                print(f"  Available: {available_mb:.1f} MB")
                return False
            return True
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
            return True  # Proceed anyway if check fails

    def _check_network_connectivity(self, host: str = "api.github.com", port: int = 443, timeout: int = 5) -> bool:
        """
        Check if network connection is available

        Args:
            host: Host to check connectivity to
            port: Port to connect to
            timeout: Connection timeout in seconds

        Returns:
            True if network is available, False otherwise
        """
        try:
            socket.create_connection((host, port), timeout=timeout)
            return True
        except (socket.error, socket.timeout):
            return False

    def get_latest_version(self) -> str:
        """
        Get the latest llama.cpp release tag from GitHub

        Returns:
            Latest release tag (e.g., "b7493")
        """
        try:
            api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
            with urlopen(api_url) as response:
                data = json.loads(response.read().decode())
                return data['tag_name']
        except Exception as e:
            print(f"Warning: Could not fetch latest version: {e}")
            print(f"Falling back to recommended version: {self.LLAMA_CPP_VERSION}")
            return self.LLAMA_CPP_VERSION

    def download_binaries(self, force: bool = False, version: Optional[str] = None) -> Path:
        """
        Download llama.cpp binaries if not already present

        Args:
            force: Force re-download even if binaries exist
            version: Specific version to download (None = use recommended LLAMA_CPP_VERSION)

        Returns:
            Path to bin directory containing executables
        """
        if not force and self._binaries_exist():
            print(f"Binaries already exist in {self.bin_dir}")
            return self.bin_dir

        # Use specified version or default to recommended
        tag = version if version else self.LLAMA_CPP_VERSION

        # Update filename to use the specified version
        os_name = self.platform_info['os']
        arch = self.platform_info['arch']

        if os_name == "win":
            build_type = "cuda-cu12.2.0" if self.platform_info.get('cuda') else "cpu"
            ext = "zip"
            filename = f"llama-{tag}-bin-{os_name}-{build_type}-{arch}.{ext}"
        elif os_name == "mac":
            variant = self.platform_info['variant']
            ext = "zip"
            filename = f"llama-{tag}-bin-{os_name}-{variant}.{ext}"
        else:  # linux
            variant = self.platform_info['variant']
            ext = "tar.gz"
            filename = f"llama-{tag}-bin-{os_name}-{variant}.{ext}"

        url = self.RELEASE_URL_TEMPLATE.format(tag=tag, filename=filename)

        print(f"Downloading llama.cpp {tag} for {os_name}-{arch}...")
        print(f"URL: {url}")

        # Check network connectivity before attempting download
        if not self._check_network_connectivity():
            raise RuntimeError(
                "No network connection available. "
                "Please check your internet connection and try again."
            )

        # Check disk space before download (estimate 1 GB needed for download + extraction)
        required_space = 1024 * 1024 * 1024  # 1 GB
        if not self._check_disk_space(required_space):
            raise RuntimeError(
                "Insufficient disk space for binary download. "
                "Please free up at least 1 GB and try again."
            )

        # Clean up old binaries BEFORE downloading
        self._cleanup_old_binaries()

        # Download to temporary file
        download_path = self.bin_dir / filename

        try:
            urlretrieve(url, download_path, reporthook=self._progress_hook)
            print()  # New line after progress
            print(f"Downloaded to {download_path}")
        except Exception as e:
            print(f"\nERROR: Failed to download binaries: {e}")
            raise RuntimeError(
                f"Failed to download llama.cpp binaries from {url}. "
                f"Please check your internet connection or download manually."
            )

        # Extract archive
        print(f"Extracting {filename}...")
        self._extract_archive(download_path)

        # Clean up archive
        download_path.unlink()
        print(f"Extraction complete")

        # Verify binaries exist
        if not self._check_binary_files_exist():
            raise RuntimeError("Binary extraction succeeded but executables not found")

        print(f"Installed binary version: {tag}")
        print(f"Binaries ready in {self.bin_dir}")
        return self.bin_dir

    def _cleanup_old_binaries(self):
        """
        Remove all old files and directories from bin_dir before installing new binaries
        """
        if not self.bin_dir.exists():
            return

        print("Removing old binaries...")
        # Remove everything in bin_dir (files and directories)
        for item in self.bin_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, onerror=remove_readonly)
            else:
                item.unlink()

    def _extract_archive(self, archive_path: Path):
        """
        Extract archive and flatten to bin_dir root (consistent across platforms)
        """
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(self.bin_dir)
        else:
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(self.bin_dir)

        # Flatten: if extracted to a subdirectory, move all files to bin root
        subdirs = [d for d in self.bin_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            # Single subdirectory - likely the llama-* folder
            subdir = subdirs[0]
            print(f"Flattening extracted directory: {subdir.name}")

            # Move all files from subdirectory to bin root
            for item in subdir.iterdir():
                dest = self.bin_dir / item.name
                # Remove destination if it exists
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest, onerror=remove_readonly)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))

            # Remove empty subdirectory
            subdir.rmdir()

        # On Unix systems, set execute permissions on binaries
        if self.platform_info['os'] != 'win':
            self._set_execute_permissions()

    def _set_execute_permissions(self):
        """
        Set execute permissions on binary files (Unix only)
        """
        import stat

        # Set execute permissions on llama-* binaries (not .so files)
        for path in self.bin_dir.glob('llama-*'):
            if path.is_file() and not path.suffix:  # No extension (not .so files)
                try:
                    # Add execute permission for owner, group, and others
                    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except Exception as e:
                    print(f"Warning: Could not set execute permission on {path}: {e}")

    def _check_binary_files_exist(self) -> bool:
        """
        Check if required binary files exist (without version check)
        """
        required = ['llama-quantize', 'llama-imatrix']

        for binary in required:
            if not self.get_binary_path(binary).exists():
                return False

        return True

    def _binaries_exist(self) -> bool:
        """
        Check if required binaries exist
        Does NOT check version - use Upgrade tab in GUI for updates
        """
        # Only check if required binary files exist
        return self._check_binary_files_exist()

    def get_binary_path(self, name: str) -> Path:
        """
        Get path to a specific binary

        Args:
            name: Binary name (e.g., 'llama-quantize', 'llama-imatrix')

        Returns:
            Path to the executable
        """
        # On Windows, add .exe extension
        if self.platform_info['os'] == 'win':
            name = f"{name}.exe"

        # Binaries are now extracted flat to bin_dir root
        return self.bin_dir / name

    def ensure_binaries(self, fallback_to_system: bool = True) -> bool:
        """
        Ensure binaries are available, download if needed

        Args:
            fallback_to_system: If True, check system PATH if download fails

        Returns:
            True if binaries are available
        """
        # Try to use existing/downloaded binaries
        if self._binaries_exist():
            return True

        # Try to download
        try:
            self.download_binaries()
            return True
        except Exception as e:
            print(f"Binary download failed: {e}")

            if not fallback_to_system:
                return False

            # Fallback: check if binaries are in system PATH
            print("Checking system PATH for llama.cpp binaries...")
            if self._check_system_binaries():
                print("Found llama.cpp binaries in system PATH")
                return True

            return False

    def _check_system_binaries(self) -> bool:
        """
        Check if llama.cpp binaries are available in system PATH
        """
        required = ['llama-quantize', 'llama-imatrix']

        for binary in required:
            if shutil.which(binary) is None:
                return False

        return True

    def _get_binaries_folder(self) -> Optional[Path]:
        """
        Determine the folder containing llama.cpp binaries

        Returns:
            Path to binaries folder, or None if using system PATH
        """
        # Check if custom binaries folder is configured
        if self.custom_binaries_folder is not None:
            # If custom folder is blank, use system PATH
            if not self.custom_binaries_folder:
                return None
            # Custom folder specified
            return Path(self.custom_binaries_folder)

        # Use auto-downloaded binaries folder
        return self.bin_dir

    def _get_binary_path_with_fallback(self, binary_name: str) -> Path:
        """
        Get path to a llama.cpp binary with fallback logic

        Args:
            binary_name: Name of binary without extension (e.g., 'llama-quantize')

        Returns:
            Path to the executable

        Raises:
            RuntimeError: If binary cannot be found
        """
        binaries_folder = self._get_binaries_folder()

        # Add .exe extension on Windows
        if self.platform_info['os'] == 'win':
            binary_filename = f'{binary_name}.exe'
        else:
            binary_filename = binary_name

        # Using system PATH
        if binaries_folder is None:
            system_path = shutil.which(binary_name)
            if system_path:
                return Path(system_path)
            else:
                raise RuntimeError(
                    f"Custom binaries enabled with blank path, but '{binary_name}' not found in system PATH"
                )

        # Using custom or auto-downloaded folder
        binary_path = binaries_folder / binary_filename

        # If custom folder, binary must exist there
        if self.custom_binaries_folder:
            if binary_path.exists():
                return binary_path
            else:
                raise RuntimeError(
                    f"{binary_name} not found in custom binaries folder: {binaries_folder}"
                )

        # For auto-downloaded binaries, check multiple possible locations
        # (binaries might be in subdirectories after extraction)
        if not binary_path.exists():
            binary_path = self.get_binary_path(binary_name)

        # Fallback to system PATH if auto-downloaded binaries don't exist
        if not binary_path.exists():
            system_path = shutil.which(binary_name)
            if system_path:
                return Path(system_path)

        return binary_path

    def get_quantize_path(self) -> Path:
        """Get path to llama-quantize executable"""
        return self._get_binary_path_with_fallback('llama-quantize')

    def get_imatrix_path(self) -> Path:
        """Get path to llama-imatrix executable"""
        return self._get_binary_path_with_fallback('llama-imatrix')

    def update_conversion_scripts(self, use_recommended: bool = True) -> Dict[str, str]:
        """
        Update llama.cpp conversion scripts to recommended or latest version

        Args:
            use_recommended: If True, checkout the recommended version (LLAMA_CPP_VERSION).
                           If False, get the latest tagged release and re-clone.

        Returns:
            Dict with 'status' ('success', 'already_updated', 'not_found', 'error')
            and 'message' keys
        """
        import subprocess
        import shutil

        project_root = Path(__file__).parent.parent
        llama_cpp_dir = project_root / "llama.cpp"

        if not llama_cpp_dir.exists():
            return {
                'status': 'not_found',
                'message': 'llama.cpp directory not found'
            }

        if not (llama_cpp_dir / ".git").exists():
            return {
                'status': 'error',
                'message': 'llama.cpp directory is not a git repository'
            }

        try:
            if use_recommended:
                # Checkout the recommended version
                target_version = self.LLAMA_CPP_VERSION
                print(f"Updating conversion scripts to recommended version: {target_version}")

                # Fetch latest tags
                fetch_result = subprocess.run(
                    ["git", "fetch", "--tags", "origin"],
                    cwd=llama_cpp_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if fetch_result.returncode != 0:
                    return {
                        'status': 'error',
                        'message': f"Failed to fetch tags: {fetch_result.stderr}"
                    }

                # Check current version
                current_result = subprocess.run(
                    ["git", "describe", "--tags", "--always"],
                    cwd=llama_cpp_dir,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                current_version = current_result.stdout.strip() if current_result.returncode == 0 else "unknown"

                if target_version in current_version:
                    print(f"Conversion scripts already at version {target_version}")
                    return {
                        'status': 'already_updated',
                        'message': f'Conversion scripts already at recommended version: {target_version}'
                    }

                # Checkout the target version
                checkout_result = subprocess.run(
                    ["git", "checkout", target_version],
                    cwd=llama_cpp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if checkout_result.returncode == 0:
                    print(f"Conversion scripts updated to version {target_version}")
                    return {
                        'status': 'success',
                        'message': f'Conversion scripts updated to recommended version: {target_version}'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f"Failed to checkout version {target_version}: {checkout_result.stderr}"
                    }
            else:
                # Get latest tagged release and re-clone
                print("Fetching latest release version...")
                latest_version = self.get_latest_version()
                print(f"Latest release: {latest_version}")

                # Check current version
                current_result = subprocess.run(
                    ["git", "describe", "--tags", "--always"],
                    cwd=llama_cpp_dir,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                current_version = current_result.stdout.strip() if current_result.returncode == 0 else "unknown"

                if latest_version in current_version:
                    print(f"Conversion scripts already at latest version: {latest_version}")
                    return {
                        'status': 'already_updated',
                        'message': f'Conversion scripts already at latest version: {latest_version}'
                    }

                # Delete and re-clone with latest version
                print(f"Updating to latest version {latest_version}...")
                print("Removing existing llama.cpp repository...")
                shutil.rmtree(llama_cpp_dir, onerror=remove_readonly)

                print(f"Cloning llama.cpp repository at version {latest_version}...")
                clone_result = subprocess.run(
                    [
                        "git", "clone",
                        "https://github.com/ggml-org/llama.cpp.git",
                        "--depth=1",
                        "--branch", latest_version,
                        str(llama_cpp_dir)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if clone_result.returncode == 0:
                    print(f"Conversion scripts updated to latest version: {latest_version}")
                    return {
                        'status': 'success',
                        'message': f'Conversion scripts updated to latest version: {latest_version}'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f"Failed to clone latest version: {clone_result.stderr}"
                    }
        except subprocess.TimeoutExpired:
            error_msg = "Update timed out"
            print(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }
        except Exception as e:
            error_msg = f"Could not update conversion scripts: {e}"
            print(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }
