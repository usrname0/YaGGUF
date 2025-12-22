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
from pathlib import Path
from typing import Optional, Dict
from urllib.request import urlretrieve, urlopen


class BinaryManager:
    """
    Manages llama.cpp binary downloads and locations
    """

    LLAMA_CPP_VERSION = "b7493"
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
            ext = 'zip'
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

    def _extract_archive(self, archive_path: Path):
        """
        Extract zip or tar archive
        """
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(self.bin_dir)
        else:
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(self.bin_dir)

        # On Unix systems, set execute permissions on binaries
        if platform.system().lower() != 'windows':
            self._set_execute_permissions()

    def _set_execute_permissions(self):
        """
        Set execute permissions on binary files (Unix only)
        """
        import stat

        # Search for llama-* binaries and set execute permissions
        for path in self.bin_dir.rglob('llama-*'):
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
        if platform.system().lower() == 'windows':
            name = f"{name}.exe"

        # Binaries might be in a subdirectory after extraction
        # Try both bin_dir root and common subdirectories
        possible_locations = [
            self.bin_dir / name,
            self.bin_dir / 'bin' / name,
            self.bin_dir / 'build' / 'bin' / name,  # Ubuntu/Linux releases
            self.bin_dir / f"llama-{self.LLAMA_CPP_VERSION}-bin" / name,
        ]

        for location in possible_locations:
            if location.exists():
                return location

        # Return default location even if doesn't exist (for error messages)
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

    def get_quantize_path(self) -> Path:
        """Get path to llama-quantize executable"""
        # Check if custom binaries folder is configured
        if self.custom_binaries_folder is not None:
            # If custom folder is blank, use system PATH
            if not self.custom_binaries_folder:
                system_path = shutil.which('llama-quantize')
                if system_path:
                    return Path(system_path)
                else:
                    raise RuntimeError("Custom binaries enabled with blank path, but 'llama-quantize' not found in system PATH")

            # Custom folder specified - look for binary there
            custom_folder = Path(self.custom_binaries_folder)
            binary_name = 'llama-quantize'
            if platform.system().lower() == 'windows':
                binary_name = 'llama-quantize.exe'

            binary_path = custom_folder / binary_name
            if binary_path.exists():
                return binary_path
            else:
                raise RuntimeError(f"llama-quantize not found in custom binaries folder: {custom_folder}")

        # Use auto-downloaded binaries
        path = self.get_binary_path('llama-quantize')

        # Fallback to system PATH if auto-downloaded binaries don't exist
        if not path.exists():
            system_path = shutil.which('llama-quantize')
            if system_path:
                return Path(system_path)

        return path

    def get_imatrix_path(self) -> Path:
        """Get path to llama-imatrix executable"""
        # Check if custom binaries folder is configured
        if self.custom_binaries_folder is not None:
            # If custom folder is blank, use system PATH
            if not self.custom_binaries_folder:
                system_path = shutil.which('llama-imatrix')
                if system_path:
                    return Path(system_path)
                else:
                    raise RuntimeError("Custom binaries enabled with blank path, but 'llama-imatrix' not found in system PATH")

            # Custom folder specified - look for binary there
            custom_folder = Path(self.custom_binaries_folder)
            binary_name = 'llama-imatrix'
            if platform.system().lower() == 'windows':
                binary_name = 'llama-imatrix.exe'

            binary_path = custom_folder / binary_name
            if binary_path.exists():
                return binary_path
            else:
                raise RuntimeError(f"llama-imatrix not found in custom binaries folder: {custom_folder}")

        # Use auto-downloaded binaries
        path = self.get_binary_path('llama-imatrix')

        # Fallback to system PATH if auto-downloaded binaries don't exist
        if not path.exists():
            system_path = shutil.which('llama-imatrix')
            if system_path:
                return Path(system_path)

        return path
