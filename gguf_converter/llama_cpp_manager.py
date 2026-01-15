"""
Manager for llama.cpp resources including binaries and conversion scripts
"""

import os
import platform
import zipfile
import tarfile
import shutil
import json
import stat
from pathlib import Path
from typing import Optional, Dict
from urllib.request import urlretrieve, urlopen
from colorama import Style
from .theme import THEME as theme


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


class LlamaCppManager:
    """
    Manages llama.cpp resources including binaries and conversion scripts
    """

    LLAMA_CPP_VERSION = "b7744"
    RELEASE_URL_TEMPLATE = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{filename}"

    def __init__(self, bin_dir: Optional[Path] = None, custom_binaries_folder: Optional[str] = None):
        """
        Initialize llama.cpp manager

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
                print(f"{theme['error']}ERROR: Insufficient disk space{Style.RESET_ALL}")
                print(f"{theme['error']}  Required: {required_mb:.1f} MB{Style.RESET_ALL}")
                print(f"{theme['error']}  Available: {available_mb:.1f} MB{Style.RESET_ALL}")
                return False
            return True
        except Exception as e:
            print(f"{theme['warning']}Warning: Could not check disk space: {e}{Style.RESET_ALL}")
            return True  # Proceed anyway if check fails

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
            print(f"{theme['warning']}Warning: Could not fetch latest version: {e}{Style.RESET_ALL}")
            print(f"{theme['info']}Falling back to recommended version: {self.LLAMA_CPP_VERSION}{Style.RESET_ALL}")
            return self.LLAMA_CPP_VERSION

    def update_binaries(self, force: bool = False, version: Optional[str] = None) -> Path:
        """
        Update llama.cpp binaries to recommended or specific version

        Args:
            force: Force re-download even if binaries exist at target version
            version: Specific version to download (None = use recommended LLAMA_CPP_VERSION)

        Returns:
            Path to bin directory containing executables
        """
        # Use specified version or default to recommended
        tag = version if version else self.LLAMA_CPP_VERSION

        # Print banner
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner_line = "=" * 80
        print(f"\n{theme['info']}{banner_line}{Style.RESET_ALL}")
        print(f"{theme['info']}{'UPDATE BINARIES'.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{timestamp.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{banner_line}{Style.RESET_ALL}\n")

        # Check if binaries exist and match the requested version
        if not force and self._binaries_exist():
            installed_version = self.get_installed_version_tag()
            if installed_version == tag:
                print(f"{theme['info']}Binaries already at version {tag}{Style.RESET_ALL}")
                print(f"{theme['success']}Binaries ready in {self.bin_dir}{Style.RESET_ALL}")
                return self.bin_dir
            elif installed_version:
                print(f"{theme['info']}Installed version {installed_version} differs from requested {tag}{Style.RESET_ALL}")
            else:
                print(f"{theme['info']}Unable to determine installed version, downloading {tag}...{Style.RESET_ALL}")

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

        print(f"{theme['info']}Downloading llama.cpp {tag} for {os_name}-{arch}...{Style.RESET_ALL}")
        print(f"{theme['highlight']}{url}{Style.RESET_ALL}")

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
            print(f"{theme['info']}Downloaded to {download_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{theme['error']}ERROR: Failed to download binaries: {e}{Style.RESET_ALL}")
            raise RuntimeError(
                f"Failed to download llama.cpp binaries from {url}. "
                f"Please check your internet connection or download manually."
            )

        # Extract archive
        print(f"{theme['info']}Extracting {filename}...{Style.RESET_ALL}")
        self._extract_archive(download_path)

        # Clean up archive
        download_path.unlink()
        print(f"{theme['info']}Extraction complete{Style.RESET_ALL}")

        # Verify binaries exist
        if not self._check_binary_files_exist():
            raise RuntimeError("Binary extraction succeeded but executables not found")

        print(f"{theme['success']}Installed binary version: {tag}{Style.RESET_ALL}")
        print(f"{theme['success']}Binaries ready in {self.bin_dir}{Style.RESET_ALL}")
        return self.bin_dir

    def _cleanup_old_binaries(self):
        """
        Remove all old files and directories from bin_dir before installing new binaries
        """
        if not self.bin_dir.exists():
            return

        print(f"{theme['info']}Removing old binaries...{Style.RESET_ALL}")
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
            print(f"{theme['info']}Flattening extracted directory: {subdir.name}{Style.RESET_ALL}")

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
        # Set execute permissions on llama-* binaries (not .so files)
        for path in self.bin_dir.glob('llama-*'):
            if path.is_file() and not path.suffix:  # No extension (not .so files)
                try:
                    # Add execute permission for owner, group, and others
                    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except Exception as e:
                    print(f"{theme['warning']}Warning: Could not set execute permission on {path}: {e}{Style.RESET_ALL}")

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

    def get_installed_version_info(self) -> Dict[str, Optional[str]]:
        """
        Get version information of currently installed binaries by running llama-cli --version

        Returns:
            Dict with 'full_version' (complete version string) and 'tag' (e.g., 'b7574')
            Returns None values if unable to determine
        """
        import subprocess
        import re

        try:
            cli_path = self.get_binary_path('llama-cli')
            if not cli_path.exists():
                return {'full_version': None, 'tag': None}

            result = subprocess.run(
                [str(cli_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse version from output (may be in stdout or stderr)
                output = result.stderr if result.stderr else result.stdout
                if output:
                    # Get the full version line
                    full_version = None
                    for line in output.split('\n'):
                        if line.startswith('version:'):
                            full_version = line.strip()
                            break
                    if not full_version:
                        full_version = output.strip().split('\n')[0]

                    # Extract version tag pattern like "b7574" or just "7574"
                    tag = None
                    match = re.search(r'\b(b?\d{4,5})\b', output)
                    if match:
                        version = match.group(1)
                        # Ensure it has the 'b' prefix
                        tag = version if version.startswith('b') else f'b{version}'

                    return {'full_version': full_version, 'tag': tag}
        except Exception:
            pass

        return {'full_version': None, 'tag': None}

    def get_installed_version_tag(self) -> Optional[str]:
        """
        Get the version tag of currently installed binaries

        Returns:
            Version tag (e.g., 'b7574') or None if unable to determine
        """
        return self.get_installed_version_info()['tag']

    def get_installed_conversion_scripts_version_info(self) -> Dict[str, Optional[str]]:
        """
        Get version information of currently installed conversion scripts by checking git

        Returns:
            Dict with 'full_version' (git describe output) and 'tag' (version tag like 'b7574')
            Returns None values if unable to determine
        """
        import subprocess

        project_root = Path(__file__).parent.parent
        llama_cpp_dir = project_root / "llama.cpp"

        if not llama_cpp_dir.exists() or not (llama_cpp_dir / ".git").exists():
            return {'full_version': None, 'tag': None}

        try:
            # Get the version tag using git describe
            result = subprocess.run(
                ["git", "describe", "--tags", "--always"],
                cwd=llama_cpp_dir,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                full_version = result.stdout.strip()

                # Extract just the tag portion (e.g., "b7574" from "b7574-123-gabcdef")
                tag = full_version.split('-')[0] if full_version else None

                return {'full_version': full_version, 'tag': tag}
        except Exception:
            pass

        return {'full_version': None, 'tag': None}

    def get_installed_conversion_scripts_version_tag(self) -> Optional[str]:
        """
        Get the version tag of currently installed conversion scripts

        Returns:
            Version tag (e.g., 'b7574') or None if unable to determine
        """
        return self.get_installed_conversion_scripts_version_info()['tag']

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
            self.update_binaries()
            return True
        except Exception as e:
            print(f"{theme['error']}Binary download failed: {e}{Style.RESET_ALL}")

            if not fallback_to_system:
                return False

            # Fallback: check if binaries are in system PATH
            print(f"{theme['info']}Checking system PATH for llama.cpp binaries...{Style.RESET_ALL}")
            if self._check_system_binaries():
                print(f"{theme['success']}Found llama.cpp binaries in system PATH{Style.RESET_ALL}")
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

    def get_server_path(self) -> Path:
        """Get path to llama-server executable"""
        return self._get_binary_path_with_fallback('llama-server')

    def update_conversion_scripts(self, force: bool = False, version: Optional[str] = None) -> Dict[str, str]:
        """
        Update llama.cpp conversion scripts to recommended or specific version

        Args:
            force: Force update even if already at target version
            version: Specific version to checkout (None = use recommended LLAMA_CPP_VERSION)

        Returns:
            Dict with 'status' ('success', 'already_updated', 'not_found', 'error')
            and 'message' keys
        """
        import subprocess
        import shutil

        # Print banner
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner_line = "=" * 80
        print(f"\n{theme['info']}{banner_line}{Style.RESET_ALL}")
        print(f"{theme['info']}{'UPDATE CONVERSION SCRIPTS'.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{timestamp.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{banner_line}{Style.RESET_ALL}\n")

        project_root = Path(__file__).parent.parent
        llama_cpp_dir = project_root / "llama.cpp"

        # Determine target version
        target_version = version if version else self.LLAMA_CPP_VERSION

        # Check if llama.cpp directory exists
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
            # Check current version if not forcing update
            if not force:
                version_info = self.get_installed_conversion_scripts_version_info()
                current_version = version_info['full_version']

                if current_version and target_version in current_version:
                    print(f"{theme['info']}Conversion scripts already at version {target_version}{Style.RESET_ALL}")
                    print(f"{theme['success']}Conversion scripts ready in {llama_cpp_dir}{Style.RESET_ALL}")
                    return {
                        'status': 'already_updated',
                        'message': f'Conversion scripts already at version {target_version}'
                    }
                elif current_version:
                    print(f"{theme['info']}Current version {current_version} differs from requested {target_version}{Style.RESET_ALL}")
                    print(f"{theme['info']}Updating to {target_version}...{Style.RESET_ALL}")

            # Fetch latest tags
            print(f"{theme['info']}Fetching latest tags...{Style.RESET_ALL}")
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

            # Checkout the target version
            print(f"{theme['info']}Checking out version {target_version}...{Style.RESET_ALL}")
            print(f"{theme['highlight']}https://github.com/ggml-org/llama.cpp/tree/{target_version}{Style.RESET_ALL}")
            checkout_result = subprocess.run(
                ["git", "checkout", target_version],
                cwd=llama_cpp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if checkout_result.returncode == 0:
                print(f"{theme['success']}Conversion scripts updated to version {target_version}{Style.RESET_ALL}")
                print(f"{theme['success']}Conversion scripts ready in {llama_cpp_dir}{Style.RESET_ALL}")
                return {
                    'status': 'success',
                    'message': f'Conversion scripts updated to version {target_version}'
                }
            else:
                return {
                    'status': 'error',
                    'message': f"Failed to checkout version {target_version}: {checkout_result.stderr}"
                }

        except subprocess.TimeoutExpired:
            error_msg = "Update timed out"
            print(f"{theme['error']}{error_msg}{Style.RESET_ALL}")
            return {
                'status': 'error',
                'message': error_msg
            }
        except Exception as e:
            error_msg = f"Could not update conversion scripts: {e}"
            print(f"{theme['error']}{error_msg}{Style.RESET_ALL}")
            return {
                'status': 'error',
                'message': error_msg
            }
