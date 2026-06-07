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

    LLAMA_CPP_VERSION = "b9544"
    RELEASE_URL_TEMPLATE = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{filename}"

    # Seconds to wait on GitHub API calls before giving up. These run on the
    # Streamlit render thread, so a missing timeout would hang the GUI.
    GITHUB_API_TIMEOUT = 10

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
            with urlopen(api_url, timeout=self.GITHUB_API_TIMEOUT) as response:
                data = json.loads(response.read().decode())
                return data['tag_name']
        except Exception as e:
            print(f"{theme['warning']}Warning: Could not fetch latest version: {e}{Style.RESET_ALL}")
            print(f"{theme['info']}Falling back to recommended version: {self.LLAMA_CPP_VERSION}{Style.RESET_ALL}")
            return self.LLAMA_CPP_VERSION

    def _get_release_asset_names(self, tag: str) -> set:
        """
        Fetch the set of asset filenames published for a given release tag.

        The assets embedded in the release-by-tag response are paginated by
        GitHub (30 per page), and llama.cpp releases ship far more assets than
        that, so we resolve the release id and page through the dedicated assets
        endpoint to get the complete set. A truncated set would otherwise drop
        CUDA variants and make valid backends look unavailable.

        Results are cached per-tag on the instance (the manager persists for the
        session). Returns an empty set on failure so callers can fail open.
        """
        if not hasattr(self, '_asset_cache'):
            self._asset_cache = {}
        if tag in self._asset_cache:
            return self._asset_cache[tag]

        names = set()
        try:
            api_url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
            with urlopen(api_url, timeout=self.GITHUB_API_TIMEOUT) as response:
                data = json.loads(response.read().decode())

            release_id = data.get('id')
            if release_id is None:
                # No id to paginate with — fall back to the embedded (possibly
                # truncated) assets array rather than nothing.
                for asset in data.get('assets', []):
                    name = asset.get('name')
                    if name:
                        names.add(name)
            else:
                # Page through the assets endpoint until a short/empty page.
                # A hard page cap guards against an unbounded loop if a server
                # ever keeps returning full pages (releases have a few hundred
                # assets, so 50 pages / 5000 assets is far more than enough).
                for page in range(1, 51):
                    assets_url = (
                        f"https://api.github.com/repos/ggml-org/llama.cpp/"
                        f"releases/{release_id}/assets?per_page=100&page={page}"
                    )
                    with urlopen(assets_url, timeout=self.GITHUB_API_TIMEOUT) as response:
                        page_data = json.loads(response.read().decode())
                    if not page_data:
                        break
                    for asset in page_data:
                        name = asset.get('name')
                        if name:
                            names.add(name)
                    if len(page_data) < 100:
                        break
        except Exception:
            names = set()

        # Only cache successful fetches so a transient failure isn't sticky
        if names:
            self._asset_cache[tag] = names
        return names

    def _bin_info_path(self) -> Path:
        return self.bin_dir / ".yagguf_bin_info.json"

    def _write_bin_info(self, tag: str, gpu_backend: str) -> None:
        try:
            with open(self._bin_info_path(), 'w') as f:
                json.dump({"version": tag, "gpu_backend": gpu_backend}, f)
        except Exception:
            pass  # Non-critical

    def _read_bin_info(self) -> Dict[str, Optional[str]]:
        try:
            with open(self._bin_info_path()) as f:
                data = json.load(f)
                return {
                    "version": data.get("version"),
                    "gpu_backend": data.get("gpu_backend"),
                }
        except Exception:
            return {"version": None, "gpu_backend": None}

    @staticmethod
    def _cuda_family(backend: str) -> str:
        """
        Normalise a CUDA backend to its major family.

        "cuda-13.3" -> "cuda-13", "cuda-13" -> "cuda-13". Non-CUDA backends
        (and anything unexpected) are returned unchanged.
        """
        if not backend or not backend.startswith("cuda"):
            return backend
        import re
        m = re.match(r'(cuda-\d+)', backend)
        return m.group(1) if m else backend

    def _resolve_cuda_variant(self, tag: str, backend: str) -> str:
        """
        Resolve a CUDA backend selection to the exact patch variant published
        for `tag` on this platform.

        Accepts a major family like "cuda-13" (what the GUI now stores) or a
        legacy exact value like "cuda-13.3", and returns the highest available
        patch for that major (e.g. "cuda-13.3"). This is what lets a saved
        "CUDA 13.x" preference keep working when llama.cpp bumps 13.1 -> 13.3.

        Returns the input unchanged for non-CUDA backends, or when the release
        assets can't be inspected (the download/validation step then surfaces
        any remaining problem).
        """
        if not backend or not backend.startswith("cuda"):
            return backend

        import re
        major = self._cuda_family(backend).split('-', 1)[1]
        os_name = self.platform_info['os']
        arch = self.platform_info['arch']
        assets = self._get_release_asset_names(tag)
        if not assets:
            return backend

        pat = re.compile(rf"^llama-{re.escape(tag)}-bin-{os_name}-(cuda-{major}\.[\d.]+)-{arch}\.")
        variants = {m.group(1) for a in assets if (m := pat.match(a))}
        if not variants:
            return backend
        # Highest patch within the chosen major (e.g. cuda-13.5 over cuda-13.3)
        return max(variants, key=lambda v: tuple(int(n) for n in re.findall(r'\d+', v)))

    def _discover_cuda_backends(self, tag: str) -> list:
        """
        Return [(label, value), ...] for the CUDA major families actually
        published for `tag` on this platform's win-x64/arm64 builds.

        Values are major families ("cuda-13") shown as "CUDA 13.x" rather than
        exact patch versions, so a saved preference survives llama.cpp bumping
        the bundled toolkit between releases (e.g. cuda-13.1 -> cuda-13.3). The
        exact patch is resolved at download time by _resolve_cuda_variant().
        Falls back to a static list when the assets can't be fetched.
        """
        import re

        os_name = self.platform_info['os']
        arch = self.platform_info['arch']
        assets = self._get_release_asset_names(tag)

        found = []
        if assets:
            pat = re.compile(rf"^llama-{re.escape(tag)}-bin-{os_name}-cuda-(\d+)\.[\d.]+-{arch}\.")
            majors = sorted({m.group(1) for a in assets if (m := pat.match(a))}, key=int)
            for major in majors:
                found.append((f"CUDA {major}.x (Nvidia)", f"cuda-{major}"))

        if found:
            return found

        # Fallback when the release can't be inspected
        return [
            ("CUDA 12.x (Nvidia)", "cuda-12"),
            ("CUDA 13.x (Nvidia)", "cuda-13"),
        ]

    def get_available_gpu_backends(self, tag: Optional[str] = None) -> list:
        """
        Return list of (label, value) tuples for available GPU backends on this platform.

        Labels are human-readable, values are used in llama.cpp release filenames.
        CUDA entries are discovered from the release assets (defaulting to the
        latest release) so they track upstream toolkit version changes. The result
        for the default (latest) lookup is memoized for the session to avoid
        repeated network calls on Streamlit reruns.
        """
        os_name = self.platform_info['os']
        if os_name == 'macos':
            return [("CPU + Metal (built-in)", "cpu")]

        use_cache = tag is None
        if use_cache and getattr(self, '_backend_list_cache', None) is not None:
            return self._backend_list_cache

        resolved_tag = tag if tag else self.get_latest_version()
        cuda_backends = self._discover_cuda_backends(resolved_tag)

        if os_name == 'win':
            backends = [("CPU", "cpu")]
            backends += cuda_backends
            backends += [
                ("Vulkan (Nvidia/AMD/Intel)", "vulkan"),
                ("HIP / ROCm (AMD)", "hip-radeon"),
                ("SYCL (Intel)", "sycl"),
            ]
        else:  # ubuntu/linux
            backends = [("CPU", "cpu")]
            backends += cuda_backends
            backends += [
                ("Vulkan (Nvidia/AMD/Intel)", "vulkan"),
                ("HIP / ROCm (AMD)", "hip-radeon"),
            ]

        if use_cache:
            self._backend_list_cache = backends
        return backends

    def update_binaries(self, force: bool = False, version: Optional[str] = None, gpu_backend: str = "cpu") -> Path:
        """
        Update llama.cpp binaries to recommended or specific version

        Args:
            force: Force re-download even if binaries exist at target version
            version: Specific version to download (None = use recommended LLAMA_CPP_VERSION)
            gpu_backend: Backend to download. CUDA may be a major family ("cuda-13",
                         shown as "CUDA 13.x") or a legacy exact value
                         ("cuda-13.3"); the exact patch is resolved against the
                         target release. Also accepts "cpu", "vulkan", etc.
                         Only relevant for Windows; ignored on macOS and Linux.

        Returns:
            Path to bin directory containing executables
        """
        # Use specified version or default to recommended
        tag = version if version else self.LLAMA_CPP_VERSION

        # CUDA selections are stored as a major family (e.g. "cuda-13"); resolve
        # the exact patch variant published for this release (e.g. "cuda-13.3").
        requested_family = self._cuda_family(gpu_backend)
        resolved_backend = self._resolve_cuda_variant(tag, gpu_backend)

        # Print banner
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner_line = "=" * 80
        print(f"\n{theme['info']}{banner_line}{Style.RESET_ALL}")
        print(f"{theme['info']}{'UPDATE BINARIES'.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{timestamp.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{banner_line}{Style.RESET_ALL}\n")

        # Check if binaries exist and match the requested version AND backend.
        # Compare CUDA backends by major family so a patch bump (13.1 -> 13.3)
        # within the same selection isn't treated as a different backend.
        if not force and self._binaries_exist():
            installed_version = self.get_installed_version_tag()
            installed_backend = self._read_bin_info().get("gpu_backend")
            installed_family = self._cuda_family(installed_backend or "")
            if installed_version == tag and installed_family == requested_family:
                print(f"{theme['info']}Binaries already at version {tag} ({requested_family}){Style.RESET_ALL}")
                print(f"{theme['success']}Binaries ready in {self.bin_dir}{Style.RESET_ALL}")
                return self.bin_dir
            elif installed_version == tag and installed_family != requested_family:
                print(f"{theme['info']}Version matches ({tag}) but backend differs: {installed_backend} → {requested_family}{Style.RESET_ALL}")
            elif installed_version:
                print(f"{theme['info']}Installed version {installed_version} differs from requested {tag}{Style.RESET_ALL}")
            else:
                print(f"{theme['info']}Unable to determine installed version, downloading {tag}...{Style.RESET_ALL}")

        # Update filename to use the specified version
        os_name = self.platform_info['os']
        arch = self.platform_info['arch']

        if os_name == "win":
            ext = "zip"
            filename = f"llama-{tag}-bin-{os_name}-{resolved_backend}-{arch}.{ext}"
        elif os_name == "macos":
            variant = self.platform_info['variant']
            ext = "zip"
            filename = f"llama-{tag}-bin-{os_name}-{variant}.{ext}"
        else:  # linux/ubuntu
            variant = self.platform_info['variant']
            ext = "tar.gz"
            filename = f"llama-{tag}-bin-{os_name}-{variant}.{ext}"

        url = self.RELEASE_URL_TEMPLATE.format(tag=tag, filename=filename)

        # Validate the asset exists for this release before touching anything.
        # llama.cpp changes CUDA toolkit versions between releases (e.g. a stored
        # cuda-13.1 backend no longer exists in a newer release that ships
        # cuda-13.3), which would otherwise 404 mid-update.
        assets = self._get_release_asset_names(tag)
        if assets and filename not in assets:
            prefix = f"llama-{tag}-bin-{os_name}-"
            suffix = f"-{arch}.{ext}" if os_name == "win" else f".{ext}"
            raw = sorted(
                a[len(prefix):-len(suffix)]
                for a in assets
                if a.startswith(prefix) and a.endswith(suffix)
            )
            # Collapse CUDA variants to families to match the GUI labels
            available = sorted({self._cuda_family(v) for v in raw})
            hint = (
                f"\nAvailable backends for {tag}: {', '.join(available)}"
                if available else ""
            )
            raise RuntimeError(
                f"llama.cpp release {tag} does not provide '{filename}'.\n"
                f"The '{requested_family}' backend likely isn't available for this "
                f"version (llama.cpp changes its bundled CUDA toolkit version "
                f"between releases).{hint}\n"
                f"Pick an available backend in the GPU Backend selector and try again."
            )

        print(f"{theme['info']}Downloading llama.cpp {tag} for {os_name}-{arch}...{Style.RESET_ALL}")
        print(f"{theme['highlight']}{url}{Style.RESET_ALL}")

        # Check disk space before download (estimate 1 GB needed for download + extraction)
        required_space = 1024 * 1024 * 1024  # 1 GB
        if not self._check_disk_space(required_space):
            raise RuntimeError(
                "Insufficient disk space for binary download. "
                "Please free up at least 1 GB and try again."
            )

        # Download FIRST, then remove old binaries — so a failed download (e.g.
        # 404 on a missing variant) leaves the existing working install intact.
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        download_path = self.bin_dir / filename

        try:
            urlretrieve(url, download_path, reporthook=self._progress_hook)
            print()  # New line after progress
            print(f"{theme['info']}Downloaded to {download_path}{Style.RESET_ALL}")
        except Exception as e:
            # Remove any partial download; keep the previously installed binaries
            if download_path.exists():
                try:
                    download_path.unlink()
                except OSError:
                    pass
            print(f"\n{theme['error']}ERROR: Failed to download binaries: {e}{Style.RESET_ALL}")
            raise RuntimeError(
                f"Failed to download llama.cpp binaries from {url}. "
                f"Please check your internet connection or download manually. "
                f"Your existing binaries were left untouched."
            )

        # Download succeeded — now safe to remove old binaries (preserve the archive)
        self._cleanup_old_binaries(exclude={download_path})

        # Extract archive
        print(f"{theme['info']}Extracting {filename}...{Style.RESET_ALL}")
        self._extract_archive(download_path)

        # Clean up archive
        download_path.unlink()
        print(f"{theme['info']}Extraction complete{Style.RESET_ALL}")

        # For Windows CUDA builds, also download the CUDA runtime DLLs
        if os_name == "win" and resolved_backend.startswith("cuda"):
            cudart_filename = f"cudart-llama-bin-win-{resolved_backend}-{arch}.zip"
            cudart_url = self.RELEASE_URL_TEMPLATE.format(tag=tag, filename=cudart_filename)
            cudart_path = self.bin_dir / cudart_filename
            print(f"{theme['info']}Downloading CUDA runtime DLLs...{Style.RESET_ALL}")
            print(f"{theme['highlight']}{cudart_url}{Style.RESET_ALL}")
            try:
                urlretrieve(cudart_url, cudart_path, reporthook=self._progress_hook)
                print()
                self._extract_archive(cudart_path)
                cudart_path.unlink()
                print(f"{theme['info']}CUDA runtime DLLs extracted{Style.RESET_ALL}")
            except Exception as e:
                print(f"{theme['warning']}Warning: Could not download CUDA runtime DLLs: {e}{Style.RESET_ALL}")
                print(f"{theme['warning']}CUDA may not work correctly without these files.{Style.RESET_ALL}")

        # Verify binaries exist
        if not self._check_binary_files_exist():
            raise RuntimeError("Binary extraction succeeded but executables not found")

        # Store the major family as the canonical backend identity so the saved
        # preference stays stable across patch bumps; show the resolved patch.
        self._write_bin_info(tag, requested_family)
        backend_display = (
            f"{requested_family} → {resolved_backend}"
            if resolved_backend != requested_family else requested_family
        )
        print(f"{theme['success']}Installed binary version: {tag} ({backend_display}){Style.RESET_ALL}")
        print(f"{theme['success']}Binaries ready in {self.bin_dir}{Style.RESET_ALL}")
        return self.bin_dir

    def _cleanup_old_binaries(self, exclude=None):
        """
        Remove all old files and directories from bin_dir before installing new binaries

        Args:
            exclude: Optional iterable of paths to preserve (e.g. a freshly
                     downloaded archive that lives inside bin_dir).
        """
        if not self.bin_dir.exists():
            return

        exclude_resolved = {Path(p).resolve() for p in (exclude or set())}

        print(f"{theme['info']}Removing old binaries...{Style.RESET_ALL}")
        # Remove everything in bin_dir (files and directories)
        for item in self.bin_dir.iterdir():
            if item.resolve() in exclude_resolved:
                continue
            if item.is_dir():
                shutil.rmtree(item, onerror=remove_readonly)
            else:
                try:
                    item.unlink()
                except PermissionError:
                    os.chmod(item, stat.S_IWRITE)
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
                        try:
                            dest.unlink()
                        except PermissionError:
                            os.chmod(dest, stat.S_IWRITE)
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

                    # Extract version tag pattern like "b7574" or just "7574".
                    # Search the "version:" line first — CUDA builds print device
                    # and driver numbers (e.g. VRAM in MiB) before the version line,
                    # and matching the whole output would grab one of those instead.
                    search_text = full_version if full_version else output
                    tag = None
                    match = re.search(r'\b(b?\d{4,6})\b', search_text)
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
