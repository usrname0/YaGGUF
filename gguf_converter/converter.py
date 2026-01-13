"""
Core GGUF conversion and quantization functionality
"""

import subprocess
import sys
import json
import io
import time
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List
from huggingface_hub import snapshot_download, HfApi
from colorama import Style
from .llama_cpp_manager import LlamaCppManager
from . import imatrix_stats
from .theme import THEME as theme
from .model_quirks import ModelQuirks



class GGUFConverter:
    """
    GGUF converter that wraps llama.cpp for conversion and quantization
    No C++ compilation required - binaries are auto-downloaded
    """

    QUANTIZATION_TYPES = [
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
        "Q2_K", "Q2_K_S",
        "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_K", "Q4_K_S", "Q4_K_M",
        "Q5_K", "Q5_K_S", "Q5_K_M",
        "Q6_K",
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
        "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
        "IQ4_XS", "IQ4_NL",
        "F16", "BF16", "F32",
    ]

    # Regex pattern for GGUF shard files (e.g., model-00001-of-00009.gguf)
    SHARD_PATTERN = re.compile(r'-(\d+)-of-(\d+)\.gguf$')

    @staticmethod
    def _clean_llama_error(error_output: str) -> str:
        """
        Extract relevant error information from llama.cpp output

        Args:
            error_output: Raw stderr from llama.cpp command

        Returns:
            Cleaned error message with only relevant information
        """
        lines = error_output.split('\n')
        relevant_lines = []
        in_error_banner = False

        for line in lines:
            # Skip metadata loader lines
            if 'llama_model_loader: - kv' in line or 'llama_model_loader: - type' in line:
                continue

            # Keep error banners (lines with ===)
            if '===' in line:
                in_error_banner = True
                continue

            # Keep lines in error banners
            if in_error_banner:
                if line.strip():
                    relevant_lines.append(line)
                else:
                    in_error_banner = False
                continue

            # Keep important error lines
            if any(keyword in line for keyword in [
                'failed to',
                'error:',
                'Error:',
                'ERROR:',
                'Missing importance matrix',
                'Please do not use',
                'bailing out',
                'CUDA error',
                'out of memory',
                'cannot',
                'unsupported',
                'invalid'
            ]):
                relevant_lines.append(line.strip())

        if relevant_lines:
            # Join with double newlines for better readability in error messages
            return '\n\n'.join(relevant_lines)

        # Fallback to original if we couldn't extract anything useful
        return error_output

    def __init__(self, custom_binaries_folder=None, custom_llama_cpp_repo=None):
        """
        Initialize the converter and llama.cpp manager

        Args:
            custom_binaries_folder: Optional path to folder containing custom llama.cpp binaries.
                                   If empty string, will use system PATH.
                                   If None, will use auto-downloaded binaries.
            custom_llama_cpp_repo: Optional path to custom llama.cpp repository.
                                   If None, will use auto-cloned repository.
        """
        self.llama_cpp_manager = LlamaCppManager(custom_binaries_folder=custom_binaries_folder)
        self.custom_llama_cpp_repo = custom_llama_cpp_repo
        if custom_binaries_folder is None:
            if not self.llama_cpp_manager.ensure_binaries():
                raise RuntimeError(
                    "Failed to get llama.cpp binaries.\n"
                    "Please check your internet connection or provide a path to a custom llama.cpp installation in the Settings tab."
                )
        if custom_llama_cpp_repo is None:
            self._ensure_llama_cpp_repo()

    def download_model(
        self,
        repo_id: str,
        output_dir: Union[str, Path],
        revision: Optional[str] = None
    ) -> Path:
        """
        Download a model from HuggingFace

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/model-name")
            output_dir: Directory to save the model
            revision: Specific revision/branch to download

        Returns:
            Path to the downloaded model directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Print banner
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner_line = "=" * 80
        print(f"\n{theme['info']}{banner_line}{Style.RESET_ALL}")
        print(f"{theme['info']}{'DOWNLOAD FROM HUGGINGFACE'.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{repo_id.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{timestamp.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{banner_line}{Style.RESET_ALL}\n")

        # First, quickly verify repository exists (fast failure on 404)
        print(f"{theme['info']}Verifying repository exists...{Style.RESET_ALL}")
        try:
            api = HfApi()
            # Quick check without files_metadata - fails fast on non-existent repos
            api.repo_info(repo_id=repo_id, revision=revision, timeout=10.0)
        except Exception as e:
            # Let the error bubble up - repo doesn't exist or network issue
            raise RuntimeError(f"Repository not found or inaccessible: {repo_id}") from e

        # Now check repository size and disk space
        print(f"{theme['info']}Checking repository size...{Style.RESET_ALL}")
        try:
            repo_info = api.repo_info(repo_id=repo_id, revision=revision, files_metadata=True, timeout=10.0)

            # Calculate total size from all files
            total_size_bytes = 0
            if hasattr(repo_info, 'siblings') and repo_info.siblings:
                for file in repo_info.siblings:
                    if hasattr(file, 'size') and file.size:
                        total_size_bytes += file.size

            if total_size_bytes > 0:
                total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
                print(f"{theme['info']}Repository size: {total_size_gb:.2f} GB{Style.RESET_ALL}")

                # Check disk space with fixed 500MB buffer for safety
                buffer_bytes = 500 * 1024 * 1024  # 500MB
                required_bytes = total_size_bytes + buffer_bytes

                stat = shutil.disk_usage(output_dir)
                available_gb = stat.free / (1024 * 1024 * 1024)
                required_gb = required_bytes / (1024 * 1024 * 1024)

                if stat.free < required_bytes:
                    # Raise exception - GUI layer will handle display
                    raise RuntimeError(
                        f"Insufficient disk space for model download.\n"
                        f"Model size: {total_size_gb:.2f} GB\n"
                        f"Required (with buffer): {required_gb:.2f} GB\n"
                        f"Available: {available_gb:.2f} GB\n"
                        f"Please free up at least {required_gb - available_gb:.2f} GB and try again."
                    )
            else:
                print(f"{theme['warning']}Warning: Could not determine repository size, proceeding anyway{Style.RESET_ALL}")

        except RuntimeError:
            raise  # Re-raise disk space errors
        except Exception as e:
            print(f"{theme['warning']}Warning: Could not check repository size: {e}{Style.RESET_ALL}")
            print(f"{theme['warning']}Proceeding with download anyway...{Style.RESET_ALL}")

        print(f"{theme['success']}Downloading {repo_id} from HuggingFace...{Style.RESET_ALL}")
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir / Path(repo_id).name,
            revision=revision
        )

        # Brief pause to let progress bars finish flushing to terminal
        # (snapshot_download returns before all parallel progress bars complete)
        import time
        time.sleep(0.5)

        return Path(model_path)

    @staticmethod
    def _get_shard_count(shard_files: List[Path]) -> Optional[int]:
        """
        Get the total shard count from a set of shard files.

        Args:
            shard_files: List of shard files (e.g., model-00001-of-00009.gguf, ...)

        Returns:
            Total shard count if consistent across all files, None otherwise
        """
        if not shard_files:
            return None

        total_count = None
        for file in shard_files:
            match = GGUFConverter.SHARD_PATTERN.search(file.name)
            if match:
                shard_total = int(match.group(2))
                if total_count is None:
                    total_count = shard_total
                elif total_count != shard_total:
                    # Inconsistent total counts
                    return None

        return total_count

    @staticmethod
    def _validate_shard_set(shard_files: List[Path]) -> bool:
        """
        Validate that a set of shard files is complete.

        Args:
            shard_files: List of shard files (e.g., model-00001-of-00009.gguf, ...)

        Returns:
            True if we have a complete set (all shards present), False otherwise
        """
        if not shard_files:
            return False

        # Extract shard numbers and total count
        shard_info = []
        total_count = None

        for file in shard_files:
            match = GGUFConverter.SHARD_PATTERN.search(file.name)
            if match:
                shard_num = int(match.group(1))
                shard_total = int(match.group(2))
                shard_info.append(shard_num)

                # Verify all files agree on total count
                if total_count is None:
                    total_count = shard_total
                elif total_count != shard_total:
                    # Inconsistent total counts - incomplete/corrupt set
                    return False

        if total_count is None:
            return False

        # Check if we have all shards from 1 to total_count
        expected_shards = set(range(1, total_count + 1))
        actual_shards = set(shard_info)

        return expected_shards == actual_shards

    def convert_to_gguf(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        output_type: str = "f16",
        vocab_only: bool = False,
        verbose: bool = False,
        split_max_size: Optional[str] = None,
        split_max_tensors: Optional[int] = None,
        mmproj_precision: str = "F16"
    ) -> Path:
        """
        Convert a model to GGUF format

        Args:
            model_path: Path to the input model (safetensors/pytorch)
            output_path: Path for the output GGUF file
            output_type: Output precision (f32, f16, bf16, q8_0, auto)
            vocab_only: Only extract vocabulary
            verbose: Enable verbose logging
            split_max_size: Maximum size per split (e.g., "2G")
            split_max_tensors: Maximum tensors per split
            mmproj_precision: Precision for mmproj (vision projector) file (F16, F32, BF16, Q8_0). Default: F16 for compatibility.

        Returns:
            Path to the created GGUF file (or first shard if split)
        """
        model_path = Path(model_path)
        output_path = Path(output_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build command to run convert_hf_to_gguf.py
        # We need to find the convert script from llama.cpp
        convert_script = self._find_convert_script()

        if not convert_script:
            raise RuntimeError(
                "Could not find convert_hf_to_gguf.py script.\n"
                "The llama.cpp repository should have been auto-cloned.\n"
                "Please check your git installation or manually clone llama.cpp."
            )

        cmd = [
            sys.executable,
            str(convert_script),
            str(model_path),
            "--outfile", str(output_path),
            "--outtype", output_type.lower(),
        ]

        if vocab_only:
            cmd.append("--vocab-only")
        if verbose:
            cmd.append("--verbose")
        if split_max_size:
            cmd.extend(["--split-max-size", split_max_size])
        if split_max_tensors:
            cmd.extend(["--split-max-tensors", str(split_max_tensors)])

        # Add model-specific flags and print detection info
        ModelQuirks.print_model_detection(model_path)
        model_flags = ModelQuirks.get_conversion_flags(model_path)
        cmd.extend(model_flags)

        print(f"{theme['info']}Converting {model_path.name} to GGUF format...{Style.RESET_ALL}")
        print(f"{theme['highlight']}Running: {' '.join(cmd)}{Style.RESET_ALL}\n")

        result = subprocess.run(cmd, capture_output=not verbose, text=True, encoding='utf-8', errors='replace')

        if result.returncode != 0:
            # Combine stdout and stderr since errors can be in either
            error_output = (result.stderr or '') + (result.stdout or '')
            raw_error = error_output.strip() if error_output.strip() else 'Unknown error'
            error_msg = self._clean_llama_error(raw_error)
            raise RuntimeError(f"Conversion failed:\n\n{error_msg}")

        if verbose and result.stdout:
            print(result.stdout)

        # Check if shards were created when splitting was requested
        actual_output_path = output_path
        if (split_max_size or split_max_tensors) and not output_path.exists():
            # Look for sharded output files
            output_dir = output_path.parent
            output_stem = output_path.stem
            sharded_files = sorted(output_dir.glob(f"{output_stem}-*-of-*.gguf"))

            if sharded_files:
                actual_output_path = sharded_files[0]
                print(f"\n{theme['success']}Conversion complete: {len(sharded_files)} shard(s) created{Style.RESET_ALL}")
                for shard in sharded_files:
                    print(f"{theme['metadata']}  - {shard.name}{Style.RESET_ALL}")
            else:
                print(f"\n{theme['success']}Conversion complete: {output_path}{Style.RESET_ALL}")
        else:
            print(f"\n{theme['success']}Conversion complete: {output_path}{Style.RESET_ALL}")

        # For vision models, generate the vision projector (mmproj) file separately
        if ModelQuirks.is_vision_model(model_path):
            print(f"\n{theme['info']}Generating vision projector (mmproj) file...{Style.RESET_ALL}")

            # Build mmproj output path using model name and mmproj precision
            mmproj_output = output_path.parent / f"mmproj-{model_path.name}-{mmproj_precision.upper()}.gguf"

            # Check if mmproj file already exists
            if mmproj_output.exists():
                print(f"{theme['warning']}Overwriting existing mmproj file: {mmproj_output.name}{Style.RESET_ALL}")

            # Build mmproj conversion command
            mmproj_cmd = [
                sys.executable,
                str(convert_script),
                str(model_path),
                "--outfile", str(mmproj_output),
                "--outtype", mmproj_precision.lower(),
            ]

            # Add model-specific flags WITH mmproj
            mmproj_flags = ModelQuirks.get_conversion_flags(model_path, include_mmproj=True)
            mmproj_cmd.extend(mmproj_flags)

            if verbose:
                mmproj_cmd.append("--verbose")

            print(f"{theme['highlight']}Running: {' '.join(mmproj_cmd)}{Style.RESET_ALL}\n")

            mmproj_result = subprocess.run(mmproj_cmd, capture_output=not verbose, text=True, encoding='utf-8', errors='replace')

            if mmproj_result.returncode != 0:
                error_output = (mmproj_result.stderr or '') + (mmproj_result.stdout or '')
                raw_error = error_output.strip() if error_output.strip() else 'Unknown error'
                error_msg = self._clean_llama_error(raw_error)
                raise RuntimeError(f"Vision projector export failed:\n\n{error_msg}")

            if verbose and mmproj_result.stdout:
                print(mmproj_result.stdout)

            print(f"{theme['success']}\nVision projector exported: {mmproj_output}{Style.RESET_ALL}")
            print(f"{theme['info']}Use both files together: {actual_output_path.name} + {mmproj_output.name}{Style.RESET_ALL}\n")

        return actual_output_path

    def _ensure_llama_cpp_repo(self):
        """
        Verify llama.cpp repository exists
        Repository should be cloned during startup by check_and_download_binaries.py
        """
        llama_cpp_dir = Path(__file__).parent.parent / "llama.cpp"
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

        if not llama_cpp_dir.exists() or not convert_script.exists():
            raise RuntimeError(
                "llama.cpp repository not found. This should have been cloned during startup.\n"
                "Please restart the application or run the startup script again."
            )

    def _find_convert_script(self) -> Optional[Path]:
        """
        Find the convert_hf_to_gguf.py script

        Note: The repo is cloned during startup by check_and_download_binaries.py, and
        _ensure_llama_cpp_repo() verifies it exists during init
        """
        # Check custom repo path first
        if self.custom_llama_cpp_repo:
            custom_script = Path(self.custom_llama_cpp_repo) / "convert_hf_to_gguf.py"
            if custom_script.exists():
                return custom_script
            # If custom path specified but script not found, still return None to trigger error
            return None

        # Primary location (managed by this tool)
        llama_cpp_dir = Path(__file__).parent.parent / "llama.cpp"
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

        if convert_script.exists():
            return convert_script

        return None


    def quantize(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        quantization_type: str = "Q4_K_M",
        imatrix_path: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = None,
        verbose: bool = True,
        parallel: bool = True,
        num_workers: Optional[int] = None,
        scalar_optimization: bool = False,
        leave_output_tensor: bool = False,
        pure_quantization: bool = False,
        keep_split: bool = False,
        output_tensor_type: Optional[str] = None,
        token_embedding_type: Optional[str] = None
    ) -> Path:
        """
        Quantize a GGUF model using llama.cpp

        Args:
            input_path: Path to input GGUF file (f16/f32)
            output_path: Path for quantized output file
            quantization_type: Quantization type (e.g., Q4_0, Q4_K_M, IQ3_XXS, etc.)
            imatrix_path: Optional importance matrix file for better quality
            num_threads: Number of threads to use (passed to llama-quantize)
            verbose: Enable verbose output
            parallel: Ignored (kept for API compatibility)
            num_workers: Ignored (kept for API compatibility)
            scalar_optimization: Ignored (kept for API compatibility)
            leave_output_tensor: Keep output.weight unquantized for better quality (increases model size)
            pure_quantization: Disable k-quant mixtures, quantize all tensors uniformly
            keep_split: Keep model in same shards as input (for multi-file models)
            output_tensor_type: Override quantization type for output.weight tensor (e.g., "Q8_0", "F16")
            token_embedding_type: Override quantization type for token embeddings (e.g., "Q8_0", "F16")

        Returns:
            Path to the quantized GGUF file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if quantization_type not in self.QUANTIZATION_TYPES:
            raise ValueError(
                f"Invalid quantization type: {quantization_type}.\n"
                f"Available: {', '.join(self.QUANTIZATION_TYPES)}"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"{theme['info']}Quantizing {input_path.name} to {quantization_type}...{Style.RESET_ALL}")

        start_time = time.time()

        # Build llama-quantize command
        # Important: Flags must come BEFORE positional arguments!
        # Usage: llama-quantize [--flags...] input.gguf output.gguf type [nthreads]
        quantize_bin = self.llama_cpp_manager.get_quantize_path()
        cmd = [str(quantize_bin)]

        # Add optional flags FIRST (before positional arguments)
        if leave_output_tensor:
            cmd.append("--leave-output-tensor")

        if pure_quantization:
            cmd.append("--pure")

        if keep_split:
            cmd.append("--keep-split")

        if imatrix_path:
            imatrix_path = Path(imatrix_path)
            if not imatrix_path.exists():
                raise FileNotFoundError(f"Imatrix file not found: {imatrix_path}")
            cmd.extend(["--imatrix", str(imatrix_path)])

        if output_tensor_type:
            cmd.extend(["--output-tensor-type", output_tensor_type])

        if token_embedding_type:
            cmd.extend(["--token-embedding-type", token_embedding_type])

        # Add positional arguments AFTER flags
        cmd.extend([str(input_path), str(output_path), quantization_type])

        # Add num_threads as final positional argument (optional)
        if num_threads:
            cmd.append(str(num_threads))

        print(f"{theme['highlight']}Running: {' '.join(cmd)}{Style.RESET_ALL}\n")
        print()

        # Run llama-quantize
        # Always capture output so we can parse errors, but print it if verbose=True
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=True
            )

            if verbose:
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)

        except subprocess.CalledProcessError as e:
            raw_error = e.stderr if e.stderr else str(e)
            error_msg = self._clean_llama_error(raw_error)

            # Check for tied embeddings issue (common with certain models and quantization types)
            # Pattern: "GGML_ASSERT(start % type_traits[type].blck_size == 0) failed"
            # Usually happens with token_embd.weight on tied embedding models
            if "blck_size" in raw_error and "failed" in raw_error:
                raise RuntimeError(
                    f"Quantization failed: Model incompatibility detected.\n\n"
                    f"This error typically occurs with certain models (e.g., models with tied embeddings)\n"
                    f"when using specific quantization types.\n\n"
                    f"Try a different quantization type or use F16/F32 instead.\n\n"
                    f"Error:\n{error_msg}"
                )

            raise RuntimeError(f"Quantization failed:\n\n{error_msg}")

        elapsed = time.time() - start_time

        # Check for output file
        # If --keep-split was used, llama-quantize creates sharded files like "model-00001-of-00001.gguf"
        actual_output_path = output_path
        sharded_files = []

        if not output_path.exists() and keep_split:
            # Look for sharded output files
            output_dir = output_path.parent
            output_stem = output_path.stem  # filename without extension

            # Pattern: model-00001-of-00001.gguf
            sharded_files = list(output_dir.glob(f"{output_stem}-*-of-*.gguf"))

            if sharded_files:
                # Use the first shard as the "output" path to return
                actual_output_path = sorted(sharded_files)[0]
                print(f"{theme['info']}Model saved as sharded files ({len(sharded_files)} shard(s)){Style.RESET_ALL}")
                for shard in sorted(sharded_files):
                    print(f"{theme['metadata']}  - {shard.name}{Style.RESET_ALL}")

        # Print summary
        if actual_output_path.exists():
            input_size = input_path.stat().st_size / (1024**3)

            # If sharded, sum all shard sizes
            if sharded_files:
                total_output_size = sum(f.stat().st_size for f in sharded_files)
                output_size = total_output_size / (1024**3)
            else:
                output_size = actual_output_path.stat().st_size / (1024**3)

            ratio = input_size / output_size if output_size > 0 else 0

            print(f"\n{theme['success']}Quantization complete: {actual_output_path.name if sharded_files else actual_output_path}{Style.RESET_ALL}")
            print(f"{theme['metadata']}Time taken: {elapsed:.2f}s ({elapsed/60:.2f} minutes){Style.RESET_ALL}")
            print(f"{theme['metadata']}Size: {input_size:.2f} GB -> {output_size:.2f} GB ({ratio:.2f}x compression){Style.RESET_ALL}")
        else:
            raise RuntimeError("Quantization appeared to succeed but output file not found")

        # Return the actual path (might be a shard file if keep_split was used)
        # This ensures the caller can actually find the file
        return actual_output_path

    def generate_imatrix(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        calibration_file: Union[str, Path],
        ctx_size: int = 512,
        num_threads: Optional[int] = None,
        verbose: bool = True,
        chunks: Optional[int] = None,
        collect_output_weight: bool = False,
        num_gpu_layers: Optional[int] = None,
        verbosity: Optional[int] = None,
        from_chunk: Optional[int] = None,
        no_ppl: bool = False,
        parse_special: bool = False,
        output_frequency: Optional[int] = None
    ) -> Path:
        """
        Generate importance matrix (imatrix) for better quantization quality

        Args:
            model_path: Path to GGUF model file (f16/f32)
            output_path: Path for output imatrix file
            calibration_file: Text file with calibration data (required)
            ctx_size: Context window size for processing (default: 512)
            num_threads: Number of threads to use
            verbose: Verbosity control (False=normal/1, True=debug/2)
            chunks: Number of chunks to process (None = process all)
            collect_output_weight: Collect importance matrix for output.weight tensor (required for IQ quantizations)
            num_gpu_layers: Number of GPU layers to offload (None = CPU only)
            verbosity: Verbosity level (0=quiet, 1=normal, 2+=debug, overrides verbose if set)
            from_chunk: Skip first N chunks (useful for resuming)
            no_ppl: Disable perplexity calculation (speeds up processing)
            parse_special: Parse special tokens (recommended for chat models, can significantly slow down generation)
            output_frequency: Save interval in chunks (default: 10)

        Returns:
            Path to the generated imatrix file
        """
        model_path = Path(model_path)
        output_path = Path(output_path)
        calibration_file = Path(calibration_file)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not calibration_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"{theme['info']}Generating importance matrix for {model_path.name}...{Style.RESET_ALL}")
        print(f"{theme['info']}This may take a while...{Style.RESET_ALL}")

        start_time = time.time()

        # Build llama-imatrix command
        imatrix_bin = self.llama_cpp_manager.get_imatrix_path()
        cmd = [
            str(imatrix_bin),
            "-m", str(model_path),
            "-o", str(output_path),
            "-c", str(ctx_size),
            "-f", str(calibration_file)
        ]

        # Add optional arguments

        if chunks:
            cmd.extend(["--chunks", str(chunks)])

        if from_chunk:
            cmd.extend(["--chunk", str(from_chunk)])

        if collect_output_weight:
            cmd.append("--process-output")

        if num_threads:
            cmd.extend(["-t", str(num_threads)])

        if num_gpu_layers is not None:
            cmd.extend(["-ngl", str(num_gpu_layers)])

        # Handle verbosity - new verbosity param overrides old verbose bool
        if verbosity is not None:
            cmd.extend(["-lv", str(verbosity)])
        else:
            # Map verbose boolean: False=1 (normal), True=2 (verbose)
            cmd.extend(["-lv", "2" if verbose else "1"])

        if no_ppl:
            cmd.append("--no-ppl")

        if parse_special:
            cmd.append("--parse-special")

        if output_frequency:
            cmd.extend(["-ofreq", str(output_frequency)])

        # Use GGUF format for compatibility with llama-quantize
        # Newer llama-quantize versions expect GGUF format, not DAT
        cmd.extend(["--output-format", "gguf"])

        print(f"{theme['highlight']}Running: {' '.join(cmd)}{Style.RESET_ALL}")
        print()

        # Run llama-imatrix
        try:
            result = subprocess.run(
                cmd,
                capture_output=not verbose,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=True
            )

            if verbose and result.stdout:
                print(result.stdout)

        except subprocess.CalledProcessError as e:
            raw_error = e.stderr if e.stderr else str(e)
            error_msg = self._clean_llama_error(raw_error)
            raise RuntimeError(f"Imatrix generation failed:\n\n{error_msg}")

        elapsed = time.time() - start_time

        # Print summary
        if output_path.exists():
            print(f"\n{theme['success']}Imatrix generation complete: {output_path}{Style.RESET_ALL}")
            print(f"{theme['metadata']}Time taken: {elapsed:.2f}s ({elapsed/60:.2f} minutes){Style.RESET_ALL}")
        else:
            raise RuntimeError("Imatrix generation appeared to succeed but output file not found")

        return output_path

    def show_imatrix_statistics(
        self,
        imatrix_path: Union[str, Path],
        verbose: bool = False
    ) -> str:
        """
        Show statistics about an imatrix file using native Python implementation

        Args:
            imatrix_path: Path to imatrix file (.gguf format)
            verbose: If True, print output to terminal in real-time

        Returns:
            Statistics output as string
        """
        imatrix_path = Path(imatrix_path)

        if not imatrix_path.exists():
            raise FileNotFoundError(f"Imatrix file not found: {imatrix_path}")

        # Print banner
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner_line = "=" * 80
        print(f"\n{theme['info']}{banner_line}{Style.RESET_ALL}")
        print(f"{theme['info']}{'IMATRIX STATISTICS'.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{imatrix_path.name.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{timestamp.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{banner_line}{Style.RESET_ALL}\n")

        # Use native Python implementation (no longer requires llama-imatrix binary or model file)
        # Capture stdout to return as string
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            # Call our Python implementation
            success = imatrix_stats.show_statistics(str(imatrix_path))

            # Get the output
            output = buffer.getvalue()

            # If verbose, also print to terminal (with colors)
            if verbose:
                print(output, file=old_stdout, end='')

            if not success:
                raise RuntimeError("Failed to compute statistics")

            # Strip ANSI color codes for GUI display
            import re
            clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)

            return clean_output if clean_output.strip() else "No statistics output (file may be empty or incompatible)"

        finally:
            # Restore stdout
            sys.stdout = old_stdout

    def convert_and_quantize(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        quantization_types: List[str] = ["Q4_K_M"],
        intermediate_type: str = "f16",
        custom_intermediate_path: Optional[Union[str, Path]] = None,
        custom_intermediate_format: Optional[str] = None,
        verbose: bool = False,
        parallel: bool = True,
        num_workers: Optional[int] = None,
        scalar_optimization: bool = False,
        imatrix_path: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = None,
        generate_imatrix: bool = False,
        imatrix_ctx_size: int = 512,
        imatrix_chunks: Optional[int] = None,
        imatrix_collect_output: bool = False,
        imatrix_calibration_file: Optional[Union[str, Path]] = None,
        imatrix_output_name: Optional[str] = None,
        imatrix_num_gpu_layers: Optional[int] = None,
        ignore_imatrix_warnings: bool = False,
        leave_output_tensor: bool = False,
        pure_quantization: bool = False,
        keep_split: bool = False,
        output_tensor_type: Optional[str] = None,
        token_embedding_type: Optional[str] = None,
        overwrite_intermediates: bool = False,
        overwrite_quants: bool = True,
        split_max_size: Optional[str] = None,
        mmproj_precision: str = "F16"
    ) -> List[Path]:
        """
        Convert to GGUF and quantize in one go

        Args:
            model_path: Path to input model or HuggingFace repo ID
            output_dir: Directory for output files
            quantization_types: List of quantization types to create
            intermediate_type: Intermediate format (f16 or f32)
            custom_intermediate_path: Path to existing intermediate GGUF file to use instead of generating
            custom_intermediate_format: Format of custom intermediate file (F16/F32/BF16)
            verbose: Enable verbose logging
            parallel: Ignored (kept for API compatibility)
            num_workers: Ignored (kept for API compatibility)
            scalar_optimization: Ignored (kept for API compatibility)
            imatrix_path: Optional importance matrix file for low-bit quants (deprecated, use generate_imatrix)
            num_threads: Number of threads for llama.cpp (None = auto)
            generate_imatrix: Auto-generate importance matrix in output directory
            imatrix_ctx_size: Context window size for imatrix generation (default: 512)
            imatrix_chunks: Number of chunks to process for imatrix (None = all)
            imatrix_collect_output: Collect output.weight tensor in imatrix (required for IQ quantizations)
            imatrix_calibration_file: Path to calibration file for imatrix generation (None = use default)
            ignore_imatrix_warnings: Allow IQ quants without imatrix (may cause degraded quality or failure)
            leave_output_tensor: Keep output.weight unquantized for better quality (increases model size)
            pure_quantization: Disable k-quant mixtures, quantize all tensors uniformly
            keep_split: Keep model in same shards as input (for multi-file models)
            output_tensor_type: Override quantization type for output.weight tensor (e.g., "Q8_0", "F16")
            token_embedding_type: Override quantization type for token embeddings (e.g., "Q8_0", "F16")
            overwrite_intermediates: Regenerate intermediate formats (F32/F16/BF16) even if they exist
            overwrite_quants: Regenerate quantized formats even if they exist
            split_max_size: Maximum size per shard when splitting (e.g., "5G" or "2.5G"). If None, no splitting.
            mmproj_precision: Precision for mmproj (vision projector) file (F16, F32, BF16, Q8_0). Default: F16 for compatibility.

        Returns:
            List of paths to created quantized files
        """
        # Keep original string to check for HuggingFace repo ID format
        model_path_str = str(model_path)
        model_path = Path(model_path)
        output_dir = Path(output_dir)

        # Track if we created the output directory
        created_output_dir = not output_dir.exists()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Print banner
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_display_name = model_path.name
        banner_line = "=" * 80
        print(f"\n{theme['info']}{banner_line}{Style.RESET_ALL}")
        print(f"{theme['info']}{'CONVERT AND QUANTIZE'.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{model_display_name.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{timestamp.center(80)}{Style.RESET_ALL}")
        print(f"{theme['info']}{banner_line}{Style.RESET_ALL}\n")

        # Print version information from actual binaries that will be used
        import re

        # Helper function to get version from binary
        def get_binary_version_and_path(binary_path, binary_name=""):
            try:
                # Try --version first
                result = subprocess.run(
                    [str(binary_path), "--version"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=5
                )

                # If --version fails, try --help (some binaries only show version in help)
                if result.returncode != 0:
                    result = subprocess.run(
                        [str(binary_path), "--help"],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=5
                    )

                if result.returncode == 0:
                    output = result.stderr if result.stderr else result.stdout

                    # Look for the version: line first (more reliable - avoids GPU model numbers)
                    version_line = None
                    for line in output.split('\n'):
                        if line.startswith('version:'):
                            version_line = line.strip()
                            break

                    # Extract version number from the version line or full output
                    search_text = version_line if version_line else output
                    match = re.search(r'\b(b?\d{4,5})\b', search_text)
                    if match:
                        version = match.group(1)
                        return (version if version.startswith('b') else f'b{version}'), binary_path
                    else:
                        return 'unknown', binary_path
                else:
                    return 'unknown', binary_path
            except Exception:
                return 'unknown', binary_path

        # Get binary version (check llama-imatrix, assume llama-quantize is same version)
        imatrix_bin = self.llama_cpp_manager.get_imatrix_path()
        binary_version, _ = get_binary_version_and_path(imatrix_bin, "llama-imatrix")

        # Get conversion scripts version from actual script location that will be used
        try:
            convert_script = self._find_convert_script()
            if convert_script:
                llama_cpp_dir = convert_script.parent
                if (llama_cpp_dir / ".git").exists():
                    result = subprocess.run(
                        ["git", "describe", "--tags", "--always"],
                        cwd=llama_cpp_dir,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        full_version = result.stdout.strip()
                        scripts_version = full_version.split('-')[0] if full_version else 'unknown'
                    else:
                        scripts_version = 'unknown'
                else:
                    scripts_version = 'unknown (not a git repo)'
                scripts_location = str(llama_cpp_dir)
            else:
                scripts_version = 'unknown'
                scripts_location = 'not found'
        except Exception:
            scripts_version = 'unknown'
            scripts_location = 'error'

        print(f"{theme['info']}Binaries: {binary_version} ({imatrix_bin.parent}){Style.RESET_ALL}")
        print(f"{theme['info']}llama.cpp repo: {scripts_version} ({scripts_location}){Style.RESET_ALL}")

        # Print output directory creation message if we created it
        if created_output_dir:
            print(f"{theme['info']}\nCreated output directory: {output_dir}{Style.RESET_ALL}")

        print()  # Empty line after version/directory info

        # Check if it's a HuggingFace repo ID (before Path conversion changes the separators)
        if not model_path.exists() and "/" in model_path_str:
            print(f"{theme['success']}Downloading from HuggingFace: {model_path_str}{Style.RESET_ALL}")
            model_path = self.download_model(model_path_str, output_dir / "downloads")

        # Validate model path - skip config.json check if using custom intermediate
        # Check both path and format to ensure we're actually using a custom intermediate
        using_custom_intermediate = custom_intermediate_path and custom_intermediate_format

        if not using_custom_intermediate:
            # Only validate source model directory when converting from source
            if model_path.is_dir():
                if not (model_path / "config.json").exists():
                    raise ValueError(
                        f"Model path must be a HuggingFace model directory with config.json.\n"
                        f"The directory '{model_path}' does not contain config.json."
                    )
            elif not model_path.is_file():
                raise ValueError(f"Model path does not exist: {model_path}")

        # Check if imatrix enforcement is disabled for IQ quants
        IMATRIX_REQUIRED_TYPES = [
            "IQ1_S", "IQ1_M",
            "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
            "IQ3_XXS", "IQ3_XS"
        ]

        using_imatrix = generate_imatrix or imatrix_path is not None
        requested_iq_quants = [q for q in quantization_types if q in IMATRIX_REQUIRED_TYPES]

        if requested_iq_quants and not using_imatrix and ignore_imatrix_warnings:
            print(f"\n{theme['warning']}WARNING: Enforce imatrix disabled!{Style.RESET_ALL}")
            print(f"\n{theme['warning']}These quantizations require an importance matrix for best quality:{Style.RESET_ALL}")
            print(f"  {', '.join(requested_iq_quants)}")
            print(f"\n{theme['warning']}Proceeding without imatrix (enforce imatrix unchecked){Style.RESET_ALL}")
            print(f"{theme['warning']}Results may have significantly degraded quality. Consider enabling imatrix generation.{Style.RESET_ALL}")

        is_already_gguf = False
        model_name = model_path.name

        # Use split settings if user requested splitting
        split_max_tensors = None
        if split_max_size:
            print(f"{theme['info']}Splitting files with max size per shard: {split_max_size}{Style.RESET_ALL}")

        # Display safetensors file information if converting from safetensors
        if not using_custom_intermediate and model_path.is_dir():
            safetensors_files = list(model_path.glob("*.safetensors"))
            if safetensors_files:
                total_size = sum(f.stat().st_size for f in safetensors_files) / (1024**3)
                print(f"{theme['success']}\nUsing safetensors files from: {model_path.name}{Style.RESET_ALL}")
                print(f"{theme['info']}File count: {len(safetensors_files)}{Style.RESET_ALL}")
                print(f"{theme['info']}Total size: {total_size:.2f} GB{Style.RESET_ALL}\n")

        # Step 1: Convert to GGUF or use custom intermediate
        if using_custom_intermediate:
            # Use provided custom intermediate file
            assert custom_intermediate_path is not None, "custom_intermediate_path must be set when using_custom_intermediate is True"
            intermediate_file = Path(custom_intermediate_path)

            # Validate file exists
            if not intermediate_file.exists():
                raise FileNotFoundError(f"Custom intermediate file not found: {intermediate_file}")

            # Override intermediate_type to match custom file
            if custom_intermediate_format:
                intermediate_type = custom_intermediate_format.upper()

            print(f"{theme['success']}\nUsing custom intermediate file: {intermediate_file.name}{Style.RESET_ALL}")
            print(f"{theme['info']}Format: {intermediate_type}{Style.RESET_ALL}")
            print(f"{theme['info']}Size: {intermediate_file.stat().st_size / (1024**3):.2f} GB{Style.RESET_ALL}")
            print(f"{theme['info']}Skipping conversion step...{Style.RESET_ALL}\n")
        else:
            # Normal conversion workflow
            intermediate_file = output_dir / f"{model_name}_{intermediate_type.upper()}.gguf"

            # Check for existing intermediate files based on keep_split mode
            if split_max_size or split_max_tensors:
                # Keep split mode: check for split intermediate files, ignore single file
                intermediate_shards = sorted(output_dir.glob(f"{intermediate_file.stem}-*-of-*.gguf"))

                if intermediate_shards:
                    # Split files always regenerate - delete old shards
                    existing_shard_count = self._get_shard_count(intermediate_shards)
                    print(f"{theme['warning']}\nExisting {existing_shard_count}-shard intermediate set found{Style.RESET_ALL}")
                    print(f"{theme['warning']}Deleting and regenerating (split mode always overwrites)...{Style.RESET_ALL}")
                    for shard in intermediate_shards:
                        print(f"{theme['warning']}  Deleting: {shard.name}{Style.RESET_ALL}")
                        shard.unlink()

                    # Don't reassign intermediate_file - keep it as base path for split mode
                    self.convert_to_gguf(
                        model_path=model_path,
                        output_path=intermediate_file,
                        output_type=intermediate_type,
                        verbose=verbose,
                        split_max_size=split_max_size,
                        split_max_tensors=split_max_tensors,
                        mmproj_precision=mmproj_precision
                    )
                else:
                    # No split files exist, create them
                    print(f"{theme['info']}Converting {model_name} to GGUF (with splitting)...{Style.RESET_ALL}")
                    # Don't reassign intermediate_file - keep it as base path for split mode
                    self.convert_to_gguf(
                        model_path=model_path,
                        output_path=intermediate_file,
                        output_type=intermediate_type,
                        verbose=verbose,
                        split_max_size=split_max_size,
                        split_max_tensors=split_max_tensors,
                        mmproj_precision=mmproj_precision
                    )
            else:
                # Normal mode: check for single intermediate file, ignore split files
                if intermediate_file.exists():
                    if overwrite_intermediates:
                        print(f"{theme['info']}Intermediate file exists: {intermediate_file.name}{Style.RESET_ALL}")
                        print(f"{theme['info']}Overwriting (overwrite_intermediates=True)...{Style.RESET_ALL}")
                        print(f"{theme['warning']}Overwriting: {intermediate_file}{Style.RESET_ALL}")
                        intermediate_file = self.convert_to_gguf(
                            model_path=model_path,
                            output_path=intermediate_file,
                            output_type=intermediate_type,
                            verbose=verbose,
                            mmproj_precision=mmproj_precision
                        )
                    else:
                        print(f"{theme['success']}Intermediate file already exists: {intermediate_file}{Style.RESET_ALL}")
                        print(f"{theme['info']}Skipping conversion, using existing file...{Style.RESET_ALL}")
                else:
                    # No single file exists, create it
                    print(f"{theme['info']}Converting {model_name} to GGUF...{Style.RESET_ALL}")
                    intermediate_file = self.convert_to_gguf(
                        model_path=model_path,
                        output_path=intermediate_file,
                        output_type=intermediate_type,
                        verbose=verbose,
                        mmproj_precision=mmproj_precision
                    )

        # Step 1.5: Generate importance matrix if requested
        if generate_imatrix:
            # Use custom name if provided, otherwise default to model_name.imatrix
            if imatrix_output_name:
                imatrix_file = output_dir / imatrix_output_name
            else:
                imatrix_file = output_dir / f"{model_name}.imatrix"

            # Always generate fresh imatrix (overwrite if exists)
            # If user wants to reuse, they should use the "Reuse existing" option instead
            if imatrix_file.exists():
                print(f"{theme['info']}Importance matrix already exists: {imatrix_file.name}{Style.RESET_ALL}")
                print(f"{theme['info']}Overwriting with fresh generation...{Style.RESET_ALL}")
                print(f"{theme['warning']}Overwriting: {imatrix_file}{Style.RESET_ALL}")

            print(f"{theme['info']}Generating importance matrix for {model_name}...{Style.RESET_ALL}")

            # Determine calibration file to use
            calibration_file = None
            if imatrix_calibration_file:
                calibration_file = Path(imatrix_calibration_file)
                if not calibration_file.exists():
                    raise FileNotFoundError(
                        f"Calibration file not found: {calibration_file}\n"
                        f"llama-imatrix requires a calibration file. Please add a calibration file\n"
                        f"(like wiki.train.raw) to the calibration_data directory."
                    )
            else:
                raise ValueError(
                    "No calibration file specified. llama-imatrix requires a calibration file.\n"
                    "Please select a calibration file in the Imatrix Settings tab."
                )

            # For imatrix generation, use first shard if in split mode
            imatrix_input_file = intermediate_file
            if split_max_size or split_max_tensors:
                intermediate_shards = sorted(output_dir.glob(f"{intermediate_file.stem}-*-of-*.gguf"))
                if intermediate_shards:
                    imatrix_input_file = intermediate_shards[0]

            self.generate_imatrix(
                model_path=imatrix_input_file,
                output_path=imatrix_file,
                calibration_file=calibration_file,
                ctx_size=imatrix_ctx_size,
                num_threads=num_threads,
                verbose=verbose,
                chunks=imatrix_chunks,
                collect_output_weight=imatrix_collect_output,
                num_gpu_layers=imatrix_num_gpu_layers
            )

            # Use the generated imatrix
            imatrix_path = imatrix_file
        elif imatrix_path is not None:
            # Reusing existing imatrix
            imatrix_path = Path(imatrix_path)
            print(f"{theme['success']}Using existing imatrix: {imatrix_path}{Style.RESET_ALL}")

        # Determine the actual intermediate file to use for quantization
        # In split mode, use the first shard; otherwise use the base path
        actual_intermediate_file = intermediate_file
        if split_max_size or split_max_tensors:
            intermediate_shards = sorted(output_dir.glob(f"{intermediate_file.stem}-*-of-*.gguf"))
            if intermediate_shards:
                actual_intermediate_file = intermediate_shards[0]

        # Step 2: Quantize to requested types
        quantized_files = []
        for quant_type in quantization_types:
            output_file = output_dir / f"{model_name}_{quant_type}.gguf"

            # Handle F16/F32/BF16 specially - these are unquantized formats
            if quant_type.upper() in ["F16", "F32", "BF16"]:
                # Check if this is the intermediate format
                if quant_type.upper() == intermediate_type.upper():
                    print(f"{theme['info']}{quant_type} is the intermediate format{Style.RESET_ALL}")
                    # If intermediate is sharded, add all shards
                    if split_max_size or split_max_tensors:
                        intermediate_stem_base = output_dir / f"{model_name}_{intermediate_type.upper()}"
                        sharded_files = sorted(output_dir.glob(f"{intermediate_stem_base.name}-*-of-*.gguf"))
                        if sharded_files:
                            quantized_files.extend(sharded_files)
                        else:
                            quantized_files.append(intermediate_file)
                    else:
                        quantized_files.append(intermediate_file)
                else:
                    # Not the intermediate format, check for existing files
                    if split_max_size or split_max_tensors:
                        # Keep split mode: check for split files, ignore single
                        output_shards = sorted(output_dir.glob(f"{output_file.stem}-*-of-*.gguf"))

                        if output_shards:
                            # Split files always regenerate - delete old shards
                            existing_shard_count = self._get_shard_count(output_shards)
                            print(f"{theme['warning']}{quant_type} existing {existing_shard_count}-shard set found{Style.RESET_ALL}")
                            print(f"{theme['warning']}Deleting and regenerating (split mode always overwrites)...{Style.RESET_ALL}")
                            for shard in output_shards:
                                print(f"{theme['warning']}  Deleting: {shard.name}{Style.RESET_ALL}")
                                shard.unlink()

                            actual_output = self.convert_to_gguf(
                                model_path=model_path,
                                output_path=output_file,
                                output_type=quant_type.lower(),
                                verbose=verbose,
                                split_max_size=split_max_size,
                                split_max_tensors=split_max_tensors,
                                mmproj_precision=mmproj_precision
                            )
                            # Get all new shards
                            new_shards = sorted(output_dir.glob(f"{output_file.stem}-*-of-*.gguf"))
                            quantized_files.extend(new_shards if new_shards else [actual_output])
                        else:
                            # No split files exist, create them
                            print(f"{theme['info']}Converting {model_name} to {quant_type} from source (with splitting)...{Style.RESET_ALL}")
                            actual_output = self.convert_to_gguf(
                                model_path=model_path,
                                output_path=output_file,
                                output_type=quant_type.lower(),
                                verbose=verbose,
                                split_max_size=split_max_size,
                                split_max_tensors=split_max_tensors,
                                mmproj_precision=mmproj_precision
                            )
                            # Get all shards
                            new_shards = sorted(output_dir.glob(f"{output_file.stem}-*-of-*.gguf"))
                            quantized_files.extend(new_shards if new_shards else [actual_output])
                    else:
                        # Normal mode: check for single file, ignore split files
                        if output_file.exists():
                            if overwrite_intermediates:
                                print(f"{theme['info']}{quant_type} file exists: {output_file.name}{Style.RESET_ALL}")
                                print(f"{theme['info']}Overwriting (overwrite_intermediates=True)...{Style.RESET_ALL}")
                                print(f"{theme['warning']}Overwriting: {output_file}{Style.RESET_ALL}")
                                actual_output = self.convert_to_gguf(
                                    model_path=model_path,
                                    output_path=output_file,
                                    output_type=quant_type.lower(),
                                    verbose=verbose,
                                    mmproj_precision=mmproj_precision
                                )
                                quantized_files.append(actual_output)
                            else:
                                print(f"{theme['success']}{quant_type} file already exists: {output_file}{Style.RESET_ALL}")
                                print(f"{theme['info']}Skipping conversion, using existing file...{Style.RESET_ALL}")
                                quantized_files.append(output_file)
                        else:
                            # No single file exists, create it
                            print(f"{theme['info']}Converting {model_name} to {quant_type} from source...{Style.RESET_ALL}")
                            actual_output = self.convert_to_gguf(
                                model_path=model_path,
                                output_path=output_file,
                                output_type=quant_type.lower(),
                                verbose=verbose,
                                mmproj_precision=mmproj_precision
                            )
                            quantized_files.append(actual_output)
            else:
                # Regular quantization types (Q4_K_M, etc.)
                if keep_split:
                    # Keep split mode: check for split files, ignore single
                    output_shards = sorted(output_dir.glob(f"{output_file.stem}-*-of-*.gguf"))

                    if output_shards:
                        # Check if we should reuse existing shards
                        is_complete = self._validate_shard_set(output_shards)
                        existing_shard_count = self._get_shard_count(output_shards)

                        if is_complete and not overwrite_quants:
                            print(f"{theme['warning']}{quant_type} existing {existing_shard_count}-shard set found{Style.RESET_ALL}")
                            print(f"{theme['warning']}Note: New max_shard_size setting may produce a different shard count{Style.RESET_ALL}")
                            print(f"{theme['info']}Reusing existing files...{Style.RESET_ALL}")
                            for shard in output_shards:
                                print(f"{theme['metadata']}  - {shard.name}{Style.RESET_ALL}")
                            quantized_files.extend(output_shards)
                            continue  # Skip to next quant type

                        # Need to regenerate - delete old shards
                        if not is_complete:
                            print(f"{theme['warning']}{quant_type} incomplete shard set found ({len(output_shards)} file(s)){Style.RESET_ALL}")
                        else:
                            print(f"{theme['info']}{quant_type} existing {existing_shard_count}-shard set found{Style.RESET_ALL}")
                            print(f"{theme['info']}Overwriting (overwrite_quants=True)...{Style.RESET_ALL}")

                        print(f"{theme['warning']}Deleting old shards...{Style.RESET_ALL}")
                        for shard in output_shards:
                            print(f"{theme['warning']}  Deleting: {shard.name}{Style.RESET_ALL}")
                            shard.unlink()

                    # Need to quantize (either no files, incomplete set, or overwrite_quants=True)
                    # Use actual_intermediate_file which is already set to first shard in split mode
                    quantize_input = actual_intermediate_file

                    actual_output_path = self.quantize(
                        input_path=quantize_input,
                        output_path=output_file,
                        quantization_type=quant_type,
                        verbose=verbose,
                        imatrix_path=imatrix_path,
                        num_threads=num_threads,
                        leave_output_tensor=leave_output_tensor,
                        pure_quantization=pure_quantization,
                        keep_split=keep_split,
                        output_tensor_type=output_tensor_type,
                        token_embedding_type=token_embedding_type
                    )

                    # Get all output shards
                    new_shards = sorted(output_dir.glob(f"{output_file.stem}-*-of-*.gguf"))
                    if new_shards:
                        quantized_files.extend(new_shards)
                    else:
                        # Fallback if no shards found (shouldn't happen in keep_split mode)
                        quantized_files.append(actual_output_path)
                else:
                    # Normal mode: check for single file, ignore split files
                    if output_file.exists() and not overwrite_quants:
                        print(f"{theme['success']}{quant_type} file already exists: {output_file}{Style.RESET_ALL}")
                        print(f"{theme['info']}Skipping quantization, using existing file...{Style.RESET_ALL}")
                        quantized_files.append(output_file)
                    else:
                        # Need to quantize
                        if output_file.exists():
                            print(f"{theme['info']}{quant_type} file exists: {output_file.name}{Style.RESET_ALL}")
                            print(f"{theme['info']}Overwriting (overwrite_quants=True)...{Style.RESET_ALL}")
                            print(f"{theme['warning']}Overwriting: {output_file}{Style.RESET_ALL}")

                        actual_output_path = self.quantize(
                            input_path=actual_intermediate_file,
                            output_path=output_file,
                            quantization_type=quant_type,
                            verbose=verbose,
                            imatrix_path=imatrix_path,
                            num_threads=num_threads,
                            leave_output_tensor=leave_output_tensor,
                            pure_quantization=pure_quantization,
                            keep_split=False,  # Explicitly disable in normal mode
                            output_tensor_type=output_tensor_type,
                            token_embedding_type=token_embedding_type
                        )

                        quantized_files.append(actual_output_path)

        return quantized_files
