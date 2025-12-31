"""
Core GGUF conversion and quantization functionality
"""

import subprocess
import sys
import json
import io
import time
import shutil
from pathlib import Path
from typing import Optional, Union, List
from huggingface_hub import snapshot_download, HfApi
from colorama import init as colorama_init, Style
from .llama_cpp_manager import LlamaCppManager
from . import imatrix_stats
from .theme import THEME as theme

# Initialize colorama for cross-platform color support
colorama_init(autoreset=True)



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
                    "Please check your internet connection or install llama.cpp manually."
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

        print(f"{theme['info']}Downloading {repo_id} from HuggingFace...{Style.RESET_ALL}")
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir / Path(repo_id).name,
            revision=revision
        )

        return Path(model_path)

    def convert_to_gguf(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        output_type: str = "f16",
        vocab_only: bool = False,
        verbose: bool = False
    ) -> Path:
        """
        Convert a model to GGUF format

        Args:
            model_path: Path to the input model (safetensors/pytorch)
            output_path: Path for the output GGUF file
            output_type: Output precision (f32, f16, bf16, q8_0, auto)
            vocab_only: Only extract vocabulary
            verbose: Enable verbose logging

        Returns:
            Path to the created GGUF file
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

        # Check if model requires special handling
        if self._is_ministral3_model(model_path):
            print(f"{theme['info']}Detected Ministral-3 model, using --mistral-format flag{Style.RESET_ALL}")
            cmd.append("--mistral-format")

        print(f"{theme['info']}Converting {model_path.name} to GGUF format...{Style.RESET_ALL}")
        print(f"{theme['highlight']}Running: {' '.join(cmd)}{Style.RESET_ALL}")

        result = subprocess.run(cmd, capture_output=not verbose, text=True)

        if result.returncode != 0:
            # Combine stdout and stderr since errors can be in either
            error_output = (result.stderr or '') + (result.stdout or '')
            raw_error = error_output.strip() if error_output.strip() else 'Unknown error'
            error_msg = self._clean_llama_error(raw_error)
            raise RuntimeError(f"Conversion failed:\n\n{error_msg}")

        if verbose and result.stdout:
            print(result.stdout)

        print(f"{theme['success']}Conversion complete: {output_path}{Style.RESET_ALL}")
        return output_path

    def _is_ministral3_model(self, model_path: Path) -> bool:
        """
        Check if the model is a Ministral-3 model that requires --mistral-format flag

        Args:
            model_path: Path to the model directory

        Returns:
            True if this is a Ministral-3 model
        """
        config_file = model_path / "config.json"
        if not config_file.exists():
            return False

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Check for Ministral-3 indicators
            architectures = config.get("architectures", [])
            model_type = config.get("model_type", "")
            text_config = config.get("text_config", {})
            text_model_type = text_config.get("model_type", "")

            # Ministral-3 has Mistral3ForConditionalGeneration architecture
            # and ministral3 as the text model type
            return (
                "Mistral3ForConditionalGeneration" in architectures or
                model_type == "mistral3" or
                text_model_type == "ministral3"
            )
        except (json.JSONDecodeError, IOError):
            return False

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

        # Fallback: check for system-wide installations
        fallback_locations = [
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.cwd() / "llama.cpp" / "convert_hf_to_gguf.py",
        ]

        for location in fallback_locations:
            if location.exists():
                return location

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
        allow_requantize: bool = False,
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
            allow_requantize: Allow quantizing already-quantized models (may reduce quality)
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
        quantize_bin = self.llama_cpp_manager.get_quantize_path()
        cmd = [str(quantize_bin), str(input_path), str(output_path), quantization_type]

        # Add num_threads (positional argument, comes before optional flags)
        if num_threads:
            cmd.append(str(num_threads))

        # Add optional flags (must come after positional arguments)
        if allow_requantize:
            cmd.append("--allow-requantize")

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

        print(f"{theme['highlight']}Running: {' '.join(cmd)}{Style.RESET_ALL}")
        print()

        # Run llama-quantize
        # Always capture output so we can parse errors, but print it if verbose=True
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
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

            # Check if imatrix was provided but llama.cpp still complains about missing imatrix
            # This indicates likely model incompatibility
            if "without an importance matrix" in raw_error and imatrix_path:
                raise RuntimeError(
                    f"Quantization failed: Model incompatibility detected.\n\n"
                    f"An importance matrix was provided, but {quantization_type} quantization still failed.\n"
                    f"This typically means the model architecture is incompatible with this quantization type.\n\n"
                    f"Error:\n{error_msg}"
                )

            raise RuntimeError(f"Quantization failed:\n\n{error_msg}")

        elapsed = time.time() - start_time

        # Print summary
        if output_path.exists():
            input_size = input_path.stat().st_size / (1024**3)
            output_size = output_path.stat().st_size / (1024**3)
            ratio = input_size / output_size if output_size > 0 else 0

            print(f"\n{theme['success']}Quantization complete: {output_path}{Style.RESET_ALL}")
            print(f"{theme['metadata']}Time taken: {elapsed:.2f}s ({elapsed/60:.2f} minutes){Style.RESET_ALL}")
            print(f"{theme['metadata']}Size: {input_size:.2f} GB -> {output_size:.2f} GB ({ratio:.2f}x compression){Style.RESET_ALL}")
        else:
            raise RuntimeError("Quantization appeared to succeed but output file not found")

        return output_path

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

        # Use native Python implementation (no longer requires llama-imatrix binary or model file)
        # Capture stdout to return as string
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            # Call our Python implementation
            success = imatrix_stats.show_statistics(str(imatrix_path))

            # Get the output
            output = buffer.getvalue()

            # If verbose, also print to terminal
            if verbose:
                print(output, file=old_stdout, end='')

            if not success:
                raise RuntimeError("Failed to compute statistics")

            return output if output.strip() else "No statistics output (file may be empty or incompatible)"

        finally:
            # Restore stdout
            sys.stdout = old_stdout

    def convert_and_quantize(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        quantization_types: List[str] = ["Q4_K_M"],
        intermediate_type: str = "f16",
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
        allow_requantize: bool = False,
        leave_output_tensor: bool = False,
        pure_quantization: bool = False,
        keep_split: bool = False,
        output_tensor_type: Optional[str] = None,
        token_embedding_type: Optional[str] = None
    ) -> List[Path]:
        """
        Convert to GGUF and quantize in one go

        Args:
            model_path: Path to input model or HuggingFace repo ID
            output_dir: Directory for output files
            quantization_types: List of quantization types to create
            intermediate_type: Intermediate format (f16 or f32)
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
            allow_requantize: Allow quantizing already-quantized models (may reduce quality)
            leave_output_tensor: Keep output.weight unquantized for better quality (increases model size)
            pure_quantization: Disable k-quant mixtures, quantize all tensors uniformly
            keep_split: Keep model in same shards as input (for multi-file models)
            output_tensor_type: Override quantization type for output.weight tensor (e.g., "Q8_0", "F16")
            token_embedding_type: Override quantization type for token embeddings (e.g., "Q8_0", "F16")

        Returns:
            List of paths to created quantized files
        """
        # Keep original string to check for HuggingFace repo ID format
        model_path_str = str(model_path)
        model_path = Path(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if it's a HuggingFace repo ID (before Path conversion changes the separators)
        if not model_path.exists() and "/" in model_path_str:
            print(f"{theme['info']}Downloading from HuggingFace: {model_path_str}{Style.RESET_ALL}")
            model_path = self.download_model(model_path_str, output_dir / "downloads")

        # Validate model path is a source model directory with config.json
        if model_path.is_dir():
            if not (model_path / "config.json").exists():
                raise ValueError(
                    f"Model path must be a HuggingFace model directory with config.json.\n"
                    f"The directory '{model_path}' does not contain config.json."
                )
        elif not model_path.is_file():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Check for imatrix warnings
        IMATRIX_REQUIRED_TYPES = [
            "IQ1_S", "IQ1_M",
            "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
            "IQ3_XXS", "IQ3_XS"
        ]

        using_imatrix = generate_imatrix or imatrix_path is not None
        requested_iq_quants = [q for q in quantization_types if q in IMATRIX_REQUIRED_TYPES]

        if requested_iq_quants and not using_imatrix and ignore_imatrix_warnings:
            print(f"\n{theme['warning']}WARNING: Imatrix warnings override enabled!{Style.RESET_ALL}")
            print(f"\n{theme['warning']}These quantizations require an importance matrix for best quality:{Style.RESET_ALL}")
            print(f"  {', '.join(requested_iq_quants)}")
            print(f"\n{theme['warning']}Proceeding without imatrix as requested (ignore_imatrix_warnings=True){Style.RESET_ALL}")
            print(f"{theme['warning']}Results may have significantly degraded quality. Consider enabling imatrix generation.{Style.RESET_ALL}")

        is_already_gguf = False
        model_name = model_path.name

        # Step 1: Convert to GGUF
        intermediate_file = output_dir / f"{model_name}_{intermediate_type.upper()}.gguf"

        # Check if intermediate file already exists
        if intermediate_file.exists():
            print(f"{theme['info']}\nIntermediate file already exists: {intermediate_file.name}{Style.RESET_ALL}")
            print(f"{theme['info']}Skipping conversion, using existing file...{Style.RESET_ALL}")
        else:
            print(f"{theme['info']}Converting {model_name} to GGUF...{Style.RESET_ALL}")
            self.convert_to_gguf(
                model_path=model_path,
                output_path=intermediate_file,
                output_type=intermediate_type,
                verbose=verbose
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

            self.generate_imatrix(
                model_path=intermediate_file,
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

        # Step 2: Quantize to requested types
        quantized_files = []
        for quant_type in quantization_types:
            output_file = output_dir / f"{model_name}_{quant_type}.gguf"

            # Handle F16/F32/BF16 specially - these are unquantized formats
            if quant_type.upper() in ["F16", "F32", "BF16"]:
                # Check if this is the intermediate format
                if quant_type.upper() == intermediate_type.upper():
                    print(f"{theme['info']}{quant_type} is the intermediate format{Style.RESET_ALL}")
                    quantized_files.append(intermediate_file)
                # Check if output already exists
                elif output_file.exists():
                    print(f"{theme['info']}{quant_type} file already exists: {output_file.name}{Style.RESET_ALL}")
                    print(f"{theme['info']}Skipping conversion, using existing file...{Style.RESET_ALL}")
                    quantized_files.append(output_file)
                # Generate from source
                else:
                    print(f"{theme['info']}Converting {model_name} to {quant_type} from source...{Style.RESET_ALL}")
                    self.convert_to_gguf(
                        model_path=model_path,
                        output_path=output_file,
                        output_type=quant_type.lower(),
                        verbose=verbose
                    )
                    quantized_files.append(output_file)
            else:
                self.quantize(
                    input_path=intermediate_file,
                    output_path=output_file,
                    quantization_type=quant_type,
                    verbose=verbose,
                    imatrix_path=imatrix_path,
                    num_threads=num_threads,
                    allow_requantize=allow_requantize,
                    leave_output_tensor=leave_output_tensor,
                    pure_quantization=pure_quantization,
                    keep_split=keep_split,
                    output_tensor_type=output_tensor_type,
                    token_embedding_type=token_embedding_type
                )
                quantized_files.append(output_file)

        return quantized_files
