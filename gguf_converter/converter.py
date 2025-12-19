"""
Core GGUF conversion and quantization functionality
"""

import os
import shutil
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, Union, List
from huggingface_hub import snapshot_download
from colorama import Fore, Style, init as colorama_init
from .binary_manager import BinaryManager

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

    # Model incompatibility registry
    MODEL_INCOMPATIBILITIES = {
        "tied_embeddings": {
            "description": "Models with tied embeddings (shared input/output embeddings)",
            "detection": {
                "config_flag": "tie_word_embeddings",
                "model_families": ["Qwen", "QWen", "qwen"],
            },
            "incompatible_quants": [
                # IQ quantizations require separate output.weight tensor
                "IQ1_S", "IQ1_M",
                "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
                "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
                "IQ4_XS", "IQ4_NL",
                "Q2_K_S",  # Also requires output.weight
            ],
            "alternatives": [
                "Q3_K_M or Q3_K_S (similar quality to IQ3)",
                "Q2_K (not Q2_K_S) for smaller files",
                "Q4_K_M (best quality/size balance)",
            ],
            "reason": "These quantizations require a separate output.weight tensor. Models with tied embeddings share the same tensor for input and output, making these quantizations incompatible.",
        },
    }

    def __init__(self, custom_binaries_folder=None):
        """
        Initialize the converter and binary manager

        Args:
            custom_binaries_folder: Optional path to folder containing custom llama.cpp binaries.
                                   If empty string, will use system PATH.
                                   If None, will use auto-downloaded binaries.
        """
        self.binary_manager = BinaryManager(custom_binaries_folder=custom_binaries_folder)
        if custom_binaries_folder is None:
            if not self.binary_manager.ensure_binaries():
                raise RuntimeError(
                    "Failed to get llama.cpp binaries. "
                    "Please check your internet connection or install llama.cpp manually."
                )
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

        print(f"Downloading {repo_id} from HuggingFace...")
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
                "Could not find convert_hf_to_gguf.py script. "
                "The llama.cpp repository should have been auto-cloned. "
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
            print(f"Detected Ministral-3 model, using --mistral-format flag")
            cmd.append("--mistral-format")

        print(f"{Fore.YELLOW}Converting {model_path.name} to GGUF format...{Style.RESET_ALL}")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=not verbose, text=True)

        if result.returncode != 0:
            # Combine stdout and stderr since errors can be in either
            error_output = (result.stderr or '') + (result.stdout or '')
            error_msg = error_output.strip() if error_output.strip() else 'Unknown error'
            raise RuntimeError(f"Conversion failed:\n{error_msg}")

        if verbose and result.stdout:
            print(result.stdout)

        print(f"{Fore.GREEN}Conversion complete: {output_path}{Style.RESET_ALL}")
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

    def _check_incompatibility_type(self, model_path: Path, incompat_type: str) -> bool:
        """
        Check if a model matches a specific incompatibility type

        Args:
            model_path: Path to the model directory
            incompat_type: Incompatibility type key from MODEL_INCOMPATIBILITIES

        Returns:
            True if model matches this incompatibility type
        """
        if incompat_type not in self.MODEL_INCOMPATIBILITIES:
            return False

        config_file = model_path / "config.json"
        if not config_file.exists():
            return False

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            incompat_info = self.MODEL_INCOMPATIBILITIES[incompat_type]
            detection = incompat_info["detection"]

            # Check config flag if specified
            if "config_flag" in detection:
                if config.get(detection["config_flag"], False):
                    return True

            # Check model families if specified
            if "model_families" in detection:
                architectures = config.get("architectures", [])
                model_type = config.get("model_type", "")

                for pattern in detection["model_families"]:
                    # Check architectures
                    if any(pattern in arch for arch in architectures):
                        return True
                    # Check model_type
                    if pattern in model_type:
                        return True

            return False

        except (json.JSONDecodeError, IOError):
            return False

    def get_incompatible_quantizations(self, model_path: Union[str, Path]) -> List[str]:
        """
        Get list of quantization types incompatible with this model

        Args:
            model_path: Path to the model directory

        Returns:
            List of incompatible quantization type names
        """
        model_path = Path(model_path)
        incompatible = []

        # Check all registered incompatibility types
        for incompat_type in self.MODEL_INCOMPATIBILITIES:
            if self._check_incompatibility_type(model_path, incompat_type):
                incompatible.extend(
                    self.MODEL_INCOMPATIBILITIES[incompat_type]["incompatible_quants"]
                )

        # Remove duplicates while preserving order
        seen = set()
        return [x for x in incompatible if not (x in seen or seen.add(x))]

    def get_incompatibility_info(self, model_path: Union[str, Path]) -> dict:
        """
        Get detailed incompatibility information for a model

        Args:
            model_path: Path to the model directory

        Returns:
            Dictionary with incompatibility details:
            {
                "has_incompatibilities": bool,
                "types": List[str],  # Matched incompatibility type keys
                "incompatible_quants": List[str],
                "alternatives": List[str],
                "reasons": List[str],
            }
        """
        model_path = Path(model_path)
        matched_types = []
        all_incompatible = []
        all_alternatives = []
        all_reasons = []

        # Check all registered incompatibility types
        for incompat_type, info in self.MODEL_INCOMPATIBILITIES.items():
            if self._check_incompatibility_type(model_path, incompat_type):
                matched_types.append(incompat_type)
                all_incompatible.extend(info["incompatible_quants"])
                all_alternatives.extend(info["alternatives"])
                all_reasons.append(f"{info['description']}: {info['reason']}")

        # Remove duplicate quants while preserving order
        seen = set()
        unique_incompatible = [x for x in all_incompatible if not (x in seen or seen.add(x))]

        return {
            "has_incompatibilities": len(matched_types) > 0,
            "types": matched_types,
            "incompatible_quants": unique_incompatible,
            "alternatives": all_alternatives,
            "reasons": all_reasons,
        }

    def _ensure_llama_cpp_repo(self):
        """
        Ensure llama.cpp repository exists (download if missing)
        Does NOT auto-update - use Upgrade tab in GUI for updates
        """
        llama_cpp_dir = Path(__file__).parent.parent / "llama.cpp"
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

        # Only clone if missing - don't auto-update
        if not llama_cpp_dir.exists() or not convert_script.exists():
            # Remove old repo if it exists but is incomplete
            if llama_cpp_dir.exists():
                print(f"Removing incomplete llama.cpp repository...")
                shutil.rmtree(llama_cpp_dir)

            # Clone fresh copy
            expected_version = self.binary_manager.LLAMA_CPP_VERSION
            print(f"Cloning llama.cpp repository (version {expected_version})...")
            try:
                subprocess.run([
                    "git", "clone",
                    "https://github.com/ggml-org/llama.cpp.git",
                    "--depth=1",
                    str(llama_cpp_dir)
                ], check=True, capture_output=True, text=True)

                # Write version file for tracking
                version_file = llama_cpp_dir / "REPO_VERSION"
                version_file.write_text(expected_version)
                print(f"llama.cpp repository ready (version {expected_version})")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to clone llama.cpp repository: {e.stderr if e.stderr else str(e)}\n"
                    f"Please check your git installation and internet connection."
                )

    def _find_convert_script(self) -> Optional[Path]:
        """
        Find the convert_hf_to_gguf.py script

        Note: _ensure_llama_cpp_repo() is called during init, so the repo
        should already be cloned and up to date
        """
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
        nthreads: Optional[int] = None,
        verbose: bool = True,
        parallel: bool = True,
        num_workers: Optional[int] = None,
        scalar_optimization: bool = False,
        allow_requantize: bool = False,
        leave_output_tensor: bool = False,
        pure_quantization: bool = False,
        keep_split: bool = False
    ) -> Path:
        """
        Quantize a GGUF model using llama.cpp

        Args:
            input_path: Path to input GGUF file (f16/f32)
            output_path: Path for quantized output file
            quantization_type: Quantization type (e.g., Q4_0, Q4_K_M, IQ3_XXS, etc.)
            imatrix_path: Optional importance matrix file for better quality
            nthreads: Number of threads to use (passed to llama-quantize)
            verbose: Enable verbose output
            parallel: Ignored (kept for API compatibility)
            num_workers: Ignored (kept for API compatibility)
            scalar_optimization: Ignored (kept for API compatibility)
            allow_requantize: Allow quantizing already-quantized models (may reduce quality)
            leave_output_tensor: Keep output.weight unquantized for better quality (increases model size)
            pure_quantization: Disable k-quant mixtures, quantize all tensors uniformly
            keep_split: Keep model in same shards as input (for multi-file models)

        Returns:
            Path to the quantized GGUF file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if quantization_type not in self.QUANTIZATION_TYPES:
            raise ValueError(
                f"Invalid quantization type: {quantization_type}. "
                f"Available: {', '.join(self.QUANTIZATION_TYPES)}"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"{Fore.YELLOW}Quantizing {input_path.name} to {quantization_type}...{Style.RESET_ALL}")

        import time
        start_time = time.time()

        # Build llama-quantize command
        quantize_bin = self.binary_manager.get_quantize_path()
        cmd = [str(quantize_bin), str(input_path), str(output_path), quantization_type]

        # Add nthreads (positional argument, comes before optional flags)
        if nthreads:
            cmd.append(str(nthreads))

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

        print(f"{Fore.CYAN}Running: {' '.join(cmd)}{Style.RESET_ALL}")
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
            error_msg = e.stderr if e.stderr else str(e)

            # Check if imatrix was provided but llama.cpp still complains about missing imatrix
            # This indicates model incompatibility (e.g., tied embeddings), not user error
            if "without an importance matrix" in error_msg and imatrix_path:
                raise RuntimeError(
                    f"Quantization failed: Model incompatibility detected.\n\n"
                    f"An importance matrix was provided, but {quantization_type} quantization still failed. "
                    f"This typically means the model architecture is incompatible with this quantization type.\n\n"
                    f"Common cause: Models with 'tied embeddings' (e.g., Qwen series) lack the output.weight tensor "
                    f"required for IQ quantizations due to a limitation in llama.cpp.\n\n"
                    f"Recommended alternatives:\n"
                    f"  - Q3_K_M or Q3_K_S (good quality, similar size)\n"
                    f"  - Q2_K (not Q2_K_S) if you need smaller files\n\n"
                    f"See KNOWN_ISSUES.md for detailed explanation.\n\n"
                    f"Original error: {error_msg}"
                )

            raise RuntimeError(f"Quantization failed: {error_msg}")

        elapsed = time.time() - start_time

        # Print summary
        if output_path.exists():
            input_size = input_path.stat().st_size / (1024**3)
            output_size = output_path.stat().st_size / (1024**3)
            ratio = input_size / output_size if output_size > 0 else 0

            print(f"\n{Fore.GREEN}Quantization complete: {output_path}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Time taken: {elapsed:.2f}s ({elapsed/60:.2f} minutes){Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Size: {input_size:.2f} GB -> {output_size:.2f} GB ({ratio:.2f}x compression){Style.RESET_ALL}")
        else:
            raise RuntimeError("Quantization appeared to succeed but output file not found")

        return output_path

    def generate_imatrix(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        calibration_file: Optional[Union[str, Path]] = None,
        ctx_size: int = 512,
        nthreads: Optional[int] = None,
        verbose: bool = True,
        chunks: Optional[int] = None,
        collect_output_weight: bool = False,
        ngl: Optional[int] = None,
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
            calibration_file: Text file with calibration data (default: uses built-in)
            ctx_size: Context window size for processing (default: 512)
            nthreads: Number of threads to use
            verbose: Verbosity control (False=normal/1, True=debug/2)
            chunks: Number of chunks to process (None = process all)
            collect_output_weight: Collect importance matrix for output.weight tensor (required for IQ quantizations)
            ngl: Number of GPU layers to offload (None = CPU only)
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

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if calibration_file:
            calibration_file = Path(calibration_file)
            if not calibration_file.exists():
                raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"{Fore.YELLOW}Generating importance matrix for {model_path.name}...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}This may take a while...{Style.RESET_ALL}")

        import time
        start_time = time.time()

        # Build llama-imatrix command
        imatrix_bin = self.binary_manager.get_imatrix_path()
        cmd = [
            str(imatrix_bin),
            "-m", str(model_path),
            "-o", str(output_path),
            "-c", str(ctx_size)
        ]

        # Add optional arguments
        if calibration_file:
            cmd.extend(["-f", str(calibration_file)])

        if chunks:
            cmd.extend(["--chunks", str(chunks)])

        if from_chunk:
            cmd.extend(["--chunk", str(from_chunk)])

        if collect_output_weight:
            cmd.append("--process-output")

        if nthreads:
            cmd.extend(["-t", str(nthreads)])

        if ngl is not None:
            cmd.extend(["-ngl", str(ngl)])

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

        print(f"{Fore.CYAN}Running: {' '.join(cmd)}{Style.RESET_ALL}")
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
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"Imatrix generation failed: {error_msg}")

        elapsed = time.time() - start_time

        # Print summary
        if output_path.exists():
            print(f"\n{Fore.GREEN}Imatrix generation complete: {output_path}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Time taken: {elapsed:.2f}s ({elapsed/60:.2f} minutes){Style.RESET_ALL}")
        else:
            raise RuntimeError("Imatrix generation appeared to succeed but output file not found")

        return output_path

    def show_imatrix_statistics(
        self,
        imatrix_path: Union[str, Path],
        model_path: Union[str, Path],
        verbose: bool = False
    ) -> str:
        """
        Show statistics about an imatrix file

        Args:
            imatrix_path: Path to imatrix file
            model_path: Path to model file (required by llama-imatrix)
            verbose: If True, print output to terminal in real-time

        Returns:
            Statistics output as string
        """
        imatrix_path = Path(imatrix_path)
        model_path = Path(model_path)

        if not imatrix_path.exists():
            raise FileNotFoundError(f"Imatrix file not found: {imatrix_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Build llama-imatrix command with --show-statistics
        # Note: llama-imatrix requires a model even for showing statistics
        imatrix_bin = self.binary_manager.get_imatrix_path()
        cmd = [
            str(imatrix_bin),
            "-m", str(model_path),
            "--in-file", str(imatrix_path),
            "--show-statistics"
        ]

        if verbose:
            print(f"\nRunning: {' '.join(cmd)}\n", flush=True)

        try:
            if verbose:
                # Stream output to terminal in real-time AND capture for GUI
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                )

                # Collect output while streaming to terminal
                output_lines = []
                assert process.stdout is not None  # We set stdout=PIPE above
                for line in process.stdout:
                    print(line, end='', flush=True)  # Real-time terminal output
                    output_lines.append(line)  # Capture for GUI

                # Wait for process to complete
                process.wait()

                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)

                output = ''.join(output_lines)
                return output if output.strip() else "No statistics output (file may be empty or incompatible)"
            else:
                # Capture output for display in GUI only
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    check=True
                )
                # Combine stdout and stderr since llama-imatrix may output to either
                output = result.stdout + result.stderr
                return output if output.strip() else "No statistics output (file may be empty or incompatible)"
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"Failed to show statistics: {error_msg}")

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
        nthreads: Optional[int] = None,
        generate_imatrix: bool = False,
        imatrix_ctx_size: int = 512,
        imatrix_chunks: Optional[int] = None,
        imatrix_collect_output: bool = False,
        imatrix_calibration_file: Optional[Union[str, Path]] = None,
        imatrix_output_name: Optional[str] = None,
        imatrix_ngl: Optional[int] = None,
        ignore_incompatibilities: bool = False,
        allow_requantize: bool = False,
        leave_output_tensor: bool = False,
        pure_quantization: bool = False,
        keep_split: bool = False
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
            nthreads: Number of threads for llama.cpp (None = auto)
            generate_imatrix: Auto-generate importance matrix in output directory
            imatrix_ctx_size: Context window size for imatrix generation (default: 512)
            imatrix_chunks: Number of chunks to process for imatrix (None = all)
            imatrix_collect_output: Collect output.weight tensor in imatrix (required for IQ quantizations)
            imatrix_calibration_file: Path to calibration file for imatrix generation (None = use default)
            ignore_incompatibilities: Skip incompatibility checks (advanced users, may cause failures)
            allow_requantize: Allow quantizing already-quantized models (may reduce quality)
            leave_output_tensor: Keep output.weight unquantized for better quality (increases model size)
            pure_quantization: Disable k-quant mixtures, quantize all tensors uniformly
            keep_split: Keep model in same shards as input (for multi-file models)

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
            print(f"Downloading from HuggingFace: {model_path_str}")
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

        # Check for incompatible quantizations using centralized registry
        incompat_info = self.get_incompatibility_info(model_path)
        if incompat_info["has_incompatibilities"] and not ignore_incompatibilities:
            incompatible = incompat_info["incompatible_quants"]
            requested_incompatible = [q for q in quantization_types if q in incompatible]

            if requested_incompatible:
                # Filter out incompatible types
                original_types = quantization_types.copy()
                quantization_types = [q for q in quantization_types if q not in incompatible]

                print(f"\n{Fore.YELLOW}WARNING: Model incompatibility detected!{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Detected issues: {', '.join(incompat_info['types'])}{Style.RESET_ALL}")
                print(f"\n{Fore.YELLOW}Incompatible quantizations requested:{Style.RESET_ALL}")
                print(f"  {', '.join(requested_incompatible)}")

                print(f"\n{Fore.YELLOW}Reason:{Style.RESET_ALL}")
                for reason in incompat_info["reasons"]:
                    print(f"  - {reason}")

                print(f"\n{Fore.CYAN}Recommended alternatives:{Style.RESET_ALL}")
                for alt in incompat_info["alternatives"]:
                    print(f"  - {alt}")

                if quantization_types:
                    print(f"\n{Fore.GREEN}Continuing with compatible quantizations: {', '.join(quantization_types)}{Style.RESET_ALL}")
                else:
                    raise ValueError(
                        f"All requested quantizations are incompatible with this model.\n"
                        f"Incompatible: {', '.join(requested_incompatible)}\n"
                        f"Alternatives: {', '.join(incompat_info['alternatives'])}\n"
                        f"See KNOWN_ISSUES.md for details."
                    )
        elif incompat_info["has_incompatibilities"] and ignore_incompatibilities:
            # Show warning but proceed
            incompatible = incompat_info["incompatible_quants"]
            requested_incompatible = [q for q in quantization_types if q in incompatible]

            if requested_incompatible:
                print(f"\n{Fore.YELLOW}⚠ WARNING: Incompatibility override enabled!{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Detected issues: {', '.join(incompat_info['types'])}{Style.RESET_ALL}")
                print(f"\n{Fore.RED}These quantizations are likely to FAIL:{Style.RESET_ALL}")
                print(f"  {', '.join(requested_incompatible)}")
                print(f"\n{Fore.YELLOW}Proceeding anyway as requested (ignore_incompatibilities=True){Style.RESET_ALL}")
                print(f"{Fore.CYAN}If conversion fails, use: {', '.join(incompat_info['alternatives'])}{Style.RESET_ALL}")

        is_already_gguf = False
        model_name = model_path.name

        # Step 1: Convert to GGUF
        intermediate_file = output_dir / f"{model_name}_{intermediate_type.upper()}.gguf"

        # Check if intermediate file already exists
        if intermediate_file.exists():
            print(f"Intermediate file already exists: {intermediate_file.name}")
            print("Skipping conversion, using existing file...")
        else:
            print(f"Converting {model_name} to GGUF...")
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
                print(f"Importance matrix already exists: {imatrix_file.name}")
                print("Overwriting with fresh generation...")

            print(f"Generating importance matrix for {model_name}...")

            # Determine calibration file to use
            calibration_file = None
            if imatrix_calibration_file:
                calibration_file = Path(imatrix_calibration_file)
                if not calibration_file.exists():
                    print(f"Warning: Specified calibration file not found: {calibration_file}")
                    print("Falling back to default calibration file...")
                    calibration_file = None

            # Use default if no file specified or if specified file doesn't exist
            if not calibration_file:
                # Look in calibration_data folder at project root (one level up from gguf_converter module)
                project_root = Path(__file__).parent.parent
                calibration_file = project_root / "calibration_data" / "_default.txt"

                if not calibration_file.exists():
                    print("Warning: _default.txt not found in calibration_data folder, using built-in calibration data")
                    calibration_file = None

            self.generate_imatrix(
                model_path=intermediate_file,
                output_path=imatrix_file,
                calibration_file=calibration_file,
                ctx_size=imatrix_ctx_size,
                nthreads=nthreads,
                verbose=verbose,
                chunks=imatrix_chunks,
                collect_output_weight=imatrix_collect_output,
                ngl=imatrix_ngl
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
                    print(f"{quant_type} is the intermediate format (already created above)")
                    quantized_files.append(intermediate_file)
                # Check if output already exists
                elif output_file.exists():
                    print(f"{quant_type} file already exists: {output_file.name}")
                    print("Skipping conversion, using existing file...")
                    quantized_files.append(output_file)
                # Generate from source
                else:
                    print(f"Converting {model_name} to {quant_type} from source...")
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
                    nthreads=nthreads,
                    allow_requantize=allow_requantize,
                    leave_output_tensor=leave_output_tensor,
                    pure_quantization=pure_quantization,
                    keep_split=keep_split
                )
                quantized_files.append(output_file)

        return quantized_files
