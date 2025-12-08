"""
Core GGUF conversion and quantization functionality
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union, List
from huggingface_hub import snapshot_download
from .binary_manager import BinaryManager


class GGUFConverter:
    """
    GGUF converter that wraps llama.cpp for conversion and quantization
    No C++ compilation required - binaries are auto-downloaded
    """

    # Supported quantization types (from llama.cpp)
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

    def __init__(self):
        """Initialize the converter and binary manager"""
        self.binary_manager = BinaryManager()
        # Ensure binaries are available on init
        if not self.binary_manager.ensure_binaries():
            raise RuntimeError(
                "Failed to get llama.cpp binaries. "
                "Please check your internet connection or install llama.cpp manually."
            )

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

        print(f"Converting {model_path.name} to GGUF format...")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=not verbose, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr if result.stderr else 'Unknown error'}")

        if verbose and result.stdout:
            print(result.stdout)

        print(f"Conversion complete: {output_path}")
        return output_path

    def _find_convert_script(self) -> Optional[Path]:
        """Find the convert_hf_to_gguf.py script"""
        # Try to find it in common locations
        possible_locations = [
            # In llama-cpp-python package
            Path(__file__).parent.parent / "llama.cpp" / "convert_hf_to_gguf.py",
            # System-wide llama.cpp installation
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            # Current directory
            Path.cwd() / "llama.cpp" / "convert_hf_to_gguf.py",
        ]

        for location in possible_locations:
            if location.exists():
                return location

        # Try to clone llama.cpp if not found
        llama_cpp_dir = Path(__file__).parent.parent / "llama.cpp"
        if not llama_cpp_dir.exists():
            print("Cloning llama.cpp repository...")
            subprocess.run([
                "git", "clone",
                "https://github.com/ggerganov/llama.cpp.git",
                "--depth=1",
                str(llama_cpp_dir)
            ], check=True)

        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        return convert_script if convert_script.exists() else None


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
        scalar_optimization: bool = False
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

        print(f"Quantizing {input_path.name} to {quantization_type}...")

        import time
        start_time = time.time()

        # Build llama-quantize command
        quantize_bin = self.binary_manager.get_quantize_path()
        cmd = [str(quantize_bin), str(input_path), str(output_path), quantization_type]

        # Add nthreads (positional argument, comes before optional flags)
        if nthreads:
            cmd.append(str(nthreads))

        # Add optional flags (must come after positional arguments)
        if imatrix_path:
            imatrix_path = Path(imatrix_path)
            if not imatrix_path.exists():
                raise FileNotFoundError(f"Imatrix file not found: {imatrix_path}")
            cmd.extend(["--imatrix", str(imatrix_path)])

        print(f"Running: {' '.join(cmd)}")
        print()

        # Run llama-quantize
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
            raise RuntimeError(f"Quantization failed: {error_msg}")

        elapsed = time.time() - start_time

        # Print summary
        if output_path.exists():
            input_size = input_path.stat().st_size / (1024**3)
            output_size = output_path.stat().st_size / (1024**3)
            ratio = input_size / output_size if output_size > 0 else 0

            print(f"\nQuantization complete: {output_path}")
            print(f"Time taken: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
            print(f"Size: {input_size:.2f} GB -> {output_size:.2f} GB ({ratio:.2f}x compression)")
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
            collect_output_weight: Collect importance matrix for output.weight tensor
            ngl: Number of GPU layers to offload (None = CPU only)
            verbosity: Verbosity level (0=quiet, 1=normal, 2+=debug, overrides verbose if set)
            from_chunk: Skip first N chunks (useful for resuming)
            no_ppl: Disable perplexity calculation (speeds up processing)
            parse_special: Parse special tokens
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

        print(f"Generating importance matrix for {model_path.name}...")
        print(f"This may take a while...")

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

        print(f"Running: {' '.join(cmd)}")
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
            print(f"\nImatrix generation complete: {output_path}")
            print(f"Time taken: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
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
        imatrix_output_name: Optional[str] = None
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
            imatrix_collect_output: Collect output.weight tensor in imatrix
            imatrix_calibration_file: Path to calibration file for imatrix generation (None = use default)

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

        # Check if input is already a GGUF file
        is_already_gguf = model_path.is_file() and model_path.suffix == '.gguf'

        if is_already_gguf:
            print(f"Input is already a GGUF file: {model_path.name}")
            print("Skipping conversion, going straight to quantization...")
            intermediate_file = model_path
            model_name = model_path.stem  # Remove .gguf extension
        else:
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
                collect_output_weight=imatrix_collect_output
            )

            # Use the generated imatrix
            imatrix_path = imatrix_file

        # Step 2: Quantize to requested types
        quantized_files = []
        for quant_type in quantization_types:
            output_file = output_dir / f"{model_name}_{quant_type}.gguf"

            # Handle F16/F32/BF16 specially - these are intermediate formats
            if quant_type.upper() in ["F16", "F32", "BF16"]:
                # If the requested format matches the intermediate format, just use that file
                if quant_type.upper() == intermediate_type.upper():
                    # Check if output file would be same as intermediate (case-insensitive check)
                    if output_file.resolve() == intermediate_file.resolve():
                        print(f"{quant_type} output already exists as intermediate file...")
                        quantized_files.append(intermediate_file)
                    else:
                        import shutil
                        print(f"Creating {quant_type} output (copying intermediate file)...")
                        shutil.copy2(intermediate_file, output_file)
                        quantized_files.append(output_file)
                else:
                    # Different format requested - need to convert (e.g., F16 -> F32)
                    # Try using quantize command which might support format conversion
                    try:
                        print(f"Converting from {intermediate_type} to {quant_type}...")
                        self.quantize(
                            input_path=intermediate_file,
                            output_path=output_file,
                            quantization_type=quant_type,
                            verbose=verbose,
                            imatrix_path=None,  # Don't use imatrix for format conversion
                            nthreads=nthreads
                        )
                        quantized_files.append(output_file)
                    except Exception as e:
                        print(f"Warning: Could not convert {intermediate_type} to {quant_type}: {e}")
                        print(f"Suggestion: Use '{quant_type.lower()}' as the intermediate format instead")
                        raise RuntimeError(
                            f"Cannot convert from {intermediate_type} to {quant_type}. "
                            f"Please set the intermediate format to '{quant_type.lower()}' instead."
                        )
            else:
                self.quantize(
                    input_path=intermediate_file,
                    output_path=output_file,
                    quantization_type=quant_type,
                    verbose=verbose,
                    imatrix_path=imatrix_path,
                    nthreads=nthreads
                )
                quantized_files.append(output_file)

        return quantized_files
