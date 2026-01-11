"""
Integration tests using real models

These tests download a small model and run the full conversion pipeline
to verify everything works end-to-end.

WARNING: These tests:
- Download ~500MB model from HuggingFace
- Take several minutes to complete
- Require significant disk space (2-3GB with all outputs)

Skip with: pytest -m "not integration"
Run only integration tests: pytest -m integration
"""

import pytest
import shutil
import subprocess
import hashlib
from pathlib import Path
from gguf_converter.converter import GGUFConverter
from gguf_converter.llama_cpp_manager import LlamaCppManager


# Small model for testing (only ~270MB)
TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TEST_MODEL_NAME = "SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def test_output_dir(tmp_path_factory, keep_outputs, custom_test_output_dir):
    """
    Create a temporary directory for test outputs
    Scope is module so it persists across all tests in this file

    Can be customized with command-line flags:
    - --keep-test-outputs: Don't clean up files after tests
    - --test-output-dir <path>: Save to specific directory
    """
    # Use custom directory if specified, otherwise use pytest temp
    if custom_test_output_dir:
        output_dir = Path(custom_test_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*80}")
        print(f"Saving test outputs to: {output_dir}")
        print(f"{'='*80}\n")
    else:
        output_dir = tmp_path_factory.mktemp("integration_tests")

    yield output_dir

    # Cleanup after all tests (unless --keep-test-outputs flag set)
    if not keep_outputs and output_dir.exists() and not custom_test_output_dir:
        print(f"\nCleaning up test outputs from {output_dir}")
        shutil.rmtree(output_dir)
    elif keep_outputs or custom_test_output_dir:
        print(f"\n{'='*80}")
        print(f"Test outputs preserved at: {output_dir}")
        print(f"{'='*80}\n")


@pytest.fixture(scope="module")
def converter():
    """
    Create a GGUFConverter instance for testing
    """
    return GGUFConverter()


@pytest.fixture(scope="module")
def downloaded_model(converter, test_output_dir):
    """
    Download the test model once and reuse across all tests
    """
    downloads_dir = test_output_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    print(f"\n[SETUP] Downloading {TEST_MODEL} (this may take a few minutes)...")
    model_path = converter.download_model(
        repo_id=TEST_MODEL,
        output_dir=downloads_dir
    )
    print(f"[SETUP] Model downloaded to {model_path}")

    return model_path


@pytest.mark.integration
@pytest.mark.slow
class TestBasicConversion:
    """
    Test basic conversion workflow
    """

    def test_download_model(self, downloaded_model):
        """
        Verify model was downloaded successfully
        """
        assert downloaded_model.exists()
        assert downloaded_model.is_dir()

        # Check for expected model files
        config_file = downloaded_model / "config.json"
        assert config_file.exists(), "Model should have config.json"

    def test_convert_to_f16(self, converter, downloaded_model, test_output_dir):
        """
        Test converting model to F16 GGUF format
        """
        output_dir = test_output_dir / "basic_conversion"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{TEST_MODEL_NAME}_F16.gguf"

        result = converter.convert_to_gguf(
            model_path=downloaded_model,
            output_path=output_file,
            output_type="f16",
            verbose=False
        )

        # Verify output
        assert result.exists()
        assert result.stat().st_size > 0

        # F16 should be roughly 0.26GB for 135M model
        size_gb = result.stat().st_size / (1024**3)
        assert 0.2 < size_gb < 0.5, f"F16 model size {size_gb:.2f}GB seems wrong"


@pytest.mark.integration
@pytest.mark.slow
class TestQuantization:
    """
    Test quantization with different types
    """

    def test_quantize_q4_k_m(self, converter, downloaded_model, test_output_dir):
        """
        Test quantizing to Q4_K_M (most common quantization)
        """
        output_dir = test_output_dir / "quantization"
        output_dir.mkdir(exist_ok=True)

        # First convert to F16
        f16_file = output_dir / f"{TEST_MODEL_NAME}_F16.gguf"
        if not f16_file.exists():
            converter.convert_to_gguf(
                model_path=downloaded_model,
                output_path=f16_file,
                output_type="f16",
                verbose=False
            )

        # Quantize to Q4_K_M
        q4_file = output_dir / f"{TEST_MODEL_NAME}_Q4_K_M.gguf"
        result = converter.quantize(
            input_path=f16_file,
            output_path=q4_file,
            quantization_type="Q4_K_M",
            verbose=False
        )

        # Verify output
        assert result.exists()
        assert result.stat().st_size > 0

        # Q4_K_M should be smaller than F16
        q4_size = result.stat().st_size
        f16_size = f16_file.stat().st_size
        compression_ratio = f16_size / q4_size

        assert compression_ratio > 1.5, f"Compression ratio {compression_ratio:.2f}x too low"
        assert compression_ratio < 5.0, f"Compression ratio {compression_ratio:.2f}x suspiciously high"

    def test_quantize_q8_0(self, converter, downloaded_model, test_output_dir):
        """
        Test quantizing to Q8_0 (high quality quantization)
        """
        output_dir = test_output_dir / "quantization"
        output_dir.mkdir(exist_ok=True)

        # First convert to F16
        f16_file = output_dir / f"{TEST_MODEL_NAME}_F16.gguf"
        if not f16_file.exists():
            converter.convert_to_gguf(
                model_path=downloaded_model,
                output_path=f16_file,
                output_type="f16",
                verbose=False
            )

        # Quantize to Q8_0
        q8_file = output_dir / f"{TEST_MODEL_NAME}_Q8_0.gguf"
        result = converter.quantize(
            input_path=f16_file,
            output_path=q8_file,
            quantization_type="Q8_0",
            verbose=False
        )

        # Verify output
        assert result.exists()
        assert result.stat().st_size > 0

        # Q8_0 should be larger than Q4_K_M but smaller than F16
        q8_size = result.stat().st_size
        f16_size = f16_file.stat().st_size

        assert q8_size < f16_size, "Q8_0 should be smaller than F16"


@pytest.mark.integration
@pytest.mark.slow
class TestOverwriteBehavior:
    """
    Test overwrite vs reuse behavior
    """

    def test_overwrite_intermediates_false(self, converter, downloaded_model, test_output_dir):
        """
        Test that existing intermediate files are reused when overwrite=False
        """
        output_dir = test_output_dir / "overwrite_test"
        output_dir.mkdir(exist_ok=True)

        # First run - create F16
        results1 = converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="f16",
            overwrite_intermediates=False,
            overwrite_quants=True,
            verbose=False
        )

        f16_file = output_dir / f"{TEST_MODEL_NAME}_F16.gguf"
        original_mtime = f16_file.stat().st_mtime

        # Second run - should reuse F16
        results2 = converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="f16",
            overwrite_intermediates=False,
            overwrite_quants=True,
            verbose=False
        )

        # F16 file should not have been regenerated
        new_mtime = f16_file.stat().st_mtime
        assert new_mtime == original_mtime, "F16 file should have been reused"

    def test_overwrite_intermediates_true(self, converter, downloaded_model, test_output_dir):
        """
        Test that existing intermediate files are regenerated when overwrite=True
        """
        output_dir = test_output_dir / "overwrite_test"
        output_dir.mkdir(exist_ok=True)

        # First run - create F16
        converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="f16",
            overwrite_intermediates=False,
            overwrite_quants=True,
            verbose=False
        )

        f16_file = output_dir / f"{TEST_MODEL_NAME}_F16.gguf"
        original_mtime = f16_file.stat().st_mtime

        import time
        time.sleep(1)  # Ensure timestamp difference

        # Second run - should regenerate F16
        converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="f16",
            overwrite_intermediates=True,
            overwrite_quants=True,
            verbose=False
        )

        # F16 file should have been regenerated
        new_mtime = f16_file.stat().st_mtime
        assert new_mtime > original_mtime, "F16 file should have been regenerated"

    def test_overwrite_quants_false(self, converter, downloaded_model, test_output_dir):
        """
        Test that existing quantized files are reused when overwrite_quants=False
        """
        output_dir = test_output_dir / "overwrite_quants_test"
        output_dir.mkdir(exist_ok=True)

        # First run - create quantized file
        converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="f16",
            overwrite_quants=True,
            verbose=False
        )

        q4_file = output_dir / f"{TEST_MODEL_NAME}_Q4_K_M.gguf"
        original_mtime = q4_file.stat().st_mtime

        # Second run - should reuse quantized file
        converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="f16",
            overwrite_quants=False,
            verbose=False
        )

        # Quantized file should not have been regenerated
        new_mtime = q4_file.stat().st_mtime
        assert new_mtime == original_mtime, "Quantized file should have been reused"


@pytest.mark.integration
@pytest.mark.slow
class TestMultipleQuantizations:
    """
    Test creating multiple quantizations in one run
    """

    def test_multiple_quant_types(self, converter, downloaded_model, test_output_dir):
        """
        Test creating multiple quantization types in a single run
        """
        output_dir = test_output_dir / "multiple_quants"
        output_dir.mkdir(exist_ok=True)

        quant_types = ["Q4_K_M", "Q5_K_M", "Q8_0"]

        results = converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=quant_types,
            intermediate_type="f16",
            verbose=False
        )

        # Verify all quantizations were created
        assert len(results) == len(quant_types)

        for result in results:
            assert result.exists()
            assert result.stat().st_size > 0

        # Verify files are in size order (Q8_0 > Q5_K_M > Q4_K_M)
        sizes = {
            result.name: result.stat().st_size
            for result in results
        }

        q4_size = sizes[f"{TEST_MODEL_NAME}_Q4_K_M.gguf"]
        q5_size = sizes[f"{TEST_MODEL_NAME}_Q5_K_M.gguf"]
        q8_size = sizes[f"{TEST_MODEL_NAME}_Q8_0.gguf"]

        assert q4_size < q5_size < q8_size, "Quantization sizes should be ordered"


@pytest.mark.integration
@pytest.mark.slow
class TestImatrixGeneration:
    """
    Test importance matrix generation and usage
    """

    def test_generate_imatrix(self, converter, downloaded_model, test_output_dir):
        """
        Test generating an importance matrix
        """
        output_dir = test_output_dir / "imatrix_test"
        output_dir.mkdir(exist_ok=True)

        # First need F16 model
        f16_file = output_dir / f"{TEST_MODEL_NAME}_F16.gguf"
        if not f16_file.exists():
            converter.convert_to_gguf(
                model_path=downloaded_model,
                output_path=f16_file,
                output_type="f16",
                verbose=False
            )

        # Find calibration file
        project_root = Path(__file__).parent.parent
        calibration_dir = project_root / "calibration_data"

        # Look for any calibration file
        calibration_files = list(calibration_dir.glob("*.txt")) if calibration_dir.exists() else []

        if not calibration_files:
            pytest.skip("No calibration file found in calibration_data/")

        calibration_file = calibration_files[0]

        # Generate imatrix with minimal settings for speed
        imatrix_file = output_dir / f"{TEST_MODEL_NAME}.imatrix"
        result = converter.generate_imatrix(
            model_path=f16_file,
            output_path=imatrix_file,
            calibration_file=calibration_file,
            ctx_size=512,
            chunks=10,  # Only process 10 chunks for speed
            verbose=False,
            no_ppl=True  # Skip perplexity calculation for speed
        )

        # Verify imatrix was created
        assert result.exists()
        assert result.stat().st_size > 0

    def test_quantize_with_imatrix(self, converter, downloaded_model, test_output_dir):
        """
        Test quantizing with an importance matrix

        This test is optional since generating imatrix takes time
        """
        output_dir = test_output_dir / "imatrix_quant_test"
        output_dir.mkdir(exist_ok=True)

        # Check if we have a calibration file
        project_root = Path(__file__).parent.parent
        calibration_dir = project_root / "calibration_data"
        calibration_files = list(calibration_dir.glob("*.txt")) if calibration_dir.exists() else []

        if not calibration_files:
            pytest.skip("No calibration file found - skipping imatrix quantization test")

        # Use convert_and_quantize with imatrix generation
        results = converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["IQ3_XXS"],  # Requires imatrix
            intermediate_type="f16",
            generate_imatrix=True,
            imatrix_chunks=10,  # Minimal for speed
            imatrix_calibration_file=calibration_files[0],
            verbose=False
        )

        # Verify quantization succeeded
        assert len(results) == 1
        assert results[0].exists()
        assert results[0].stat().st_size > 0


@pytest.fixture(scope="module")
def llama_manager():
    """
    Create a LlamaCppManager instance for testing
    """
    return LlamaCppManager()


@pytest.mark.integration
@pytest.mark.slow
class TestFunctionalCorrectness:
    """
    Test that the converted models are functionally correct
    """

    def test_q4_k_m_checksum(self, converter, llama_manager, downloaded_model, test_output_dir):
        """
        Verify Q4_K_M model output against a golden checksum
        """
        # --- 1. Locate the correct binary ---
        binary_name = "llama-completion"
        llama_bin_path = llama_manager.get_binary_path(binary_name)
        
        if not llama_bin_path or not llama_bin_path.exists():
             llama_bin_path = llama_manager.get_binary_path("main")

        if not llama_bin_path or not llama_bin_path.exists():
            pytest.skip(f"{binary_name} (or 'main') binary not found, skipping checksum test.")

        # --- 2. Prepare Model ---
        output_dir = test_output_dir / "correctness_Q4_K_M"
        output_dir.mkdir(exist_ok=True)

        quantized_file = output_dir / f"{TEST_MODEL_NAME}_Q4_K_M.gguf"

        if not quantized_file.exists():
            print(f"\n[CHECKSUM] Quantizing to Q4_K_M for correctness test...")
            results = converter.convert_and_quantize(
                model_path=downloaded_model,
                output_dir=output_dir,
                quantization_types=["Q4_K_M"],
                intermediate_type="f16",
                verbose=False,
            )
            assert results, f"Failed to create Q4_K_M model"
            assert quantized_file.exists()

        # --- 3. Generate Text ---
        prompt = "Explain the golden ratio"
        seed = 42
        n_predict = 30

        cmd = [
            str(llama_bin_path),
            "-m", str(quantized_file),
            "-p", prompt,
            "--seed", str(seed),
            "-n", str(n_predict),
            "--temp", "0.0",
            "-ngl", "0",      # Force CPU mode (prevents shader hangs)
            "-no-cnv"         # Force Raw Mode (disables 'user/assistant' chat wrappers)
        ]

        print(f"\n[CHECKSUM] Running {binary_name}: {' '.join(cmd)}")
        
        # KEY FIX: capture_output=True is required to populate result.stdout
        result = subprocess.run(
            cmd,
            capture_output=True, # <--- THIS MUST BE TRUE
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL # <--- Keep this to kill interactive loops
        )

        generated_text_with_prompt = result.stdout.strip()
        
        if not generated_text_with_prompt:
             # If this fails, print stderr to see why
             pytest.fail(f"llama-completion produced no output. Stderr:\n{result.stderr}")

        # Extract only the generated part
        if prompt in generated_text_with_prompt:
             generated_text = generated_text_with_prompt.split(prompt, 1)[-1].strip()
        else:
             generated_text = generated_text_with_prompt

        # Basic sanity checks before checksum validation
        assert len(generated_text) > 10, "Generated text is too short - model may be broken"
        assert len(generated_text) < 5000, "Generated text is suspiciously long - possible corruption"
        assert not generated_text.startswith("�"), "Output contains unicode errors - model may be corrupt"

        # --- 4. Validate Checksum ---
        project_root = Path(__file__).parent.parent
        golden_checksum_path = (
            project_root / "tests" / "golden_outputs" / "smollm2_135m_q4_k_m_checksum.txt"
        )

        current_checksum = hashlib.sha256(generated_text.encode("utf-8")).hexdigest()
        print(f"\n[CHECKSUM] Generated text ({len(generated_text)} chars):")
        print(f"[CHECKSUM]   '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
        print(f"[CHECKSUM] Current checksum: {current_checksum}")

        if golden_checksum_path.exists():
            golden_checksum = golden_checksum_path.read_text().strip()
            print(f"[CHECKSUM] Golden checksum:  {golden_checksum}")

            if current_checksum != golden_checksum:
                print(f"\n[CHECKSUM] MISMATCH DETECTED!")
                print(f"[CHECKSUM] This could mean:")
                print(f"[CHECKSUM]   - llama.cpp version changed (update golden checksum)")
                print(f"[CHECKSUM]   - Model quantization is broken (investigate!)")
                print(f"[CHECKSUM]   - Platform differences (rare with seed + temp=0)")

            assert current_checksum == golden_checksum, \
                f"Checksum mismatch!\n  Expected: {golden_checksum}\n  Got:      {current_checksum}\n  Full text: {generated_text}"
        else:
            print(f"[CHECKSUM] Golden checksum file not found. Creating {golden_checksum_path}")
            golden_checksum_path.parent.mkdir(parents=True, exist_ok=True)
            golden_checksum_path.write_text(current_checksum)
            pytest.skip(
                f"Created new golden checksum file. Please commit: {golden_checksum_path}. Rerun to verify."
            )

@pytest.mark.integration
@pytest.mark.slow
class TestEdgeCases:
    """
    Test edge cases and error handling
    """

    def test_invalid_quantization_type(self, converter, downloaded_model, test_output_dir):
        """
        Test that invalid quantization types raise appropriate errors
        """
        output_dir = test_output_dir / "invalid_quant"
        output_dir.mkdir(exist_ok=True)

        with pytest.raises(ValueError, match="Invalid quantization type"):
            converter.convert_and_quantize(
                model_path=downloaded_model,
                output_dir=output_dir,
                quantization_types=["INVALID_QUANT_TYPE"],
                intermediate_type="f16",
                verbose=False
            )

    def test_bf16_intermediate(self, converter, downloaded_model, test_output_dir):
        """
        Test using BF16 as intermediate format
        """
        output_dir = test_output_dir / "bf16_test"
        output_dir.mkdir(exist_ok=True)

        results = converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="bf16",  # Use BF16 instead of F16
            verbose=False
        )

        # Verify conversion succeeded
        assert len(results) == 1
        assert results[0].exists()

        # BF16 intermediate file should also exist
        bf16_file = output_dir / f"{TEST_MODEL_NAME}_BF16.gguf"
        assert bf16_file.exists()

    def test_f32_intermediate(self, converter, downloaded_model, test_output_dir):
        """
        Test using F32 as intermediate format
        """
        output_dir = test_output_dir / "f32_test"
        output_dir.mkdir(exist_ok=True)

        results = converter.convert_and_quantize(
            model_path=downloaded_model,
            output_dir=output_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="f32",  # Use F32 instead of F16
            verbose=False
        )

        # Verify conversion succeeded
        assert len(results) == 1
        assert results[0].exists()

        # F32 intermediate should be largest
        f32_file = output_dir / f"{TEST_MODEL_NAME}_F32.gguf"
        assert f32_file.exists()

        q4_size = results[0].stat().st_size
        f32_size = f32_file.stat().st_size
        assert f32_size > q4_size, "F32 should be larger than Q4_K_M"