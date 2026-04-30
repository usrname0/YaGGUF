"""
Tests for the Run GGUF tab — command builders, binary smoke tests, and integration.

Unit tests cover all argument combinations for the pure command builder functions.
Smoke tests verify the new binaries (llama-server, llama-bench, llama-fit-params)
are invocable. Integration tests run llama-bench in both auto and manual hardware
modes against a real model to catch llama.cpp API changes end-to-end.
"""

import platform
import subprocess
import shutil
import pytest
from pathlib import Path

from gguf_converter.gui_tabs.run_gguf import (
    build_server_cmd,
    build_cli_cmd,
    build_bench_cmd,
    _parse_fit_output,
    _find_binary,
)
from gguf_converter.llama_cpp_manager import LlamaCppManager


# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------

def _server_params(**overrides):
    return {
        "auto_fit": True, "fit_target": 1024, "fit_ctx": 0,
        "ngl": 99, "ctx": 4096, "flash_attn": False,
        "cache_type_k": "f16", "cache_type_v": "f16",
        "port": 8080, "host": "127.0.0.1", "parallel": 1,
        "temp": 0.8, "top_k": 40, "top_p": 0.95,
        "min_p": 0.05, "presence_penalty": 0.0,
        "jinja": False, "reasoning": "off",
        **overrides,
    }


def _bench_params(**overrides):
    return {
        "auto_fit": True, "ngl": 99,
        "bench_n_prompt": 512, "bench_n_gen": 128,
        "bench_batch_size": 2048, "bench_ubatch_size": 512,
        "bench_repetitions": 5, "bench_threads": 0,
        "bench_cache_type_k": "f16", "bench_cache_type_v": "f16",
        "bench_output": "md", "bench_no_warmup": False,
        "bench_progress": False, "flash_attn": False,
        **overrides,
    }


# ---------------------------------------------------------------------------
# build_server_cmd
# ---------------------------------------------------------------------------

class TestBuildServerCmd:

    def test_binary_and_model_at_start(self):
        cmd = build_server_cmd("llama-server", "/models/test.gguf", _server_params())
        assert cmd[0] == "llama-server"
        assert cmd[1:3] == ["-m", "/models/test.gguf"]

    def test_auto_fit_with_targets(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            auto_fit=True, fit_target=2048, fit_ctx=8192
        ))
        assert "--fit-target" in cmd
        assert cmd[cmd.index("--fit-target") + 1] == "2048"
        assert "--fit-ctx" in cmd
        assert cmd[cmd.index("--fit-ctx") + 1] == "8192"
        assert "-ngl" not in cmd
        assert "-c" not in cmd

    def test_auto_fit_zero_targets_omitted(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            auto_fit=True, fit_target=0, fit_ctx=0
        ))
        assert "--fit-target" not in cmd
        assert "--fit-ctx" not in cmd

    def test_manual_mode_ngl_and_ctx(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            auto_fit=False, ngl=32, ctx=8192
        ))
        assert "-ngl" in cmd
        assert cmd[cmd.index("-ngl") + 1] == "32"
        assert "-c" in cmd
        assert cmd[cmd.index("-c") + 1] == "8192"
        assert "--fit-target" not in cmd

    def test_manual_flash_attn_on(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            auto_fit=False, flash_attn=True
        ))
        assert "-fa" in cmd
        assert cmd[cmd.index("-fa") + 1] == "on"

    def test_manual_flash_attn_off_omitted(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            auto_fit=False, flash_attn=False
        ))
        assert "-fa" not in cmd

    def test_cache_type_k_non_default(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(cache_type_k="q8_0"))
        assert "-ctk" in cmd
        assert cmd[cmd.index("-ctk") + 1] == "q8_0"

    def test_cache_type_v_non_default(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(cache_type_v="q4_0"))
        assert "-ctv" in cmd
        assert cmd[cmd.index("-ctv") + 1] == "q4_0"

    def test_cache_type_f16_omitted(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            cache_type_k="f16", cache_type_v="f16"
        ))
        assert "-ctk" not in cmd
        assert "-ctv" not in cmd

    def test_port_host_parallel(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            port=11434, host="0.0.0.0", parallel=4
        ))
        assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "11434"
        assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "0.0.0.0"
        assert "-np" in cmd and cmd[cmd.index("-np") + 1] == "4"

    def test_sampling_params(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            temp=0.7, top_k=20, top_p=0.9, min_p=0.1, presence_penalty=0.5
        ))
        assert cmd[cmd.index("--temp") + 1] == "0.7"
        assert cmd[cmd.index("--top-k") + 1] == "20"
        assert cmd[cmd.index("--top-p") + 1] == "0.9"
        assert cmd[cmd.index("--min-p") + 1] == "0.1"
        assert cmd[cmd.index("--presence-penalty") + 1] == "0.5"

    def test_jinja_disabled_suppresses_reasoning(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            jinja=False, reasoning="on"
        ))
        assert "--jinja" not in cmd
        assert "--reasoning" not in cmd

    def test_jinja_enabled_reasoning_auto_omitted(self):
        # "auto" is llama.cpp's default — we must NOT emit --reasoning
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            jinja=True, reasoning="auto"
        ))
        assert "--jinja" in cmd
        assert "--reasoning" not in cmd

    def test_jinja_enabled_reasoning_on(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            jinja=True, reasoning="on"
        ))
        assert "--jinja" in cmd
        assert "--reasoning" in cmd
        assert cmd[cmd.index("--reasoning") + 1] == "on"

    def test_jinja_enabled_reasoning_off(self):
        cmd = build_server_cmd("llama-server", "m.gguf", _server_params(
            jinja=True, reasoning="off"
        ))
        assert "--jinja" in cmd
        assert cmd[cmd.index("--reasoning") + 1] == "off"


# ---------------------------------------------------------------------------
# build_cli_cmd
# ---------------------------------------------------------------------------

class TestBuildCliCmd:

    def test_binary_and_model_at_start(self):
        cmd = build_cli_cmd("llama-cli", "/models/test.gguf", _server_params())
        assert cmd[0] == "llama-cli"
        assert cmd[1:3] == ["-m", "/models/test.gguf"]

    def test_cnv_always_present(self):
        cmd = build_cli_cmd("llama-cli", "m.gguf", _server_params())
        assert "-cnv" in cmd

    def test_no_server_args(self):
        cmd = build_cli_cmd("llama-cli", "m.gguf", _server_params())
        assert "--port" not in cmd
        assert "--host" not in cmd
        assert "-np" not in cmd

    def test_auto_fit(self):
        cmd = build_cli_cmd("llama-cli", "m.gguf", _server_params(
            auto_fit=True, fit_target=512, fit_ctx=4096
        ))
        assert "--fit-target" in cmd
        assert "--fit-ctx" in cmd
        assert "-ngl" not in cmd
        assert "-c" not in cmd

    def test_manual_mode(self):
        cmd = build_cli_cmd("llama-cli", "m.gguf", _server_params(
            auto_fit=False, ngl=16, ctx=2048, flash_attn=True
        ))
        assert cmd[cmd.index("-ngl") + 1] == "16"
        assert cmd[cmd.index("-c") + 1] == "2048"
        assert "-fa" in cmd

    def test_jinja_reasoning_auto_omitted(self):
        cmd = build_cli_cmd("llama-cli", "m.gguf", _server_params(
            jinja=True, reasoning="auto"
        ))
        assert "--jinja" in cmd
        assert "--reasoning" not in cmd

    def test_jinja_reasoning_on(self):
        cmd = build_cli_cmd("llama-cli", "m.gguf", _server_params(
            jinja=True, reasoning="on"
        ))
        assert "--jinja" in cmd
        assert cmd[cmd.index("--reasoning") + 1] == "on"


# ---------------------------------------------------------------------------
# build_bench_cmd
# ---------------------------------------------------------------------------

class TestBuildBenchCmd:

    def test_binary_and_model_at_start(self):
        cmd = build_bench_cmd("llama-bench", "/models/test.gguf", _bench_params())
        assert cmd[0] == "llama-bench"
        assert cmd[1:3] == ["-m", "/models/test.gguf"]

    def test_auto_fit_no_ngl(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(auto_fit=True))
        assert "-ngl" not in cmd

    def test_manual_includes_ngl(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(auto_fit=False, ngl=0))
        assert "-ngl" in cmd
        assert cmd[cmd.index("-ngl") + 1] == "0"

    def test_manual_ngl_all_layers(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(auto_fit=False, ngl=999))
        assert cmd[cmd.index("-ngl") + 1] == "999"

    def test_bench_core_params(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(
            bench_n_prompt=256, bench_n_gen=64,
            bench_batch_size=512, bench_ubatch_size=128,
            bench_repetitions=3,
        ))
        assert cmd[cmd.index("-p") + 1] == "256"
        assert cmd[cmd.index("-n") + 1] == "64"
        assert cmd[cmd.index("-b") + 1] == "512"
        assert cmd[cmd.index("-ub") + 1] == "128"
        assert cmd[cmd.index("-r") + 1] == "3"

    def test_threads_zero_omitted(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(bench_threads=0))
        assert "-t" not in cmd

    def test_threads_nonzero_included(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(bench_threads=8))
        assert "-t" in cmd
        assert cmd[cmd.index("-t") + 1] == "8"

    def test_cache_types_always_emitted(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(
            bench_cache_type_k="q8_0", bench_cache_type_v="q4_0"
        ))
        assert cmd[cmd.index("-ctk") + 1] == "q8_0"
        assert cmd[cmd.index("-ctv") + 1] == "q4_0"

    def test_flash_attn_on(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(flash_attn=True))
        assert "-fa" in cmd
        assert cmd[cmd.index("-fa") + 1] == "1"

    def test_flash_attn_off(self):
        cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(flash_attn=False))
        assert "-fa" in cmd
        assert cmd[cmd.index("-fa") + 1] == "0"

    def test_output_format(self):
        for fmt in ["csv", "json", "jsonl", "sql"]:
            cmd = build_bench_cmd("llama-bench", "m.gguf", _bench_params(bench_output=fmt))
            assert cmd[cmd.index("-o") + 1] == fmt

    def test_no_warmup_flag(self):
        cmd_nw = build_bench_cmd("llama-bench", "m.gguf", _bench_params(bench_no_warmup=True))
        cmd_w  = build_bench_cmd("llama-bench", "m.gguf", _bench_params(bench_no_warmup=False))
        assert "--no-warmup" in cmd_nw
        assert "--no-warmup" not in cmd_w

    def test_progress_flag(self):
        cmd_p  = build_bench_cmd("llama-bench", "m.gguf", _bench_params(bench_progress=True))
        cmd_np = build_bench_cmd("llama-bench", "m.gguf", _bench_params(bench_progress=False))
        assert "--progress" in cmd_p
        assert "--progress" not in cmd_np


# ---------------------------------------------------------------------------
# _parse_fit_output
# ---------------------------------------------------------------------------

class TestParseFitOutput:

    def test_typical_output(self):
        ngl, ctx = _parse_fit_output("-c 40192 -ngl -1")
        assert ngl == -1
        assert ctx == 40192

    def test_different_order(self):
        ngl, ctx = _parse_fit_output("-ngl 32 -c 8192")
        assert ngl == 32
        assert ctx == 8192

    def test_empty_string(self):
        ngl, ctx = _parse_fit_output("")
        assert ngl is None
        assert ctx is None

    def test_only_ngl(self):
        ngl, ctx = _parse_fit_output("-ngl 0")
        assert ngl == 0
        assert ctx is None

    def test_only_ctx(self):
        ngl, ctx = _parse_fit_output("-c 4096")
        assert ngl is None
        assert ctx == 4096

    def test_all_layers_negative_one(self):
        ngl, ctx = _parse_fit_output("-c 65536 -ngl -1")
        assert ngl == -1
        assert ctx == 65536


# ---------------------------------------------------------------------------
# _find_binary
# ---------------------------------------------------------------------------

class TestFindBinary:

    def test_none_dir_returns_name(self):
        assert _find_binary(None, "llama-server") == "llama-server"

    def test_existing_binary_returns_full_path(self, tmp_path):
        suffix = ".exe" if platform.system() == "Windows" else ""
        binary = tmp_path / f"llama-server{suffix}"
        binary.touch()
        result = _find_binary(tmp_path, "llama-server")
        assert result == str(binary)

    def test_missing_binary_returns_name(self, tmp_path):
        result = _find_binary(tmp_path, "llama-server")
        assert result == "llama-server"


# ---------------------------------------------------------------------------
# Smoke tests for Run GGUF binaries
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llama_manager():
    return LlamaCppManager()


@pytest.mark.requires_binaries
class TestRunGgufBinarySmoke:

    def test_llama_server_version(self, llama_manager):
        path = llama_manager.get_binary_path("llama-server")
        if not path or not path.exists():
            pytest.skip("llama-server not found")
        result = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "version" in output.lower() or "llama" in output.lower()

    def test_llama_bench_runs(self, llama_manager):
        path = llama_manager.get_binary_path("llama-bench")
        if not path or not path.exists():
            pytest.skip("llama-bench not found")
        result = subprocess.run(
            [str(path), "--help"], capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        assert len(output) > 0, "llama-bench produced no output"
        assert "bench" in output.lower() or "usage" in output.lower()

    def test_llama_fit_params_help(self, llama_manager):
        path = llama_manager.get_binary_path("llama-fit-params")
        if not path or not path.exists():
            pytest.skip("llama-fit-params not present in this llama.cpp build")
        result = subprocess.run(
            [str(path), "--help"], capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        assert len(output) > 0, "llama-fit-params produced no output"


# ---------------------------------------------------------------------------
# Integration: llama-bench via build_bench_cmd against real model
# ---------------------------------------------------------------------------

_TEST_MODEL     = "HuggingFaceTB/SmolLM2-135M-Instruct"
_TEST_MODEL_NAME = "SmolLM2-135M-Instruct"

# Minimal bench params that run fast even on CPU
_FAST_BENCH = dict(
    bench_n_prompt=32,
    bench_n_gen=16,
    bench_batch_size=32,
    bench_ubatch_size=32,
    bench_repetitions=1,
    bench_no_warmup=True,
)


@pytest.fixture(scope="module")
def run_gguf_test_dir(tmp_path_factory, keep_outputs, custom_test_output_dir):
    if custom_test_output_dir:
        d = Path(custom_test_output_dir)
        d.mkdir(parents=True, exist_ok=True)
    else:
        d = tmp_path_factory.mktemp("run_gguf_tests")
    yield d
    if not keep_outputs and not custom_test_output_dir and d.exists():
        shutil.rmtree(d)


@pytest.fixture(scope="module")
def q4_model_path(run_gguf_test_dir):
    """Download SmolLM2-135M and quantize to Q4_K_M, reusing file if it already exists."""
    from gguf_converter.converter import GGUFConverter

    quantized = run_gguf_test_dir / f"{_TEST_MODEL_NAME}_Q4_K_M.gguf"
    if not quantized.exists():
        converter = GGUFConverter()
        dl_dir = run_gguf_test_dir / "downloads"
        dl_dir.mkdir(exist_ok=True)
        model_path = converter.download_model(repo_id=_TEST_MODEL, output_dir=dl_dir)
        converter.convert_and_quantize(
            model_path=model_path,
            output_dir=run_gguf_test_dir,
            quantization_types=["Q4_K_M"],
            intermediate_type="f16",
            verbose=False,
        )
    assert quantized.exists(), "Failed to create Q4_K_M model for bench integration test"
    return quantized


@pytest.mark.integration
@pytest.mark.slow
class TestRunGgufBenchIntegration:
    """
    Run llama-bench via build_bench_cmd in both hardware modes against a real model.
    Validates that the auto/manual distinction actually reaches the binary correctly.
    """

    def test_manual_mode_bench_runs(self, llama_manager, q4_model_path):
        """Manual mode: -ngl 0 forces CPU, command and binary both work."""
        binary = llama_manager.get_binary_path("llama-bench")
        if not binary or not binary.exists():
            pytest.skip("llama-bench not found")

        params = _bench_params(auto_fit=False, ngl=0, **_FAST_BENCH)
        cmd = build_bench_cmd(str(binary), str(q4_model_path), params)

        assert "-ngl" in cmd, "Manual mode must emit -ngl"
        assert cmd[cmd.index("-ngl") + 1] == "0"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"llama-bench manual mode failed:\n{output}"
        assert len(output) > 0

    def test_auto_fit_mode_bench_runs(self, llama_manager, q4_model_path):
        """Auto mode: no -ngl in command, llama.cpp auto-fits hardware."""
        binary = llama_manager.get_binary_path("llama-bench")
        if not binary or not binary.exists():
            pytest.skip("llama-bench not found")

        params = _bench_params(auto_fit=True, **_FAST_BENCH)
        cmd = build_bench_cmd(str(binary), str(q4_model_path), params)

        assert "-ngl" not in cmd, "Auto mode must NOT emit -ngl"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"llama-bench auto mode failed:\n{output}"
        assert len(output) > 0
