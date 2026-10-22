# SPDX-License-Identifier: Apache-2.0
"""CPU-only baseline regressions: real WAV decoding and loopback HTTP failure/cancellation.

Synthetic test signals are not music, quality evidence, or a model-generation substitute.
"""

import copy
import importlib.util
import io
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose" / "yue2-3b"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "establish_baseline.py").is_file(),
    reason="independent YuE2 deployment checkout is not present",
)


@pytest.fixture
def baseline():
    spec = importlib.util.spec_from_file_location(
        "yue2_full_song_baseline", DEPLOY / "establish_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def healthy_resources():
    return {
        "available_bytes": 8 * 1024**3,
        "service_tree_rss_bytes": 3 * 1024**3,
        "disk_free_bytes": 4 * 1024**3,
        "processes": [],
    }


@pytest.fixture
def wav_bytes():
    # Six seconds, stereo, finite/non-silent, including exactly 0.1s silence at both ends.
    t = np.arange(6 * 48000, dtype=np.float64) / 48000
    signal = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    signal[:4800] = 0
    signal[-4800:] = 0
    samples = np.column_stack((signal, signal))
    buffer = io.BytesIO()
    sf.write(buffer, samples, 48000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@contextmanager
def local_server(body, *, truncated="false", behavior="audio"):
    """A real loopback socket; never invokes HTTP/GPU/model services outside this fixture."""
    requests = []
    disconnected = threading.Event()
    received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append({"path": self.path, "body": json.loads(raw)})
            received.set()
            if behavior == "wait_for_disconnect":
                self.connection.settimeout(4)
                try:
                    if self.connection.recv(1) == b"":
                        disconnected.set()
                except (ConnectionResetError, BrokenPipeError):
                    disconnected.set()
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            if truncated is not None:
                self.send_header("X-Audio-Truncated", truncated)
            self.send_header("Connection", "close")
            self.end_headers()
            if behavior == "incomplete":
                self.wfile.write(body[:100])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
            else:
                self.wfile.write(body)
            self.close_connection = True

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="baseline-local-test-server")
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, disconnected, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


SAMPLE = {
    "title": "CPU fixture, not model input evidence",
    "lyrics": "[Verse]\nfixture",
    "style": "CPU fixture",
    "seed": 12300,
    "cot": "full",
}


def test_natural_song_requires_explicit_nontruncation_and_real_decodable_wav(
    baseline, wav_bytes, tmp_path
):
    directory = tmp_path / "off"
    with local_server(wav_bytes) as (url, requests, _, _):
        result = baseline.generate_song(url, SAMPLE, "off", directory, healthy_resources)
    assert result["complete"] and result["naturally_ended"] and result["truncated"] is False
    assert result["audio"]["duration_seconds"] == 6
    assert result["audio"]["sample_rate"] == 48000 and result["audio"]["channels"] == 2
    assert result["audio"]["rms"] > 0.1 and result["audio"]["near_full_scale_fraction"] == 0
    assert result["listening_review"] == "not_performed"
    assert (directory / "audio.wav").read_bytes() == wav_bytes
    assert result["transport"]["resources"]["monitor_joined"]


@pytest.mark.parametrize(
    "flag,reason",
    [
        ("true", "native_frame_budget_truncated"),
        (None, "missing_or_invalid_X-Audio-Truncated"),
        ("0", "missing_or_invalid_X-Audio-Truncated"),
    ],
)
def test_truncated_or_unverified_song_keeps_original_wav_and_fails(
    baseline, wav_bytes, tmp_path, flag, reason
):
    directory = tmp_path / "off"
    with local_server(wav_bytes, truncated=flag) as (url, _, _, _):
        with pytest.raises(baseline.BaselineFailure, match="完整自然结束"):
            baseline.generate_song(url, SAMPLE, "off", directory, healthy_resources)
    result = json.loads((directory / "result.json").read_text())
    assert result["status"] == "failed" and not result["complete"]
    assert reason in result["failure_reasons"]
    assert (directory / "audio.wav").read_bytes() == wav_bytes
    assert not (tmp_path / "completed.json").exists()


def test_network_body_failure_propagates_and_cannot_write_completed(baseline, wav_bytes, tmp_path):
    with local_server(wav_bytes, behavior="incomplete") as (url, _, _, _):
        with pytest.raises((baseline.BaselineFailure, OSError)):
            baseline.run_baseline("off", url, DEPLOY, SAMPLE, tmp_path, healthy_resources)
    failed = json.loads((tmp_path / "failed.json").read_text())
    transport = json.loads((tmp_path / "off/transport.json").read_text())
    assert failed["status"] == "failed" and failed["completed_stages"] == []
    assert transport["status"] == "failed" and transport["cancellation_requested"]
    assert transport["resources"]["monitor_joined"]
    assert not (tmp_path / "completed.json").exists()


def test_refused_network_connection_is_not_swallowed(baseline, tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with pytest.raises(OSError):
        baseline.http_request(
            f"http://127.0.0.1:{port}",
            "/v1/completions",
            {},
            tmp_path / "planning",
            healthy_resources,
            timeout=2,
        )
    result = json.loads((tmp_path / "planning/transport.json").read_text())
    assert result["status"] == "failed" and result["resources"]["monitor_joined"]


def test_runtime_resource_failure_interrupts_actual_blocking_socket_and_joins_monitor(
    baseline, tmp_path
):
    samples = 0

    def declining_resources():
        nonlocal samples
        samples += 1
        result = healthy_resources()
        if samples > 1:
            result["available_bytes"] = baseline.MIN_AVAILABLE - 1
        return result

    with local_server(b"", behavior="wait_for_disconnect") as (url, _, disconnected, received):
        with pytest.raises(baseline.ResourceFailure, match="低于4GiB"):
            baseline.http_request(
                url, "/v1/audio/speech", {}, tmp_path / "off", declining_resources, timeout=4
            )
        assert received.is_set() and disconnected.wait(2)
    result = json.loads((tmp_path / "off/transport.json").read_text())
    assert result["cancellation_requested"] and result["server_cancellation_acknowledged"] is False
    assert result["resources"]["monitor_joined"]
    assert result["resources"]["minimum_available_bytes"] == baseline.MIN_AVAILABLE - 1
    assert "ResourceFailure" == result["error_type"]


def test_deadline_interrupts_waiting_http_without_leaving_request_thread(baseline, tmp_path):
    with local_server(b"", behavior="wait_for_disconnect") as (url, _, disconnected, _):
        with pytest.raises((baseline.ResourceFailure, TimeoutError)):
            baseline.http_request(
                url, "/v1/audio/speech", {}, tmp_path / "off", healthy_resources, timeout=0.8
            )
        assert disconnected.wait(2)
    result = json.loads((tmp_path / "off/transport.json").read_text())
    assert result["status"] == "failed" and result["resources"]["monitor_joined"]


def test_planning_validation_failure_prevents_full_request_without_mocking_pipeline(
    baseline, tmp_path
):
    # Real native request builder rejects this before any helper/tokenizer/model import.
    invalid = {**SAMPLE, "lyrics": ""}
    with local_server(b"") as (url, requests, _, _):
        with pytest.raises(ValueError, match="lyrics"):
            baseline.run_baseline("full", url, DEPLOY, invalid, tmp_path, healthy_resources)
    assert requests == []
    assert json.loads((tmp_path / "planning/result.json").read_text())["status"] == "failed"
    assert json.loads((tmp_path / "failed.json").read_text())["completed_stages"] == []
    assert not (tmp_path / "full").exists() and not (tmp_path / "completed.json").exists()


@pytest.fixture
def original_mode():
    environment = {
        "YUE2_METAL_QUANTIZATION": "off",
        "VLLM_OMNI_METAL_ANE": "off",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    plist = {"EnvironmentVariables": environment.copy()}
    profile = {
        "pipeline": "yue2",
        "dtype": "bfloat16",
        "stages": [
            {"max_model_len": 24576, "max_num_seqs": 1, "gpu_memory_utilization": 0.30},
        ],
    }
    return plist, profile, environment


def test_original_mode_requires_both_configured_and_actual_off_modes(baseline, original_mode):
    baseline.validate_original_mode(*original_mode)
    for key, bad in (("YUE2_METAL_QUANTIZATION", "dynamic"), ("VLLM_OMNI_METAL_ANE", "auto")):
        for actual in (False, True):
            plist, profile, environment = copy.deepcopy(original_mode)
            target = environment if actual else plist["EnvironmentVariables"]
            target[key] = bad
            with pytest.raises(baseline.BaselineFailure, match=key):
                baseline.validate_original_mode(plist, profile, environment)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_model_len", 6144),
        ("gpu_memory_utilization", 0.4),
        ("kv_cache_memory_bytes", 3 * 1024**3),
        ("quantization", "w8"),
        ("dtype", "float16"),
        ("quantization_config", {}),
    ],
)
def test_original_mode_rejects_changed_context_budget_or_precision(
    baseline, original_mode, key, value
):
    plist, profile, environment = copy.deepcopy(original_mode)
    profile["stages"][0][key] = value
    with pytest.raises(baseline.BaselineFailure):
        baseline.validate_original_mode(plist, profile, environment)


@pytest.mark.parametrize("subtype", ["PCM_16", "FLOAT", "DOUBLE"])
def test_pcm_statistics_use_decoded_sample_amplitudes(baseline, tmp_path, subtype):
    samples = np.zeros((48000, 2), dtype=np.float32)
    samples[2400:45600] = 0.25
    samples[2400] = [1.0, -1.0]
    path = tmp_path / "signal.wav"
    sf.write(path, samples, 48000, subtype=subtype)
    result = baseline.analyze_audio(path, save_npy=True)
    decoded, rate = sf.read(path, dtype="float64", always_2d=True)
    assert rate == 48000 and result["frames"] == 48000
    assert result["finite"] and result["non_silent"]
    assert result["peak_abs"] == 1.0
    assert result["rms"] == pytest.approx(
        float(np.sqrt(np.sum(decoded * decoded) / decoded.size)), rel=1e-7
    )
    assert result["near_full_scale_fraction"] == pytest.approx(2 / samples.size)
    assert result["leading_silence_seconds"] == pytest.approx(0.05)
    assert result["trailing_silence_seconds"] == pytest.approx(0.05)
    np.testing.assert_array_equal(np.load(path.with_suffix(".npy")), decoded.astype(np.float32))
    json.dumps(result, allow_nan=False)


def test_nonfinite_silent_and_clipped_float_signals_cannot_pass_objective_audio(baseline, tmp_path):
    for name, value in (("silence", 0.0), ("nan", float("nan")), ("clipped", 1.1)):
        path = tmp_path / f"{name}.wav"
        sf.write(path, np.full((6 * 48000, 2), value, dtype=np.float32), 48000, subtype="FLOAT")
        result = baseline.analyze_audio(path)
        assert not result["valid_objective_audio"]
        if name == "nan":
            assert not result["finite"] and result["peak_abs"] is None
        elif name == "clipped":
            assert result["near_full_scale_fraction"] == 1 and result["out_of_range_fraction"] == 1
        else:
            assert not result["non_silent"] and result["rms"] == 0
        json.dumps(result, allow_nan=False)
