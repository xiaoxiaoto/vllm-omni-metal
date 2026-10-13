# SPDX-License-Identifier: Apache-2.0
"""Acceptance of upstream native stages, not a bundled model implementation."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import psutil
import pytest
import yaml

from vllm_omni_metal import is_available


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("VLLM_OMNI_METAL_NATIVE_MODEL") or not is_available(),
    reason="requires Apple Silicon and an explicitly supplied local native model",
)
def test_upstream_native_speech(tmp_path):
    import vllm_omni

    model = Path(os.environ["VLLM_OMNI_METAL_NATIVE_MODEL"])
    assert model.is_dir()
    # Change resource limits only; preserve the upstream pipeline and connectors.
    deploy = Path(vllm_omni.__file__).parent / "deploy/higgs_multimodal_qwen3.yaml"
    config = yaml.safe_load(deploy.read_text())
    for stage in config["stages"]:
        stage.update(
            max_num_seqs=1,
            enable_prefix_caching=False,
            attention_backend="CUSTOM",
            gpu_memory_utilization=0.3,
        )
    profile = tmp_path / "deploy.yaml"
    profile.write_text(yaml.safe_dump(config))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    logfile = root / "outputs/native-stage-acceptance.log"
    logfile.parent.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
    )
    command = [
        str(Path(sys.executable).parent / "vllm-omni"),
        "serve",
        str(model),
        "--omni",
        "--served-model-name",
        "native-acceptance",
        "--deploy-config",
        str(profile),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    with logfile.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + 240
            ready = False
            while process.poll() is None and time.monotonic() < deadline:
                assert psutil.virtual_memory().available > 3 * 1024**3, "memory safety stop"
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                        ready = r.status == 200
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if ready:
                    break
                time.sleep(0.5)
            assert ready, logfile.read_text()

            def generate(seed):
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/audio/speech",
                    data=json.dumps(
                        {
                            "model": "native-acceptance",
                            "input": "Hello, this is a test on Apple Silicon.",
                            "response_format": "wav",
                            "seed": seed,
                            "max_new_tokens": 512,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    assert response.status == 200
                    return response.read()

            first = generate(42)
            assert first == generate(42), "per-request seed or model state is not isolated"
            with wave.open(BytesIO(first)) as wav:
                assert wav.getframerate() == 24000 and wav.getnchannels() == 1
                assert wav.getsampwidth() == 2 and wav.getnframes() > 2400
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
                assert np.isfinite(samples).all() and samples.astype(np.float32).std() > 1
            (root / "outputs/native-stage-acceptance.wav").write_bytes(first)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
