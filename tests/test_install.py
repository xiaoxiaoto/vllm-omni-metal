# SPDX-License-Identifier: Apache-2.0
"""Installer command/safety tests with no downloads or dependency changes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def installer(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    log = tmp_path / "commands.jsonl"
    scripts = {
        "uname": '#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo "${TEST_ARCH:-arm64}";; esac\n',
        "sw_vers": '#!/bin/sh\necho "${TEST_MACOS:-15.0}"\n',
        "uv": f"#!{sys.executable}\n"
        + """import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['TEST_LOG'], 'a') as file:
    file.write(json.dumps({'args': args, 'target': os.getenv('VLLM_OMNI_TARGET_DEVICE'),
                           'version': os.getenv('VLLM_OMNI_VERSION_OVERRIDE')}) + '\\n')
if args[0] == 'venv':
    path = pathlib.Path(args[1]) / 'bin'
    path.mkdir(parents=True)
    # Command orchestration only; actual package validation is not simulated.
    python = path / 'python'
    python.write_text('#!/bin/sh\\nexit 0\\n')
    python.chmod(0o755)
if args[:2] == ['pip', 'install'] and os.getenv('TEST_FAIL_INSTALL'):
    sys.exit(7)
""",
    }
    for name, content in scripts.items():
        command = commands / name
        command.write_text(content)
        command.chmod(0o755)
    env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}", TEST_LOG=str(log))

    def run(*args, piped=False, **extra):
        command = ["bash", str(ROOT / "install.sh"), *args]
        text = None
        if piped:
            command = ["bash", "-s", "--", *args]
            text = (ROOT / "install.sh").read_text()
        return subprocess.run(
            command, input=text, env=env | extra, capture_output=True, text=True, timeout=20
        )

    return run, log


@pytest.mark.parametrize("piped", [False, True])
def test_install_pins_native_wheels_and_uses_only_private_python(installer, tmp_path, piped):
    run, log = installer
    venv = tmp_path / "private env"
    result = run("--venv", str(venv), piped=piped)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0]["args"] == ["venv", str(venv), "--python", "3.12", "--seed"]
    installs = [call for call in calls if call["args"][:2] == ["pip", "install"]]
    assert len(installs) == 2
    for call in installs:
        assert call["args"][2:4] == ["--python", str(venv / "bin/python")]
        assert call["target"] == "cpu" and call["version"] == "0.30.0+cpu"
    core, metal, omni = installs[0]["args"][4:]
    assert "0.30.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl#sha256=" in core
    assert "vllm_metal-0.30.0-cp312-cp312-macosx_15_0_arm64.whl#sha256=" in metal
    assert omni.endswith("@a8576ccb725c4e21cd13c3eb5f9a546b21149d2b")
    plugin = installs[1]["args"][4:]
    if piped:
        assert len(plugin) == 1 and "#sha256=" in plugin[0]
    else:
        assert plugin == ["-e", str(ROOT)]
    assert calls[-1]["args"] == ["pip", "check", "--python", str(venv / "bin/python")]
    assert (venv / ".vllm-omni-metal-installer").is_file()
    # Reruns reuse only an installer-owned environment, never --clear it.
    assert run("--venv", str(venv), piped=piped).returncode == 0
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(call["args"][0] == "venv" for call in calls) == 1


def test_install_refuses_unrelated_environment(installer, tmp_path):
    run, log = installer
    venv = tmp_path / "existing"
    venv.mkdir()
    sentinel = venv / "user-file"
    sentinel.write_text("keep")
    result = run("--venv", str(venv))
    assert result.returncode != 0 and "拒绝修改" in result.stderr
    assert sentinel.read_text() == "keep" and not log.exists()


@pytest.mark.parametrize("overrides", [{"TEST_ARCH": "x86_64"}, {"TEST_MACOS": "14.5"}])
def test_install_rejects_unsupported_platform_before_any_install(installer, tmp_path, overrides):
    run, log = installer
    result = run("--venv", str(tmp_path / "unused"), **overrides)
    assert result.returncode != 0 and not log.exists()


def test_failed_dependency_install_stops_before_plugin_and_is_retryable(installer, tmp_path):
    run, log = installer
    venv = tmp_path / "partial"
    result = run("--venv", str(venv), TEST_FAIL_INSTALL="1")
    assert result.returncode == 7
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2 and calls[-1]["args"][:2] == ["pip", "install"]
    assert run("--venv", str(venv)).returncode == 0


def test_formula_and_script_pins_match():
    script = (ROOT / "install.sh").read_text()
    formula = (ROOT / "Formula/vllm-omni-metal.rb").read_text()
    for pin in (
        "fd9adfd566a8afa4ecbdf36b04bab10ebf43cd9ca18eca4b0a0bf2013b8de135",
        "3955de8b2c75b633ef889b70fa469a4db661a0c5bb2f78e36272d68b16da9353",
        "a8576ccb725c4e21cd13c3eb5f9a546b21149d2b",
        "81a8a7e30cc5cd7f2e41f3aabf5dd9d710f218b0",
        "ee68d7b122ccb2934e571b5041ee493aca12382eabb95fc43a8cd2a797b3cca6",
    ):
        assert pin in script and pin in formula
    assert 'bin.install_symlink libexec/"bin/vllm-omni"' in formula
    assert 'bin.install_symlink libexec/"bin/vllm"' not in formula
