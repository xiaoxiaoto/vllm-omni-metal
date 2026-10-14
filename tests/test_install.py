# SPDX-License-Identifier: Apache-2.0
"""Installer command/safety tests with no downloads or dependency changes."""

import json
import os
import runpy
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError
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
    assert len(installs) == 1
    for call in installs:
        assert call["args"][2:4] == ["--python", str(venv / "bin/python")]
        assert call["target"] == "cpu" and call["version"] == "0.31.0.dev0+cpu"
    core, metal, omni, *plugin = installs[0]["args"][4:]
    assert "0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl#sha256=" in core
    assert "vllm_metal-0.31.0-cp312-cp312-macosx_15_0_arm64.whl#sha256=" in metal
    assert omni.endswith("@2ec5f265c69520cdd5e5f0a19db485d38885843e")
    if piped:
        assert plugin == [
            "vllm-omni-metal @ git+https://github.com/xiaoxiaoto/vllm-omni-metal.git@main"
        ]
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


def test_plugin_version_matches_omni_and_source_fallback(monkeypatch):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert f"vllm-omni=={project['version']}" in project["dependencies"]

    def missing_package(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing_package)
    source = runpy.run_path(str(ROOT / "vllm_omni_metal/__init__.py"))
    assert source["__version__"] == project["version"]


def test_formula_and_script_pins_match():
    script = (ROOT / "install.sh").read_text()
    formula = (ROOT / "Formula/vllm-omni-metal.rb").read_text()
    for pin in (
        "21378516a51ff15029f71168e827b94660e21b39484eb52277166f01ac7dd686",
        "e2ab31296f8c42d8b65c8d5f5490f160087d1070178444870a5672e44aaacf6e",
        "2ec5f265c69520cdd5e5f0a19db485d38885843e",
        "0.31.0.dev0+cpu",
    ):
        assert pin in script and pin in formula
    version_check = "assert version('vllm-omni-metal') == version('vllm-omni').split('+', 1)[0]"
    assert version_check in script and version_check in formula
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(item.startswith("vllm==0.31.0;") for item in dependencies)
    assert any(item.startswith("vllm-metal==0.31.0;") for item in dependencies)
    assert "vllm-omni==0.31.0.dev0" in dependencies
    assert 'head "https://github.com/xiaoxiaoto/vllm-omni-metal.git", branch: "main"' in formula
    assert '  url "https://codeload.github.com/xiaoxiaoto/' not in formula
    assert 'bin.install_symlink libexec/"bin/vllm-omni"' in formula
    assert 'bin.install_symlink libexec/"bin/vllm"' not in formula
