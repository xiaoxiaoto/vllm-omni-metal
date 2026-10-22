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
        "uname": '#!/bin/sh\ncase "$1" in -s) echo "${TEST_OS:-Darwin}";; -m) echo "${TEST_ARCH:-arm64}";; esac\n',
        "sw_vers": '#!/bin/sh\necho "${TEST_MACOS:-15.0}"\n',
        "uv": "#!/usr/bin/env python3\n"
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
    python.write_text(
        '#!/bin/sh\\n[ "$1" != -I ] || shift\\ncase "$1" in\\n'
        '  -m) exit "${TEST_FAIL_BUILD:-0}" ;;\\n'
        '  -c) exit "${TEST_FAIL_LOAD:-0}" ;;\\n'
        'esac\\nexit 0\\n'
    )
    python.chmod(0o755)
if args[:2] == ['pip', 'install'] and os.getenv('TEST_FAIL_INSTALL'):
    sys.exit(7)
""",
        "clang++": "#!/bin/sh\nexit 19\n",
        "swift": "#!/bin/sh\nexit 19\n",
        "xcrun": "#!/bin/sh\nexit 19\n",
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


@pytest.mark.parametrize("piped", [False, True])
def test_install_refuses_unrelated_environment(installer, tmp_path, piped):
    run, log = installer
    venv = tmp_path / "existing"
    venv.mkdir()
    sentinel = venv / "user-file"
    sentinel.write_text("keep")
    result = run("--venv", str(venv), "--with-mps-attention", piped=piped)
    assert result.returncode != 0 and "拒绝修改" in result.stderr
    assert sentinel.read_text() == "keep" and not log.exists()


@pytest.mark.parametrize("piped", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [{"TEST_OS": "Linux"}, {"TEST_ARCH": "x86_64"}, {"TEST_MACOS": "14.5"}],
)
def test_install_rejects_unsupported_platform_before_any_install(
    installer, tmp_path, overrides, piped
):
    run, log = installer
    result = run("--venv", str(tmp_path / "unused"), "--with-vpipe", piped=piped, **overrides)
    assert result.returncode != 0 and not log.exists()
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("piped", [False, True])
@pytest.mark.parametrize(
    "invalid", [("--unknown",), ("--venv",), ("--venv", ""), ("--venv", "--with-vpipe")]
)
def test_install_rejects_invalid_arguments_before_any_changes(installer, tmp_path, piped, invalid):
    run, log = installer
    venv = tmp_path / "unused"
    result = run("--venv", str(venv), *invalid, piped=piped)
    assert result.returncode != 0
    assert not log.exists() and not venv.exists()


def test_failed_dependency_install_stops_before_plugin_and_is_retryable(installer, tmp_path):
    run, log = installer
    venv = tmp_path / "partial"
    result = run("--venv", str(venv), TEST_FAIL_INSTALL="1")
    assert result.returncode == 7
    assert "安装完成" not in result.stdout
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2 and calls[-1]["args"][:2] == ["pip", "install"]
    assert run("--venv", str(venv)).returncode == 0


def test_install_uses_requested_official_omni_snapshot(installer, tmp_path):
    run, log = installer
    commit = "a" * 40
    result = run("--venv", str(tmp_path / "latest"), VLLM_OMNI_COMMIT=commit)
    assert result.returncode == 0, result.stderr
    installs = [json.loads(line) for line in log.read_text().splitlines()]
    install = next(call for call in installs if call["args"][:2] == ["pip", "install"])
    assert any(arg.endswith("@" + commit) for arg in install["args"])


@pytest.mark.parametrize("commit", ["main", "abcd", "-invalid", "a" * 40 + ";bad"])
def test_install_rejects_non_commit_snapshot_before_any_changes(installer, tmp_path, commit):
    run, log = installer
    result = run("--venv", str(tmp_path / "unused"), VLLM_OMNI_COMMIT=commit)
    assert result.returncode != 0 and "完整 Git commit" in result.stderr
    assert not log.exists() and not (tmp_path / "unused").exists()


def test_plugin_version_matches_omni_and_source_fallback(monkeypatch):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert f"vllm-omni=={project['version']}" in project["dependencies"]

    def missing_package(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing_package)
    source = runpy.run_path(str(ROOT / "vllm_omni_metal/__init__.py"))
    assert source["__version__"] == project["version"]


@pytest.mark.parametrize("piped", [False, True])
def test_default_install_does_not_require_optional_compilers(installer, tmp_path, piped):
    """Boundary stubs reject builds; this does not simulate real package installation."""
    run, _ = installer
    result = run(
        "--venv",
        str(tmp_path / "no compiler"),
        piped=piped,
        TEST_FAIL_BUILD="19",
        TEST_FAIL_LOAD="19",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("piped", [False, True])
@pytest.mark.parametrize("flag", ["--with-mps-attention", "--with-vpipe", "--with-ane"])
def test_requested_build_failure_prevents_install_success(installer, tmp_path, piped, flag):
    run, _ = installer
    venv = tmp_path / "build failure"
    result = run("--venv", str(venv), flag, piped=piped, TEST_FAIL_BUILD="7")
    assert result.returncode == 7
    assert "安装完成" not in result.stdout
    assert (venv / ".vllm-omni-metal-installer").is_file()
    # An optional failure leaves the owned environment retryable, not cleared.
    sentinel = venv / "keep-on-retry"
    sentinel.write_text("keep")
    assert run("--venv", str(venv), flag, piped=piped).returncode == 0
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("piped", [False, True])
def test_attention_load_failure_prevents_install_success(installer, tmp_path, piped):
    run, _ = installer
    result = run(
        "--venv",
        str(tmp_path / "load failure"),
        "--with-mps-attention",
        piped=piped,
        TEST_FAIL_LOAD="9",
    )
    assert result.returncode == 9
    assert "安装完成" not in result.stdout


@pytest.mark.parametrize("module", ["ops.mps_attention", "ops.vpipe_kernels"])
@pytest.mark.parametrize("arguments,expected", [(["--help"], 0), (["--unknown-option"], 2)])
def test_optional_build_cli_does_not_build_for_help_or_invalid_arguments(
    tmp_path, module, arguments, expected
):
    result = subprocess.run(
        [sys.executable, "-m", f"vllm_omni_metal.{module}", *arguments],
        cwd=tmp_path,
        env=dict(os.environ, HOME=str(tmp_path), PYTHONPATH=str(ROOT)),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == expected, result.stderr
    assert not (tmp_path / ".cache/vllm-omni-metal").exists()


@pytest.mark.parametrize("resource", ["install.sh", "Formula/vllm-omni-metal.rb"])
def test_install_smoke_checks_include_dynamic_quantization_modules(resource):
    source = (ROOT / resource).read_text()
    for module in ("linear", "engine", "capture", "adapter", "runtime"):
        assert f"'quantization.{module}'" in source
    assert '"$python" -I -' in source if resource == "install.sh" else 'bin/python", "-I"' in source


def test_quantization_python_sources_use_existing_package_discovery():
    setuptools = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    assert "vllm_omni_metal*" in setuptools["packages"]["find"]["include"]
    package = ROOT / "vllm_omni_metal" / "quantization"
    for module in ("__init__", "linear", "engine", "capture", "adapter", "runtime"):
        assert (package / f"{module}.py").is_file()
