# SPDX-License-Identifier: Apache-2.0
"""Deployment readiness/activation with CPU-only packages and isolated interpreters.

Network, wheel installation and native compilation are boundaries here, not simulated
installation evidence. Real wheel/native availability is checked separately by integration.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "prepare_backend.py").is_file(),
    reason="independent deployment checkout is not present",
)

# A tiny executable package exercises interpreter/cache identity and failure propagation;
# it deliberately does not implement a native launcher or perform any GPU work.
ATTENTION = """from pathlib import Path
import sys

root = Path(__file__).parent
artifact = Path(sys.prefix) / "launcher.json"

def get_mps_ops():
    if (root / "load-failure").exists():
        raise RuntimeError("launcher load rejected")
    if not artifact.exists():
        return None
    if artifact.read_text() != sys.prefix:
        raise RuntimeError("launcher belongs to another interpreter")
    return object()

def build_mps_launcher():
    if (root / "build-failure").exists():
        raise RuntimeError("launcher build rejected")
    artifact.write_text(sys.prefix)
"""


def load_script(monkeypatch, path):
    monkeypatch.syspath_prepend(str(DEPLOY))
    spec = importlib.util.spec_from_file_location("deployment_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_package(site, name, **files):
    package = site / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    for filename, text in files.items():
        path = package / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return package


def make_environment(runtime):
    venv.create(runtime, with_pip=False, symlinks=True)
    return (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


@pytest.fixture
def backend(monkeypatch, tmp_path):
    script = tmp_path / "prepare_backend.py"
    shutil.copyfile(DEPLOY / "prepare_backend.py", script)
    module = load_script(monkeypatch, script)
    source = tmp_path / "source"
    source.mkdir()
    for name in ("install.sh", "pyproject.toml"):
        (source / name).write_text("fixture")
    package = write_package(
        source,
        "vllm_omni_metal",
        **{
            "mps_attention.py": ATTENTION,
            "_native/metal/header.h": "header",
            "_native/pytorch_backend/mps_ops.mm": "launcher",
            "_native/ane/bridge.swift": "bridge",
            "_native/vpipe.metal": "shader",
        },
    )
    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "CACHE", tmp_path / "cache")
    latest = dict(vllm=module.EXPECTED["vllm"], omni_commit="a" * 40)
    monkeypatch.setattr(module, "official_versions", lambda: latest)
    monkeypatch.setattr(module, "stop_models", lambda: None)
    failure = SimpleNamespace(stage=None)

    def commands(*command, **kwargs):
        if command[0] == "bash":
            if failure.stage == "build":
                raise subprocess.CalledProcessError(7, command)
            runtime = Path(command[command.index("--venv") + 1])
            site = make_environment(runtime)
            shutil.copytree(package, site / package.name)
            if "--with-mps-attention" in command:
                (runtime / "launcher.json").write_text(str(runtime))
        elif "-c" in command:
            subprocess.run(command, check=True, capture_output=True, text=True)
        # Official version and pip dependency checks are separate external boundaries.

    def install_wheel(runtime):
        if failure.stage == "load":
            site = (
                runtime
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            (site / package.name / "load-failure").touch()

    monkeypatch.setattr(module, "run", commands)
    monkeypatch.setattr(module, "install_plugin", install_wheel)
    return module, latest, package, failure


@pytest.mark.parametrize("failure_stage", ["build", "load"])
def test_backend_failure_does_not_publish_or_switch_current(backend, failure_stage):
    module, _, _, failure = backend
    module.CACHE.mkdir()
    old = module.CACHE / "previous"
    old.mkdir()
    current = module.CACHE / "current"
    current.symlink_to(old, target_is_directory=True)
    failure.stage = failure_stage
    with pytest.raises(subprocess.CalledProcessError):
        module.prepare()
    assert current.resolve() == old
    assert not list(module.CACHE.rglob(".ready"))


@pytest.mark.parametrize(
    "changed",
    [
        "preparer",
        "mps_attention.py",
        "_native/metal/header.h",
        "_native/pytorch_backend/mps_ops.mm",
        "_native/ane/bridge.swift",
        "_native/vpipe.metal",
    ],
)
def test_backend_changes_create_new_verified_environment(backend, changed):
    module, _, package, _ = backend
    first = module.prepare()
    target = Path(module.__file__) if changed == "preparer" else package / changed
    target.write_text(target.read_text() + "\n# changed\n")
    second = module.prepare()
    assert first != second
    assert (second / ".ready").read_text() == second.name
    assert (module.CACHE / "current").resolve() == second


def test_backend_ready_reuse_still_requires_loadable_launcher(backend):
    module, _, _, _ = backend
    runtime = module.prepare()
    (runtime / "launcher.json").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        module.prepare()
    assert not (runtime / ".ready").exists()
    assert (module.CACHE / "current").resolve() == runtime


@pytest.fixture
def overlay(monkeypatch, tmp_path):
    project = tmp_path / "deployment" / "yue2-3b"
    (project / "patches").mkdir(parents=True)
    (project / "patches/yue2-mps.patch").write_text("fixture patch")
    script = project / "prepare_runtime.py"
    shutil.copyfile(DEPLOY / "yue2-3b/prepare_runtime.py", script)
    module = load_script(monkeypatch, script)
    base_site = tmp_path / "base-site"
    package = write_package(
        base_site,
        "vllm_omni_metal",
        **{
            "mps_attention.py": ATTENTION,
            "_native/metal/header.h": "header",
        },
    )
    metadata = SimpleNamespace(
        read_text=lambda name: json.dumps(
            {
                "url": "https://github.com/vllm-project/vllm-omni.git",
                "vcs_info": {"commit_id": "a" * 40},
            }
        ),
        locate_file=lambda name: base_site / name,
    )
    monkeypatch.setattr(module, "distribution", lambda name: metadata)
    monkeypatch.setattr(module, "sysconfig", SimpleNamespace(get_path=lambda name: str(base_site)))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        module, "venv", SimpleNamespace(create=lambda path, **kwargs: make_environment(path))
    )

    def commands(*command, **kwargs):
        python = Path(command[0])
        if python.parent.name == "bin" and python.parent.parent.parent.name == "yue2-3b":
            runtime = python.parent.parent
            if "-c" in command:
                subprocess.run(command, check=True, capture_output=True, text=True)
            elif "install" in command:
                site = (
                    runtime
                    / "lib"
                    / f"python{sys.version_info.major}.{sys.version_info.minor}"
                    / "site-packages"
                )
                write_package(site, "vllm_omni")
        # Git, pip and base version/API checks are external setup boundaries.

    monkeypatch.setattr(module, "run", commands)
    return module, package


@pytest.mark.parametrize("stage", ["build", "load"])
def test_overlay_failure_never_publishes_ready(overlay, stage):
    module, package = overlay
    (package / f"{stage}-failure").touch()
    with pytest.raises(subprocess.CalledProcessError):
        module.prepare()
    assert not list(Path.home().rglob(".ready"))


def test_overlay_ready_reuse_rejects_missing_or_broken_launcher(overlay):
    module, package = overlay
    runtime = module.prepare()
    (runtime / "launcher.json").unlink()
    # Fault flags are outside the source tree so this exercises the same ready environment.
    (runtime / "launcher.json").write_text("another interpreter")
    with pytest.raises(subprocess.CalledProcessError):
        module.prepare()
    assert not (runtime / ".ready").exists()


@pytest.mark.parametrize("changed", ["preparer", "_native/metal/header.h"])
def test_overlay_source_changes_do_not_reuse_old_ready(overlay, changed):
    module, package = overlay
    first = module.prepare()
    target = Path(module.__file__) if changed == "preparer" else package / changed
    target.write_text(target.read_text() + "\n# changed\n")
    second = module.prepare()
    assert first != second
    assert (second / ".ready").read_text() == second.name


def test_overlay_uses_selected_interpreter_not_pythonpath_launcher(overlay, monkeypatch, tmp_path):
    module, _ = overlay
    wrong = tmp_path / "wrong-environment"
    write_package(
        wrong,
        "vllm_omni_metal",
        **{
            "mps_attention.py": "raise RuntimeError('wrong launcher environment')\n",
        },
    )
    monkeypatch.setenv("PYTHONPATH", str(wrong))
    runtime = module.prepare()
    assert (runtime / "launcher.json").read_text() == str(runtime)
    assert (runtime / ".ready").read_text() == runtime.name
