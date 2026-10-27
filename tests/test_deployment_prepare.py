# SPDX-License-Identifier: Apache-2.0
"""Deployment readiness/activation with CPU-only packages and isolated interpreters.

Network, wheel installation and native compilation are boundaries here, not simulated
installation evidence. Real wheel/native availability is checked separately by integration.
"""

import base64
import hashlib
import importlib.util
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tomllib
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

root = Path(__file__).parents[1]
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
            "ops/__init__.py": "",
            "ops/mps_attention.py": ATTENTION,
            "quantization/__init__.py": "",
            "quantization/linear.py": "QuantizedLinear = object\nvalidate_profile = object\n",
            "quantization/engine.py": "DynamicEngine = object\n",
            "quantization/capture.py": "CaptureSession = object\n",
            "quantization/adapter.py": "DenseModelAdapter = object\n",
            "quantization/runtime.py": "load_profile = object\nconfigure_model_quantization = object\n",
            "ane/__init__.py": "",
            "ane/projection.py": "DenseProjection = object\nprepare_projection = object\n",
            "_native/metal/header.h": "header",
            "_native/pytorch_backend/mps_ops.mm": "launcher",
            "_native/ane/bridge.swift": "bridge",
            "_native/elementwise.metal": "shader",
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
        "ops/mps_attention.py",
        "quantization/capture.py",
        "quantization/adapter.py",
        "quantization/runtime.py",
        "_native/metal/header.h",
        "_native/pytorch_backend/mps_ops.mm",
        "_native/ane/bridge.swift",
        "_native/elementwise.metal",
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


@pytest.mark.parametrize("module_name", ["capture", "adapter", "runtime"])
def test_backend_ready_reuse_rejects_missing_dynamic_api(backend, module_name):
    module, _, _, _ = backend
    runtime = module.prepare()
    site = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (site / "vllm_omni_metal" / "quantization" / f"{module_name}.py").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        module.prepare()
    assert not (runtime / ".ready").exists()


@pytest.fixture
def overlay(monkeypatch, tmp_path):
    project = tmp_path / "deployment" / "yue2-3b"
    (project / "patches").mkdir(parents=True)
    (project / "patches/yue2-mps.patch").write_text("fixture patch")
    (project / "yue2_dynamic_quantization.py").write_text("def install(model, *, scope): pass\n")
    (project / "prepare_dynamic_quantization.py").write_text("# offline search\n")
    script = project / "prepare_runtime.py"
    shutil.copyfile(DEPLOY / "yue2-3b/prepare_runtime.py", script)
    module = load_script(monkeypatch, script)
    base_site = tmp_path / "base-site"
    package = write_package(
        base_site,
        "vllm_omni_metal",
        **{
            "ops/__init__.py": "",
            "ops/mps_attention.py": ATTENTION,
            "quantization/__init__.py": "",
            "quantization/linear.py": "QuantizedLinear = object\nvalidate_profile = object\n",
            "quantization/engine.py": "DynamicEngine = object\n",
            "quantization/capture.py": "CaptureSession = object\n",
            "quantization/adapter.py": "DenseModelAdapter = object\n",
            "quantization/runtime.py": "load_profile = object\nconfigure_model_quantization = object\n",
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
        if command[0] == "git" and "checkout" in command:
            (Path(command[2]) / "pyproject.toml").write_text(
                "[tool.setuptools]\ninclude-package-data = false\n"
                '[tool.setuptools.packages.find]\ninclude = ["vllm_omni*"]\n'
            )
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
                source = Path(command[-1])
                if source.suffix == ".whl":
                    shutil.copytree(package, site / package.name)
                else:
                    write_package(site, "vllm_omni")
                    for name in ("yue2_dynamic_quantization.py", "prepare_dynamic_quantization.py"):
                        shutil.copyfile(source / name, site / name)
                    metadata = site / "vllm_omni-0.31.0.dev0.dist-info"
                    metadata.mkdir()
                    (metadata / "METADATA").write_text(
                        "Metadata-Version: 2.1\nName: vllm-omni\nVersion: 0.31.0.dev0\n"
                    )
                    (metadata / "RECORD").write_text(
                        "".join(
                            f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256((site / name).read_bytes()).digest()).decode().rstrip('=')},{(site / name).stat().st_size}\n"
                            for name in (
                                "yue2_dynamic_quantization.py",
                                "prepare_dynamic_quantization.py",
                            )
                        )
                    )
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


@pytest.mark.parametrize(
    "changed",
    [
        "preparer",
        "_native/metal/header.h",
        "yue2_dynamic_quantization.py",
        "prepare_dynamic_quantization.py",
    ],
)
def test_overlay_source_changes_do_not_reuse_old_ready(overlay, changed):
    module, package = overlay
    first = module.prepare()
    if changed == "preparer":
        target = Path(module.__file__)
    elif changed in {"yue2_dynamic_quantization.py", "prepare_dynamic_quantization.py"}:
        target = Path(module.__file__).parent / changed
    else:
        target = package / changed
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
            "ops/__init__.py": "",
            "ops/mps_attention.py": "raise RuntimeError('wrong launcher environment')\n",
        },
    )
    monkeypatch.setenv("PYTHONPATH", str(wrong))
    runtime = module.prepare()
    assert (runtime / "launcher.json").read_text() == str(runtime)
    assert (runtime / ".ready").read_text() == runtime.name


def test_overlay_private_module_is_declared_in_omni_wheel(overlay, tmp_path):
    module, _ = overlay
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    config = checkout / "pyproject.toml"
    config.write_text(
        "[tool.setuptools]\ninclude-package-data = false\n"
        '[tool.setuptools.packages.find]\ninclude = ["vllm_omni*"]\n'
    )
    module.package_private_module(checkout)
    setuptools = tomllib.loads(config.read_text())["tool"]["setuptools"]
    assert setuptools["py-modules"] == ["yue2_dynamic_quantization", "prepare_dynamic_quantization"]
    assert setuptools["packages"]["find"]["include"] == ["vllm_omni*"]
    assert not setuptools["include-package-data"]
    assert (checkout / "yue2_dynamic_quantization.py").read_bytes() == (
        Path(module.__file__).parent / "yue2_dynamic_quantization.py"
    ).read_bytes()
    assert (checkout / "prepare_dynamic_quantization.py").read_bytes() == (
        Path(module.__file__).parent / "prepare_dynamic_quantization.py"
    ).read_bytes()


def test_overlay_ready_reuse_rejects_tampered_private_module(overlay):
    module, _ = overlay
    runtime = module.prepare()
    installed = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/yue2_dynamic_quantization.py"
    )
    installed.write_text(installed.read_text() + "\n# installed source changed\n")
    with pytest.raises(subprocess.CalledProcessError):
        module.prepare()
    assert not (runtime / ".ready").exists()


def test_overlay_explicit_backend_wheel_is_private_and_remembered(overlay, monkeypatch, tmp_path):
    module, package = overlay
    wheel = tmp_path / "vllm_omni_metal-0.31.0.dev0-py3-none-any.whl"
    wheel.write_bytes(b"wheel boundary fixture")
    monkeypatch.setenv("VLLM_OMNI_METAL_WHEEL", str(wheel))
    original = (package / "quantization/linear.py").read_bytes()
    first = module.prepare()
    selection = Path.home() / "Library/Application Support/yue2-3b/backend-wheel"
    assert selection.read_text().strip() == str(wheel)
    assert selection.stat().st_mode & 0o777 == 0o600
    monkeypatch.delenv("VLLM_OMNI_METAL_WHEEL")
    assert module.prepare() == first
    wheel.write_bytes(b"new wheel boundary fixture")
    assert module.prepare() != first
    assert (package / "quantization/linear.py").read_bytes() == original


def test_overlay_migrates_flat_base_without_importing_new_api_there(overlay, monkeypatch, tmp_path):
    module, package = overlay
    legacy = package / "ane.py"
    legacy.write_text("DenseSwiGLU = object\n")
    wheel = tmp_path / "vllm_omni_metal-0.31.0.dev0-py3-none-any.whl"
    wheel.write_bytes(b"private backend upgrade boundary fixture")
    monkeypatch.setenv("VLLM_OMNI_METAL_WHEEL", str(wheel))
    original_run = module.run

    def run(*command, **kwargs):
        if Path(command[0]) == Path(sys.executable) and "-c" in command:
            arguments = list(command)
            index = arguments.index("-c") + 1
            arguments[index] = (
                f"import sys; sys.path.insert(0, {str(package.parent)!r}); " + arguments[index]
            )
            subprocess.run(arguments, check=True, capture_output=True, text=True)
        else:
            original_run(*command, **kwargs)

    monkeypatch.setattr(module, "run", run)
    runtime = module.prepare()
    assert (runtime / ".ready").read_text() == runtime.name
    assert legacy.read_text() == "DenseSwiGLU = object\n"
    assert (
        Path.home() / "Library/Application Support/yue2-3b/backend-wheel"
    ).read_text().strip() == str(wheel)


def test_overlay_missing_selected_wheel_cannot_fall_back_to_shared_backend(
    overlay,
    monkeypatch,
    tmp_path,
):
    module, _ = overlay
    monkeypatch.setenv("VLLM_OMNI_METAL_WHEEL", str(tmp_path / "missing.whl"))
    with pytest.raises(ValueError, match="wheel"):
        module.prepare()
    assert not list(Path.home().rglob(".ready"))


@pytest.mark.parametrize("fault", ["record", "engine", "linear"])
def test_overlay_ready_reuse_requires_wheel_ownership_and_generic_apis(overlay, fault):
    module, package = overlay
    runtime = module.prepare()
    if fault == "record":
        record = next(runtime.glob("lib/python*/site-packages/vllm_omni-*.dist-info/RECORD"))
        record.write_text("")
    else:
        # Keep identity fixed to exercise the ready probe after an inherited API failure.
        module.runtime_fingerprint = lambda *args: runtime.name
        (package / f"quantization/{fault}.py").write_text("raise ImportError('API unavailable')\n")
    with pytest.raises(subprocess.CalledProcessError):
        module.prepare()
    assert not (runtime / ".ready").exists()


@pytest.mark.parametrize("mode", ["off", "dynamic"])
def test_yue2_launch_configuration_uses_private_profile_without_documents_cwd(tmp_path, mode):
    # Execute only the configuration-writing block, never launchctl or model preparation.
    script = (DEPLOY / "yue2-3b/start.sh").read_text()
    source = script.rsplit("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    project = tmp_path / "Documents/deploy"
    project.mkdir(parents=True)
    (project / "metal-deploy.yaml").write_text("fixture deployment profile\n")
    policy = project / "dynamic-profile.json"
    if mode == "dynamic":
        policy.write_text('{"calibration":{"approved":true}}\n')
        policy.chmod(0o600)
    state = tmp_path / "Library/Application Support/yue2-3b"
    target = state / "com.local.yue2-3b-metal.plist"
    environment = dict(
        os.environ,
        HOME=str(tmp_path),
        YUE2_METAL_QUANTIZATION=mode,
        YUE2_METAL_QUANTIZATION_PROFILE=str(policy),
        VLLM_OMNI_METAL_ANE="off",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            source,
            "/overlay/bin/vllm-omni",
            "/cached/model",
            "/cached/vae",
            "fixture-yue2",
            str(target),
        ],
        cwd=project,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    config = plistlib.loads(target.read_bytes())
    env = config["EnvironmentVariables"]
    assert env["YUE2_METAL_QUANTIZATION"] == mode
    assert env["YUE2_VAE"] == "/cached/vae"
    assert config["WorkingDirectory"] == str(state)
    assert "PYTHONPATH" not in env
    assert not config.get("KeepAlive")
    if mode == "dynamic":
        installed = Path(env["YUE2_METAL_QUANTIZATION_PROFILE"])
        assert installed.parent == state
        assert installed.read_bytes() == policy.read_bytes()
        assert installed.stat().st_mode & 0o777 == 0o600
    else:
        assert "YUE2_METAL_QUANTIZATION_PROFILE" not in env
        assert not (state / "dynamic-profile.json").exists()


@pytest.mark.parametrize("bundle", [False, True])
@pytest.mark.parametrize(
    ("mode", "global_mode", "policy_permissions", "succeeds"),
    [
        ("dynamic", "off", 0o600, True),
        ("off", "dynamic", None, True),
        ("invalid", "off", None, False),
        ("manual-w8a16", "off", None, False),
        ("dynamic", "off", None, False),
        ("dynamic", "off", 0o644, False),
        (None, "off", None, True),
        (None, "off", 0o644, True),
        (None, "dynamic", 0o600, True),
        (None, "dynamic", None, False),
    ],
)
def test_yue2_start_mode_and_profile_gates_without_service_operations(
    tmp_path,
    mode,
    global_mode,
    policy_permissions,
    succeeds,
    bundle,
):
    script = DEPLOY / "yue2-3b/start.sh"
    # Stop before temporary setup/preparation; commands below are preflight boundaries only.
    prefix = script.read_text().split('workdir="$(mktemp -d)"', 1)[0]
    commands = tmp_path / "bin"
    commands.mkdir()
    stubs = {
        "uname": '#!/bin/sh\nif [ "$1" = "-s" ]; then echo Darwin; else echo arm64; fi\n',
        "python3": ('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nexec "$TEST_PYTHON" "$@"\n'),
    }
    for name in ("git", "modelscope", "launchctl", "curl", "lsof", "plutil"):
        stubs[name] = "#!/bin/sh\nexit 1\n"
    for name, source in stubs.items():
        executable = commands / name
        executable.write_text(source)
        executable.chmod(0o700)
    policy = tmp_path / "policy.json"
    if policy_permissions is not None:
        approved = {"calibration": {"approved": True}}
        policy.write_text(
            json.dumps({"profiles": {"actual-scope": approved}} if bundle else approved)
        )
        policy.chmod(policy_permissions)
    wheel = tmp_path / "selected.whl"
    wheel.write_bytes(b"selection boundary")
    environment = dict(
        os.environ,
        HOME=str(tmp_path),
        PATH=f"{commands}:{os.defpath}",
        TEST_PYTHON=sys.executable,
        VLLM_OMNI_METAL_QUANTIZATION=global_mode,
        VLLM_OMNI_METAL_QUANTIZATION_PROFILE=str(policy),
        VLLM_OMNI_METAL_WHEEL=str(wheel),
        VLLM_OMNI_METAL_ANE="auto",
    )
    environment.pop("YUE2_METAL_QUANTIZATION", None)
    environment.pop("YUE2_METAL_QUANTIZATION_PROFILE", None)
    if mode is not None:
        environment["YUE2_METAL_QUANTIZATION"] = mode
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            prefix + '\nprintf "%s|%s" "$YUE2_METAL_QUANTIZATION" "$VLLM_OMNI_METAL_ANE"\n',
            str(script),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == succeeds
    if succeeds:
        expected = mode if mode is not None else global_mode
        assert result.stdout == ("dynamic|off" if expected == "dynamic" else "off|auto")


@pytest.mark.parametrize(
    "content,error",
    [
        ('{"approved":true,"calibration":{"approved":false}}\n', "完整批准"),
        ("x" * (4 * 1024**2 + 1), "4MiB"),
    ],
    ids=("unapproved", "oversized"),
)
def test_yue2_start_rejects_unsafe_profile_without_service_operations(tmp_path, content, error):
    script = DEPLOY / "yue2-3b/start.sh"
    # Execute the actual preflight gate only; no launchd or installation calls.
    source = script.read_text().split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    profile = tmp_path / "candidate-profile.json"
    profile.write_text(content)
    profile.chmod(0o600)
    result = subprocess.run(
        [sys.executable, "-I", "-c", source],
        env=dict(
            os.environ,
            YUE2_METAL_QUANTIZATION="dynamic",
            YUE2_METAL_QUANTIZATION_PROFILE=str(profile),
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert error in result.stderr
