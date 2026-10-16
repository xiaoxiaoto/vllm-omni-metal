# SPDX-License-Identifier: Apache-2.0
require "uri"

class VllmOmniMetal < Formula
  desc "Native multi-stage vLLM-Omni inference on Apple Silicon"
  homepage "https://github.com/xiaoxiaoto/vllm-omni-metal"
  # No published plugin release matches the 0.31 stack yet; install HEAD.
  license "Apache-2.0"
  head "https://github.com/xiaoxiaoto/vllm-omni-metal.git", branch: "main"

  depends_on arch: :arm64
  depends_on macos: :sequoia
  depends_on "python@3.12"

  # Match vllm-metal: avoid expanding relative Mach-O library IDs.
  preserve_rpath

  resource "vllm" do
    url "https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl",
        using: :nounzip
    sha256 "21378516a51ff15029f71168e827b94660e21b39484eb52277166f01ac7dd686"
  end

  resource "vllm-metal" do
    url "https://github.com/vllm-project/vllm-metal/releases/download/v0.31.0/vllm_metal-0.31.0-cp312-cp312-macosx_15_0_arm64.whl",
        using: :nounzip
    sha256 "e2ab31296f8c42d8b65c8d5f5490f160087d1070178444870a5672e44aaacf6e"
  end

  def install
    system "python3.12", "-m", "venv", libexec
    wheels = %w[vllm vllm-metal].map do |name|
      filename = URI::DEFAULT_PARSER.unescape(File.basename(URI.parse(resource(name).url).path))
      wheel = buildpath/filename
      cp resource(name).cached_download, wheel
      wheel
    end
    with_env(VLLM_OMNI_TARGET_DEVICE: "cpu", VLLM_OMNI_VERSION_OVERRIDE: "0.31.0.dev0+cpu") do
      system libexec/"bin/pip", "install", *wheels,
             "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@2ec5f265c69520cdd5e5f0a19db485d38885843e",
             buildpath
    end
    system libexec/"bin/pip", "check"

    # Homebrew removes RECORD files, but TVM-FFI uses these to find libraries.
    site_packages = libexec/"lib/python3.12/site-packages"
    %w[apache_tvm_ffi xgrammar].each do |package|
      record = site_packages.glob("#{package}-*.dist-info/RECORD").fetch(0)
      (pkgshare/"#{package}.record").write record.read
    end
    # Do not take over the vllm executable owned by vllm-metal.
    bin.install_symlink libexec/"bin/vllm-omni"
  end

  def post_install
    site_packages = libexec/"lib/python3.12/site-packages"
    %w[apache_tvm_ffi xgrammar].each do |package|
      metadata = site_packages.glob("#{package}-*.dist-info").fetch(0)
      cp pkgshare/"#{package}.record", metadata/"RECORD"
    end
  end

  def caveats
    <<~EOS
      Start an Omni server with a Metal-compatible deploy profile:
        vllm-omni serve /path/to/model --omni --deploy-config /path/to/metal-profile.yaml

      Python and all backends are installed in a private libexec environment.
      No environment activation is needed. The existing vllm command is unchanged.

      Optional tools are not built by installation, imports, or --help.
      Prepare them explicitly with this libexec Python after installation:
        #{libexec}/bin/python -I -m vllm_omni_metal.mps_attention
        #{libexec}/bin/python -I -c 'from vllm_omni_metal.mps_attention import get_mps_ops; assert get_mps_ops() is not None'
        #{libexec}/bin/python -I -m vllm_omni_metal.vpipe_kernels --diffusion
        #{libexec}/bin/python -I -m vllm_omni_metal.ane_build --bridge
      Attention and ANE host tools require existing Apple Command Line Tools.
      vpipe additionally requires an installed Apple Metal compiler; no tools
      are downloaded automatically. Missing vpipe compiler is not an install failure.
      No startup/request compilation, model downloads, conversion, or calibration.

      ANE auto keeps GPU without approved weights and a whole-pipeline profile.
      Host tools do not establish ANE availability: the real YuE2 graph was
      rejected (relative L2 5.499% exceeds 1%); no ANE profile was approved.
      YuE2 generation evidence uses a separate model overlay, not stock Omni.
      This generic package bundles no model patches, registry, or media.
      Existing environments and running services are not migrated or restarted;
      stop services using this libexec before upgrading or preparing its tools.
    EOS
  end

  test do
    system libexec/"bin/python", "-I", "-c", <<~PYTHON
      from importlib import import_module
      from importlib.metadata import entry_points, version
      import torch
      from vllm_omni_metal import is_available
      from vllm_omni_metal.ane import DenseSwiGLU, TiledDenseSwiGLU, execution_scope
      from importlib.resources import files
      import sys
      from pathlib import Path
      import vllm_omni_metal
      assert Path(vllm_omni_metal.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
      for module in ('mps_attention', 'vpipe_kernels', 'normalization', 'activation', 'rotary',
                     'ane_build', 'diffusion_ops'):
          import_module(f'vllm_omni_metal.{module}')
      for resource in (
          '_native/pytorch_backend/mps_ops.mm',
          '_native/metal/metal_device.h',
          '_native/metal/paged_attention_kernels.h',
          '_native/vpipe_elementwise.metal',
          '_native/diffusion_rotary.metal',
          '_native/VPIPE.md',
          '_native/ane/ane-emitter.cc',
          '_native/ane/ane-emitter.h',
          '_native/ane/emit.cc',
          '_native/ane/bridge.swift',
          '_native/ane/LICENSE',
      ):
          assert files('vllm_omni_metal').joinpath(resource).is_file(), resource
      assert version('vllm') == '0.31.0+cpu'
      assert version('vllm-metal') == '0.31.0'
      assert version('vllm-omni') == '0.31.0.dev0+cpu'
      assert version('vllm-omni-metal') == version('vllm-omni').split('+', 1)[0]
      assert is_available()
      assert torch.Generator(device='mps') is not None
      assert any(e.name == 'metal' and e.value == 'vllm_omni_metal:register_platform'
                 for e in entry_points(group='vllm_omni.platform_plugins'))
    PYTHON
    system libexec/"bin/python", "-I", "-m", "vllm_omni_metal.mps_attention", "--help"
    system libexec/"bin/python", "-I", "-m", "vllm_omni_metal.vpipe_kernels", "--help"
    system libexec/"bin/python", "-I", "-m", "vllm_omni_metal.ane_build", "--help"
    system libexec/"bin/python", "-I", bin/"vllm-omni", "--help"
  end
end
