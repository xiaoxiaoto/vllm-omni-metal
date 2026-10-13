# SPDX-License-Identifier: Apache-2.0
require "uri"

class VllmOmniMetal < Formula
  desc "Native multi-stage vLLM-Omni inference on Apple Silicon"
  homepage "https://github.com/xiaoxiaoto/vllm-omni-metal"
  # The current published source snapshot; no release wheel is advertised yet.
  url "https://codeload.github.com/xiaoxiaoto/vllm-omni-metal/tar.gz/81a8a7e30cc5cd7f2e41f3aabf5dd9d710f218b0"
  version "0.1.0"
  sha256 "ee68d7b122ccb2934e571b5041ee493aca12382eabb95fc43a8cd2a797b3cca6"
  license "Apache-2.0"
  head "https://github.com/xiaoxiaoto/vllm-omni-metal.git", branch: "main"

  depends_on arch: :arm64
  depends_on macos: :sequoia
  depends_on "python@3.12"

  # Match vllm-metal: avoid expanding relative Mach-O library IDs.
  preserve_rpath

  resource "vllm" do
    url "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl",
        using: :nounzip
    sha256 "fd9adfd566a8afa4ecbdf36b04bab10ebf43cd9ca18eca4b0a0bf2013b8de135"
  end

  resource "vllm-metal" do
    url "https://github.com/vllm-project/vllm-metal/releases/download/v0.30.0/vllm_metal-0.30.0-cp312-cp312-macosx_15_0_arm64.whl",
        using: :nounzip
    sha256 "3955de8b2c75b633ef889b70fa469a4db661a0c5bb2f78e36272d68b16da9353"
  end

  def install
    system "python3.12", "-m", "venv", libexec
    wheels = %w[vllm vllm-metal].map do |name|
      filename = URI::DEFAULT_PARSER.unescape(File.basename(URI.parse(resource(name).url).path))
      wheel = buildpath/filename
      cp resource(name).cached_download, wheel
      wheel
    end
    with_env(VLLM_OMNI_TARGET_DEVICE: "cpu", VLLM_OMNI_VERSION_OVERRIDE: "0.30.0+cpu") do
      system libexec/"bin/pip", "install", *wheels,
             "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@a8576ccb725c4e21cd13c3eb5f9a546b21149d2b",
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
    EOS
  end

  test do
    system libexec/"bin/python", "-c", <<~PYTHON
      from importlib.metadata import entry_points, version
      import torch
      from vllm_omni_metal import is_available
      assert version('vllm') == '0.30.0+cpu'
      assert version('vllm-metal') == '0.30.0'
      assert version('vllm-omni') == '0.30.0+cpu'
      assert is_available()
      assert torch.Generator(device='mps') is not None
      assert any(e.name == 'metal' and e.value == 'vllm_omni_metal:register_platform'
                 for e in entry_points(group='vllm_omni.platform_plugins'))
    PYTHON
    system bin/"vllm-omni", "--help"
  end
end
