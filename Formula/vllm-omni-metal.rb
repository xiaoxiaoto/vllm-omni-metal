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
    EOS
  end

  test do
    system libexec/"bin/python", "-c", <<~PYTHON
      from importlib.metadata import entry_points, version
      import torch
      from vllm_omni_metal import is_available
      assert version('vllm') == '0.31.0+cpu'
      assert version('vllm-metal') == '0.31.0'
      assert version('vllm-omni') == '0.31.0.dev0+cpu'
      assert version('vllm-omni-metal') == version('vllm-omni').split('+', 1)[0]
      assert is_available()
      assert torch.Generator(device='mps') is not None
      assert any(e.name == 'metal' and e.value == 'vllm_omni_metal:register_platform'
                 for e in entry_points(group='vllm_omni.platform_plugins'))
    PYTHON
    system bin/"vllm-omni", "--help"
  end
end
