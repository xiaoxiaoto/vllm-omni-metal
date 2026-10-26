# SPDX-License-Identifier: Apache-2.0
"""Private 7B numerical boundaries; small blocks do not approve whole-model quality."""

import pytest
import torch


def test_fp16_text_gate_retains_large_residual_through_next_normalization():
    nadit = pytest.importorskip("vllm_omni.diffusion.models.seedvr2.nadit")
    if not hasattr(nadit, "NormalMLP"):
        pytest.skip("Requires the installed private SeedVR2 7B wheel")
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="numerical-boundary-only",
        dtype=torch.float16,
        enforce_eager=True,
        distributed_executor_backend="uni",
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
    )
    with set_current_diffusion_config(config):
        block = nadit.NaMMSRTransformerBlock(
            vid_dim=8,
            txt_dim=8,
            emb_dim=48,
            heads=1,
            head_dim=8,
            expand_ratio=4,
            norm_eps=1e-5,
            qk_bias=False,
            mlp_type="normal",
            shared_weights=False,
            rope_dim=6,
            rope_type="rope3d",
            is_last_layer=False,
            use_varlen_kernel=False,
        ).to(dtype=torch.float16)
    with torch.inference_mode():
        for parameter in block.parameters():
            parameter.zero_()
        block.mlp.txt.proj_out.bias.fill_(-20544)
        block.ada.txt.mlp_gate.fill_(4)
        video = torch.ones(4, 8, dtype=torch.float16)
        text = torch.full((2, 8), -1500, dtype=torch.float16)
        embedding = torch.zeros(1, 48, dtype=torch.float16)
        runtime = nadit.SeedVR2WindowRuntime((1, 2, 2), text_len=2, num_layers=1)
        context = runtime.context(runtime.layout_for_layer(0), video.device)
        expected = text.float()
        for _ in range(2):
            actual_video, text = block(video, text, embedding, context, runtime)
            expected = expected - 20544 * 4
            assert text.dtype == torch.float32
            torch.testing.assert_close(text, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_video, video, rtol=0, atol=0)
        assert block.mlp.txt.proj_in.weight.dtype == torch.float16
        assert block.mlp.txt.proj_out.weight.dtype == torch.float16
