# SPDX-License-Identifier: Apache-2.0
"""Actual Metal inference, using locally built tiny random weights (no downloads)."""

import asyncio
import os

import numpy as np
import pytest
import torch

from vllm_omni_metal import is_available

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("VLLM_OMNI_METAL_INTEGRATION") != "1" or not is_available(),
        reason="requires Apple Silicon and VLLM_OMNI_METAL_INTEGRATION=1",
    ),
]


@pytest.fixture
def tiny_sd_model(tmp_path):
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    with torch.random.fork_rng():
        torch.manual_seed(0)
        pipe = StableDiffusionPipeline(
            vae=AutoencoderKL(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                block_out_channels=(32, 64),
                layers_per_block=1,
                norm_num_groups=8,
                sample_size=64,
            ),
            text_encoder=CLIPTextModel(
                CLIPTextConfig(
                    vocab_size=2,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    max_position_embeddings=77,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=1,
                )
            ),
            tokenizer=CLIPTokenizer(
                vocab={"<|startoftext|>": 0, "<|endoftext|>": 1},
                merges=[],
                model_max_length=77,
            ),
            unet=UNet2DConditionModel(
                sample_size=32,
                in_channels=4,
                out_channels=4,
                layers_per_block=1,
                block_out_channels=(32, 64),
                norm_num_groups=8,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
                attention_head_dim=8,
            ),
            scheduler=DDIMScheduler(clip_sample=False, steps_offset=1),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    model = tmp_path / "model"
    pipe.save_pretrained(model)
    return str(model)


def test_diffusion_engine_on_mps(tiny_sd_model, tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    config = OmniDiffusionConfig(
        model=tiny_sd_model,
        diffusion_load_format="diffusers",
        dtype="float32",
        distributed_executor_backend="uni",
        num_gpus=1,
    )
    # Two lifecycles verify distributed cleanup and reinitialization, too.
    first = None
    for lifecycle in range(2):
        engine = DiffusionEngine(config)
        loop = asyncio.Runner()
        try:
            worker = engine.executor.driver_worker.worker
            assert worker.device.type == "mps"
            assert (
                next(worker.model_runner.pipeline._pipeline.unet.parameters()).device.type == "mps"
            )

            async def generate(request_id, seed, engine=engine):
                request = OmniDiffusionRequest(
                    prompt={"prompt": "robot", "modalities": ["image"]},
                    sampling_params=OmniDiffusionSamplingParams(
                        seed=seed,
                        num_inference_steps=2,
                        height=64,
                        width=64,
                    ),
                    request_id=request_id,
                )
                outputs = []
                async for batch in engine.step_streaming(request):
                    outputs = batch
                images = extract_images_from_outputs(outputs)
                assert len(images) == 1 and images[0].size == (64, 64)
                image = np.asarray(images[0])
                assert image.std() > 0
                images[0].save(tmp_path / f"metal-{request_id}.png")
                return image

            a = loop.run(asyncio.wait_for(generate(f"a{lifecycle}", 42), timeout=30))
            b = loop.run(asyncio.wait_for(generate(f"b{lifecycle}", 42), timeout=30))
            assert np.array_equal(a, b)
            if first is not None:
                assert np.array_equal(first, a)
            first = a
            assert not np.array_equal(
                a, loop.run(asyncio.wait_for(generate(f"c{lifecycle}", 43), timeout=30))
            )
        finally:
            engine.close()
            loop.close()
