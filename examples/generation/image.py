# SPDX-License-Identifier: Apache-2.0
"""Offline generation through the upstream Omni entrypoint, not a separate backend."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image with vLLM-Omni on Metal/MPS.")
    parser.add_argument("--model", default="hf-internal-testing/tiny-stable-diffusion-pipe")
    parser.add_argument("--prompt", default="a small red robot")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/robot.png"))
    args = parser.parse_args()
    if min(args.steps, args.width, args.height) <= 0 or args.width % 8 or args.height % 8:
        parser.error("steps must be positive; width and height must be positive multiples of 8")

    from vllm_omni import Omni
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from vllm_omni_metal import is_available

    if not is_available():
        parser.error("Apple Silicon with an available PyTorch MPS backend is required")
    omni = Omni(
        model=args.model,
        diffusion_load_format="diffusers",
        dtype=args.dtype,
        num_gpus=1,
        distributed_executor_backend="uni",
        diffusion_attention_backend="TORCH_SDPA",
    )
    try:
        outputs = omni.generate(
            {"prompt": args.prompt, "modalities": ["image"]},
            OmniDiffusionSamplingParams(
                seed=args.seed, num_inference_steps=args.steps, height=args.height, width=args.width
            ),
        )
        images = extract_images_from_outputs(outputs)
        if not images:
            raise RuntimeError("Omni returned no images")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(args.output)
        print(args.output.resolve())
    finally:
        omni.close()


if __name__ == "__main__":
    main()
