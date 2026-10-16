# SPDX-License-Identifier: Apache-2.0
"""Bounded A/B generation benchmark through real spawned native workers, without downloads."""

import argparse
import json
import statistics
import tempfile
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import psutil
import torch


class BenchmarkHooks:
    """Explicit experiment controls; shader compilation is never a production fallback."""

    def prepare_benchmark(self, compile_for_test):
        from vllm_omni_metal import vpipe_kernels

        self._benchmark_library = vpipe_kernels._library
        self._benchmark_supports_silu = vpipe_kernels.supports_silu
        self._benchmark_supports_norm = vpipe_kernels.supports_add_rms_norm
        self._benchmark_shader = None
        if compile_for_test:
            self._benchmark_shader = torch.mps.compile_shader(
                "#define VPIPE_ELT bfloat\n" + vpipe_kernels._SOURCE.read_text()
            )
        elif vpipe_kernels._library(torch.bfloat16) is None:
            raise RuntimeError("build metallibs first, or explicitly select --compile-for-test")
        self._benchmark_calls = {"silu": 0, "norm": 0}
        self._benchmark_silu = vpipe_kernels.silu_and_mul

        def silu(x):
            self._benchmark_calls["silu"] += 1
            return self._benchmark_silu(x)

        vpipe_kernels.silu_and_mul = silu
        self._benchmark_norm = vpipe_kernels.add_rms_norm

        def norm(*args):
            self._benchmark_calls["norm"] += 1
            return self._benchmark_norm(*args)

        vpipe_kernels.add_rms_norm = norm

    def select_benchmark_mode(self, mode):
        from vllm_omni_metal import vpipe_kernels

        if mode not in ("baseline", "activation", "normalization", "combined"):
            raise ValueError(mode)
        vpipe_kernels._library = (
            (lambda dtype: None)
            if mode == "baseline"
            else (lambda dtype: self._benchmark_shader)
            if self._benchmark_shader is not None
            else self._benchmark_library
        )
        vpipe_kernels.supports_silu = (
            self._benchmark_supports_silu if mode in ("activation", "combined") else lambda x: False
        )
        vpipe_kernels.supports_add_rms_norm = (
            self._benchmark_supports_norm
            if mode in ("normalization", "combined")
            else lambda *args: False
        )
        self._benchmark_calls = {"silu": 0, "norm": 0}

    def benchmark_evidence(self):
        return dict(self._benchmark_calls)

    def restore_benchmark(self):
        from vllm_omni_metal import vpipe_kernels

        vpipe_kernels._library = self._benchmark_library
        vpipe_kernels.supports_silu = self._benchmark_supports_silu
        vpipe_kernels.silu_and_mul = self._benchmark_silu
        vpipe_kernels.supports_add_rms_norm = self._benchmark_supports_norm
        vpipe_kernels.add_rms_norm = self._benchmark_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["llama", "qwen3"], default="llama")
    parser.add_argument("--model", type=Path, help="Existing local checkpoint; never downloaded")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--compile-for-test", action="store_true")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS required; no CPU fallback")
    if not (
        1 <= args.prompt_tokens <= 256
        and 1 <= args.output_tokens <= 64
        and 1 <= args.batch_size <= 4
        and 4 <= args.iterations <= 20
    ):
        parser.error("budget: prompt 1..256, output 1..64, batch 1..4, iterations 4..20")
    weight_bytes = 0
    if args.model is not None:
        args.model = args.model.expanduser().resolve()
        if not args.model.is_dir():
            parser.error("--model must be an existing local checkpoint directory")
        weights = [
            path
            for path in args.model.rglob("*")
            if path.is_file() and path.suffix in (".safetensors", ".bin")
        ]
        if not weights:
            parser.error("local checkpoint has no weight files")
        weight_bytes = sum(path.stat().st_size for path in weights)
    required_bytes = max(6 * 1024**3, 3 * 1024**3 + 2 * weight_bytes)
    available = psutil.virtual_memory().available
    if available < required_bytes:
        parser.error(
            f"available {available / 1024**3:.2f} GiB; requires "
            f"{required_bytes / 1024**3:.2f} GiB including weights and system headroom"
        )
    from transformers import (
        AutoConfig,
        LlamaConfig,
        LlamaForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )
    from vllm import LLM, SamplingParams

    torch.manual_seed(83)
    if args.model is None:
        config_cls, model_cls = {
            "llama": (LlamaConfig, LlamaForCausalLM),
            "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        }[args.architecture]
        config = config_cls(
            vocab_size=128,
            hidden_size=512,
            intermediate_size=8192,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=384,
            bos_token_id=1,
            eos_token_id=2,
        )
    else:
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text_config = config.get_text_config()
    prompts = [
        {"prompt_token_ids": [3 + (i + row * 7) % 120 for i in range(args.prompt_tokens)]}
        for row in range(args.batch_size)
    ]
    params = SamplingParams(
        temperature=0, max_tokens=args.output_tokens, ignore_eos=True, detokenize=False
    )
    modes = ("baseline", "activation", "normalization", "combined")
    samples = {mode: [] for mode in modes}
    calls = {}
    reference_tokens = None
    seeded_tokens = None
    available_before = psutil.virtual_memory().available / 1024**3
    directory_context = (
        tempfile.TemporaryDirectory(prefix="omni-metal-benchmark-")
        if args.model is None
        else nullcontext(str(args.model))
    )
    parameter_count = None
    with directory_context as directory:
        if args.model is None:
            model = model_cls(config).to(torch.bfloat16).eval()
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            model.save_pretrained(directory)
            del model
        llm = LLM(
            model=str(Path(directory)),
            worker_cls="vllm_omni_metal.native_worker.MetalARWorker",
            worker_extension_cls="benchmark_native_optimizations.BenchmarkHooks",
            distributed_executor_backend="uni",
            dtype="bfloat16",
            enforce_eager=True,
            compilation_config={"custom_ops": ["none", "+silu_and_mul", "-rotary_embedding"]},
            enable_prefix_caching=False,
            async_scheduling=False,
            skip_tokenizer_init=True,
            max_model_len=384,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=min(1024, 384 * args.batch_size),
            num_gpu_blocks_override=128,
        )
        prepared = False
        try:
            llm.collective_rpc("prepare_benchmark", args=(args.compile_for_test,), timeout=60)
            prepared = True
            for iteration in range(args.iterations + 2):
                order = modes if iteration % 2 == 0 else tuple(reversed(modes))
                for mode in order:
                    if psutil.virtual_memory().available < 3 * 1024**3:
                        raise MemoryError("benchmark stopped at 3 GiB system headroom")
                    llm.collective_rpc("select_benchmark_mode", args=(mode,), timeout=30)
                    start = perf_counter()
                    output = llm.generate(prompts, params, use_tqdm=False)
                    elapsed_ms = (perf_counter() - start) * 1000
                    tokens = [list(result.outputs[0].token_ids) for result in output]
                    if reference_tokens is None:
                        reference_tokens = tokens
                    if tokens != reference_tokens:
                        raise AssertionError(f"{mode} changed greedy output")
                    if any(len(row) != args.output_tokens for row in tokens):
                        raise AssertionError("generation did not produce every requested token")
                    if iteration >= 2:
                        samples[mode].append(elapsed_ms)
                    (calls[mode],) = llm.collective_rpc("benchmark_evidence", timeout=30)
                    activation_enabled = mode in ("activation", "combined")
                    if calls[mode]["silu"] and not activation_enabled:
                        raise AssertionError(f"{mode} unexpectedly selected fused activation")
                    # Separate short requests need not share a prefill batch.
                    if (
                        activation_enabled
                        and args.prompt_tokens * text_config.intermediate_size >= 262144
                        and not calls[mode]["silu"]
                    ):
                        raise AssertionError(f"{mode} did not exercise eligible activation")
                    if bool(calls[mode]["norm"]) != (mode in ("normalization", "combined")):
                        raise AssertionError(
                            f"{mode} normalization route did not match the workload"
                        )
            seeded_params = SamplingParams(
                temperature=0.7,
                seed=42,
                max_tokens=args.output_tokens,
                ignore_eos=True,
                detokenize=False,
            )
            for mode in modes:
                llm.collective_rpc("select_benchmark_mode", args=(mode,), timeout=30)
                output = llm.generate(prompts, seeded_params, use_tqdm=False)
                tokens = [list(result.outputs[0].token_ids) for result in output]
                if seeded_tokens is None:
                    seeded_tokens = tokens
                if tokens != seeded_tokens:
                    raise AssertionError(f"{mode} changed seeded output")
        finally:
            try:
                if prepared:
                    llm.collective_rpc("restore_benchmark", timeout=30)
            finally:
                llm.llm_engine.engine_core.shutdown()
    baseline = statistics.median(samples["baseline"])
    print(
        json.dumps(
            {
                "architecture": config.architectures[0]
                if config.architectures
                else args.architecture,
                "weights": "local_random_not_pretrained"
                if args.model is None
                else "local_pretrained",
                "parameter_count": parameter_count,
                "checkpoint_weight_bytes": weight_bytes if args.model is not None else None,
                "dtype": "bfloat16",
                "batch_size": args.batch_size,
                "prompt_tokens_per_request": args.prompt_tokens,
                "output_tokens_per_request": args.output_tokens,
                "iterations": args.iterations,
                "warmup_rounds": 2,
                "balanced_order": True,
                "test_only_compilation": args.compile_for_test,
                "available_gib_before": available_before,
                "greedy_and_seeded_outputs_equal": True,
                "results": {
                    mode: {
                        "median_ms": statistics.median(values),
                        "min_ms": min(values),
                        "max_ms": max(values),
                        "samples_ms": values,
                        "speedup_over_baseline": baseline / statistics.median(values),
                        "paired_speedup_median": statistics.median(
                            baseline_ms / candidate_ms
                            for baseline_ms, candidate_ms in zip(
                                samples["baseline"], values, strict=True
                            )
                        ),
                        "kernel_calls_per_generation": calls[mode],
                    }
                    for mode, values in samples.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
