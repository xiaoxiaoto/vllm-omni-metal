# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams

from vllm_omni_metal import is_available
from vllm_omni_metal.native_runner import MetalNativeModelRunner


class PlainLanguageModel(nn.Module):
    """No Omni kwargs: exercises the ordinary vLLM language model signature."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        self.inputs.append((input_ids.tolist(), positions.tolist()))
        return torch.nn.functional.one_hot((input_ids + 1) % 16, 16).float()

    def compute_logits(self, hidden):
        return hidden * 10


def make_output(new=(), counts=None, cached=(), computed=(), finished=()):
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        num_scheduled_tokens=counts or {},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(cached),
            num_computed_tokens=list(computed),
            new_block_ids=[None] * len(cached),
            resumed_req_ids=set(),
        ),
        finished_req_ids=set(finished),
    )


def request(rid, tokens, **sampling):
    return SimpleNamespace(
        req_id=rid,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(temperature=0, detokenize=False, **sampling),
        num_computed_tokens=0,
        block_ids=(),
        mm_features=[],
    )


@pytest.fixture
def runner():
    config = VllmConfig()
    runner = MetalNativeModelRunner(config, torch.device("cpu"))
    runner.model = PlainLanguageModel()
    yield runner
    runner.shutdown_omni_connectors()


def test_standard_signature_and_cached_decode(runner):
    result = runner.execute_model(make_output([request("a", [1, 2])], {"a": 2}))
    assert result.sampled_token_ids == [[3]]
    result = runner.execute_model(make_output(counts={"a": 1}, cached=["a"], computed=[2]))
    assert result.sampled_token_ids == [[4]]
    assert runner.model.inputs == [([1, 2], [0, 1]), ([3], [2])]
    runner.execute_model(make_output(finished=["a"]))
    assert not runner.requests and not runner.model_intermediate_buffer


def test_batch_rows_are_not_mixed(runner):
    result = runner.execute_model(
        make_output([request("a", [1, 2]), request("b", [7])], {"a": 2, "b": 1})
    )
    assert result.req_ids == ["a", "b"]
    assert result.sampled_token_ids == [[3], [8]]


def test_allowed_tokens_and_bias_are_not_ignored(runner):
    result = runner.execute_model(
        make_output([request("a", [2], allowed_token_ids=[9], logit_bias={9: 20})], {"a": 1})
    )
    assert result.sampled_token_ids == [[9]]


def test_min_tokens_stop_mask_is_not_ignored(runner):
    result = runner.execute_model(
        make_output([request("a", [2], stop_token_ids=[3], min_tokens=2, max_tokens=4)], {"a": 1})
    )
    assert result.sampled_token_ids != [[3]]


def test_preprocess_postprocess_and_hidden_payload_use_request_spans(runner):
    class HookedModel(PlainLanguageModel):
        def preprocess_batch(self, req_ids, model_intermediate_buffer, device):
            for rid in req_ids:
                model_intermediate_buffer[rid]["batch_hook"] = True

        def preprocess(self, input_ids, input_embeds, **info):
            assert info["_omni_is_prefill"] and info["batch_hook"]
            embeddings = torch.nn.functional.one_hot(input_ids % 16, 16).float()
            return input_ids, embeddings, {"runtime": {"seen": len(input_ids)}}

        def postprocess(self, hidden, **info):
            assert len(hidden) == info["runtime"]["seen"]
            return {"post": {"rows": len(hidden)}}

    runner.model = HookedModel()
    result = runner.execute_model(
        make_output([request("a", [1, 2]), request("b", [7])], {"a": 2, "b": 1})
    )
    assert runner.model_intermediate_buffer["a"]["post"]["rows"] == 2
    assert runner.model_intermediate_buffer["b"]["post"]["rows"] == 1
    assert [len(payload["hidden_states"]) for payload in result.inter_stage_outputs] == [2, 1]


def test_payload_nonfinite_values_rejected_before_transport():
    with pytest.raises(ValueError, match="non-finite"):
        MetalNativeModelRunner._validate_payload({"values": [torch.tensor([float("nan")])]})


def test_prepare_runner_inputs_receives_request_local_sampling_and_discard_mask(runner):
    class PreparedModel(PlainLanguageModel):
        accepts_runner_sampling_extra_args = True

        def prepare_runner_inputs(self, **kwargs):
            self.prepared = kwargs
            return kwargs["input_ids"], kwargs["positions"]

    runner.model = PreparedModel()
    runner.execute_model(make_output([request("a", [1, 2, 3], extra_args={"seed": 83})], {"a": 2}))
    assert runner.model.prepared["sampling_extra_args"] == [{"seed": 83}]
    assert runner.model.prepared["discard_mask"] == [True]
    runner.execute_model(make_output(counts={"a": 1}, cached=["a"], computed=[2]))
    assert runner.model.prepared["num_computed_tokens"] == [2]
    assert runner.model.prepared["discard_mask"] == [False]


def test_prepare_runner_inputs_cannot_change_scheduled_shape(runner):
    class InvalidModel(PlainLanguageModel):
        def prepare_runner_inputs(self, **kwargs):
            return kwargs["input_ids"][:0], kwargs["positions"]

    runner.model = InvalidModel()
    with pytest.raises(ValueError, match="scheduled token shape"):
        runner.execute_model(make_output([request("a", [1])], {"a": 1}))


class SparseModel(PlainLanguageModel):
    prefer_model_sampler = True

    def make_omni_output(self, output, **kwargs):
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        self.payload = {
            "model_outputs": [],
            "sr": torch.tensor([48000]),
            "meta": {"req_id": [], "sparse_audio": ["1"]},
        }
        return OmniOutput(text_hidden_states=output, multimodal_outputs=self.payload)

    def sample(self, logits, sampling_metadata):
        from vllm.v1.outputs import SamplerOutput

        # Reverse sparse order; batch indexing must not swap audio between requests.
        self.payload["meta"]["req_id"] = ["b", "a"]
        self.payload["model_outputs"] = [torch.tensor([7.0]), torch.tensor([2.0])]
        return SamplerOutput(sampled_token_ids=logits.argmax(-1)[:, None], logprobs_tensors=None)


def test_terminal_audio_from_model_sampler_is_routed_by_request_id(runner):
    runner.model = SparseModel()
    result = runner.execute_model(
        make_output([request("a", [1]), request("b", [6])], {"a": 1, "b": 1})
    )
    payloads = result.inter_stage_outputs
    assert payloads[0]["model_outputs"].tolist() == [2.0]
    assert payloads[1]["model_outputs"].tolist() == [7.0]
    assert all("hidden_states" not in payload for payload in payloads)


@pytest.mark.parametrize("case", ["empty", "subset", "duplicate", "unknown", "length", "marker"])
def test_sparse_output_validation_and_empty_steps(runner, case):
    class SparseCase(SparseModel):
        def sample(self, logits, sampling_metadata):
            result = super().sample(logits, sampling_metadata)
            meta = self.payload["meta"]
            if case == "empty":
                meta["req_id"] = []
                self.payload["model_outputs"] = []
            elif case == "subset":
                meta["req_id"] = ["b"]
                self.payload["model_outputs"] = [torch.tensor([7.0])]
            elif case == "duplicate":
                meta["req_id"] = ["b", "b"]
            elif case == "unknown":
                meta["req_id"] = ["b", "other"]
            elif case == "length":
                self.payload["model_outputs"] = []
            elif case == "marker":
                meta["sparse_audio"] = torch.tensor(1)
            return result

    runner.model = SparseCase()
    scheduled = make_output([request("a", [1]), request("b", [6])], {"a": 1, "b": 1})
    if case in ("empty", "subset"):
        output = runner.execute_model(scheduled)
        assert output.inter_stage_outputs[0] == {}
        if case == "empty":
            assert output.inter_stage_outputs[1] == {}
        else:
            assert output.inter_stage_outputs[1]["model_outputs"].tolist() == [7.0]
    else:
        with pytest.raises(ValueError):
            runner.execute_model(scheduled)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("VLLM_OMNI_METAL_INTEGRATION") != "1" or not is_available(),
    reason="requires Apple Silicon and explicit integration opt-in",
)
def test_prepare_and_sparse_sampler_on_default_mps_device(runner):
    class PreparedSparseModel(SparseModel):
        accepts_runner_sampling_extra_args = True

        def prepare_runner_inputs(self, **kwargs):
            assert kwargs["input_ids"].device.type == "mps"
            assert kwargs["sampling_extra_args"] == [{"seed": 83}, {"seed": 42}]
            return kwargs["input_ids"], kwargs["positions"]

    runner.device = torch.device("mps")
    runner.model = PreparedSparseModel()
    result = runner.execute_model(
        make_output(
            [
                request("a", [1], extra_args={"seed": 83}),
                request("b", [6], extra_args={"seed": 42}),
            ],
            {"a": 1, "b": 1},
        )
    )
    assert result.sampled_token_ids == [[2], [7]]
    assert result.inter_stage_outputs[0]["model_outputs"].tolist() == [2.0]
    assert result.inter_stage_outputs[1]["model_outputs"].tolist() == [7.0]


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("VLLM_OMNI_METAL_INTEGRATION") != "1" or not is_available(),
    reason="requires Apple Silicon and explicit integration opt-in",
)
@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_two_native_language_architectures_on_mps(tmp_path, architecture):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from vllm import LLM

    configs = {"llama": (LlamaConfig, LlamaForCausalLM), "qwen3": (Qwen3Config, Qwen3ForCausalLM)}
    config_cls, model_cls = configs[architecture]
    torch.manual_seed(83)
    config = config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    reference = model_cls(config).eval()
    reference.save_pretrained(tmp_path)
    prompts = [[3, 4, 5], [9, 10]]
    expected = []
    with torch.inference_mode():
        for prompt in prompts:
            tokens = list(prompt)
            for _ in range(6):
                logits = reference(torch.tensor([tokens])).logits[0, -1]
                tokens.append(int(logits.argmax()))
            expected.append(tokens[len(prompt) :])
    llm = LLM(
        model=str(tmp_path),
        worker_cls="vllm_omni_metal.native_worker.MetalARWorker",
        distributed_executor_backend="uni",
        dtype="float32",
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        skip_tokenizer_init=True,
        max_model_len=64,
        max_num_seqs=2,
        num_gpu_blocks_override=32,
    )
    try:
        output = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts],
            SamplingParams(temperature=0, max_tokens=6, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert [result.outputs[0].token_ids for result in output] == expected
        params = SamplingParams(
            temperature=0.7, seed=42, max_tokens=6, ignore_eos=True, detokenize=False
        )
        batch = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts], params, use_tqdm=False
        )
        solo = llm.generate({"prompt_token_ids": prompts[0]}, params, use_tqdm=False)
        assert batch[0].outputs[0].token_ids == solo[0].outputs[0].token_ids
    finally:
        llm.llm_engine.engine_core.shutdown()
