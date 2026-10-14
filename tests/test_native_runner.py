# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs

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


def decode_wire(result):
    payloads = result.multimodal_outputs
    if payloads is None:
        payloads = [None] * len(result.req_ids)
    message = OmniEngineCoreOutputs(
        outputs=[
            OmniEngineCoreOutput(request_id=rid, new_token_ids=[], multimodal_output=payload)
            for rid, payload in zip(result.req_ids, payloads, strict=True)
        ]
    )
    return MsgpackDecoder(OmniEngineCoreOutputs).decode(MsgpackEncoder().encode(message))


def test_terminal_scalar_metadata_matches_tensor_wire_contract(runner):
    class ScalarMetadataModel(SparseModel):
        def sample(self, logits, sampling_metadata):
            result = super().sample(logits, sampling_metadata)
            self.payload["meta"].update(truncated=[0, 1], error=[0, 0])
            return result

    runner.model = ScalarMetadataModel()
    result = runner.execute_model(
        make_output([request("a", [1]), request("b", [6])], {"a": 1, "b": 1})
    )
    decoded = decode_wire(result)
    for output, audio, truncated in zip(decoded.outputs, [2.0, 7.0], [1, 0], strict=True):
        payload = output.multimodal_output
        assert all(isinstance(value, torch.Tensor) for value in payload.values())
        assert payload["model_outputs"].item() == audio
        assert payload["meta.truncated"].item() == truncated
        assert payload["meta.error"].item() == 0
    assert result.inter_stage_outputs[0]["meta.truncated"] == 1
    assert isinstance(result.inter_stage_outputs[0]["meta.truncated"], int)


def test_numeric_wire_normalization_preserves_inter_stage_types(runner, caplog):
    payload = {
        "model_outputs": torch.empty(2, 0),
        "sr": 48000,
        "meta.error": True,
        "gain": 0.5,
        "codes": [1, 2],
        "shape": (2, 0),
        "absent": None,
        "meta.label": "conditioning",
    }
    result = runner._output(["a"], [[]], [payload])
    received = decode_wire(result).outputs[0].multimodal_output
    assert tuple(received["model_outputs"].shape) == (2, 0)
    assert received["sr"].item() == 48000
    assert received["meta.error"].item() is True
    assert received["gain"].item() == 0.5
    assert received["codes"].tolist() == [1, 2]
    assert received["shape"].tolist() == [2, 0]
    assert "absent" not in received and "meta.label" not in received
    assert "meta.label" in caplog.text
    assert result.inter_stage_outputs == [payload]
    assert isinstance(payload["sr"], int)
    assert payload["meta.label"] == "conditioning"


def test_async_wire_normalization_keeps_partition_and_absent_requests(runner):
    runner._async_chunk = True
    payloads = [
        {"meta.label": "first"},
        {"model_outputs": torch.ones(2, 16), "sr": 48000, "meta.label": "second"},
    ]
    result = runner._output(["a", "b"], [[], []], payloads)
    decoded = decode_wire(result)
    assert decoded.outputs[0].multimodal_output is None
    assert decoded.outputs[1].multimodal_output["sr"].item() == 48000
    assert result.inter_stage_outputs == [{"meta.label": "first"}, {"meta.label": "second"}]
    result = runner._output(["a"], [[]], [{}])
    assert result.multimodal_outputs is None and result.inter_stage_outputs is None
    assert runner._output([], [], []).multimodal_outputs is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [float("nan")]])
def test_nonfinite_numeric_metadata_rejected_after_wire_normalization(runner, value):
    with pytest.raises(ValueError, match="non-finite"):
        runner._output(["a"], [[]], [{"value": value}])


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
@pytest.mark.parametrize("head_size", [8, 64])
def test_two_native_language_architectures_on_mps(tmp_path, architecture, head_size, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from vllm import LLM

    configs = {"llama": (LlamaConfig, LlamaForCausalLM), "qwen3": (Qwen3Config, Qwen3ForCausalLM)}
    config_cls, model_cls = configs[architecture]
    torch.manual_seed(83)
    config = config_cls(
        vocab_size=64,
        hidden_size=4 * head_size,
        intermediate_size=8 * head_size,
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
    if head_size == 64:
        sdpa = torch.nn.functional.scaled_dot_product_attention

        def forbid_mps_sdpa(query, *args, **kwargs):
            if query.device.type == "mps":
                raise AssertionError("supported geometry must use upstream paged kernels")
            return sdpa(query, *args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbid_mps_sdpa)
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
