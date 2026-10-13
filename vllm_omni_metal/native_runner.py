# SPDX-License-Identifier: Apache-2.0
"""Native scheduler-driven Torch runner; model implementations remain upstream."""

import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.utils.mm_outputs import build_mm_cpu, partition_payload_list, snapshot_mm_payload
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin
from vllm_omni.worker.output.payload_build import build_omni_mm_payload
from vllm_omni.worker.sampling_utils import call_model_sampler
from vllm_omni.worker.sparse_audio import resolve_sparse_mm_req_ids

from vllm_omni_metal.native_attention import MetalAttentionMetadata
from vllm_omni_metal.native_compat import adapt_mps_capture_queries


@dataclass
class NativeRequestState:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: object
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    generator: object = None
    additional_information_cpu: dict = field(default_factory=dict)
    external_req_id: str | None = None
    finished: bool = False

    @property
    def request_id(self):
        return self.req_id

    def is_finished(self):
        return self.finished


class MetalNativeModelRunner(OmniConnectorModelRunnerMixin):
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.model = None
        self.kv_group_ids = {}
        self.requests = {}
        self.model_intermediate_buffer = {}
        self.input_batch = SimpleNamespace(req_ids=[], req_id_to_index={})
        self.init_omni_connectors(self.model_config)
        self.sampler = Sampler()
        self.is_generation = getattr(self.model_config, "worker_type", None) == "generation"

    def load_model(self):
        with set_current_vllm_config(self.vllm_config):
            self.model = get_model(vllm_config=self.vllm_config)
        if any(p.device.type != self.device.type for p in self.model.parameters()):
            raise RuntimeError("native model weights were not loaded onto the target device")
        if self.device.type == "mps":
            adapt_mps_capture_queries(self.model)

    def _output(self, ids, sampled, payloads):
        inter_stage, multimodal = (
            partition_payload_list(payloads) if self._async_chunk else (payloads, payloads)
        )
        result = OmniModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
            multimodal_outputs=multimodal,
            inter_stage_outputs=inter_stage,
            num_nans_in_logits={},
        )
        result.omni_connector_output = self.get_omni_connector_output()
        return result

    def _update_states(self, scheduler_output):
        finished = scheduler_output.finished_req_ids
        for rid in finished:
            state = self.requests.get(rid)
            if state is not None:
                state.finished = True
        self.flush_full_payload_outputs(finished)
        on_finished = getattr(self.model, "on_requests_finished", None)
        if finished and callable(on_finished):
            on_finished(finished)
        for rid in finished:
            self.cleanup_finished_request(rid)
            self.requests.pop(rid, None)
            self.model_intermediate_buffer.pop(rid, None)
        for data in scheduler_output.scheduled_new_reqs:
            params = data.sampling_params
            if data.mm_features or getattr(data, "prompt_embeds", None) is not None:
                raise NotImplementedError(
                    "native runner does not yet support encoder multimodal inputs"
                )
            if getattr(data, "lora_request", None) is not None:
                raise NotImplementedError("native Metal runner does not support LoRA")
            if params is not None and (
                params.logprobs is not None or params.prompt_logprobs is not None
            ):
                raise NotImplementedError("native Metal logprobs are not yet supported")
            if params is not None and params.structured_outputs is not None:
                raise NotImplementedError("native Metal structured decoding is not yet supported")
            state = NativeRequestState(
                data.req_id,
                list(data.prompt_token_ids or []),
                params,
                data.block_ids,
                data.num_computed_tokens,
            )
            state.external_req_id = getattr(data, "external_req_id", None) or data.req_id
            if params is not None and params.seed is not None:
                state.generator = torch.Generator(device=self.device).manual_seed(params.seed)
            self.requests[data.req_id] = state
            info = (
                deserialize_additional_information(getattr(data, "additional_information", None))
                or {}
            )
            self.model_intermediate_buffer[data.req_id] = info
            state.additional_information_cpu = info
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            state = self.requests[rid]
            state.num_computed_tokens = cached.num_computed_tokens[i]
            prompt = getattr(cached, "prompt_token_ids", {}).get(rid)
            if prompt is not None:
                state.prompt_token_ids = list(prompt)
            info = getattr(cached, "additional_information", {}).get(rid)
            if info is not None:
                self.model_intermediate_buffer[rid] = deserialize_additional_information(info)
                state.additional_information_cpu = self.model_intermediate_buffer[rid]
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                if rid in cached.resumed_req_ids:
                    state.block_ids = new_blocks
                else:
                    state.block_ids = tuple(
                        a + b for a, b in zip(state.block_ids, new_blocks, strict=True)
                    )
        if self._async_chunk:
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_chunk()
        self.recv_full_payload_inputs(scheduler_output)
        with self._lock:
            local_payloads = {
                rid: dict(payload) for rid, payload in self._local_stage_payload_cache.items()
            }
        for rid, payload in local_payloads.items():
            if rid in self.requests:
                self.model_intermediate_buffer[rid] = payload
                self.requests[rid].additional_information_cpu = payload

    def _sampling_metadata(self, ids, vocab_size):
        params = [self.requests[rid].sampling_params for rid in ids]

        def tensor(values):
            return torch.tensor(values, dtype=torch.float32, device=self.device)

        histories = [self.requests[rid].output_token_ids for rid in ids]
        no_penalties = all(
            p.frequency_penalty == 0 and p.presence_penalty == 0 and p.repetition_penalty == 1
            for p in params
        )
        prompts = None
        if not no_penalties:
            prompt_lists = [self.requests[rid].prompt_token_ids for rid in ids]
            prompts = torch.full(
                (len(ids), max(map(len, prompt_lists))),
                vocab_size,
                dtype=torch.long,
                device=self.device,
            )
            for i, prompt in enumerate(prompt_lists):
                prompts[i, : len(prompt)] = torch.tensor(prompt, device=self.device).clamp(
                    max=vocab_size
                )
        allowed = None
        if any(p.allowed_token_ids is not None for p in params):
            allowed = torch.zeros((len(ids), vocab_size), dtype=torch.bool, device=self.device)
            for i, p in enumerate(params):
                if p.allowed_token_ids is not None:
                    if any(token < 0 or token >= vocab_size for token in p.allowed_token_ids):
                        raise ValueError("allowed token outside model vocabulary")
                    allowed[i].fill_(True)
                    allowed[i, p.allowed_token_ids] = False
        return SamplingMetadata(
            temperature=tensor([p.temperature for p in params]),
            all_greedy=all(p.temperature == 0 for p in params),
            all_random=all(p.temperature > 0 for p in params),
            top_p=tensor([p.top_p for p in params]),
            top_k=torch.tensor(
                [vocab_size if p.top_k <= 0 else min(p.top_k, vocab_size) for p in params],
                dtype=torch.int32,
                device=self.device,
            ),
            generators={
                i: self.requests[rid].generator
                for i, rid in enumerate(ids)
                if self.requests[rid].generator is not None
            },
            max_num_logprobs=None,
            no_penalties=no_penalties,
            prompt_token_ids=prompts,
            frequency_penalties=tensor([p.frequency_penalty for p in params]),
            presence_penalties=tensor([p.presence_penalty for p in params]),
            repetition_penalties=tensor([p.repetition_penalty for p in params]),
            output_token_ids=histories,
            allowed_token_ids_mask=allowed,
            bad_words_token_ids={
                i: p.bad_words_token_ids for i, p in enumerate(params) if p.bad_words_token_ids
            },
            logitsprocs=LogitsProcessors(),
        )

    @staticmethod
    def _validate_payload(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise ValueError("native model emitted non-finite payload")
        elif isinstance(value, dict):
            for item in value.values():
                MetalNativeModelRunner._validate_payload(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                MetalNativeModelRunner._validate_payload(item)

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        self._update_states(scheduler_output)
        ids = [rid for rid, count in scheduler_output.num_scheduled_tokens.items() if count > 0]
        if not ids:
            return self._output([], [], [])
        self.input_batch.req_ids = ids
        self.input_batch.req_id_to_index = {rid: i for i, rid in enumerate(ids)}
        tokens, positions, starts, lengths = [], [], [0], []
        for rid in ids:
            state = self.requests[rid]
            count = scheduler_output.num_scheduled_tokens[rid]
            begin = state.num_computed_tokens
            history = state.prompt_token_ids + state.output_token_ids
            segment = history[begin : begin + count]
            if len(segment) != count:
                raise RuntimeError("scheduler scheduled tokens missing from request state")
            tokens.extend(segment)
            positions.extend(range(begin, begin + count))
            starts.append(len(tokens))
            lengths.append(begin + count)
        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device)
        query_starts = torch.tensor(starts, dtype=torch.int32, device=self.device)
        preprocess_batch = getattr(self.model, "preprocess_batch", None)
        if callable(preprocess_batch):
            with set_current_vllm_config(self.vllm_config):
                preprocess_batch(
                    req_ids=ids,
                    model_intermediate_buffer=self.model_intermediate_buffer,
                    device=self.device,
                )
        info = [self.model_intermediate_buffer[rid] for rid in ids]
        inputs_embeds = None
        preprocess = getattr(self.model, "preprocess", None)
        if callable(preprocess):
            embedding_rows = []
            with set_current_vllm_config(self.vllm_config):
                for i, rid in enumerate(ids):
                    begin, end = starts[i : i + 2]
                    state = self.requests[rid]
                    request_info = dict(info[i])
                    request_info.update(
                        _omni_is_prefill=state.num_computed_tokens < len(state.prompt_token_ids),
                        _omni_max_tokens=getattr(state.sampling_params, "max_tokens", None),
                        _omni_seed=getattr(state.sampling_params, "seed", None),
                    )
                    new_ids, embeds, updates = preprocess(
                        input_ids=input_ids[begin:end], input_embeds=None, **request_info
                    )
                    if len(new_ids) != end - begin or len(embeds) != end - begin:
                        raise ValueError("model preprocess changed scheduled token count")
                    if "mtp_inputs" in updates:
                        raise NotImplementedError(
                            "native Metal auxiliary talker-MTP execution is not yet supported"
                        )
                    input_ids[begin:end] = new_ids
                    embedding_rows.append(embeds)
                    for key, value in updates.items():
                        if isinstance(value, dict):
                            info[i].setdefault(key, {}).update(value)
                        else:
                            info[i][key] = value
            inputs_embeds = torch.cat(embedding_rows, dim=0)
        prepare = getattr(self.model, "prepare_runner_inputs", None)
        if callable(prepare):
            prepare_kwargs = dict(
                req_ids=ids,
                num_computed_tokens=[self.requests[rid].num_computed_tokens for rid in ids],
                num_scheduled_tokens=[b - a for a, b in zip(starts[:-1], starts[1:], strict=True)],
                input_ids=input_ids,
                positions=position_ids,
                inputs_embeds=inputs_embeds,
                input_ids_buffer=input_ids,
            )
            if getattr(self.model, "accepts_runner_sampling_extra_args", False):
                prepare_kwargs.update(
                    sampling_extra_args=[
                        self.requests[rid].sampling_params.extra_args or {} for rid in ids
                    ],
                    discard_mask=[
                        lengths[i] < len(self.requests[rid].prompt_token_ids)
                        for i, rid in enumerate(ids)
                    ],
                )
            with set_current_vllm_config(self.vllm_config):
                input_ids, position_ids = prepare(**prepare_kwargs)
            if any(
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != (len(tokens),)
                or tensor.device.type != self.device.type
                or (self.device.index is not None and tensor.device.index != self.device.index)
                or tensor.dtype not in (torch.int32, torch.int64)
                for tensor in (input_ids, position_ids)
            ):
                raise ValueError(
                    "model prepare_runner_inputs changed scheduled token shape/device/dtype"
                )
        kwargs = dict(
            input_ids=input_ids,
            positions=position_ids,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
            runtime_additional_information=info,
            model_intermediate_buffer=info,
            seq_token_counts=[b - a for a, b in zip(starts[:-1], starts[1:], strict=True)],
            omni_query_start_loc=query_starts,
            req_ids=ids,
        )
        update = getattr(self.model, "update_decode_step_metadata", None)
        if getattr(self.model, "supports_omni_decode_step_metadata", False) and callable(update):
            update(
                input_ids=input_ids,
                positions=position_ids,
                omni_query_start_loc=query_starts,
                req_ids=ids,
            )
        context = {
            name: MetalAttentionMetadata(
                starts, lengths, [self.requests[rid].block_ids[group] for rid in ids]
            )
            for name, group in self.kv_group_ids.items()
        }
        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(context, self.vllm_config, num_tokens=len(tokens)),
        ):
            signature = inspect.signature(self.model.forward)
            forward_kwargs = kwargs
            if not any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            ):
                forward_kwargs = {
                    key: value for key, value in kwargs.items() if key in signature.parameters
                }
            output = self.model(**forward_kwargs)
            if isinstance(output, OmniOutput):
                hidden = output.text_hidden_states
            else:
                hidden = output
            make_output = getattr(self.model, "make_omni_output", None)
            if callable(make_output) and not self.is_generation:
                # Model-owned sampling may append terminal audio to this step's payload.
                output = make_output(output, **kwargs)
            sampled = []
            if not self.is_generation:
                sample_hidden = hidden[
                    torch.tensor([end - 1 for end in starts[1:]], device=self.device)
                ]
                logits = self.model.compute_logits(sample_hidden)
                metadata = self._sampling_metadata(ids, logits.shape[-1])
                for i, rid in enumerate(ids):
                    state = self.requests[rid]
                    params = state.sampling_params
                    if params.logit_bias:
                        for token, bias in params.logit_bias.items():
                            if token < 0 or token >= logits.shape[-1]:
                                raise ValueError("logit bias token outside model vocabulary")
                            logits[i, token] += bias
                    if len(state.output_token_ids) < params.min_tokens:
                        stops = [
                            token
                            for token in params.all_stop_token_ids
                            if 0 <= token < logits.shape[-1]
                        ]
                        logits[i, stops] = float("-inf")
                self.input_batch.sampling_metadata = metadata
                if getattr(self.model, "prefer_model_sampler", False):
                    sample = call_model_sampler(
                        self.model,
                        self.model.sample,
                        logits,
                        metadata,
                        input_batch=self.input_batch,
                        requests=self.requests,
                    )
                else:
                    sample = self.sampler(logits, metadata)
                sampled = sample.sampled_token_ids.cpu().tolist()
                for rid, row in zip(ids, sampled, strict=True):
                    self.requests[rid].output_token_ids.extend(row)
            postprocess = getattr(self.model, "postprocess", None)
            if callable(postprocess):
                for i, rid in enumerate(ids):
                    updates = postprocess(
                        hidden[starts[i] : starts[i + 1]], **self.model_intermediate_buffer[rid]
                    )
                    for key, value in updates.items():
                        if isinstance(value, dict):
                            self.model_intermediate_buffer[rid].setdefault(key, {}).update(value)
                        else:
                            self.model_intermediate_buffer[rid][key] = value
        mm = output.multimodal_outputs if isinstance(output, OmniOutput) else None
        sparse_ids = resolve_sparse_mm_req_ids(mm)
        sparse = sparse_ids is not None
        sparse_index = {rid: i for i, rid in enumerate(sparse_ids or [])}
        if any(rid not in ids for rid in sparse_index):
            raise ValueError("sparse multimodal output refers to an unscheduled request")
        flat_mm = flatten_payload(mm) if mm else {}
        if sparse:
            for key, value in flat_mm.items():
                if key not in ("meta.req_id", "meta.sparse_audio") and isinstance(value, list):
                    if len(value) != len(sparse_ids):
                        raise ValueError(
                            "sparse multimodal payload length does not match request ids"
                        )
        mm_cpu = build_mm_cpu(flat_mm)
        payloads = []
        for i, rid in enumerate(ids):
            begin, end = starts[i : i + 2]
            payload = (
                {}
                if sparse and rid not in sparse_index
                else build_omni_mm_payload(
                    combined_multimodal_outputs=None,
                    mm_cpu=mm_cpu,
                    rid=rid,
                    idx=i,
                    start=begin,
                    end=end,
                    audio_sparse_output=sparse,
                    sparse_mm_index=sparse_index,
                    hidden_seq_len=len(tokens),
                    scheduled_seq_len=len(tokens),
                )
            )
            if (
                not sparse
                and not self.is_generation
                and getattr(self.model, "omni_pooler_payload_include_hidden", True)
            ):
                payload["hidden_states"] = hidden[begin:end].detach().cpu().contiguous()
            payloads.append(snapshot_mm_payload(payload))
        for payload in payloads:
            self._validate_payload(payload)
        if not self._async_chunk and self._should_accumulate_full_payload_output():
            for rid, payload in zip(ids, payloads, strict=True):
                self.accumulate_full_payload_output(rid, payload, self.requests[rid])
        return self._output(ids, sampled, payloads)
