# vpipe generic elementwise ports

Source: <https://github.com/tgo-app-dev/vpipe/tree/6a721d5c3b74ded70001f7c02121f14ae8f5321a>

License: Apache-2.0 (the same full license is distributed at this project's root).
Independent kernels from `gpu-kernels/metal/ops/llm_elementwise.metal` and
`gpu-kernels/metal/ops/rope.metal`, plus the standalone CoreML ANE emitter from
`apple-silicon/coreml/ane-emitter.cc`, are included. The emitter/Swift bridge is
adapted under `_native/ane/`; its manager-dependent self-test was removed and the
C ABI/host tool owns integrity checks. No bundled media/model components,
MLX steel headers, GPL libraries, or model-specific fusions are copied.

`vpipe_elementwise.metal` is **adapted**, not an unchanged source snapshot:

- SwiGLU: packed `[gate, up]` row indexing; SiLU casts to the input dtype before
  multiplying up, as vLLM's Torch operator does. The original final-only cast fails
  81 FP16 / 72 BF16 positions in the 257-element discriminating sample.
- Cached NeoX RoPE: each product casts to input dtype before addition/subtraction.
  Frequencies are never recomputed; upstream cache/scaling remains authoritative.
  Partial rotary dimensions preserve the unrotated tail and input storage.

Source tests: upstream `tests/unit-tests/mlp-fuse-ab.cc`, Metal LM attention/decode
checks. No public GitHub Actions configuration was found at the pinned revision;
this is not a claim that all vpipe kernels or upstream CI passed.

Local tests compile isolated shaders explicitly. Production workers **only load**
private prebuilt `.metallib` artifacts. Run
`python -m vllm_omni_metal.vpipe_kernels` explicitly to build using an already
installed Apple Metal compiler. No compiler/tool download or worker JIT fallback.
Missing artifacts retain the original paths; invalid artifacts fail explicitly.

Not ported: model-specific Gemma post-norm residual addition, frequency-recomputing
RoPE, projection/gate packing, sliding ring KV writes, quantization, video attention,
model-specific ANE worker/controller or online scheduler,
and M5-only NAX kernels. They do not match this plugin's current contracts/hardware.
