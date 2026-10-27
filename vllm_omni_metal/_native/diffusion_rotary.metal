// SPDX-License-Identifier: Apache-2.0
// Adjacent FP32 complex pairs, NOT the split-half/rounded NeoX contract.
#include <metal_stdlib>
using namespace metal;
#ifndef DIFFUSION_ELT
#define DIFFUSION_ELT half
#endif
#pragma clang fp contract(off)

kernel void interleaved_rotary(
    const device DIFFUSION_ELT* x [[buffer(0)]],
    const device float* frequencies [[buffer(1)]],
    device DIFFUSION_ELT* out [[buffer(2)]],
    constant long* shape [[buffer(3)]],
    uint pair [[thread_position_in_grid]])
{
    const uint sequence = uint(shape[0]), heads = uint(shape[1]);
    const uint pairs = uint(shape[2]), total = uint(shape[3]);
    if (pair >= total) return;
    const uint token = pair / (heads * pairs);
    const uint batch = token / sequence;
    const uint frequency = ((shape[4] ? batch * sequence : 0) + token % sequence)
                           * pairs + pair % pairs;
    const float real = float(x[2 * pair]), imag = float(x[2 * pair + 1]);
    const float cosine = frequencies[2 * frequency], sine = frequencies[2 * frequency + 1];
    // No intermediate low-precision product casts and no fused multiply-add.
    out[2 * pair] = DIFFUSION_ELT(real * cosine - imag * sine);
    out[2 * pair + 1] = DIFFUSION_ELT(imag * cosine + real * sine);
}

// Independent implementation: rounded-residual AdaLN and paired QK RMS/RoPE.
// Fixed-size reduction scratch handles arbitrary widths, not a model width.
inline float diffusion_sum(float value, threadgroup float* scratch,
                           uint lane, uint simd, uint tid) {
    value = simd_sum(value);
    if (lane == 0) scratch[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = tid < 8 ? scratch[tid] : 0.0f;
    if (simd == 0) value = simd_sum(value);
    if (tid == 0) scratch[0] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

inline ulong diffusion_broadcast_index(ulong flat, constant long* meta, uint which) {
    const uint rank = uint(meta[0]);
    ulong offset = 0;
    for (uint axis = rank; axis > 0; --axis) {
        const ulong size = ulong(meta[axis]);
        offset += (flat % size) * ulong(meta[1 + rank + which * rank + axis - 1]);
        flat /= size;
    }
    return offset;
}

kernel void gated_residual_rms_adaln(
    const device DIFFUSION_ELT* residual [[buffer(0)]],
    const device DIFFUSION_ELT* branch [[buffer(1)]],
    const device DIFFUSION_ELT* gate [[buffer(2)]],
    const device DIFFUSION_ELT* weight [[buffer(3)]],
    const device DIFFUSION_ELT* shift [[buffer(4)]],
    const device DIFFUSION_ELT* scale [[buffer(5)]],
    device DIFFUSION_ELT* out [[buffer(6)]],
    device DIFFUSION_ELT* rounded [[buffer(7)]],
    constant long* meta [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[8];
    // 28 KiB row + 32-byte reduction scratch stays below 32 KiB for every dtype.
    threadgroup DIFFUSION_ELT staged[28672 / sizeof(DIFFUSION_ELT)];
    const uint width = uint(meta[meta[0]]);
    const ulong base = ulong(group.y) * width;
    float sum = 0;
    for (uint col = tid; col < width; col += 256) {
        const ulong index = base + col;
        const DIFFUSION_ELT value = DIFFUSION_ELT(
            float(residual[index]) + float(branch[index]) *
            float(gate[diffusion_broadcast_index(index, meta, 0)]));
        rounded[index] = value;
        staged[col] = value;
        sum += float(value) * float(value);
    }
    const float inverse = rsqrt(diffusion_sum(sum, scratch, lane, simd, tid) / width + eps);
    for (uint col = tid; col < width; col += 256) {
        const ulong index = base + col;
        const DIFFUSION_ELT norm = DIFFUSION_ELT(float(staged[col]) * inverse);
        const DIFFUSION_ELT affine = DIFFUSION_ELT(float(norm) * float(weight[col]));
        const DIFFUSION_ELT factor = DIFFUSION_ELT(
            1.0f + float(scale[diffusion_broadcast_index(index, meta, 2)]));
        const DIFFUSION_ELT product = DIFFUSION_ELT(float(affine) * float(factor));
        out[index] = DIFFUSION_ELT(
            float(product) + float(shift[diffusion_broadcast_index(index, meta, 1)]));
    }
}

kernel void qk_norm_rotary(
    const device DIFFUSION_ELT* query [[buffer(0)]],
    const device DIFFUSION_ELT* key [[buffer(1)]],
    const device DIFFUSION_ELT* qw [[buffer(2)]],
    const device DIFFUSION_ELT* kw [[buffer(3)]],
    const device float* frequencies [[buffer(4)]],
    device DIFFUSION_ELT* qout [[buffer(5)]],
    device DIFFUSION_ELT* kout [[buffer(6)]],
    constant long* meta [[buffer(7)]],
    constant float& qeps [[buffer(8)]],
    constant float& keps [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[8];
    const uint seq = uint(meta[0]), qheads = uint(meta[1]), kheads = uint(meta[2]);
    const uint width = uint(meta[3]), heads = qheads + kheads;
    const uint token = group.y / heads, head = group.y % heads;
    const bool isq = head < qheads, weighted = meta[isq ? 5 : 6] != 0;
    const device DIFFUSION_ELT* x = isq ? query : key;
    const device DIFFUSION_ELT* weight = isq ? qw : kw;
    device DIFFUSION_ELT* out = isq ? qout : kout;
    const ulong base = (ulong(token) * (isq ? qheads : kheads) +
                        (isq ? head : head - qheads)) * width;
    float sum = 0;
    for (uint col = tid; col < width; col += 256) {
        const float value = float(x[base + col]);
        sum += value * value;
    }
    const float inverse = rsqrt(
        diffusion_sum(sum, scratch, lane, simd, tid) / width + (isq ? qeps : keps));
    const ulong frequency_base = ulong(meta[4] ? token : token % seq) * (width / 2);
    for (uint pair = tid; pair < width / 2; pair += 256) {
        const uint col = 2 * pair;
        const DIFFUSION_ELT nr = DIFFUSION_ELT(float(x[base + col]) * inverse);
        const DIFFUSION_ELT ni = DIFFUSION_ELT(float(x[base + col + 1]) * inverse);
        const float real = weighted ? float(DIFFUSION_ELT(float(nr) * float(weight[col]))) : float(nr);
        const float imag = weighted ? float(DIFFUSION_ELT(float(ni) * float(weight[col + 1]))) : float(ni);
        const float cosine = frequencies[2 * (frequency_base + pair)];
        const float sine = frequencies[2 * (frequency_base + pair) + 1];
        out[base + col] = DIFFUSION_ELT(real * cosine - imag * sine);
        out[base + col + 1] = DIFFUSION_ELT(imag * cosine + real * sine);
    }
}
