// SPDX-License-Identifier: Apache-2.0
// Adapted from tgo-app-dev/vpipe 6a721d5c3b74ded70001f7c02121f14ae8f5321a.
// ops/llm_elementwise.metal: packed input stride and cast-before-up for Torch.
// ops/rope.metal: preserve each Torch product rounding before add/subtract.
// ops/rms_norm.metal: rms_norm_fast_f16 SIMD reduction, adapted to add BEFORE
// variance, preserve cast-before-weight, and return a separate rounded sum.
#include <metal_stdlib>
using namespace metal;
#ifndef ELEMENTWISE_ELT
#define ELEMENTWISE_ELT half
#endif

inline void swiglu_generic_(const device ELEMENTWISE_ELT* gate,
                            const device ELEMENTWISE_ELT* up, device ELEMENTWISE_ELT* out,
                            int total, int width, int out_stride, int out_off,
                            uint gid)
{
  if (gid >= (uint)total) { return; }
  const uint src = width > 0 ? (gid / width) * (2 * width) + gid % width : gid;
  const float g = float(gate[src]);
  const float silu = g / (1.0f + exp(-g));
  const ELEMENTWISE_ELT v = ELEMENTWISE_ELT(float(ELEMENTWISE_ELT(silu)) * float(up[src]));
  if (out_stride <= 0) {
    out[gid] = v;
  } else {
    const int row = (int)gid / width;
    const int col = (int)gid % width;
    out[(long)row * out_stride + out_off + col] = v;
  }
}

// Separate contiguous inputs; staged dtype rounding differs from the original.
kernel void swiglu_f16(
    const device ELEMENTWISE_ELT* gate [[buffer(0)]],
    const device ELEMENTWISE_ELT* up   [[buffer(1)]],
    device ELEMENTWISE_ELT*       out   [[buffer(2)]],
    constant int&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  swiglu_generic_(gate, up, out, n, 0, 0, 0, gid);
}

// Packed input rows have stride 2*width; up points at the second half.
// 0:gate 1:up 2:out 3:rows 4:width 5:out_stride 6:out_off
kernel void swiglu_rs_f16(
    const device ELEMENTWISE_ELT* gate [[buffer(0)]],
    const device ELEMENTWISE_ELT* up   [[buffer(1)]],
    device ELEMENTWISE_ELT*       out  [[buffer(2)]],
    constant int&      rows [[buffer(3)]],
    constant int&      width [[buffer(4)]],
    constant int&      out_stride [[buffer(5)]],
    constant int&      out_off    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  swiglu_generic_(gate, up, out, rows * width, width, out_stride, out_off, gid);
}

kernel void rope_half_part_ftab_inplace_f16(
    device ELEMENTWISE_ELT*       x    [[buffer(0)]],
    const device float*     cosb [[buffer(1)]],
    const device float*     sinb [[buffer(2)]],
    constant int&      H         [[buffer(3)]],
    constant int&      S        [[buffer(4)]],
    constant int&      D         [[buffer(5)]],
    constant int&      rot       [[buffer(6)]],
    constant int&      stride    [[buffer(7)]],
    constant int&      off       [[buffer(8)]],
    constant int&      head_stride [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  // Both halves of the pair are read before the barrier and the result
  // written after it, so `live` gates the work and never the barrier.
  const bool live = d < D && t < S && h < H && d < rot;
  const int half_r = rot / 2;
  const uint src = (uint)t * (uint)stride + (uint)off +
                   (uint)h * (uint)head_stride;
  float x1 = 0.0f, x2 = 0.0f, c = 0.0f, s = 0.0f;
  if (live) {
    const int i = (d < half_r) ? d : (d - half_r);
    const uint cb = (uint)t * (uint)half_r + (uint)i;
    c = cosb[cb];
    s = sinb[cb];
    x1 = float(x[src + (uint)(d < half_r ? d : d - half_r)]);
    x2 = float(x[src + (uint)(d < half_r ? d + half_r : d)]);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (live) {
    x[src + (uint)d] =
        ELEMENTWISE_ELT((d < half_r) ? (float(ELEMENTWISE_ELT(x1 * c)) - float(ELEMENTWISE_ELT(x2 * s))) : (float(ELEMENTWISE_ELT(x2 * c)) + float(ELEMENTWISE_ELT(x1 * s))));
  }
}

// Adapted from vllm-project/vllm-metal f2e26b342c93d6c705ebee0950a0baf8025ee46d,
// metal/kernels_v2/rope.metal (Apache-2.0). Cache rows are already index-validated
// by Torch; packed Q/K share one dispatch and need no input clones.
kernel void rope_qk_cached(
    const device ELEMENTWISE_ELT* q [[buffer(0)]],
    const device ELEMENTWISE_ELT* k [[buffer(1)]],
    const device ELEMENTWISE_ELT* cache [[buffer(2)]],
    device ELEMENTWISE_ELT* out_q [[buffer(3)]],
    device ELEMENTWISE_ELT* out_k [[buffer(4)]],
    constant long* shape [[buffer(5)]],
    uint2 index [[thread_position_in_grid]])
{
  const uint q_width = shape[0], k_width = shape[1], head = shape[2];
  const uint pair = index.x, row = index.y, half_head = head / 2;
  if (pair >= (q_width + k_width) / 2) return;
  const bool is_q = pair < q_width / 2;
  const uint local_pair = is_q ? pair : pair - q_width / 2;
  const uint lane = local_pair % half_head;
  const uint col = local_pair / half_head * head + lane;
  const device ELEMENTWISE_ELT* input = is_q ? q : k;
  device ELEMENTWISE_ELT* output = is_q ? out_q : out_k;
  const uint width = is_q ? q_width : k_width;
  const long base = long(row) * width + col;
  const long cache_row = long(row) * head;
  const float c = float(cache[cache_row + lane]);
  const float s = float(cache[cache_row + lane + half_head]);
  const float x = float(input[base]), y = float(input[base + half_head]);
  output[base] = ELEMENTWISE_ELT(float(ELEMENTWISE_ELT(x * c)) - float(ELEMENTWISE_ELT(y * s)));
  output[base + half_head] = ELEMENTWISE_ELT(float(ELEMENTWISE_ELT(y * c)) + float(ELEMENTWISE_ELT(x * s)));
}

// Adapted from rms_norm_fast_f16, NOT the semantically different rms_add_f16.
// One fixed-size threadgroup per row; width is independent of threadgroup size.
kernel void add_rms_norm_f16(
    const device ELEMENTWISE_ELT* x        [[buffer(0)]],
    const device ELEMENTWISE_ELT* residual [[buffer(1)]],
    const device ELEMENTWISE_ELT* weight   [[buffer(2)]],
    device ELEMENTWISE_ELT* out            [[buffer(3)]],
    device ELEMENTWISE_ELT* summed         [[buffer(4)]],
    constant int& H                 [[buffer(5)]],
    constant float& eps             [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint3 tptg     [[threads_per_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]])
{
  const uint base = tid.y * uint(H);
  const uint lid = ltid.x;
  const uint tg = tptg.x;
  const uint nsg = tg / 32;
  float acc = 0.0f;
  for (uint i = lid; i < uint(H); i += tg) {
    const float v = float(x[base + i]) + float(residual[base + i]);
    acc += v * v;
  }
  acc = simd_sum(acc);
  threadgroup float partial[32];
  if (simd_lid == 0) { partial[simd_gid] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < nsg) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(H) + eps);
  for (uint i = lid; i < uint(H); i += tg) {
    // Re-read the FP32 sum, never the rounded residual output.
    const float v = float(x[base + i]) + float(residual[base + i]);
    const ELEMENTWISE_ELT normalized = ELEMENTWISE_ELT(v * inv);
    out[base + i] = ELEMENTWISE_ELT(float(normalized) * float(weight[i]));
    summed[base + i] = ELEMENTWISE_ELT(v);
  }
}
