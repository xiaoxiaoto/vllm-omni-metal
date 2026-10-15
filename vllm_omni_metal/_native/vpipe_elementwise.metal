// SPDX-License-Identifier: Apache-2.0
// Adapted from tgo-app-dev/vpipe 6a721d5c3b74ded70001f7c02121f14ae8f5321a.
// ops/llm_elementwise.metal: packed input stride and cast-before-up for Torch.
// ops/rope.metal: preserve each Torch product rounding before add/subtract.
#include <metal_stdlib>
using namespace metal;
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

inline void swiglu_generic_(const device VPIPE_ELT* gate,
                            const device VPIPE_ELT* up, device VPIPE_ELT* out,
                            int total, int width, int out_stride, int out_off,
                            uint gid)
{
  if (gid >= (uint)total) { return; }
  const uint src = width > 0 ? (gid / width) * (2 * width) + gid % width : gid;
  const float g = float(gate[src]);
  const float silu = g / (1.0f + exp(-g));
  const VPIPE_ELT v = VPIPE_ELT(float(VPIPE_ELT(silu)) * float(up[src]));
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
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up   [[buffer(1)]],
    device VPIPE_ELT*       out   [[buffer(2)]],
    constant int&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  swiglu_generic_(gate, up, out, n, 0, 0, 0, gid);
}

// Packed input rows have stride 2*width; up points at the second half.
// 0:gate 1:up 2:out 3:rows 4:width 5:out_stride 6:out_off
kernel void swiglu_rs_f16(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    constant int&      rows [[buffer(3)]],
    constant int&      width [[buffer(4)]],
    constant int&      out_stride [[buffer(5)]],
    constant int&      out_off    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  swiglu_generic_(gate, up, out, rows * width, width, out_stride, out_off, gid);
}

kernel void rope_half_part_ftab_inplace_f16(
    device VPIPE_ELT*       x    [[buffer(0)]],
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
        VPIPE_ELT((d < half_r) ? (float(VPIPE_ELT(x1 * c)) - float(VPIPE_ELT(x2 * s))) : (float(VPIPE_ELT(x2 * c)) + float(VPIPE_ELT(x1 * s))));
  }
}

