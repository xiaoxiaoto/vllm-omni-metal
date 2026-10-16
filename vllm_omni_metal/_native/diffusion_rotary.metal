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
