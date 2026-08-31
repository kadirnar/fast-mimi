"""Fused SEANet residual blocks (MimiResnetBlock) in exact fp32: one CUDA C++ launch per block instead of eight.

Stock block (fp32, causal, stride 1):  out = x + conv_k1( ELU( conv_k3( pad_left2( ELU(x) ) ) ) )
One CTA per (time tile of BT frames, batch item) holds all C input, H = C/2 hidden and C output channels:
  1. the x tile with its 2-frame left halo is staged into shared memory through the ELU once (pad = 0), loads
     batched so they stay in flight;
  2. conv1: every thread owns an (MJ = 4 hidden x 4 frames) micro-tile and accumulates fmaf in the reference's K
     order (input channel major, tap minor, from 0) from the shared x tile and W1 chunks (rows (ci, k), [j]
     contiguous) that stream through a double-buffered shared-memory pipeline with 16-byte cp.async;
     + b1, ELU, into shared h2[H][BT];
  3. conv2: (8 out x 4 frames) micro-tiles over j = 0..H-1 with W2 chunks ([j][o]) in the same pipeline,
     + b2, + x (residual), stored.
ELU is torch's formula x > 0 ? x : expm1(x); expm1 is a line-by-line port of libdevice's __nv_expm1f (the function
torch's CUDA ELU kernel evaluates), written with non-contracting intrinsics and exact bit-pattern constants, verified
bit-identical to torch.nn.functional.elu on a 36M-point sweep including denormals and infinities. The block output
is bit-identical to the stock module on every workload shape (the conv sums are fixed-order fp32 FMA chains, no
atomics -> deterministic).
Tiles per channel count come from a graph-replay sweep (_cfg); 512-channel blocks shorter than 600 frames keep the
incumbent chain (a time-only tiling cannot fill the machine there and the split-K convs are faster).
Padding: MimiConv1d(k=3, stride 1, causal) pads 2 on the left and 0 on the right (length preserved); the k=1 conv
pads nothing. Weights are prepacked once per block (W1 -> (3C, H) with row ci*3+k, W2 -> (H, C)).
"""
from __future__ import annotations

import types

import torch

_MOD = None
SUPPORTED = {64: 32, 128: 64, 256: 128, 512: 256}     # C -> H instantiations compiled below

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

// ---- libdevice __nv_expm1f, ported instruction by instruction (non-FTZ variants, as in torch's build) ----
__device__ __forceinline__ float ex2_approx(float a) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float expm1_dev(float x) {
    float n = rintf(__fmul_rn(x, __int_as_float(0x3FB8AA3B)));        // x * log2(e), round to nearest even
    if (fabsf(x) < __int_as_float(0x3ED1EB85)) n = 0.f;                // |x| < 0.41: no range reduction
    const float n2 = (n == 128.f) ? 127.f : n;
    const float e = ex2_approx(n2);
    float t = fmaf(-n, __int_as_float(0x3F317200), x);                 // x - n*ln2_hi
    t = fmaf(-n, __int_as_float(0x35BFBE8E), t);                       //   - n*ln2_lo
    float q = fmaf(__int_as_float(0x3AB5EBE6), t, __int_as_float(0x3C095663));
    q = fmaf(q, t, __int_as_float(0x3D2AABE3));
    q = fmaf(q, t, __int_as_float(0x3E2AA9F6));
    q = fmaf(q, t, __int_as_float(0x3EFFFFFE));
    const float p = fmaf(__fmul_rn(t, q), t, t);                       // expm1(t) ~ t + t^2 q
    float r = fmaf(p, e, __fadd_rn(e, -1.0f));                          // p * 2^n + (2^n - 1)
    r = (n == 128.f) ? __fadd_rn(r, r) : r;
    r = (n2 > 128.f) ? CUDART_INF_F : r;
    r = (n2 < -25.f) ? -1.0f : r;
    r = (x == 0.f) ? __fadd_rn(x, x) : r;
    return r;
}
__device__ __forceinline__ float elu_dev(float x) { return x > 0.f ? x : expm1_dev(x); }

// ---- fused residual block ----
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// one chunk of weight rows (contiguous in global, NF floats) -> smem, issued by all NTHR threads, 16 B each
template <int NF, int NTHR>
__device__ __forceinline__ void issue_chunk(float* dst, const float* __restrict__ src, int tid) {
    static_assert(NF % 4 == 0, "chunk must be a multiple of 4 floats");
    #pragma unroll
    for (int i = tid; i < NF / 4; i += NTHR) cp_async_16(dst + 4 * i, src + 4 * i);
    cp_async_commit();
}

// C input channels, H = C/2 hidden, BT frames per CTA, MJ hidden rows per thread (MO = 2 MJ output rows),
// NT frames per thread, CK input channels per W1 chunk, CJ hidden channels per W2 chunk (chunks double-buffered).
//
// NT sets how much FMA work each shared-memory load feeds, which is what the block is bound by: per input channel
// the thread issues 3 weight loads (one per tap) plus ceil((NT + 2) / 4) + 1 loads of its x window, and gets
// 3 * MJ * NT fused multiply-adds out of them.  At NT = 4, MJ = 4 that is 48 FMAs per 5 loads -- around 10 FMAs
// per shared-memory instruction, which caps the block near 30% of the FMA pipe however the weight chunks are
// staged (measured: growing CK / CJ does not help).  NT = 8 reuses each weight across twice as many frames for
// one extra x load: 96 FMAs per 6 loads.  The FMA order per output is unchanged (input channel major, tap minor,
// both ascending), so the result stays bit-identical to the stock module.
template <int C, int H, int BT, int MJ, int NT, int CK, int CJ, bool VEC, bool EOUT>
__global__ void __launch_bounds__((BT / NT) * (H / MJ))
resnet_block_kernel(const float* __restrict__ x, const float* __restrict__ w1, const float* __restrict__ b1,
                    const float* __restrict__ w2, const float* __restrict__ b2, float* __restrict__ out, int T) {
    constexpr int XS = BT + 4;                 // padded x row (frames t0-2 .. t0+BT+1)
    static_assert(NT % 4 == 0 && BT % NT == 0, "NT must be a multiple of 4 and divide BT");
    constexpr int TT = BT / NT;                // time tiles per CTA
    constexpr int MO = 2 * MJ;                 // C = 2H -> the same thread grid serves conv2
    constexpr int NTHR = TT * (H / MJ);        // threads per CTA
    constexpr int W1CH = CK * 3 * H;           // floats per W1 chunk (rows (ci, k) for CK channels)
    constexpr int W2CH = CJ * C;               // floats per W2 chunk (rows j for CJ hidden channels)
    constexpr int WBUF = (W1CH > W2CH ? W1CH : W2CH);
    constexpr int NCH1 = C / CK, NCH2 = H / CJ;
    static_assert(TT * (C / MO) == NTHR && NTHR % 32 == 0 && NTHR <= 1024, "thread tiling");
    static_assert(MJ % 4 == 0 && C == 2 * H && C % CK == 0 && H % CJ == 0 && WBUF % 4 == 0, "tiling");
    extern __shared__ __align__(16) float smem[];
    float* Xs = smem;                          // [C][XS]
    float* H2 = Xs + C * XS;                   // [H][BT]
    float* Wb = H2 + H * BT;                   // [2][WBUF] weight chunks
    const int tid = threadIdx.x;
    const int b = blockIdx.y;
    const int t0 = blockIdx.x * BT;
    const float* xb = x + (size_t)b * C * T;
    float* ob = out + (size_t)b * C * T;

    // 0. the first two W1 chunks start streaming right away
    issue_chunk<W1CH, NTHR>(Wb, w1, tid);
    if (NCH1 > 1) issue_chunk<W1CH, NTHR>(Wb + WBUF, w1 + W1CH, tid);

    // 1. stage ELU(x) with the causal halo (2 zero frames before t = 0); all loads of a batch are issued
    //    before any is consumed so that a thread keeps STG loads in flight (latency-bound otherwise)
    {
        constexpr int NEL = C * (BT + 2);
        constexpr int PER = (NEL + NTHR - 1) / NTHR;
        constexpr int STG = 8;
        #pragma unroll 1
        for (int p0 = 0; p0 < PER; p0 += STG) {
            float v[STG];
            #pragma unroll
            for (int q = 0; q < STG; ++q) {
                const int idx = tid + (p0 + q) * NTHR;
                const int c = idx / (BT + 2), i = idx - c * (BT + 2);
                const int t = t0 - 2 + i;
                v[q] = 0.f;
                if (idx < NEL && t >= 0 && t < T) v[q] = xb[(size_t)c * T + t];
            }
            #pragma unroll
            for (int q = 0; q < STG; ++q) {
                const int idx = tid + (p0 + q) * NTHR;
                const int c = idx / (BT + 2), i = idx - c * (BT + 2);
                const int t = t0 - 2 + i;
                if (idx < NEL) Xs[c * XS + i] = (t >= 0 && t < T) ? elu_dev(v[q]) : 0.f;
            }
        }
    }

    // 2. conv1 (k = 3, causal): every thread owns an (MJ x NT) tile, K order = (ci, k) ascending from 0
    const int ttile = tid % TT, jtile = tid / TT;
    const int tt = ttile * NT, j0 = jtile * MJ;
    float acc[MJ][NT];
    #pragma unroll
    for (int i = 0; i < MJ; ++i)
        #pragma unroll
        for (int n = 0; n < NT; ++n) acc[i][n] = 0.f;
    for (int ch = 0; ch < NCH1; ++ch) {
        if (ch + 1 < NCH1) cp_async_wait_1(); else cp_async_wait_0();
        __syncthreads();                                   // chunk ch landed (and Xs is complete at ch == 0)
        const float* wc = Wb + (ch & 1) * WBUF + j0;
        #pragma unroll 2
        for (int cc = 0; cc < CK; ++cc) {
            const int ci = ch * CK + cc;
            float xw[NT + 2];                              // frames tt .. tt + NT + 1 (the 3-tap window)
            #pragma unroll
            for (int q = 0; q < NT / 4; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(&Xs[ci * XS + tt + 4 * q]);
                xw[4 * q] = v.x; xw[4 * q + 1] = v.y; xw[4 * q + 2] = v.z; xw[4 * q + 3] = v.w;
            }
            {
                const float2 v = *reinterpret_cast<const float2*>(&Xs[ci * XS + tt + NT]);
                xw[NT] = v.x; xw[NT + 1] = v.y;
            }
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                const float* wr = wc + (cc * 3 + k) * H;
                #pragma unroll
                for (int i = 0; i < MJ; i += 4) {
                    const float4 w4 = *reinterpret_cast<const float4*>(wr + i);
                    #pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        acc[i][n] = fmaf(w4.x, xw[n + k], acc[i][n]);
                        acc[i + 1][n] = fmaf(w4.y, xw[n + k], acc[i + 1][n]);
                        acc[i + 2][n] = fmaf(w4.z, xw[n + k], acc[i + 2][n]);
                        acc[i + 3][n] = fmaf(w4.w, xw[n + k], acc[i + 3][n]);
                    }
                }
            }
        }
        __syncthreads();                                   // everyone is done with buffer ch & 1
        if (ch + 2 < NCH1) issue_chunk<W1CH, NTHR>(Wb + (ch & 1) * WBUF, w1 + (size_t)(ch + 2) * W1CH, tid);
    }
    // W2 chunks 0 and 1 stream while the hidden tile is finished
    issue_chunk<W2CH, NTHR>(Wb, w2, tid);
    if (NCH2 > 1) issue_chunk<W2CH, NTHR>(Wb + WBUF, w2 + W2CH, tid);
    #pragma unroll
    for (int i = 0; i < MJ; ++i) {
        const float bias = b1[j0 + i];
        #pragma unroll
        for (int q = 0; q < NT / 4; ++q) {
            float4 hv;
            hv.x = elu_dev(acc[i][4 * q] + bias);
            hv.y = elu_dev(acc[i][4 * q + 1] + bias);
            hv.z = elu_dev(acc[i][4 * q + 2] + bias);
            hv.w = elu_dev(acc[i][4 * q + 3] + bias);
            *reinterpret_cast<float4*>(&H2[(j0 + i) * BT + tt + 4 * q]) = hv;
        }
    }

    // 3. conv2 (k = 1) over j = 0..H-1, + b2 + residual
    const int o0 = jtile * MO;
    float acc2[MO][NT];
    #pragma unroll
    for (int m = 0; m < MO; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n) acc2[m][n] = 0.f;
    for (int ch = 0; ch < NCH2; ++ch) {
        if (ch + 1 < NCH2) cp_async_wait_1(); else cp_async_wait_0();
        __syncthreads();
        const float* wc = Wb + (ch & 1) * WBUF + o0;
        #pragma unroll 2
        for (int jj = 0; jj < CJ; ++jj) {
            const int j = ch * CJ + jj;
            float hv[NT];
            #pragma unroll
            for (int q = 0; q < NT / 4; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(&H2[j * BT + tt + 4 * q]);
                hv[4 * q] = v.x; hv[4 * q + 1] = v.y; hv[4 * q + 2] = v.z; hv[4 * q + 3] = v.w;
            }
            const float* wr = wc + jj * C;
            #pragma unroll
            for (int m = 0; m < MO; m += 4) {
                const float4 w4 = *reinterpret_cast<const float4*>(wr + m);
                #pragma unroll
                for (int n = 0; n < NT; ++n) {
                    acc2[m][n] = fmaf(w4.x, hv[n], acc2[m][n]);
                    acc2[m + 1][n] = fmaf(w4.y, hv[n], acc2[m + 1][n]);
                    acc2[m + 2][n] = fmaf(w4.z, hv[n], acc2[m + 2][n]);
                    acc2[m + 3][n] = fmaf(w4.w, hv[n], acc2[m + 3][n]);
                }
            }
        }
        __syncthreads();
        if (ch + 2 < NCH2) issue_chunk<W2CH, NTHR>(Wb + (ch & 1) * WBUF, w2 + (size_t)(ch + 2) * W2CH, tid);
    }
    const int t = t0 + tt;
    #pragma unroll
    for (int m = 0; m < MO; ++m) {
        const int o = o0 + m;
        const float bias = b2[o];
        const size_t off = (size_t)o * T + t;
        if (VEC && t + NT <= T) {
            #pragma unroll
            for (int q = 0; q < NT / 4; ++q) {
                const float4 xv = *reinterpret_cast<const float4*>(xb + off + 4 * q);
                float4 ov;
                ov.x = xv.x + (acc2[m][4 * q] + bias); ov.y = xv.y + (acc2[m][4 * q + 1] + bias);
                ov.z = xv.z + (acc2[m][4 * q + 2] + bias); ov.w = xv.w + (acc2[m][4 * q + 3] + bias);
                if (EOUT) { ov.x = elu_dev(ov.x); ov.y = elu_dev(ov.y); ov.z = elu_dev(ov.z); ov.w = elu_dev(ov.w); }
                *reinterpret_cast<float4*>(ob + off + 4 * q) = ov;
            }
        } else {
            #pragma unroll
            for (int n = 0; n < NT; ++n) if (t + n < T) {
                const float ov = xb[off + n] + (acc2[m][n] + bias);
                ob[off + n] = EOUT ? elu_dev(ov) : ov;
            }
        }
    }
}

template <int C, int H, int BT, int MJ, int NT, int CK, int CJ>
void launch(const torch::Tensor& x, const torch::Tensor& w1, const torch::Tensor& b1, const torch::Tensor& w2,
            const torch::Tensor& b2, torch::Tensor& out, int B, int T, bool eout) {
    constexpr int XS = BT + 4;
    constexpr int W1CH = CK * 3 * H, W2CH = CJ * C, WBUF = (W1CH > W2CH ? W1CH : W2CH);
    const size_t smem = (size_t)(C * XS + H * BT + 2 * WBUF) * sizeof(float);
    // EOUT folds the consumer's ELU into this epilogue: one elu_dev per output element, exactly the value the
    // standalone kernel would have written, and one whole read+write pass over the activation saved.
    void* ker[2][2] = {{(void*)resnet_block_kernel<C, H, BT, MJ, NT, CK, CJ, false, false>,
                        (void*)resnet_block_kernel<C, H, BT, MJ, NT, CK, CJ, false, true>},
                       {(void*)resnet_block_kernel<C, H, BT, MJ, NT, CK, CJ, true, false>,
                        (void*)resnet_block_kernel<C, H, BT, MJ, NT, CK, CJ, true, true>}};
    static bool attr = false;
    if (!attr) {
        for (int v = 0; v < 2; ++v)
            for (int e = 0; e < 2; ++e) cudaFuncSetAttribute(ker[v][e], cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = true;
    }
    constexpr int NTHR = (BT / NT) * (H / MJ);
    const dim3 grid((T + BT - 1) / BT, B);
    auto stream = c10::cuda::getCurrentCUDAStream();
    void* fn = ker[(T % NT == 0) ? 1 : 0][eout ? 1 : 0];
    float* xp = x.data_ptr<float>(); float* w1p = w1.data_ptr<float>(); float* b1p = b1.data_ptr<float>();
    float* w2p = w2.data_ptr<float>(); float* b2p = b2.data_ptr<float>(); float* op = out.data_ptr<float>();
    void* args[] = {&xp, &w1p, &b1p, &w2p, &b2p, &op, &T};
    cudaLaunchKernel(fn, grid, dim3(NTHR), args, smem, stream);
}

__global__ void elu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = elu_dev(x[i]);
}

}  // namespace

torch::Tensor resnet_block(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, int64_t cfg, bool eout) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous() && x.dim() == 3, "x must be contiguous fp32 (B, C, T)");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "16-byte alignment");
    const int B = (int)x.size(0), C = (int)x.size(1), T = (int)x.size(2);
    TORCH_CHECK(w1.is_contiguous() && w1.size(0) == 3 * C && w1.size(1) * 2 == C, "w1 must be (3C, H) with H = C/2");
    TORCH_CHECK(w2.is_contiguous() && w2.size(0) * 2 == C && w2.size(1) == C, "w2 must be (H, C)");
    auto out = torch::empty_like(x);
    // cfg selects the tile / weight-chunk variant (see _cfg in the Python module).  Larger CK / CJ give each
    // double-buffered weight chunk enough FMAs to cover its global-load latency, which is what the long-form
    // shapes (tens of thousands of CTAs, weights resident in L2) need; the short-form entries keep the incumbent.
    switch (C) {
        case 64:
            switch (cfg) {
                case 2: launch<64, 32, 64, 4, 4, 16, 16>(x, w1, b1, w2, b2, out, B, T, eout); break;
                case 3: launch<64, 32, 128, 4, 4, 16, 16>(x, w1, b1, w2, b2, out, B, T, eout); break;
                case 4: launch<64, 32, 128, 4, 8, 4, 8>(x, w1, b1, w2, b2, out, B, T, eout); break;
                default: launch<64, 32, 64, 4, 4, 4, 8>(x, w1, b1, w2, b2, out, B, T, eout); break;
            } break;
        case 128:
            switch (cfg) {
                case 1: launch<128, 64, 32, 4, 4, 8, 8>(x, w1, b1, w2, b2, out, B, T, eout); break;
                case 2: launch<128, 64, 48, 4, 4, 8, 16>(x, w1, b1, w2, b2, out, B, T, eout); break;
                case 3: launch<128, 64, 64, 4, 4, 8, 16>(x, w1, b1, w2, b2, out, B, T, eout); break;
                case 4: launch<128, 64, 64, 4, 8, 4, 8>(x, w1, b1, w2, b2, out, B, T, eout); break;
                default: launch<128, 64, 48, 4, 4, 4, 8>(x, w1, b1, w2, b2, out, B, T, eout); break;
            } break;
        case 256:
            switch (cfg) {
                case 1: launch<256, 128, 24, 4, 4, 2, 4>(x, w1, b1, w2, b2, out, B, T, eout); break;
                case 2: launch<256, 128, 32, 4, 4, 8, 8>(x, w1, b1, w2, b2, out, B, T, eout); break;
                case 4: launch<256, 128, 32, 4, 8, 4, 4>(x, w1, b1, w2, b2, out, B, T, eout); break;
                default: launch<256, 128, 16, 4, 4, 4, 4>(x, w1, b1, w2, b2, out, B, T, eout); break;
            } break;
        case 512:
            switch (cfg) {
                case 2: launch<512, 256, 16, 4, 4, 8, 4>(x, w1, b1, w2, b2, out, B, T, eout); break;
                case 4: launch<512, 256, 16, 4, 8, 4, 4>(x, w1, b1, w2, b2, out, B, T, eout); break;
                default: launch<512, 256, 16, 4, 4, 4, 4>(x, w1, b1, w2, b2, out, B, T, eout); break;
            } break;
        default: TORCH_CHECK(false, "unsupported channel count");
    }
    return out;
}

torch::Tensor elu_exact(torch::Tensor x) {
    auto y = torch::empty_like(x);
    const int n = (int)x.numel();
    elu_kernel<<<(n + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""
CPP_SRC = """
torch::Tensor resnet_block(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, int64_t cfg, bool eout);
torch::Tensor elu_exact(torch::Tensor x);
"""


def _ext():
    """Compile once (torch caches by source hash under .fast-kernel/build; -O3 only, no fast-math)."""
    global _MOD
    if _MOD is None:
        from .._compat import build_dir, ensure_cuda_home
        ensure_cuda_home()                      # must precede the cpp_extension import (it reads CUDA_HOME on import)
        from torch.utils.cpp_extension import load_inline
        build = build_dir(None) / "fk_resnet_block"
        build.mkdir(parents=True, exist_ok=True)
        _MOD = load_inline(name="fk_resnet_block", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                           functions=["resnet_block", "elu_exact"], extra_cuda_cflags=["-O3"], build_directory=str(build))
    return _MOD


def _cfg(C: int, T: int) -> int:
    """Tile variant per (channels, length) from a graph-replay sweep on the model's shapes; -1 = keep the stock chain
    (512 channels at short lengths: a time-only tiling cannot fill the machine, the split-K convs are faster).

    cfg 4 is the NT = 8 probe: twice the frames per thread, so each shared-memory load feeds twice the FMAs
    (10 -> 16 per load) for half the threads.  It loses by 1.6-1.8x at every width and length, with no register
    spilling (48-64 -> 96-106 registers, 0 bytes spilled), which says the block is bound by latency and traffic
    rather than by shared-memory issue rate: halving the resident threads costs more than the extra reuse wins.
    The same conclusion falls out of cfg 2/3, where larger weight chunks do not help either.  Kept so the
    measurement is reproducible; the selection stays on NT = 4."""
    if C == 64:
        return 0                               # BT=64,  W1 chunks of 4 channels
    if C == 128:
        return 0 if T >= 4000 else 1           # BT=48 / BT=32
    if C == 256:
        return -1                              # the conv chain measures faster at this width (30.7 vs 34.9 us at 1 s)
    if C == 512:
        return 0 if T >= 600 else -1           # BT=16 / stock chain
    return -1


def resnet_block_fp32(x: torch.Tensor, w1p: torch.Tensor, b1: torch.Tensor, w2p: torch.Tensor, b2: torch.Tensor, cfg: int,
                      elu_out: bool = False) -> torch.Tensor:
    """x: (B, C, T) fp32 contiguous; w1p: (3C, H) row ci*3+k; b1: (H,); w2p: (H, C); b2: (C,). Returns (B, C, T),
    or ELU of it when elu_out (folded into the kernel's epilogue: same value, one activation pass saved)."""
    return _ext().resnet_block(x, w1p, b1, w2p, b2, cfg, elu_out)


def _fused_forward(self, hidden_states, padding_cache=None):
    x = hidden_states
    elu_x = getattr(self, "_fk_elu_input", None)          # ELU(x) already emitted by the producer's epilogue (chain blocks)
    elu_out = getattr(self, "_fk_elu_out", False)          # the consumer wants ELU(block output) (chain blocks)
    self._fk_elu_input, self._fk_elu_out = None, False
    if (padding_cache is not None or not x.is_cuda or x.dtype != torch.float32 or x.dim() != 3
            or x.shape[1] != self._fk_c or x.shape[2] < 1):
        out = self._fk_forward(hidden_states, padding_cache)
        return torch.nn.functional.elu(out) if elu_out else out
    cfg = _cfg(x.shape[1], x.shape[2])
    if cfg < 0:
        return _chain_forward(self, x, elu_x, elu_out)
    return resnet_block_fp32(x.contiguous(), self._fk_w1, self._fk_b1, self._fk_w2, self._fk_b2, cfg, elu_out)


CHAIN_ELU_IN_LOAD = False       # (a) ELU in the convs' B loads: measured 2-3x slower GEMMs (breaks Triton's load pipelining)
CHAIN_ELU_OUT = True            # (a') the second ELU in the k3 conv's epilogue (once per output element, bit-exact expm1)
CHAIN_RESIDUAL_FUSED = True     # (b) residual add in the k1 conv's epilogue


def _chain_forward(self, x, elu_x=None, elu_out=False):
    """The conv chain of one block at widths/lengths where the fused kernel loses: ELU -> k3 conv -> ELU -> k1 conv + x,
    through the convs' tuned plans: the first ELU comes from the producer's epilogue when available (elu_x), the second
    from the k3 conv's epilogue, the residual add (and the consumer's ELU when elu_out) from the k1 conv's epilogue --
    the same fp32 FMAs in the same order as the separate launches; the inner MimiConv1d modules are untouched."""
    c1, c2 = self.block[1], self.block[3]
    p1, p2 = getattr(c1.conv, "_fk_plan", None), getattr(c2.conv, "_fk_plan", None)
    if p1 is None or p2 is None or not hasattr(c1, "_fk_padding_total"):
        out = self._fk_forward(x, None)
        return torch.nn.functional.elu(out) if elu_out else out
    x = x.contiguous()
    if elu_x is not None:
        h = p1(elu_x.contiguous(), c1._fk_padding_total, 0, elu_out=CHAIN_ELU_OUT)
    else:
        h = p1(x, c1._fk_padding_total, 0, elu=True, elu_in_load=CHAIN_ELU_IN_LOAD, elu_out=CHAIN_ELU_OUT)
    if CHAIN_RESIDUAL_FUSED:
        return p2(h, 0, 0, elu=not CHAIN_ELU_OUT, elu_in_load=CHAIN_ELU_IN_LOAD, residual=x, elu_out=elu_out)
    out = x + p2(h, 0, 0, elu=not CHAIN_ELU_OUT, elu_in_load=CHAIN_ELU_IN_LOAD)
    return torch.nn.functional.elu(out) if elu_out else out


def patch_resnet_blocks(model, ctx) -> dict:
    """Replace forward on every MimiResnetBlock of the plain form ELU-conv3-ELU-conv1 + identity shortcut."""
    from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiResnetBlock
    count = 0
    for name, m in model.named_modules():
        if not isinstance(m, MimiResnetBlock):
            continue
        ok = (len(m.block) == 4 and isinstance(m.block[0], torch.nn.ELU) and isinstance(m.block[2], torch.nn.ELU)
              and isinstance(m.block[1], MimiConv1d) and isinstance(m.block[3], MimiConv1d)
              and isinstance(m.shortcut, torch.nn.Identity)
              and m.block[0].alpha == 1.0 and m.block[2].alpha == 1.0)
        if ok:
            c1, c2 = m.block[1], m.block[3]
            H, C, _ = c1.conv.weight.shape
            ok = (c1.causal and c1.pad_mode == "constant" and int(c1.padding_total) == 2 and c1.conv.kernel_size == (3,)
                  and c1.conv.stride == (1,) and c1.conv.dilation == (1,) and c1.conv.groups == 1 and c1.conv.bias is not None
                  and c2.conv.kernel_size == (1,) and c2.conv.stride == (1,) and c2.conv.groups == 1 and c2.conv.bias is not None
                  and int(c2.padding_total) == 0 and SUPPORTED.get(C) == H and tuple(c2.conv.weight.shape) == (C, H, 1))
        if not ok:
            ctx.log(f"resnet block {name}: unexpected structure, left untouched")
            continue
        with torch.no_grad():
            m._fk_w1 = c1.conv.weight.detach().float().permute(1, 2, 0).reshape(3 * C, H).contiguous()   # [(ci, k)][j]
            m._fk_b1 = c1.conv.bias.detach().float().contiguous()
            m._fk_w2 = c2.conv.weight.detach().float().reshape(C, H).t().contiguous()                  # [j][o]
            m._fk_b2 = c2.conv.bias.detach().float().contiguous()
        m._fk_c = C
        m._fk_forward = m.forward
        m.forward = types.MethodType(_fused_forward, m)
        count += 1
    if count:
        _ext()          # build ahead of the warm-up so compilation never overlaps a timed run
    return {"blocks": count, "launches_per_block": 1, "stock_chain_below": {512: 600}}
