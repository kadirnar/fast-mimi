"""Fused fp32 transformer layers for Mimi's encoder_transformer / decoder_transformer.

Per layer (x = fp32 residual stream [B, T, 512], M = B*T rows) six launches replace ~40 stock kernels:
  1. qkv = xn1 @ Wqkv^T                CUDA skinny GEMM (M <= 32; q|k|v stacked, natural [N, K] layout); cuBLAS on the pre-transposed
                                        [K, N] copy for longer inputs (M > 32); it L2-prefetches the layer's per-head Wo once its
                                        own slab has landed (attention+O kernel 5.9 -> 5.5 us)
  2. part[h] = attention_h(qkv) @ Wo_h^T   CUDA: one warp per (batch, head, query row): RoPE, causal + sliding-window fp32 softmax; the
                                        block then multiplies its head's attention rows by the head's 64-column slice of Wo (prepacked
                                        per head, streamed while the attention runs) and L2-prefetches the layer's FC1 / FC2 weights
  3. h1 = x + ls_attn * sum_h part[h] ; xn2 = LN2(h1)        Triton row kernel (fixed head order)
  4. g = gelu(xn2 @ W1^T)              CUDA skinny GEMM with torch's erf GELU formula (x * 0.5 * (1 + erff(x / sqrt 2)), fp32) applied
                                        to the stored outputs: one value per lane per finished 4-row group through a branch-free
                                        predicated store, so the separate GELU launch (0.9 us + gap) is gone at no measurable cost.
                                        erff differs from torch's GELU kernel by 1 ulp in 1-8 % of values (within the audio tolerance;
                                        the codes stay identical on every gated workload)
  5. m[z] = g[:, chunk z] @ W2[:, chunk z]^T   CUDA skinny GEMM, 4 K-chunk partials (K = 2048)
  6. h2 = h1 + ls_mlp * sum_z m[z] ; xn1' = LN1_next(h2)     Triton row kernel (fixed-order partial sum; the last layer writes h2 only)
  FC1 and FC2 each L2-prefetch one half of the NEXT layer's Wqkv once their own slab has landed (QKV stays L2-warm, 8.1 -> 5.0 us).
(+ one LayerNorm row kernel for the first layer's strided input.)
All six launches of the fused chain are programmatic dependent launches (cudaLaunchKernelExC with programmatic stream
serialization; Triton launch_pdl for the row kernels): each dependent streams its input-independent preamble first (the GEMMs
their first two weight groups, the attention kernel its Wo slice), executes griddepcontrol.wait, and only then touches what the
previous kernel wrote (x rows, qkv, partials, residuals; qkv through coherent __ldcg loads, never .nc); the primaries trigger
(griddepcontrol.launch_dependents) as soon as their own dependent-visible work no longer needs the SM: the GEMMs once their x
rows are in registers, the attention kernel after its q/k/v staging, the row kernels at their start. Only the whole chain pays:
each family alone is neutral or worse (row-only 27.8, GEMM-only 27.8-28.6, attention-only 27.4 vs 27.3 us/layer), all three
27.3 -> 25.0-25.3 us/layer; trigger at kernel start (27.1) and prefetch hints issued before the wait (27.5) lose.

The skinny GEMM (out[m, n] = sum_k x[m, k] w[n, k], exact fp32): every earlier exact kernel here fed each FMA from shared memory /
L1 and sat at the LDS:FFMA balance point (7.5-9 us per 4 MB weight), while cuBLAS pads M = 25 to a 128-row tile (~5x the useful
FMAs, 6.4-8.8 us). This kernel keeps the ACTIVATION rows in registers: a block of 8 warps owns 32 (QKV: 24) weight rows and one
512-wide K chunk; warp = (row group, M-quarter); lane l = 16 h + c holds x[4 rows, 32 consecutive k] (128 registers, loaded once),
so every weight float4 streamed through shared memory feeds 16 FMAs from registers; a 16-lane XOR butterfly (5 shuffles per
warp-row, rows XOR-permuted in the register slots so no selects are needed) reduces the partials, interleaved one stage per row
with the next rows' FMAs. x and the W slab arrive by cp.async tracked by mbarriers (no block-wide waits after the prologue; each
warp starts when its own x quarter has landed; the x region is recycled for the second half of the slab). 64 blocks per GEMM.
Measured by graph replay at M = 25 (per call, cold-rotated / L2-warm weights): QKV 6.5 / 4.9 us (cuBLAS 8.1 / 6.5), FC1 7.9 / 5.5
(8.8 / 6.6), FC2 7.7 / 5.2 (8.1 / 6.4); in the real 8-layer chain 35.9-37.0 -> 28.9-29.0 us/layer, and 28.5-28.8 -> 27.1-27.2 with
the GELU epilogue + Wo prefetch (a GELU applied to FC2's x rows instead costs +2.9 us per call: 128 erff per lane, twice per block). Thread-block-cluster multicast /
DSMEM sharing of x across SMs measured far slower than plain per-block fetches on this machine (not used).

Numerics: everything fp32 (TF32 off as in the reference); every dot product is a fixed fmaf chain, softmax uses expf and IEEE
division, the per-head and K-chunk partials are summed in a fixed order, LayerNorm is a two-pass fp32 mean / variance; no atomics
-> bit-identical results run after run (the GEMM summation order differs from cuBLAS by ~1e-7 relative, within the strict gates).
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.cuda.gdc import gdc_launch_dependents, gdc_wait

D_MODEL = 512
N_HEADS = 8
HEAD_DIM = 64
D_FF = 2048
MAX_T = 32            # CUDA attention kernel: it pairs with the register-resident skinny GEMM chain (M <= 32), whose
                      # PDL edges and W1/W2 prefetch it carries.  Above that the GEMMs are cuBLAS, PDL is off, and
                      # attn_o's gridDim.x = ceil(T/8) replicas re-read Wo once each (16 MB/layer at T=125): the
                      # windowed Triton kernel + one cuBLAS O projection is faster from T = 34 up (measured:
                      # 1.3 s 1.796 -> 1.783 ms, 2 s 2.109 -> 2.072, 3 s 3.155 -> 3.007, 5 s 4.345 -> 4.012)
LONG_ATTN = True      # T > MAX_T: Triton windowed attention + cuBLAS O projection instead of the stock HF layer
LONG_BK = 32          # key tile of the windowed kernel (swept: 32 wins at every length)
LONG_BM = 32          # query tile (16 for the shortest long-form inputs)
SKINNY_MAX_M = 32     # register-resident skinny GEMM covers M = B*T <= 32 rows; cuBLAS above
S_QKV, S_FC1, S_FC2 = 24, 32, 32   # weight rows per block (64 blocks at the model's shapes)
PDL_GEMM, PDL_ATTN = 1 | 8, 1 | 2   # programmatic dependent launch: GEMMs trigger once x is in registers, attention after its staging
ROW_WARPS = 4         # warps of the residual + LayerScale + LayerNorm row kernel, one CTA per row
                      # (swept at T = 25: 1 -> 217.2 us, 2 -> 217.9, 4 -> 212.9, 8 -> 213.9 for the 8-layer stack)
QW = 8                # query rows (warps) per attention block
NCB = 2               # column blocks of the O-projection per (batch, head, query group)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
constexpr int DM = 512, NH = 8, HD = 64;
constexpr int QW = 8;              // query rows per block (one warp each)
constexpr int NT = QW * 32;        // threads per block
constexpr int NCB = 2;             // O-projection column blocks per (batch, head, query group)
constexpr int CPB = DM / NCB;      // output columns per block (256 = one per thread)
constexpr unsigned FULL = 0xffffffffu;

// programmatic dependent launch: the dependent grid may start once every block of this grid has triggered (or exited);
// griddepcontrol.wait blocks until the prerequisite grid has fully completed and its memory is visible.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// launch with (or without) the programmatic-stream-serialization attribute (captured by CUDA graphs as programmatic edges)
static void launch_ex(const void* func, dim3 grid, dim3 block, size_t smem, bool pdl, void** args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelExC(&cfg, func, args);
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchKernelExC failed: ", cudaGetErrorString(err));
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
// L2 prefetch hints for a later kernel's weight: one 128-byte line per thread, spread over the whole grid
// (fire-and-forget; issued once this kernel's own stream has landed so the memory system is otherwise idle)
__device__ __forceinline__ void prefetch_lines(const char* __restrict__ pf, size_t pf_bytes, size_t gtid, size_t nthr) {
    for (size_t off = gtid * 128; off < pf_bytes; off += nthr * 128)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(pf + off));
}

// grid (ceil(T / QW), B * NH, NCB); block NT threads.
// smem: rotated transposed keys [HD][T], values [T][HD], rotated queries [QW][HD], attention rows [QW][HD].
// wo4: Wo prepacked per head as [NH][HD/4][DM][4] (float4 index (h, k4, n)), so thread n streams 16 coalesced float4.
// part[h][m][n] = sum_k attn_h[m][k] * Wo[n][h*HD + k]  (fixed k order), m = b*T + t.
template <bool PDL>
__global__ void __launch_bounds__(NT) attn_o_kernel(const float* __restrict__ qkv, const float* __restrict__ cosb,
        const float* __restrict__ sinb, const float4* __restrict__ wo4, float* __restrict__ part, int T, int window,
        const char* __restrict__ pf1, size_t pb1, const char* __restrict__ pf2, size_t pb2, int pdl) {
    extern __shared__ float sm[];
    float* kT = sm;
    float* vs = kT + HD * T;
    float* qs = vs + T * HD;
    float* os = qs + QW * HD;
    const int bh = blockIdx.y, b = bh / NH, h = bh % NH;
    const int t0 = blockIdx.x * QW;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = blockIdx.z * CPB + tid;                  // this thread's output column
    const size_t nthr = (size_t)gridDim.x * gridDim.y * gridDim.z * NT;
    const size_t gtid = (((size_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * NT + tid;
    // 0. the O-projection weights of column n for head h: in flight during the attention
    float4 w[HD / 4];
    #pragma unroll
    for (int k4 = 0; k4 < HD / 4; ++k4) w[k4] = __ldg(wo4 + ((size_t)h * (HD / 4) + k4) * DM + n);
    // 1. stage keys (rotated, transposed), values and the block's queries (rotated): qkv was written by the previous kernel ->
    //    as a programmatic dependent, coherent loads (ld.global.cg) only after griddepcontrol.wait (a .nc load could be hoisted
    //    above the wait by the assembler); the plain instantiation keeps the L1-cached .nc loads (T = 125: 256 blocks share rows)
    if constexpr (PDL) pdl_wait();
    auto ld4 = [](const float* q) { if constexpr (PDL) return __ldcg(reinterpret_cast<const float4*>(q)); else return __ldg(reinterpret_cast<const float4*>(q)); };
    const float* base = qkv + (size_t)b * T * (3 * DM);
    for (int i = tid; i < T * (HD / 4); i += NT) {
        const int j = i / (HD / 4), d = 4 * (i - j * (HD / 4));
        const float* row = base + (size_t)j * (3 * DM) + DM + h * HD;
        const float4 kd = ld4(row + d);
        const float4 kp = ld4(row + ((d < 32) ? d + 32 : d - 32));
        const float4 c = __ldg(reinterpret_cast<const float4*>(cosb + j * 32 + (d & 31)));
        const float4 sn = __ldg(reinterpret_cast<const float4*>(sinb + j * 32 + (d & 31)));
        const float sg = (d < 32) ? -1.f : 1.f;                              // rotate_half: cat(-x2, x1)
        kT[(d + 0) * T + j] = kd.x * c.x + (sg * kp.x) * sn.x;
        kT[(d + 1) * T + j] = kd.y * c.y + (sg * kp.y) * sn.y;
        kT[(d + 2) * T + j] = kd.z * c.z + (sg * kp.z) * sn.z;
        kT[(d + 3) * T + j] = kd.w * c.w + (sg * kp.w) * sn.w;
        *reinterpret_cast<float4*>(vs + j * HD + d) = ld4(row + DM + d);
    }
    for (int i = tid; i < QW * (HD / 4); i += NT) {
        const int r = i / (HD / 4), d = 4 * (i - r * (HD / 4)), t = t0 + r;
        if (t < T) {
            const float* row = base + (size_t)t * (3 * DM) + h * HD;
            const float4 qd = ld4(row + d);
            const float4 qp = ld4(row + ((d < 32) ? d + 32 : d - 32));
            const float4 c = __ldg(reinterpret_cast<const float4*>(cosb + t * 32 + (d & 31)));
            const float4 sn = __ldg(reinterpret_cast<const float4*>(sinb + t * 32 + (d & 31)));
            const float sg = (d < 32) ? -1.f : 1.f;
            float4 o;
            o.x = qd.x * c.x + (sg * qp.x) * sn.x;
            o.y = qd.y * c.y + (sg * qp.y) * sn.y;
            o.z = qd.z * c.z + (sg * qp.z) * sn.z;
            o.w = qd.w * c.w + (sg * qp.w) * sn.w;
            *reinterpret_cast<float4*>(qs + r * HD + d) = o;
        }
    }
    __syncthreads();
    if (PDL && (pdl & 2)) pdl_trigger();                                 // staging done: the row kernel may launch and wait
    // the K/V/Q tiles have landed and the Wo float4 loads are nearly done: pull the layer's FC1 weight into L2 while
    // the attention computes (the FC1 GEMM then streams it from L2; measured 35.0 -> 34.0 us per layer)
    if (pf1 != nullptr) prefetch_lines(pf1, pb1, gtid, nthr);
    // 2. attention: one warp per query row; lane j scores keys j, j+32, ...; lane l accumulates dims 2l, 2l+1
    const int t = t0 + warp;
    if (t < T) {
        const float* q = qs + warp * HD;
        constexpr int KPL = 4;
        float p[KPL];
        float m = -CUDART_INF_F;
        #pragma unroll
        for (int kk = 0; kk < KPL; ++kk) {
            const int j = lane + 32 * kk;
            float sc = -CUDART_INF_F;
            if (j < T && j <= t && j > t - window) {
                float a = 0.f;
                #pragma unroll 8
                for (int d = 0; d < HD; ++d) a = fmaf(q[d], kT[d * T + j], a);
                sc = a * 0.125f;                                                // 1/sqrt(64): exact
            }
            p[kk] = sc;
            m = fmaxf(m, sc);
        }
        m = warp_max(m);
        float l = 0.f;
        #pragma unroll
        for (int kk = 0; kk < KPL; ++kk) { p[kk] = expf(p[kk] - m); l += p[kk]; }
        l = warp_sum(l);
        float o0 = 0.f, o1 = 0.f;
        #pragma unroll
        for (int kk = 0; kk < KPL; ++kk) {
            for (int jj = 0; jj < 32; ++jj) {
                const int j = kk * 32 + jj;
                if (j >= T) break;
                const float pj = __shfl_sync(FULL, p[kk], jj);
                const float2 vv = *reinterpret_cast<const float2*>(vs + j * HD + 2 * lane);
                o0 = fmaf(pj, vv.x, o0);
                o1 = fmaf(pj, vv.y, o1);
            }
        }
        *reinterpret_cast<float2*>(os + warp * HD + 2 * lane) = make_float2(o0 / l, o1 / l);
    }
    __syncthreads();
    // the FC2 weight queues behind FC1's lines and lands during the row kernel / FC1 GEMM (34.0 -> 33.4 us per layer)
    if (pf2 != nullptr) prefetch_lines(pf2, pb2, gtid, nthr);
    // 3. O-projection of the block's rows for column n (fixed k order)
    float acc[QW];
    #pragma unroll
    for (int r = 0; r < QW; ++r) acc[r] = 0.f;
    #pragma unroll
    for (int k4 = 0; k4 < HD / 4; ++k4) {
        #pragma unroll
        for (int r = 0; r < QW; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(os + r * HD + 4 * k4);
            acc[r] = fmaf(a.x, w[k4].x, acc[r]);
            acc[r] = fmaf(a.y, w[k4].y, acc[r]);
            acc[r] = fmaf(a.z, w[k4].z, acc[r]);
            acc[r] = fmaf(a.w, w[k4].w, acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < QW; ++r) {
        const int tt = t0 + r;
        if (tt < T) part[((size_t)h * (gridDim.y / NH) * T + (size_t)b * T + tt) * DM + n] = acc[r];
    }
}
constexpr int SLOT = 512;                 // floats per 2 KB smem row slot
constexpr int XSLOTS = 32;                // x region: rows 0..31 (quarter q = slots 8q..8q+7)
constexpr int WSLOTS0 = 16;               // W rows 0..15 -> slots 32..47; W rows 16..31 -> slots 0..15 (x quarters 0/1, once consumed)
constexpr int SMEM_BYTES = (XSLOTS + WSLOTS0) * SLOT * 4;   // 96 KB

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void mbar_init(unsigned addr, unsigned count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count)); }
// the executing thread's prior cp.async operations arrive on the barrier when they complete (no pending-count increment)
__device__ __forceinline__ void cp_async_arrive(unsigned addr) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(addr) : "memory"); }
__device__ __forceinline__ void mbar_wait(unsigned addr, unsigned parity) {
    unsigned done;
    do {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(addr), "r"(parity) : "memory");
    } while (!done);
}
// torch's erf GELU formula (x * 0.5 * (1 + erf(x / sqrt 2))) in fp32
__device__ __forceinline__ float gelu_f(float v) { return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f)); }
// predicated global store (no branch: the value's math stays straight-line and overlaps the neighbouring FMAs)
__device__ __forceinline__ void st_pred(float* ptr, float v, bool pred) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; @p st.global.f32 [%0], %1; }" :: "l"(ptr), "f"(v), "r"((int)pred) : "memory");
}
__device__ __forceinline__ float sel4(float a, float b, float c, float d, int j) { return (j & 2) ? ((j & 1) ? d : c) : ((j & 1) ? b : a); }
__device__ __forceinline__ int sel4i(int a, int b, int c, int d, int j) { return (j & 2) ? ((j & 1) ? d : c) : ((j & 1) ? b : a); }
__device__ __forceinline__ int wslot(int r) { return r < WSLOTS0 ? XSLOTS + r : r - WSLOTS0; }

// XOR butterfly over the 16 lanes of a row half (lane bit 4 selects the half): stage k has lane distance 8 >> k; while slots remain
// (h = V >> (k + 1) >= 1) it is a pair stage (slot i += partner's slot i + h), afterwards a single-slot stage.
template <int V, int K>
__device__ __forceinline__ void stage_shfl(const float* pend, float* r) {
    constexpr int h = V >> (K + 1);
    if constexpr (h >= 1) {
        #pragma unroll
        for (int i = 0; i < h; ++i) r[i] = __shfl_xor_sync(FULL, pend[i + h], 8 >> K);
    } else {
        r[0] = __shfl_xor_sync(FULL, pend[0], 8 >> K);
    }
}
template <int V, int K>
__device__ __forceinline__ void stage_add(float* pend, const float* r) {
    constexpr int h = V >> (K + 1);
    if constexpr (h >= 1) {
        #pragma unroll
        for (int i = 0; i < h; ++i) pend[i] += r[i];
    } else {
        pend[0] += r[0];
    }
}
template <int V, int J>
__device__ __forceinline__ void fma_chunk(float* acc, const float4 (*xs)[8], const float4* wv) {
    #pragma unroll
    for (int s = 0; s < V; ++s) {
        acc[s] = fmaf(xs[s][J].x, wv[J].x, acc[s]);
        acc[s] = fmaf(xs[s][J].y, wv[J].y, acc[s]);
        acc[s] = fmaf(xs[s][J].z, wv[J].z, acc[s]);
        acc[s] = fmaf(xs[s][J].w, wv[J].w, acc[s]);
    }
}

// out[kz, m, n] = sum_{k in chunk kz} x[m, k] w[n, k] for the block's rows n in [n0, n0 + S), S multiple of 8 <= 32.
// grid (ceil(N / S), K / 512); block 256 threads = 8 warps: warp = 4 * grp + mw; group grp (0/1) takes slab rows 2 i + grp,
// M-quarter mw covers x rows mw * 2V .. mw * 2V + 2V - 1 (V slots per row half; 8 rows per warp for M > 16).
// Lane l = 16 h + c: half h takes rows mw * 2V + V h + (s ^ p) in slot s (p = (l >> (4 - log2 V)) & (V - 1)); chunk c owns the
// 32 consecutive k = kb + 32 c + 4 (j ^ (c & 7)) + {0..3} for register index j < 8 (the XOR of the float4 index keeps the LDS of
// 16 lanes reading 16 different 128-byte chunks conflict-free). Per row n a lane does 4 x 32 FMAs from registers; the 16 lanes of a
// half reduce by the XOR butterfly (xor 8, 4 halving the slots; xor 2, 1), after which lane l holds out[mw * 2V + (l >> 2), n]
// (V = 4). The butterfly of row i is interleaved with the FMA chunks of row i + 1; its final add + store happen at row i + 2.
// Data flow: x quarter q (8 rows) -> smem slots 8q..8q+7 (mbarrier xq[q]: each warp waits only for its own rows); W rows 0..15 ->
// slots 32..47 (wg[0], wg[1]); the four mw < 2 warps free slots 0..15 once their x is in registers and stream W rows 16..31 there
// (wg[2], wg[3]). Issue order x q0, W g0, x q1, x q2, x q3, W g1. Fixed fmaf order per lane, fixed butterfly -> deterministic.
// pdl bits: 1 = launched as a programmatic dependent (W groups 0/1 stream before griddepcontrol.wait, x rows after it),
// 8 = trigger the next kernel once this block's x rows are in registers (measured best: after the slab landed 25.9, at start 27.1,
// after x in registers 25.1 us/layer)
template <int V, int NG, int ACT>
__global__ void __launch_bounds__(256, 1) skinny_kernel(const float* __restrict__ x, int ldx, const float* __restrict__ w, int ldw,
        float* __restrict__ out, int ldo, long long ldz, int M, int N, int S, const char* __restrict__ pf, size_t pf_bytes, int pdl) {
    extern __shared__ __align__(16) float smem[];
    __shared__ __align__(8) unsigned long long bars[8];          // xq[0..3], wg[0..3]
    constexpr int LOGV = (V == 4) ? 2 : (V == 2) ? 1 : 0;
    constexpr int H = (V >= 2) ? (V / 2) : 1;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int grp = warp >> 2, mw = warp & 3;
    const int hh = lane >> 4, c = lane & 15, sw = c & 7;
    const int p = (V > 1) ? ((lane >> (4 - LOGV)) & (V - 1)) : 0;
    const int kz = blockIdx.y, kb = kz * SLOT;
    const int n0 = blockIdx.x * S;
    out += (long long)kz * ldz;
    constexpr int ngroups = NG;
    const unsigned bar0 = (unsigned)__cvta_generic_to_shared(bars);
    auto xq = [&](int q) { return bar0 + 8u * (unsigned)q; };
    auto wg = [&](int g) { return bar0 + 32u + 8u * (unsigned)g; };
    if (tid == 0) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) mbar_init(xq(q), 256);
        mbar_init(wg(0), 256); mbar_init(wg(1), 256); mbar_init(wg(2), (V == 4) ? 128 : 256); mbar_init(wg(3), (V == 4) ? 128 : 256);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    auto issue_x_quarter = [&](int q) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = q * 1024 + i * 256 + tid;
            const int m = idx >> 7;
            if (m < M) cp_async16(smem + (size_t)idx * 4, x + (size_t)m * ldx + kb + (idx & 127) * 4);
        }
        cp_async_arrive(xq(q));
    };
    auto issue_w_group_all = [&](int g) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = g * 1024 + i * 256 + tid;
            const int r = idx >> 7, cc = idx & 127;
            const int n = min(n0 + r, N - 1);
            cp_async16(smem + (size_t)(wslot(r) * SLOT) + cc * 4, w + (size_t)n * ldw + kb + cc * 4);
        }
        cp_async_arrive(wg(g));
    };
    if (pdl & 1) {
        issue_w_group_all(0);                                           // weights: independent of the previous kernel
        if (ngroups > 1) issue_w_group_all(1);
        pdl_wait();                                                     // x rows were written by the previous kernel
        issue_x_quarter(0); issue_x_quarter(1); issue_x_quarter(2); issue_x_quarter(3);
    } else {
        issue_x_quarter(0);
        issue_w_group_all(0);
        issue_x_quarter(1);
        issue_x_quarter(2);
        issue_x_quarter(3);
        if (ngroups > 1) issue_w_group_all(1);
    }
    // ---- x rows of this warp -> registers, as soon as their quarter has landed ----
    mbar_wait(xq((mw * 2 * V) >> 3), 0);
    float4 xs[V][8];
    #pragma unroll
    for (int s = 0; s < V; ++s) {
        const int m = mw * 2 * V + V * hh + (s ^ p);
        const bool ok = m < M;
        const float4* xr = reinterpret_cast<const float4*>(smem + (size_t)(ok ? m : 0) * SLOT + 32 * c);
        #pragma unroll
        for (int j = 0; j < 8; ++j) xs[s][j] = ok ? xr[j ^ sw] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    constexpr int NF = (V == 4) ? 128 : 256;                      // threads issuing W rows 16.. into the freed x slots
    if (ngroups > 2 && (V < 4 || mw < 2)) {
        if constexpr (V == 4) asm volatile("bar.sync 1, 128;" ::: "memory"); else __syncthreads();
        const int li = (V == 4) ? ((warp >= 4) ? (tid - 64) : tid) : tid;
        for (int g = 2; g < ngroups; ++g) {
            #pragma unroll
            for (int i = 0; i < 1024 / NF; ++i) {
                const int idx = (g - 2) * 1024 + i * NF + li;
                const int rr = idx >> 7, cc = idx & 127;
                const int n = min(n0 + 16 + rr, N - 1);
                cp_async16(smem + (size_t)rr * SLOT + cc * 4, w + (size_t)n * ldw + kb + cc * 4);
            }
            cp_async_arrive(wg(g));
        }
    }
    if (pdl & 8) pdl_trigger();                                     // x is in registers: the next kernel may launch and stream its slab
    __syncwarp();                                                   // reconverge after the warp-uniform hand-off branch
    // ---- compute: the four chains of the previous group are reduced one stage per row of the current group ----
    const int m_out = mw * 2 * V + (lane >> (4 - LOGV));
    const int jl = lane & 3;                                        // row of the finished group this lane stores
    const bool out_lane = (m_out < M) && ((lane & ((16 >> LOGV) - 1)) < 4);   // V = 4: every lane; V = 2: lanes 0-3 of each 8; V = 1: of each 16
    float* orow = out + (size_t)m_out * ldo;
    float pend[2][4][V], red[4][H];
    int pend_n[2][4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        pend_n[0][q] = -1; pend_n[1][q] = -1;
        #pragma unroll
        for (int s = 0; s < V; ++s) { pend[0][q][s] = 0.f; pend[1][q][s] = 0.f; }
        #pragma unroll
        for (int i = 0; i < H; ++i) red[q][i] = 0.f;
    }
    float4 wb[2][8];
    mbar_wait(wg(0), 0);
    {
        const float4* wr = reinterpret_cast<const float4*>(smem + (size_t)wslot(grp) * SLOT + 32 * c);
        #pragma unroll
        for (int j = 0; j < 8; ++j) wb[0][j] = wr[j ^ sw];
    }
    #pragma unroll
    for (int g = 0; g < NG; ++g) {
        #pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            constexpr int dummy = 0; (void)dummy;
            const int r = 8 * g + 2 * ii + grp;
            const int n = n0 + r;
            float* pcur = pend[g & 1][0];          // chains of the previous group (being reduced)
            float acc[V];
            #pragma unroll
            for (int s = 0; s < V; ++s) acc[s] = 0.f;
            const bool same_group = (ii < 3);
            const float4* wr = reinterpret_cast<const float4*>(smem + (size_t)wslot(same_group ? r + 2 : r) * SLOT + 32 * c);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (ii == 0) stage_shfl<V, 0>(pend[g & 1][q], red[q]);
                else if (ii == 1) stage_shfl<V, 1>(pend[g & 1][q], red[q]);
                else if (ii == 2) stage_shfl<V, 2>(pend[g & 1][q], red[q]);
                else stage_shfl<V, 3>(pend[g & 1][q], red[q]);
            }
            fma_chunk<V, 0>(acc, xs, wb[ii & 1]);
            fma_chunk<V, 1>(acc, xs, wb[ii & 1]);
            fma_chunk<V, 2>(acc, xs, wb[ii & 1]);
            if (same_group) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) wb[(ii + 1) & 1][j] = wr[j ^ sw];
            }
            fma_chunk<V, 3>(acc, xs, wb[ii & 1]);
            fma_chunk<V, 4>(acc, xs, wb[ii & 1]);
            if (same_group) {
                #pragma unroll
                for (int j = 4; j < 8; ++j) wb[(ii + 1) & 1][j] = wr[j ^ sw];
            }
            fma_chunk<V, 5>(acc, xs, wb[ii & 1]);
            fma_chunk<V, 6>(acc, xs, wb[ii & 1]);
            fma_chunk<V, 7>(acc, xs, wb[ii & 1]);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (ii == 0) stage_add<V, 0>(pend[g & 1][q], red[q]);
                else if (ii == 1) stage_add<V, 1>(pend[g & 1][q], red[q]);
                else if (ii == 2) stage_add<V, 2>(pend[g & 1][q], red[q]);
                else stage_add<V, 3>(pend[g & 1][q], red[q]);
            }
            if (ii == 3) {
                // the four chains are done: lane 4q + jl stores row jl of the group for its x row (one value per lane)
                const float v = sel4(pend[g & 1][0][0], pend[g & 1][1][0], pend[g & 1][2][0], pend[g & 1][3][0], jl);
                const int nn = sel4i(pend_n[g & 1][0], pend_n[g & 1][1], pend_n[g & 1][2], pend_n[g & 1][3], jl);
                st_pred(orow + nn, (ACT == 1) ? gelu_f(v) : v, nn >= 0 && out_lane);
            }
            #pragma unroll
            for (int s = 0; s < V; ++s) pend[(g + 1) & 1][ii][s] = acc[s];
            pend_n[(g + 1) & 1][ii] = (n < N) ? n : -1;
            (void)pcur;
        }
        if (g + 1 < NG) {
            mbar_wait(wg(g + 1), 0);
            if (g + 2 == NG && pf != nullptr) {
                const size_t nthr = (size_t)gridDim.x * gridDim.y * 256;
                const size_t gtid = ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 256 + tid;
                prefetch_lines(pf, pf_bytes, gtid, nthr);
            }
            const float4* wr = reinterpret_cast<const float4*>(smem + (size_t)wslot(8 * (g + 1) + grp) * SLOT + 32 * c);
            #pragma unroll
            for (int j = 0; j < 8; ++j) wb[0][j] = wr[j ^ sw];
        }
    }
    // drain: the last group's four chains
    {
        float (*pl)[V] = pend[NG & 1];
        int* pln = pend_n[NG & 1];
        #pragma unroll
        for (int q = 0; q < 4; ++q) stage_shfl<V, 0>(pl[q], red[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) stage_add<V, 0>(pl[q], red[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) stage_shfl<V, 1>(pl[q], red[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) stage_add<V, 1>(pl[q], red[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) stage_shfl<V, 2>(pl[q], red[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) stage_add<V, 2>(pl[q], red[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) stage_shfl<V, 3>(pl[q], red[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) stage_add<V, 3>(pl[q], red[q]);
        {
            const float v = sel4(pl[0][0], pl[1][0], pl[2][0], pl[3][0], jl);
            const int nn = sel4i(pln[0], pln[1], pln[2], pln[3], jl);
            st_pred(orow + nn, (ACT == 1) ? gelu_f(v) : v, nn >= 0 && out_lane);
        }
    }
}

template <int V, int NG, int ACT>
void skinny_launch_ng(dim3 grid, cudaStream_t st, const float* x, int ldx, const float* w, int ldw, float* out, int ldo, long long ldz, int M, int N, int S,
                      const char* pf, size_t pf_bytes, int pdl) {
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(skinny_kernel<V, NG, ACT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES); attr = true; }
    void* args[] = {&x, &ldx, &w, &ldw, &out, &ldo, &ldz, &M, &N, &S, &pf, &pf_bytes, &pdl};
    launch_ex((const void*)skinny_kernel<V, NG, ACT>, grid, dim3(256), SMEM_BYTES, (pdl & 1) != 0, args);
}
template <int V, int ACT>
void skinny_launch_act(dim3 grid, cudaStream_t st, const float* x, int ldx, const float* w, int ldw, float* out, int ldo, long long ldz, int M, int N, int S,
                       const char* pf, size_t pf_bytes, int pdl) {
    switch (S >> 3) {
        case 1: skinny_launch_ng<V, 1, ACT>(grid, st, x, ldx, w, ldw, out, ldo, ldz, M, N, S, pf, pf_bytes, pdl); break;
        case 2: skinny_launch_ng<V, 2, ACT>(grid, st, x, ldx, w, ldw, out, ldo, ldz, M, N, S, pf, pf_bytes, pdl); break;
        case 3: skinny_launch_ng<V, 3, ACT>(grid, st, x, ldx, w, ldw, out, ldo, ldz, M, N, S, pf, pf_bytes, pdl); break;
        default: skinny_launch_ng<V, 4, ACT>(grid, st, x, ldx, w, ldw, out, ldo, ldz, M, N, S, pf, pf_bytes, pdl); break;
    }
}
template <int V>
void skinny_launch(dim3 grid, cudaStream_t st, const float* x, int ldx, const float* w, int ldw, float* out, int ldo, long long ldz, int M, int N, int S,
                   const char* pf, size_t pf_bytes, int act, int pdl) {
    if (act) skinny_launch_act<V, 1>(grid, st, x, ldx, w, ldw, out, ldo, ldz, M, N, S, pf, pf_bytes, pdl);
    else skinny_launch_act<V, 0>(grid, st, x, ldx, w, ldw, out, ldo, ldz, M, N, S, pf, pf_bytes, pdl);
}
}  // namespace

// pdl bits: 1 = launched as a programmatic dependent (griddepcontrol.wait before reading qkv), 2 = trigger the next kernel
// after the q/k/v staging
void attn_o(torch::Tensor qkv, torch::Tensor cosb, torch::Tensor sinb, torch::Tensor wo4, torch::Tensor part, int64_t T, int64_t window,
            c10::optional<torch::Tensor> pf1, c10::optional<torch::Tensor> pf2, int64_t pdl) {
    TORCH_CHECK(qkv.is_cuda() && qkv.dtype() == torch::kFloat32 && qkv.is_contiguous() && qkv.dim() == 2 && qkv.size(1) == 3 * DM, "qkv must be contiguous fp32 [M, 1536]");
    const int M = (int)qkv.size(0);
    TORCH_CHECK(T >= 1 && T <= 128 && M % T == 0, "T must be in [1, 128] and divide M");
    const int B = M / (int)T;
    TORCH_CHECK(cosb.is_contiguous() && sinb.is_contiguous() && cosb.size(0) >= T && cosb.size(1) == 32 && sinb.sizes() == cosb.sizes(), "cos/sin tables [T, 32]");
    TORCH_CHECK(wo4.is_contiguous() && wo4.dtype() == torch::kFloat32 && wo4.numel() == (int64_t)NH * HD * DM, "wo4 must be the per-head prepacked Wo");
    TORCH_CHECK(part.is_contiguous() && part.dtype() == torch::kFloat32 && part.sizes() == torch::IntArrayRef({(int64_t)NH, (int64_t)M, (int64_t)DM}), "part must be [8, M, 512]");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(qkv.data_ptr()) & 15) == 0 && (reinterpret_cast<uintptr_t>(wo4.data_ptr()) & 15) == 0, "16-byte alignment");
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(attn_o_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (2 * HD * 128 + 2 * QW * HD) * (int)sizeof(float));
        cudaFuncSetAttribute(attn_o_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (2 * HD * 128 + 2 * QW * HD) * (int)sizeof(float));
        attr = true;
    }
    const size_t smem = (size_t)(2 * HD * T + 2 * QW * HD) * sizeof(float);
    dim3 grid(((int)T + QW - 1) / QW, B * NH, NCB);
    auto ptr = [](const c10::optional<torch::Tensor>& t) { return t.has_value() ? reinterpret_cast<const char*>(t->data_ptr()) : nullptr; };
    auto nbytes = [](const c10::optional<torch::Tensor>& t) { return t.has_value() ? (size_t)t->numel() * t->element_size() : (size_t)0; };
    const float* qkvp = qkv.data_ptr<float>(); const float* cp = cosb.data_ptr<float>(); const float* sp = sinb.data_ptr<float>();
    const float4* wp = reinterpret_cast<const float4*>(wo4.data_ptr<float>()); float* pp = part.data_ptr<float>();
    int Ti = (int)T, wi = (int)window, pdli = (int)pdl;
    const char* p1 = ptr(pf1); size_t b1 = nbytes(pf1); const char* p2 = ptr(pf2); size_t b2 = nbytes(pf2);
    void* args[] = {&qkvp, &cp, &sp, &wp, &pp, &Ti, &wi, &p1, &b1, &p2, &b2, &pdli};
    if (pdl & 1) launch_ex((const void*)attn_o_kernel<true>, grid, dim3(NT), smem, true, args);
    else launch_ex((const void*)attn_o_kernel<false>, grid, dim3(NT), smem, false, args);
}

void skinny_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t S, c10::optional<torch::Tensor> pf, int64_t act, int64_t pdl) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.dim() == 2 && x.stride(1) == 1, "x must be fp32 [M, K] with unit inner stride");
    TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kFloat32 && w.is_contiguous() && w.dim() == 2, "w must be contiguous fp32 [N, K]");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    TORCH_CHECK(w.size(1) == K && K % 512 == 0, "K must be a multiple of 512 and match");
    const int KZ = K / 512;
    TORCH_CHECK(out.is_contiguous() && out.dtype() == torch::kFloat32 && out.numel() == (int64_t)KZ * M * N, "out must be [KZ, M, N]");
    TORCH_CHECK(M >= 1 && M <= 32, "M must be in [1, 32]");
    TORCH_CHECK(S % 8 == 0 && S >= 8 && S <= 32, "S must be a multiple of 8 in [8, 32]");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 && (x.stride(0) % 4) == 0 && (reinterpret_cast<uintptr_t>(w.data_ptr()) & 15) == 0, "16-byte alignment");
    const int V = (M > 16) ? 4 : (M > 8) ? 2 : 1;
    dim3 grid((N + (int)S - 1) / (int)S, KZ);
    auto st = c10::cuda::getCurrentCUDAStream();
    const float* xp = x.data_ptr<float>();
    const int ldx = (int)x.stride(0);
    float* op = out.data_ptr<float>();
    const long long ldz = (long long)M * N;
    const char* pfp = pf.has_value() ? reinterpret_cast<const char*>(pf->data_ptr()) : nullptr;
    const size_t pfb = pf.has_value() ? (size_t)pf->numel() * pf->element_size() : (size_t)0;
    switch (V) {
        case 4: skinny_launch<4>(grid, st, xp, ldx, w.data_ptr<float>(), K, op, N, ldz, M, N, (int)S, pfp, pfb, (int)act, (int)pdl); break;
        case 2: skinny_launch<2>(grid, st, xp, ldx, w.data_ptr<float>(), K, op, N, ldz, M, N, (int)S, pfp, pfb, (int)act, (int)pdl); break;
        default: skinny_launch<1>(grid, st, xp, ldx, w.data_ptr<float>(), K, op, N, ldz, M, N, (int)S, pfp, pfb, (int)act, (int)pdl); break;
    }
}
"""
CPP_SRC = ("void skinny_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t S, c10::optional<torch::Tensor> pf, int64_t act, int64_t pdl);\n"
           "void attn_o(torch::Tensor qkv, torch::Tensor cosb, torch::Tensor sinb, torch::Tensor wo4, torch::Tensor part, int64_t T, int64_t window, "
           "c10::optional<torch::Tensor> pf1, c10::optional<torch::Tensor> pf2, int64_t pdl);\n")

_MOD = None


def _ext():
    """Compile the attention + O-projection kernel once (torch caches by source hash under .fast-kernel/build; no fast-math)."""
    global _MOD
    if _MOD is None:
        from .._compat import build_dir, ensure_cuda_home
        ensure_cuda_home()                      # must precede the cpp_extension import (it reads CUDA_HOME on import)
        from torch.utils.cpp_extension import load_inline
        build = build_dir(None) / "fk_tf_pdl4"
        build.mkdir(parents=True, exist_ok=True)
        _MOD = load_inline(name="fk_tf_pdl4", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC], functions=["attn_o", "skinny_gemm"],
                           extra_cuda_cflags=["-O3"], build_directory=str(build))
    return _MOD


# ----------------------------------------------------------------------------------------------------------
# Triton row kernel: residual (+ fixed-order sum of S partials) + LayerScale + LayerNorm
# ----------------------------------------------------------------------------------------------------------
@triton.jit
def _row_kernel(x_ptr, sxb, sxt, sxk, o_ptr, so, ls_ptr, lnw_ptr, lnb_ptr, h_ptr, shb, sht, shk, xn_ptr, eps, T,
                N: tl.constexpr, S: tl.constexpr, HAS_RES: tl.constexpr, HAS_LN: tl.constexpr, PDL: tl.constexpr):
    """Per row (b, t): h = x + ls * sum_s o[s] (HAS_RES, stored with h strides); xn = LayerNorm(h) (HAS_LN, contiguous)."""
    if PDL:
        gdc_launch_dependents()          # the next GEMM may launch now and stream its weight slab while this kernel runs
        gdc_wait()                       # everything below reads the previous kernel's outputs
    row = tl.program_id(0)
    b = row // T
    t = row % T
    rn = tl.arange(0, N)
    h = tl.load(x_ptr + b * sxb + t * sxt + rn * sxk)
    if HAS_RES:
        o = tl.load(o_ptr + row * N + rn)
        for s in tl.static_range(1, S):
            o += tl.load(o_ptr + s * so + row * N + rn)
        ls = tl.load(ls_ptr + rn)
        h = h + ls * o
        tl.store(h_ptr + b * shb + t * sht + rn * shk, h)
    if HAS_LN:
        mean = libdevice.div_rn(tl.sum(h, axis=0), N * 1.0)
        d = h - mean
        rstd = libdevice.rsqrt(libdevice.div_rn(tl.sum(d * d, axis=0), N * 1.0) + eps)
        xn = tl.load(lnw_ptr + rn) * (rstd * d) + tl.load(lnb_ptr + rn)
        tl.store(xn_ptr + row * N + rn, xn)


def _row(x, xstr, o, ls, lnw, lnb, h, hstr, xn, eps: float, B: int, T: int, pdl: bool = False):
    """x: rows with strides xstr; o: [S, B*T, N] or [B*T, N] contiguous; h: strides hstr; xn: [B*T, N] contiguous."""
    has_res = o is not None
    has_ln = xn is not None
    hb, ht, hk = hstr if has_res else (0, 0, 0)
    S = o.shape[0] if has_res and o.dim() == 3 else 1
    so = o.stride(0) if has_res and o.dim() == 3 else 0
    _row_kernel[(B * T,)](x, xstr[0], xstr[1], xstr[2], o if has_res else x, so, ls if has_res else x,
                          lnw if has_ln else x, lnb if has_ln else x, h if has_res else x, hb, ht, hk,
                          xn if has_ln else x, eps, T, N=D_MODEL, S=S, HAS_RES=has_res, HAS_LN=has_ln, PDL=pdl, num_warps=ROW_WARPS,
                          launch_pdl=pdl)


# ----------------------------------------------------------------------------------------------------------
# Triton sliding-window attention for long inputs (T > MAX_T): the fused chain's fp32 attention at any length
# ----------------------------------------------------------------------------------------------------------
@triton.jit
def _rope_qk_kernel(qkv_ptr, cos_ptr, sin_ptr, rq_ptr, rkt_ptr, M, T,
                    BR: tl.constexpr, HD: tl.constexpr, HH: tl.constexpr, DM: tl.constexpr):
    """RoPE applied once per layer: rq[m, h*HD + d] and rkT[h*HD + d, m] (K transposed so the attention's key tiles
    are contiguous along the key index).  Same expression as the reference's rotate_half."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < M
    pos = rows % T
    d = tl.arange(0, HD)
    dh = d % HH
    half = d < HH
    partner = tl.where(half, d + HH, d - HH)
    sgn = tl.where(half, -1.0, 1.0)
    c = tl.load(cos_ptr + pos[:, None] * HH + dh[None, :], mask=rmask[:, None], other=0.0)
    sn = tl.load(sin_ptr + pos[:, None] * HH + dh[None, :], mask=rmask[:, None], other=0.0)
    base = qkv_ptr + rows[:, None] * (3 * DM) + h * HD
    qd = tl.load(base + d[None, :], mask=rmask[:, None], other=0.0)
    qp = tl.load(base + partner[None, :], mask=rmask[:, None], other=0.0)
    tl.store(rq_ptr + rows[:, None] * DM + (h * HD + d)[None, :], qd * c + (sgn[None, :] * qp) * sn, mask=rmask[:, None])
    kd = tl.load(base + DM + d[None, :], mask=rmask[:, None], other=0.0)
    kp = tl.load(base + DM + partner[None, :], mask=rmask[:, None], other=0.0)
    tl.store(rkt_ptr + (h * HD + d)[None, :] * M + rows[:, None], kd * c + (sgn[None, :] * kp) * sn, mask=rmask[:, None])


@triton.jit
def _attn_window_kernel(rq_ptr, rkt_ptr, v_ptr, out_ptr, M, T, WINDOW,
                        BM: tl.constexpr, BK: tl.constexpr, HD: tl.constexpr, DM: tl.constexpr, NH: tl.constexpr):
    """out[b*T + t, h*HD + d] = softmax_j(q_t . k_j / 8) . v_j over j in (t - WINDOW, t], fp32 throughout.

    Two passes over the key band (row maximum first, then exp / sum / weighted sum): every partial is accumulated in a
    fixed tile order, so the result is deterministic and needs no online rescaling."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NH
    h = bh % NH
    t0 = pid_m * BM
    tq = t0 + tl.arange(0, BM)
    rows = b * T + tq
    rmask = tq < T
    d = h * HD + tl.arange(0, HD)
    q = tl.load(rq_ptr + rows[:, None] * DM + d[None, :], mask=rmask[:, None], other=0.0)
    lo = tl.maximum(t0 - WINDOW + 1, 0)          # earliest key any query in this block attends to
    lo = (lo // BK) * BK
    hi = tl.minimum(t0 + BM, T)
    ninf = float("-inf")
    m_i = tl.full((BM,), ninf, tl.float32)
    for j0 in range(lo, hi, BK):
        jj = j0 + tl.arange(0, BK)
        jv = jj < T
        kt = tl.load(rkt_ptr + d[:, None] * M + (b * T + jj)[None, :], mask=jv[None, :], other=0.0)
        sc = tl.dot(q, kt, input_precision="ieee") * 0.125
        keep = (jj[None, :] <= tq[:, None]) & (jj[None, :] > tq[:, None] - WINDOW) & jv[None, :]
        m_i = tl.maximum(m_i, tl.max(tl.where(keep, sc, ninf), 1))
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, HD), tl.float32)
    for j0 in range(lo, hi, BK):
        jj = j0 + tl.arange(0, BK)
        jv = jj < T
        kt = tl.load(rkt_ptr + d[:, None] * M + (b * T + jj)[None, :], mask=jv[None, :], other=0.0)
        sc = tl.dot(q, kt, input_precision="ieee") * 0.125
        keep = (jj[None, :] <= tq[:, None]) & (jj[None, :] > tq[:, None] - WINDOW) & jv[None, :]
        p = tl.where(keep, libdevice.exp(sc - m_i[:, None]), 0.0)
        l_i += tl.sum(p, 1)
        vv = tl.load(v_ptr + (b * T + jj)[:, None] * (3 * DM) + (2 * DM + d)[None, :], mask=jv[:, None], other=0.0)
        acc = tl.dot(p, vv, acc, input_precision="ieee")
    tl.store(out_ptr + rows[:, None] * DM + d[None, :], acc / l_i[:, None], mask=rmask[:, None])


def _attn_window(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, B: int, T: int, window: int,
                 out: torch.Tensor, rq: torch.Tensor, rkt: torch.Tensor, bm: int, bk: int) -> torch.Tensor:
    """qkv: (B*T, 3*D_MODEL) contiguous; out: (B*T, D_MODEL) attention output before the O projection."""
    M = B * T
    _rope_qk_kernel[(triton.cdiv(M, 64), N_HEADS)](qkv, cos, sin, rq, rkt, M, T,
                                                   BR=64, HD=HEAD_DIM, HH=HEAD_DIM // 2, DM=D_MODEL, num_warps=4)
    _attn_window_kernel[(triton.cdiv(T, bm), B * N_HEADS)](rq, rkt, qkv, out, M, T, window,
                                                           BM=bm, BK=bk, HD=HEAD_DIM, DM=D_MODEL, NH=N_HEADS,
                                                           num_warps=4, num_stages=2)
    return out


# ----------------------------------------------------------------------------------------------------------
# model integration
# ----------------------------------------------------------------------------------------------------------
class _Layer:
    __slots__ = ("ln1_w", "ln1_b", "wqkv", "wqkv_t", "wo4", "wo_t", "ls1", "ln2_w", "ln2_b", "w1", "w1_t", "w2", "w2_t", "ls2")

    def __init__(self, layer):
        attn, mlp = layer.self_attn, layer.mlp
        self.ln1_w = layer.input_layernorm.weight.detach().contiguous()
        self.ln1_b = layer.input_layernorm.bias.detach().contiguous()
        # weight-prepack (the module parameters stay untouched): q|k|v stacked and pre-transposed to [K, N] for cuBLAS;
        # Wo regrouped per head as [H][HD/4][N][4] so the fused kernel streams it with coalesced float4 loads
        self.wqkv = torch.cat([attn.q_proj.weight.detach(), attn.k_proj.weight.detach(), attn.v_proj.weight.detach()], 0).contiguous()
        self.wqkv_t = self.wqkv.t().contiguous()
        self.wo4 = attn.o_proj.weight.detach().view(D_MODEL, N_HEADS, HEAD_DIM // 4, 4).permute(1, 2, 0, 3).contiguous()
        self.wo_t = attn.o_proj.weight.detach().t().contiguous()      # long-T path: one cuBLAS GEMM after the attention
        self.ls1 = layer.self_attn_layer_scale.scale.detach().contiguous()
        self.ln2_w = layer.post_attention_layernorm.weight.detach().contiguous()
        self.ln2_b = layer.post_attention_layernorm.bias.detach().contiguous()
        self.w1 = mlp.fc1.weight.detach().contiguous()
        self.w2 = mlp.fc2.weight.detach().contiguous()
        self.w1_t = self.w1.t().contiguous()
        self.w2_t = self.w2.t().contiguous()
        self.ls2 = layer.mlp_layer_scale.scale.detach().contiguous()


_INFO: dict = {"transformers": 0, "layers": 0, "fused_calls": 0, "fallback_calls": 0, "launches_per_layer": 6,
               "kernels": ["cuda:skinny_qkv", "cuda:attention_rope+o_proj+l2_prefetch(w1,w2)", "triton:headsum+residual+layerscale+layernorm",
                           "cuda:skinny_fc1+gelu_epilogue", "cuda:skinny_fc2(4 K-chunk partials)", "triton:residual+layerscale+layernorm+partialsum"]}


def _rope_tables(tm, x, T: int):
    """cos/sin[:, :32] for positions 0..T-1 from the module's own MimiRotaryEmbedding (identical values)."""
    cache = tm._fk_rope
    tabs = cache.get(T)
    if tabs is None:
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        cos, sin = tm.rotary_emb(x, position_ids)                 # [1, T, 64] fp32, emb = cat(freqs, freqs)
        tabs = (cos[0, :, : HEAD_DIM // 2].contiguous(), sin[0, :, : HEAD_DIM // 2].contiguous())
        cache[T] = tabs
    return tabs


def _fused_forward(tm, x):
    B, T, D = x.shape
    M = B * T
    ext = tm._fk_ext
    layers = tm._fk_layers
    eps, window = tm._fk_eps, tm._fk_window
    cos, sin = _rope_tables(tm, x, T)
    dev = x.device
    f32 = torch.float32
    xn = torch.empty((M, D), device=dev, dtype=f32)
    part = None if (LONG_ATTN and T > MAX_T) else torch.empty((N_HEADS, M, D), device=dev, dtype=f32)
    h1 = torch.empty((M, D), device=dev, dtype=f32)
    bufs = (torch.empty((M, D), device=dev, dtype=f32), torch.empty((M, D), device=dev, dtype=f32))
    final = torch.empty((B, D, T), device=dev, dtype=f32).transpose(1, 2)   # .transpose(1, 2) downstream is contiguous
    contig = [T * D, D, 1]
    final_str = [D * T, 1, T]
    skinny = M <= SKINNY_MAX_M
    long_t = T > MAX_T                                              # windowed Triton attention + cuBLAS O projection
    pdl_row = skinny                                                # the row kernels join the programmatic chain (M <= 32)
    _row(x, list(x.stride()), None, None, layers[0].ln1_w, layers[0].ln1_b, None, None, xn, eps, B, T, pdl_row)   # xn = LN1(x)
    cur, cur_str = x, list(x.stride())
    last = len(layers) - 1
    if skinny:
        qkv = torch.empty((M, 3 * D), device=dev, dtype=f32)
        f = torch.empty((M, D_FF), device=dev, dtype=f32)
        mp = torch.empty((D_FF // 512, M, D), device=dev, dtype=f32)     # FC2 K-chunk partials, summed by the row kernel
    if long_t:
        av = torch.empty((M, D), device=dev, dtype=f32)                  # attention output (before the O projection)
        rq = torch.empty((M, D), device=dev, dtype=f32)                  # RoPE'd queries
        rkt = torch.empty((D, M), device=dev, dtype=f32)                 # RoPE'd keys, transposed
    for i, L in enumerate(layers):
        if skinny:
            ext.skinny_gemm(xn, L.wqkv, qkv, S_QKV, L.wo4, 0, PDL_GEMM if skinny else 0)   # qkv = xn @ Wqkv^T [M, 1536]; L2 prefetch of Wo
        else:
            qkv = torch.mm(xn, L.wqkv_t)
        if long_t:
            _attn_window(qkv, cos, sin, B, T, window, av, rq, rkt, 16 if T <= 400 else LONG_BM, LONG_BK)   # windowed fp32 attention
            o_at = torch.mm(av, L.wo_t)                                       # O projection (one GEMM)
        else:
            ext.attn_o(qkv, cos, sin, L.wo4, part, T, window, L.w1, L.w2, PDL_ATTN if skinny else 0)   # part[h] = attn_h @ Wo_h^T; L2 prefetch of W1, W2
            o_at = part
        _row(cur, cur_str, o_at, L.ls1, L.ln2_w, L.ln2_b, h1, contig, xn, eps, B, T, pdl_row)    # h1 = cur + ls1 * (sum_h) o ; xn = LN2(h1)
        if skinny:
            nxt = layers[i + 1].wqkv if i < last else None                              # next layer's Wqkv: half prefetched from each FC kernel
            ext.skinny_gemm(xn, L.w1, f, S_FC1, nxt[: 3 * D // 2] if i < last else None, 1, PDL_GEMM if skinny else 0)   # f = gelu(xn @ W1^T) [M, 2048]
            ext.skinny_gemm(f, L.w2, mp, S_FC2, nxt[3 * D // 2 :] if i < last else None, 0, PDL_GEMM if skinny else 0)  # mp[z] = f[:, chunk z] @ W2[:, chunk z]^T [4, M, 512]
            m = mp
        else:
            g = F.gelu(torch.mm(xn, L.w1_t))
            m = torch.mm(g, L.w2_t)
        if i < last:
            out, nxt = bufs[i % 2], layers[i + 1]
            _row(h1, contig, m, L.ls2, nxt.ln1_w, nxt.ln1_b, out, contig, xn, eps, B, T, pdl_row)   # out = h1 + ls2 * sum_z m[z] ; xn = LN1_next(out)
            cur, cur_str = out, contig
        else:
            _row(h1, contig, m, L.ls2, None, None, final, final_str, None, eps, B, T, pdl_row)  # final = h1 + ls2 * sum_z m[z]
    return final


def _patched_forward(self, hidden_states=None, attention_mask=None, position_ids=None, past_key_values=None,
                     use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
    cfg = self.config
    use_cache_ = use_cache if use_cache is not None else cfg.use_cache
    out_attn = output_attentions if output_attentions is not None else cfg.output_attentions
    out_hid = output_hidden_states if output_hidden_states is not None else cfg.output_hidden_states
    rd = return_dict if return_dict is not None else cfg.return_dict
    if (past_key_values is None and not use_cache_ and not out_attn and not out_hid and attention_mask is None
            and position_ids is None and not kwargs and not self.training and isinstance(hidden_states, torch.Tensor)
            and hidden_states.is_cuda and hidden_states.dtype == torch.float32 and hidden_states.dim() == 3
            and hidden_states.shape[-1] == D_MODEL and 0 < hidden_states.shape[1] <= self._fk_max_t):
        _INFO["fused_calls"] += 1
        h = _fused_forward(self, hidden_states)
        if not rd:
            return (h,)
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=None, hidden_states=None, attentions=None)
    _INFO["fallback_calls"] += 1
    return self._fk_stock_forward(hidden_states, attention_mask, position_ids, past_key_values, use_cache,
                                  output_attentions, output_hidden_states, return_dict, **kwargs)


WARMUP_T = (2, 7, 8, 13, 14, 25, 26, 125, 126, 313)   # 313: compiles the windowed kernel before any capture


def patch_transformers(model, ctx, warmup_t=WARMUP_T) -> dict:
    """Route MimiTransformerModel.forward (encoder + decoder transformers) through the fused path."""
    from transformers.models.mimi.modeling_mimi import MimiTransformerModel
    count = 0
    for name in ("encoder_transformer", "decoder_transformer"):
        tm = getattr(model, name, None)
        if not isinstance(tm, MimiTransformerModel) or getattr(tm, "_fk_layers", None) is not None:
            continue
        c = tm.config
        assert c.hidden_size == D_MODEL and c.num_attention_heads == N_HEADS and c.head_dim == HEAD_DIM \
            and c.intermediate_size == D_FF and c.hidden_act == "gelu" and c.num_key_value_heads == N_HEADS \
            and not c.attention_bias and c.rope_parameters.get("rope_type", "default") == "default"
        tm._fk_ext = _ext()                     # compiled (or loaded from the build cache) before any warm-up / capture
        tm._fk_layers = [_Layer(layer) for layer in tm.layers]
        dev = tm._fk_layers[0].wo4.device
        tm._fk_eps = float(c.norm_eps)
        tm._fk_window = int(c.sliding_window) if c.sliding_window else 1 << 30
        tm._fk_max_t = (1 << 30) if LONG_ATTN else min(tm._fk_window, MAX_T)
        tm._fk_rope = {}
        tm._fk_stock_forward = tm.forward
        tm.forward = types.MethodType(_patched_forward, tm)
        count += 1
        _INFO["layers"] += len(tm._fk_layers)
        if warmup_t:
            with torch.inference_mode():
                for t in warmup_t:
                    tm.forward(torch.zeros((1, t, D_MODEL), device=dev, dtype=torch.float32))
            torch.cuda.synchronize(dev)
    _INFO["transformers"] += count
    return _INFO
