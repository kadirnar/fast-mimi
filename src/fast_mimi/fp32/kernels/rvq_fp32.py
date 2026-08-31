"""Fused residual vector quantizer (Mimi RVQ) in exact fp32, v5: one launch per codebook stage (CUDA C++), the
stages chained with programmatic dependent launch (PDL); 8-warp blocks with a register-blocked distance phase,
hardware (redux) argmins, and the residual rows published to the next stage before its dependency wait.

Encode, per stage s (sequential), grid = C / 32 blocks of 8 compute warps (+ 1 publisher warp), block b owning codes
[32b, 32b + 32):
  launch          : stages s >= 1 and the finalize kernel are launched with cudaLaunchKernelExC and the
                    cudaLaunchAttributeProgrammaticStreamSerialization attribute (captured by the CUDA graph as
                    programmatic edges).  A stage may therefore start while the previous one is still running: it
                    issues its slab stream first, executes griddepcontrol.wait, and only then touches the previous
                    stage's partials / residual (full completion + memory flush of that grid).  Blocks use 48 KB of
                    shared memory and 128 registers per thread so that two stages' blocks are resident on one SM.
  residual rows   : block 0 of stage s publishes r_out (tile 0) through a per-stage flag (st.release by a ninth,
                    otherwise idle warp right after the prologue barrier); the blocks of stage s + 1 acquire that flag
                    and stream the rows in BEFORE their griddepcontrol.wait, while stage s is still computing.  Stage
                    s + 2 (whose launch implies every block of s + 1 has polled) clears the flag again, and the finalize
                    kernel clears the last one, so the flag buffer is allocated once and never needs a fill kernel.
  slab stream     : the block's (32, 256) fp32 slab -> shared memory with 16-byte cp.async; the 16-byte groups are
                    XOR-swizzled per row (g ^ (j & 7)) so the per-lane column reads are bank-conflict free, and the
                    partial sums alias the slab once it has been read into registers.  The slab -> register copy is
                    split around the prologue's two dependent L2 round trips (issued while those loads are in flight),
                    and once landed the block L2-prefetches its slab of stage s + 2 (stage 0: s + 1 and s + 2), so that
                    every later stage streams from L2 while the DRAM fetch has two stage-times to complete.
  fused prologue  : the previous stage's argmin is finished here: two rows per warp (16 rows per tile) read the NB
                    exact per-block minima of stage s-1 (coherent __ldcg loads: see the note in the kernel), take the
                    global min (ties -> lowest code index, exactly like torch.argmin), gather embed[s-1][code] and do
                    the reference's single fp32 subtraction r = r - e.  Block 0 writes the code and the new residual.
  exact distances : d[n, j] = sum_d (r[n, d] - e[j, d])^2 in fp32: warp w owns d chunk w = [32w, 32w + 32) for all
                    rows, lane l owns code j = l and keeps its 32 codebook values in registers; the residual rows are
                    read from shared memory as float4 broadcasts, four rows per group so that four independent fmaf
                    chains are in flight per thread.  Per (n, j) the 32-term chain is the same fixed sequence of fmaf as
                    before, and the 8 chunk partials are summed in the same fixed order -> bit-identical distances and
                    codes (run after run, and versus v2-v5), no atomics.
  block argmin    : two rows per warp, lane = code; the (value, index) argmins (here and in the prologue) are two
                    redux.sync.min.u32 reductions each - the distances are >= +0 so their bit patterns order like their
                    values, and the second reduction picks the lowest index among the minima (torch.argmin's rule);
                    the block emits (min, index) per row.
  finalize        : one small kernel (PDL, griddepcontrol.wait first) reduces the last partials, writes its codes.
Decode: one CUDA block per (b, t, codebook set): the codes go to shared memory, every thread then issues all its
row loads at once (float2 per code, up to 32 in flight) and adds them in the reference's sequential order
0 + embed[0][c0] + embed[1][c1] + ...; the previous one-warp Triton loop paid one dependent round trip per codebook.

Measured (cold-rotated codebooks, graph replay): 3.90 -> 3.55 us per stage at 1 s (13 rows), 13.45 -> 11.9 at 5 s,
3.53 -> 3.24 at 0.25 s; decode gather 4.1 -> 1.8 us per call.  Per stage at 1 s (median block, SM cycles): release
0.26 us, slab landed 0.10, partials 0.16 + argmin 0.06, gather + residual update + barrier 0.5-0.7 (L2 -> SM bandwidth:
every block reads all rows), distance loop 1.2 (max block 1.7), tail 0.25.

Past TILED_MIN_ROWS frames that per-stage design inverts: one shared-memory load per multiply-add and 64 blocks
each re-reading every residual row cost more than the 2 MB codebook slab they were hiding.  `rvq_tiled_dist_kernel`
gives every thread a TCT x RT tile of (code, row) instead, so a 32-dim chunk costs RT + TCT shared loads rather
than RT * TCT, and one launch pair replaces the PDL chain.  The arithmetic is the same eight 32-wide sequential
fmaf chains summed in chunk order, so its codes are bit-identical to the chain above (tests/test_v4_parity.py).

Codebooks are stacked once as (S, C, D); the extension is compiled once (torch caches it by source hash) and
`rvq_prepare` builds it ahead of time from apply().
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_MOD = None
_EXTRA_FLAGS: list[str] = []
PDL = True          # programmatic dependent launch between the stage kernels (captured into the CUDA graph as programmatic edges)
PREFETCH = True     # L2 prefetch of the next slab once ours has landed (two stages ahead measured slightly slower)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int D = 256;          // codebook dimension
constexpr int BC = 32;          // codes per block (one per lane)
constexpr int NW = 8;           // compute warps: warp w computes d chunk w for every row
constexpr int NTC = NW * 32;    // compute threads
constexpr int NT = NTC + 32;    // + one publisher warp (block 0: releases the residual-rows flag right after the
                                //   prologue barrier, so no compute warp stalls on the release fence)
constexpr int NTF = 256;        // threads of the finalize kernel
constexpr int NCH = 8;          // 32-wide d chunks, summed in a fixed order
constexpr int BN = 16;          // residual rows per tile (two per warp in the prologue, one per half-warp in the argmin)
constexpr int EST = D;          // slab row stride (no padding: 48 KB in total so that two stages' blocks share an SM under PDL);
                                // 16-byte groups are XOR-swizzled per row (g ^ (j & 7)) so the per-lane column reads are bank-conflict free
constexpr int NRMAX = 5;        // residual rows per register-blocked group of the distance loop (independent fmaf chains per thread)
constexpr int NBMAX = 64;       // partial entries per row read by one warp (2 per lane)

// shared memory: the slab (32 KB) is read once into registers, then the partial sums alias it
constexpr int SMEM_E = BC * EST, SMEM_R = BN * D, SMEM_P = NCH * BN * BC, SMEM_FLOATS = SMEM_E + SMEM_R;
static_assert(SMEM_P <= SMEM_E, "partials alias the slab");
static_assert(NW == NCH && 2 * NW == BN, "8 compute warps: one chunk per warp, two prologue rows per warp, two argmin rows per warp");
// programmatic dependent launch: let the next stage's grid launch / wait for the previous grid's completion
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// (value, index) argmin over the warp (WIDTH lanes: 32, or 16 for the two halves independently); ties -> lowest
// index.  The pair order is total, so the butterfly is exact and every lane of the group ends with the same result.
template <int WIDTH>
__device__ __forceinline__ void warp_argmin(float& v, int& i) {
    #pragma unroll
    for (int off = WIDTH / 2; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, v, off);
        int oi = __shfl_xor_sync(0xffffffffu, i, off);
        if (ov < v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
}

// cross-grid publication of the residual rows: block 0 of stage s sets flag[s] (release) once r_out is written;
// the dependent stage's blocks wait for it (acquire) BEFORE griddepcontrol.wait, so the residual rows stream in
// while the previous stage is still computing (they were written in its prologue, long before its end)
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
    int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release_gpu(int* p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// (value, index) argmin over the full warp in two hardware reductions: the distances are sums of squares (>= +0,
// never negative or NaN), so the order of their IEEE bit patterns as unsigned integers is their numeric order;
// the second reduction takes the lowest index among the lanes holding the minimum -> exactly torch.argmin's
// first-index rule.  (Measured: ~155 cycles vs ~350-500 for the 5-round shuffle butterfly, which is latency-bound.)
__device__ __forceinline__ unsigned redux_min_u32(unsigned v) {
    unsigned r; asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v)); return r;
}
__device__ __forceinline__ void redux_argmin(float& v, int& i) {
    const unsigned vb = __float_as_uint(v);
    const unsigned vmin = redux_min_u32(vb);
    const unsigned imin = redux_min_u32(vb == vmin ? (unsigned)i : 0x7fffffffu);
    v = __uint_as_float(vmin); i = (int)imin;
}
// two independent argmins (the reductions interleave, hiding each other's latency)
__device__ __forceinline__ void redux_argmin2(float& v0, int& i0, float& v1, int& i1) {
    const unsigned vb0 = __float_as_uint(v0), vb1 = __float_as_uint(v1);
    const unsigned vmin0 = redux_min_u32(vb0), vmin1 = redux_min_u32(vb1);
    const unsigned imin0 = redux_min_u32(vb0 == vmin0 ? (unsigned)i0 : 0x7fffffffu);
    const unsigned imin1 = redux_min_u32(vb1 == vmin1 ? (unsigned)i1 : 0x7fffffffu);
    v0 = __uint_as_float(vmin0); i0 = (int)imin0; v1 = __uint_as_float(vmin1); i1 = (int)imin1;
}

// global min over the NB per-block partials of one row (lane-strided), ties -> lowest code index
__device__ __forceinline__ int reduce_code(const float* __restrict__ pd, const int* __restrict__ pi, int NB, int lane) {
    float bv = CUDART_INF_F;
    int bi = 0x7fffffff;
    for (int b = lane; b < NB; b += 32) {
        float v = __ldcg(pd + b);
        int i = __ldcg(pi + b);
        if (v < bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    redux_argmin(bv, bi);
    return bi;
}

// the codebook values of this lane's code for 16-byte groups [Q0, Q1) of its warp's chunk (swizzled slab -> registers)
template <int Q0, int Q1>
__device__ __forceinline__ void load_e(const float* E, float (&e)[32], int lane, int chunk) {
    #pragma unroll
    for (int q = Q0; q < Q1; ++q) {
        const float4 v = *reinterpret_cast<const float4*>(&E[lane * EST + (((8 * chunk + q) ^ (lane & 7)) << 2)]);
        e[4 * q] = v.x; e[4 * q + 1] = v.y; e[4 * q + 2] = v.z; e[4 * q + 3] = v.w;
    }
}

// the loads of one prologue row: this warp's NB partials of the previous stage (2 per lane) and the residual row
// (2 float4 per lane).  All of it was written by another grid -> coherent loads (see stage_body).
__device__ __forceinline__ void load_partials(const float* __restrict__ pd_prev, const int* __restrict__ pi_prev, int n, int NB, bool ha, int lane,
                                              float (&pv)[2], int (&px)[2]) {
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int b = lane + 32 * k;
        pv[k] = (ha && b < NB) ? __ldcg(pd_prev + (size_t)n * NB + b) : CUDART_INF_F;
        px[k] = (ha && b < NB) ? __ldcg(pi_prev + (size_t)n * NB + b) : 0x7fffffff;
    }
}
__device__ __forceinline__ void load_residual(const float* __restrict__ r_in, int rs, int n, bool hr, int lane, float4& r0, float4& r1) {
    const float* rrow = r_in + (size_t)n * rs;
    r0 = hr ? __ldcg(reinterpret_cast<const float4*>(rrow + lane * 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
    r1 = hr ? __ldcg(reinterpret_cast<const float4*>(rrow + 128 + lane * 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
}

// L2 prefetch of this block's slab of the stage at pf_base (pf bit 0; normally s + 1, or stage 0 of the chain that
// follows) and the one after it (bit 1): one 128-byte line per thread and stage, fire-and-forget, issued while the
// memory system idles during the compute phase
__device__ __forceinline__ void prefetch_slabs(const float* pf_base, size_t cbs, int blk, int pf, int tid) {
    if (tid >= (BC * D * 4) / 128) return;
    #pragma unroll
    for (int which = 0; which < 2; ++which) {
        if ((pf >> which) & 1) {
            const char* p = reinterpret_cast<const char*>(pf_base + (size_t)which * cbs + (size_t)blk * BC * D) + (size_t)tid * 128;
            asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
        }
    }
}

// exact partial distances of rows nn0 .. nn0 + NR - 1 over this warp's 32-wide d range: NR independent fixed-order
// fmaf chains per lane (the residual float4 broadcasts are shared by the 32 codes)
template <int NR>
__device__ __forceinline__ void dist_rows(const float* R, const float (&e)[32], float* P, int chunk, int lane, int nn0) {
    float acc[NR];
    const float4* rp = reinterpret_cast<const float4*>(&R[nn0 * D + 32 * chunk]);
    #pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = 0.f;
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
        float4 r4[NR];
        #pragma unroll
        for (int i = 0; i < NR; ++i) r4[i] = rp[i * (D / 4) + q];
        #pragma unroll
        for (int i = 0; i < NR; ++i) {
            float t;
            t = r4[i].x - e[4 * q + 0]; acc[i] = fmaf(t, t, acc[i]);
            t = r4[i].y - e[4 * q + 1]; acc[i] = fmaf(t, t, acc[i]);
            t = r4[i].z - e[4 * q + 2]; acc[i] = fmaf(t, t, acc[i]);
            t = r4[i].w - e[4 * q + 3]; acc[i] = fmaf(t, t, acc[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < NR; ++i) P[(chunk * BN + nn0 + i) * BC + lane] = acc[i];
}

// one stage of one RVQ problem for block `blk` of NB.  r_in rows are `rs` floats apart (the projection GEMM's row
// layout is used directly: no copy kernel); r_out is dense (N, D).
__device__ __forceinline__ void
stage_body(const float* __restrict__ r_in, int rs, float* __restrict__ r_out,
           const float* __restrict__ cb_s, const float* __restrict__ cb_prev,
           const float* __restrict__ pd_prev, const int* __restrict__ pi_prev,
           float* __restrict__ pd, int* __restrict__ pi, long long* __restrict__ idx_prev,
           const int* flag_wait, int* flag_set, int* flag_reset, const float* __restrict__ pf_base,
           int N, int NB, int C, int has_prev, int pf, int blk, float* smem) {
    float* E = smem;
    float* R = smem + SMEM_E;
    float* P = smem;                             // aliases E (read into registers before P is written)
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int chunk = warp;
    const size_t cbs = (size_t)C * D;
    const bool pub = warp == NW;                 // the publisher warp: only block 0 keeps it (exited warps skip barriers)
    if (pub && (blk != 0 || flag_set == nullptr)) return;
    // the flag of stage s - 2 has been consumed by every block of stage s - 1 (they all polled it before triggering
    // this launch): clear it for the next call, so the flag buffer never needs a fill kernel
    if (flag_reset != nullptr && blk == 0 && tid == 0) *flag_reset = 0;

    // 0. stream this block's slab first (independent of the previous stage, so it is issued before the dependency
    //    wait); with programmatic dependent launch this happens while the previous stage is still running
    {
        const float* src = cb_s + (size_t)blk * BC * D;
        if (warp < NW) {
            #pragma unroll
            for (int e4 = tid; e4 < BC * D / 4; e4 += NTC) {
                const int j = e4 >> 6, g = (e4 & 63);
                cp_async_16(&E[j * EST + ((g ^ (j & 7)) << 2)], src + e4 * 4);
            }
        }
        cp_async_commit();
    }
    // 0b. the residual rows of tile 0: published by the previous stage's block 0 during ITS prologue (flag, release /
    //     acquire), or written before the chain started (stages 0 and 1) -> loaded before the dependency wait
    float4 r0[2], r1[2];
    if (flag_wait != nullptr) {
        if (lane == 0) { while (ld_acquire_gpu(flag_wait) == 0) __nanosleep(64); }
        __syncwarp();
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) load_residual(r_in, rs, warp + 8 * h, !pub && warp + 8 * h < N, lane, r0[h], r1[h]);
    float e[32];
    // 1. everything below reads the previous stage's partials / residual: wait for that grid (no-op without PDL).
    //    Those reads go through __ldcg (coherent ld.global.cg): a plain load of const __restrict__ data becomes
    //    ld.global.nc, which ptxas treats as invariant for the kernel's lifetime and may schedule ABOVE
    //    griddepcontrol.wait (observed: two pd_prev/pi_prev loads hoisted -> stale partials -> out-of-range gather).
    pdl_wait();
    // 2. fused prologue, two rows per warp: n = n0 + warp and n0 + warp + 8; loads first
    float pv[2][2]; int px[2][2];
    int n0 = 0;
    #pragma unroll
    for (int h = 0; h < 2; ++h) load_partials(pd_prev, pi_prev, warp + 8 * h, NB, !pub && has_prev && warp + 8 * h < N, lane, pv[h], px[h]);
    {
        // the slab has (normally) landed by now: its first half goes to registers while the partials are in flight
        cp_async_wait_all();
        __syncthreads();
        if (!pub) load_e<0, 4>(E, e, lane, chunk);
        prefetch_slabs(pf_base, cbs, blk, pf, tid);
        pdl_trigger();                         // the next stage may launch: its slab stream overlaps our compute
    }
    for (;;) {
        const int rows = min(BN, N - n0);
        if (has_prev) {
            int bi[2]; float bv[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                bv[h] = pv[h][0]; bi[h] = px[h][0];
                if (pv[h][1] < bv[h] || (pv[h][1] == bv[h] && px[h][1] < bi[h])) { bv[h] = pv[h][1]; bi[h] = px[h][1]; }
            }
            redux_argmin2(bv[0], bi[0], bv[1], bi[1]);
            float4 e0[2], e1[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                e0[h] = make_float4(0.f, 0.f, 0.f, 0.f); e1[h] = e0[h];
                if (!pub && warp + 8 * h < rows) {
                    const float* erow = cb_prev + (size_t)bi[h] * D;
                    e0[h] = *reinterpret_cast<const float4*>(erow + lane * 4);
                    e1[h] = *reinterpret_cast<const float4*>(erow + 128 + lane * 4);
                }
            }
            if (n0 == 0 && !pub) load_e<4, 8>(E, e, lane, chunk);   // second half overlaps the gather's round trip
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (!pub && warp + 8 * h < rows) {
                    const int n = n0 + warp + 8 * h;
                    r0[h].x -= e0[h].x; r0[h].y -= e0[h].y; r0[h].z -= e0[h].z; r0[h].w -= e0[h].w;
                    r1[h].x -= e1[h].x; r1[h].y -= e1[h].y; r1[h].z -= e1[h].z; r1[h].w -= e1[h].w;
                    if (blk == 0) {
                        *reinterpret_cast<float4*>(r_out + (size_t)n * D + lane * 4) = r0[h];
                        *reinterpret_cast<float4*>(r_out + (size_t)n * D + 128 + lane * 4) = r1[h];
                        if (lane == 0) idx_prev[n] = (long long)bi[h];
                    }
                }
            }
        } else if (n0 == 0 && !pub) {
            load_e<4, 8>(E, e, lane, chunk);
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (!pub && warp + 8 * h < rows) {
                *reinterpret_cast<float4*>(&R[(warp + 8 * h) * D + lane * 4]) = r0[h];
                *reinterpret_cast<float4*>(&R[(warp + 8 * h) * D + 128 + lane * 4]) = r1[h];
            }
        }
        __syncthreads();                           // R complete; every lane holds its slab values: P may overwrite E
        if (pub) {                                 // block 0's publisher warp: r_out rows of tile 0 are written -> release
            if (lane == 0) st_release_gpu(flag_set, 1);
            return;                                // (an exited warp no longer takes part in the barriers below)
        }
        // 3. exact partial distances over this warp's 32-wide d range, rows in groups (fixed fmaf order)
        int k = 0;
        if (NRMAX >= 5) for (; k + 5 <= rows; k += 5) dist_rows<5>(R, e, P, chunk, lane, k);
        if (NRMAX >= 4) for (; k + 4 <= rows; k += 4) dist_rows<4>(R, e, P, chunk, lane, k);
        if (NRMAX >= 3) for (; k + 3 <= rows; k += 3) dist_rows<3>(R, e, P, chunk, lane, k);
        if (NRMAX >= 2) for (; k + 2 <= rows; k += 2) dist_rows<2>(R, e, P, chunk, lane, k);
        for (; k < rows; ++k) dist_rows<1>(R, e, P, chunk, lane, k);
        __syncthreads();
        // 4. fixed-order sum of the 8 warp partials, block argmin (ties -> lowest index): rows 2w and 2w + 1 per
        //    warp, lane = code, the two hardware reductions interleaved; emit
        {
            const int nn0 = 2 * warp, nn1 = 2 * warp + 1;
            float v0 = 0.f, v1 = 0.f;
            #pragma unroll
            for (int c = 0; c < NCH; ++c) {
                v0 += P[(c * BN + nn0) * BC + lane];
                v1 += P[(c * BN + nn1) * BC + lane];
            }
            int i0 = blk * BC + lane, i1 = i0;
            redux_argmin2(v0, i0, v1, i1);
            if (lane == 0) {
                if (nn0 < rows) { pd[(size_t)(n0 + nn0) * NB + blk] = v0; pi[(size_t)(n0 + nn0) * NB + blk] = i0; }
                if (nn1 < rows) { pd[(size_t)(n0 + nn1) * NB + blk] = v1; pi[(size_t)(n0 + nn1) * NB + blk] = i1; }
            }
        }
        n0 += BN;
        if (n0 >= N) break;
        __syncthreads();                           // R and P are reused by the next tile
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = n0 + warp + 8 * h;
            load_partials(pd_prev, pi_prev, n, NB, has_prev && n < N, lane, pv[h], px[h]);
            load_residual(r_in, rs, n, n < N, lane, r0[h], r1[h]);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Register-tiled search for the row counts where the per-stage kernel above becomes ALU/L2 bound.
//
// That kernel gives every thread ONE code and streams the rows past it, so each MAC costs one shared-memory load
// (r), one sub and one fma, and every one of the 64 blocks re-reads all N residual rows and re-gathers the previous
// stage's embedding rows.  Both costs are invisible at 13 rows (2 MB of codebook dominates) and dominate by 313.
//
// Here a thread owns an RT x CT tile of (row, code), so a chunk's 32 dims cost RT + CT shared loads instead of
// RT * CT, and the residual is read once per code block instead of once per code.  The arithmetic per (n, j) is
// unchanged and therefore bit-identical: eight 32-wide chunks, each one sequential fmaf chain in increasing d,
// summed into acc in chunk order 0..7 -- exactly what dist_rows + the P reduction do above.
// ---------------------------------------------------------------------------------------------------------------
#ifndef FK_TCT
#define FK_TCT 4
#endif
constexpr int TCT = FK_TCT;     // codes per thread
constexpr int TBC = 16 * TCT;   // codes per block (16 code-threads x TCT)
constexpr int TNT = 256;        // threads per block (16 code-threads x 16 row-threads)
#ifndef FK_J32
#define FK_J32 1
#endif
#ifndef FK_J64
#define FK_J64 16
#endif
#ifndef FK_J128
#define FK_J128 1
#endif
#ifndef FK_TBR128
#define FK_TBR128 350   // rows above which the 128-row tile (RT=8) beats the 64-row one
#endif
#ifndef FK_TBR64
#define FK_TBR64 64      // ... and above which the 64-row tile (RT=4) beats the 32-row one.  Both thresholds
                         // keep the grid at 2 row-blocks x 32 code-blocks = 64 CTAs wherever they can: a third
                         // row-block is 96 CTAs on 70 SMs and costs ~45% (75 rows: 0.483 vs 0.334 ms)
#endif
constexpr int ESTP = 33;        // shared row stride in floats: +1 so the per-thread column reads hit 33 distinct banks

template <int TBR, int JSTEP>
__global__ void __launch_bounds__(TNT)
rvq_tiled_dist_kernel(const float* __restrict__ r_in, int rs, const float* __restrict__ cb_s,
                      float* __restrict__ pd, int* __restrict__ pi, int N, int C, int CB) {
    constexpr int RT = TBR / 16;                 // rows per thread
    static_assert(TBR % 16 == 0 && RT >= 1, "row tiling");
    extern __shared__ __align__(16) float tsm[];
    float* Es = tsm;                             // [TBC][ESTP]
    float* Rs = tsm + TBC * ESTP;                // [TBR][ESTP]
    const int tid = threadIdx.x, tc = tid & 15, tr = tid >> 4;
    const int bc = blockIdx.x, rb = blockIdx.y;
    const int n_base = rb * TBR, j_base = bc * TBC;
    // A thread's TCT codes are jj0, jj0 + JSTEP, ...  JSTEP = 16 makes the 16 code-threads of a warp half read 16
    // consecutive Es rows -- banks (row + d) % 32 at ESTP = 33, conflict-free -- where the contiguous JSTEP = 1
    // mapping has them read rows 4 * tc + j and hit every bank exactly twice.  It wins only at the 64-row tile
    // (250 rows 0.956 -> 0.861 ms, 188 0.882 -> 0.849); at 32 and 128 rows the contiguous mapping measures faster
    // (125 rows 0.619 vs 0.716, 388 rows 1.377 vs 1.406), so each tile keeps its own.  Either way the thread's
    // codes ascend with j, so the sequential scan and the butterfly keep torch.argmin's lowest-index rule.
    const int nn0 = tr * RT, jj0 = (JSTEP == 1) ? tc * TCT : tc;

    // the chunk after the one being computed is fetched into registers first, so its global latency sits under the
    // ~1300 FMAs of the current chunk instead of in front of the next barrier.  The shared rows are 33 floats apart
    // (conflict-free for the column reads below) and therefore not 16-byte aligned, which rules out cp.async here.
    constexpr int NE = TBC * 8 / TNT, NR = TBR * 8 / TNT;
    static_assert(TBC * 8 == NE * TNT && TBR * 8 == NR * TNT, "chunk staging must divide the block");
    float4 ebuf[NE], rbuf[NR];
    #define FK_LOAD_CHUNK(cc)                                                                                    \
        {                                                                                                        \
            const int dd = 32 * (cc);                                                                            \
            _Pragma("unroll") for (int u = 0; u < NE; ++u) {                                                      \
                const int q = tid + u * TNT, j = q >> 3, g = q & 7;                                               \
                ebuf[u] = *reinterpret_cast<const float4*>(cb_s + (size_t)(j_base + j) * D + dd + 4 * g);         \
            }                                                                                                    \
            _Pragma("unroll") for (int u = 0; u < NR; ++u) {                                                      \
                const int q = tid + u * TNT, n = q >> 3, g = q & 7;                                               \
                rbuf[u] = (n_base + n < N)                                                                        \
                    ? *reinterpret_cast<const float4*>(r_in + (size_t)(n_base + n) * rs + dd + 4 * g)             \
                    : make_float4(0.f, 0.f, 0.f, 0.f);                                                            \
            }                                                                                                    \
        }
    float acc[RT][TCT];
    FK_LOAD_CHUNK(0)
    for (int chunk = 0; chunk < NCH; ++chunk) {
        #pragma unroll
        for (int u = 0; u < NE; ++u) {
            const int q = tid + u * TNT, j = q >> 3, g = q & 7;
            float* dst = &Es[j * ESTP + 4 * g];
            dst[0] = ebuf[u].x; dst[1] = ebuf[u].y; dst[2] = ebuf[u].z; dst[3] = ebuf[u].w;
        }
        #pragma unroll
        for (int u = 0; u < NR; ++u) {
            const int q = tid + u * TNT, n = q >> 3, g = q & 7;
            float* dst = &Rs[n * ESTP + 4 * g];
            dst[0] = rbuf[u].x; dst[1] = rbuf[u].y; dst[2] = rbuf[u].z; dst[3] = rbuf[u].w;
        }
        __syncthreads();
        if (chunk + 1 < NCH) FK_LOAD_CHUNK(chunk + 1)
        float p[RT][TCT];
        #pragma unroll
        for (int i = 0; i < RT; ++i)
            #pragma unroll
            for (int j = 0; j < TCT; ++j) p[i][j] = 0.f;
        #pragma unroll 4
        for (int d = 0; d < 32; ++d) {
            float rv[RT], ev[TCT];
            #pragma unroll
            for (int i = 0; i < RT; ++i) rv[i] = Rs[(nn0 + i) * ESTP + d];
            #pragma unroll
            for (int j = 0; j < TCT; ++j) ev[j] = Es[(jj0 + JSTEP * j) * ESTP + d];
            #pragma unroll
            for (int i = 0; i < RT; ++i)
                #pragma unroll
                for (int j = 0; j < TCT; ++j) { const float t = rv[i] - ev[j]; p[i][j] = fmaf(t, t, p[i][j]); }
        }
        #pragma unroll
        for (int i = 0; i < RT; ++i)
            #pragma unroll
            for (int j = 0; j < TCT; ++j) acc[i][j] = (chunk == 0) ? p[i][j] : acc[i][j] + p[i][j];
        __syncthreads();
    }
    #undef FK_LOAD_CHUNK
    // per-row argmin over this block's TBC codes: CT sequential (increasing code), then the 16 code-threads
    // (lanes tc = 0..15 of each warp half) -- ties resolve to the lowest code index, as torch.argmin does
    #pragma unroll
    for (int i = 0; i < RT; ++i) {
        float bv = acc[i][0];
        int bi = j_base + jj0;
        #pragma unroll
        for (int j = 1; j < TCT; ++j) if (acc[i][j] < bv) { bv = acc[i][j]; bi = j_base + jj0 + JSTEP * j; }
        warp_argmin<16>(bv, bi);
        const int n = n_base + nn0 + i;
        if (tc == 0 && n < N) { pd[(size_t)n * CB + bc] = bv; pi[(size_t)n * CB + bc] = bi; }
    }
}

// reduce the CB per-code-block minima of each row, emit the code, and apply the reference's residual update
// r_out = r_in - embed[code]  (one warp per row, 8 floats per lane)
__global__ void __launch_bounds__(256)
rvq_tiled_final_kernel(const float* __restrict__ pd, const int* __restrict__ pi, const float* __restrict__ r_in, int rs,
                       float* __restrict__ r_out, const float* __restrict__ cb_s, long long* __restrict__ idx,
                       int N, int CB) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = blockIdx.x * 8 + w;
    if (n >= N) return;
    const int code = reduce_code(pd + (size_t)n * CB, pi + (size_t)n * CB, CB, lane);
    if (lane == 0) idx[n] = (long long)code;
    const float* rr = r_in + (size_t)n * rs;
    const float* er = cb_s + (size_t)code * D;
    float* orow = r_out + (size_t)n * D;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int o = lane * 8 + 4 * k;
        const float4 a = *reinterpret_cast<const float4*>(rr + o);
        const float4 b = *reinterpret_cast<const float4*>(er + o);
        *reinterpret_cast<float4*>(orow + o) = make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    }
}

void tiled_stage(const float* r_in, int rs, float* r_out, const float* cb_s, float* pd, int* pi, long long* idx,
                 int N, int C, int CB, cudaStream_t stream) {
    constexpr int SM_E = TBC * ESTP;
    if (N > FK_TBR128) {
        const size_t sm = (SM_E + 128 * ESTP) * sizeof(float);
        rvq_tiled_dist_kernel<128, FK_J128><<<dim3(CB, (N + 127) / 128), TNT, sm, stream>>>(r_in, rs, cb_s, pd, pi, N, C, CB);
    } else if (N > FK_TBR64) {
        const size_t sm = (SM_E + 64 * ESTP) * sizeof(float);
        rvq_tiled_dist_kernel<64, FK_J64><<<dim3(CB, (N + 63) / 64), TNT, sm, stream>>>(r_in, rs, cb_s, pd, pi, N, C, CB);
    } else {
        const size_t sm = (SM_E + 32 * ESTP) * sizeof(float);
        rvq_tiled_dist_kernel<32, FK_J32><<<dim3(CB, (N + 31) / 32), TNT, sm, stream>>>(r_in, rs, cb_s, pd, pi, N, C, CB);
    }
    rvq_tiled_final_kernel<<<(N + 7) / 8, 256, 0, stream>>>(pd, pi, r_in, rs, r_out, cb_s, idx, N, CB);
}

__global__ void __launch_bounds__(NT, 2)
rvq_stage_kernel(const float* __restrict__ r_in, int rs, float* __restrict__ r_out,
                 const float* __restrict__ cb_s, const float* __restrict__ cb_prev,
                 const float* __restrict__ pd_prev, const int* __restrict__ pi_prev,
                 float* __restrict__ pd, int* __restrict__ pi, long long* __restrict__ idx_prev,
                 const int* flag_wait, int* flag_set, int* flag_reset, const float* __restrict__ pf_base, int N, int NB, int C, int has_prev, int pf) {
    extern __shared__ __align__(16) float smem[];
    stage_body(r_in, rs, r_out, cb_s, cb_prev, pd_prev, pi_prev, pd, pi, idx_prev, flag_wait, flag_set, flag_reset, pf_base,
               N, NB, C, has_prev, pf, blockIdx.x, smem);
}

__global__ void __launch_bounds__(NTF)
rvq_final_kernel(const float* __restrict__ pd, const int* __restrict__ pi, long long* __restrict__ idx, int* flag_reset, int N, int NB) {
    pdl_wait();
    if (flag_reset != nullptr && blockIdx.x == 0 && threadIdx.x == 0) *flag_reset = 0;   // the last stage's consumers are done
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * (NTF / 32) + (threadIdx.x >> 5);
    if (n >= N) return;
    const int code = reduce_code(pd + (size_t)n * NB, pi + (size_t)n * NB, NB, lane);
    if (lane == 0) idx[n] = (long long)code;
}

// decode gather: block (n = b*T + t, h = codebook set), 128 threads x float2 = one 256-wide output row
constexpr int GS_MAX = 32;      // codebooks per set
__global__ void __launch_bounds__(128)
rvq_gather_kernel(const long long* __restrict__ codes, const float* __restrict__ cb0, const float* __restrict__ cb1,
                  float* __restrict__ out, int T, int S0, int S1, int C, long long sb, long long sk, long long st, long long out_sb) {
    __shared__ int sc[GS_MAX];
    const int n = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int b = n / T, t = n - b * T;
    const int S = h == 0 ? S0 : S1, koff = h == 0 ? 0 : S0;
    const float* cb = h == 0 ? cb0 : cb1;
    if (tid < S) sc[tid] = (int)codes[b * sb + (long long)(koff + tid) * sk + t * st];
    __syncthreads();
    float2 v[GS_MAX];
    #pragma unroll
    for (int s = 0; s < GS_MAX; ++s)
        v[s] = (s < S) ? *reinterpret_cast<const float2*>(cb + ((size_t)s * C + sc[s]) * D + 2 * tid) : make_float2(0.f, 0.f);
    float2 acc = make_float2(0.f, 0.f);
    #pragma unroll
    for (int s = 0; s < GS_MAX; ++s)
        if (s < S) { acc.x += v[s].x; acc.y += v[s].y; }
    float* o = out + (size_t)b * out_sb + (size_t)(h * D + 2 * tid) * T + t;
    o[0] = acc.x; o[T] = acc.y;
}

bool g_attr_set = false;

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

}  // namespace

static void set_stage_attrs() {
    if (!g_attr_set) {
        const int smem = (int)(SMEM_FLOATS * sizeof(float));
        cudaFuncSetAttribute(rvq_stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_attr_set = true;
    }
}

static void check_rows(const torch::Tensor& r, int D2) {
    TORCH_CHECK(r.is_cuda() && r.dtype() == torch::kFloat32 && r.dim() == 2 && r.stride(1) == 1 && r.size(1) >= D2 && r.stride(0) % 4 == 0, "residual rows: fp32, unit column stride, 16-byte aligned rows");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(r.data_ptr()) & 15) == 0, "16-byte alignment");
}

void rvq_stage(torch::Tensor r_in, torch::Tensor r_out, torch::Tensor cb, int64_t stage,
               torch::Tensor pd, torch::Tensor pi, torch::Tensor idx, torch::Tensor flags, bool has_prev, bool has_next,
               torch::Tensor pf_cb, int64_t pf_stage, int64_t pf, bool pdl) {
    TORCH_CHECK(cb.is_cuda() && cb.dtype() == torch::kFloat32 && cb.is_contiguous() && cb.dim() == 3, "cb must be a contiguous fp32 (S, C, D) CUDA tensor");
    TORCH_CHECK(cb.size(2) == D && cb.size(1) % BC == 0 && cb.size(1) / BC <= NBMAX, "codebook must be (S, C, 256) with C a multiple of 32, C <= 2048");
    check_rows(r_in, D);
    TORCH_CHECK(r_out.is_contiguous() && (reinterpret_cast<uintptr_t>(r_out.data_ptr()) & 15) == 0, "r_out must be contiguous, 16-byte aligned");
    TORCH_CHECK(pd.is_contiguous() && pi.is_contiguous() && idx.is_contiguous(), "workspaces must be contiguous");
    TORCH_CHECK(flags.is_contiguous() && flags.dtype() == torch::kInt32 && flags.numel() >= stage + 1, "flags: int32 (num_q,), zeroed per call");
    const int N = (int)r_in.size(0), C = (int)cb.size(1), NB = C / BC;
    TORCH_CHECK(pd.size(1) == N && pd.size(2) == NB && pi.sizes() == pd.sizes() && idx.size(1) == N, "workspace shapes");
    TORCH_CHECK(stage >= 0 && stage < cb.size(0) && stage < pd.size(0) && (!has_prev || stage >= 1), "stage out of range");
    TORCH_CHECK(pf_cb.is_cuda() && pf_cb.dtype() == torch::kFloat32 && pf_cb.is_contiguous() && pf_cb.dim() == 3 && pf_cb.size(1) == cb.size(1) && pf_cb.size(2) == D, "prefetch codebook: same (C, D)");
    TORCH_CHECK(!(pf & 1) || pf_stage < pf_cb.size(0), "prefetch stage out of range");
    TORCH_CHECK(!(pf & 2) || pf_stage + 1 < pf_cb.size(0), "prefetch stage + 1 out of range");
    set_stage_attrs();
    const size_t smem = (size_t)SMEM_FLOATS * sizeof(float);
    const float* cbp = cb.data_ptr<float>();
    const size_t cbs = (size_t)C * D, ps = (size_t)N * NB;
    const int64_t prev = has_prev ? stage - 1 : stage;
    const float* a_rin = r_in.data_ptr<float>(); int a_rs = (int)r_in.stride(0); float* a_rout = r_out.data_ptr<float>();
    const float* a_cbs = cbp + stage * cbs; const float* a_cbp = cbp + prev * cbs;
    const float* a_pdp = pd.data_ptr<float>() + prev * ps; const int* a_pip = pi.data_ptr<int>() + prev * ps;
    float* a_pd = pd.data_ptr<float>() + stage * ps; int* a_pi = pi.data_ptr<int>() + stage * ps;
    long long* a_idx = reinterpret_cast<long long*>(idx.data_ptr<int64_t>()) + prev * (size_t)N;
    // r_in of stage s >= 2 is r_out of stage s - 1 (published by its block 0 through flags[s - 1]); stages 0 and 1 read
    // the projection rows, complete before the chain started.  Stage s >= 1 publishes its own r_out through flags[s].
    TORCH_CHECK(flags.numel() > stage, "flags: int32 (>= num_q,), zero between calls");
    // a flag is set only when a stage s + 1 of THIS chain will consume (and stage s + 2 / the finalize kernel clear) it
    const int* a_fw = (has_prev && stage >= 2) ? flags.data_ptr<int>() + (stage - 1) : nullptr;           // wait for r_out of s - 1
    int* a_fs = (has_prev && has_next) ? flags.data_ptr<int>() + stage : nullptr;                          // publish r_out of s
    int* a_fr = (stage >= 3) ? flags.data_ptr<int>() + (stage - 2) : nullptr;                              // clear the consumed flag
    const float* a_pfb = pf_cb.data_ptr<float>() + (size_t)pf_stage * cbs;
    int a_N = N, a_NB = NB, a_C = C, a_hp = has_prev ? 1 : 0, a_pf = (int)pf;
    void* args[] = {&a_rin, &a_rs, &a_rout, &a_cbs, &a_cbp, &a_pdp, &a_pip, &a_pd, &a_pi, &a_idx, &a_fw, &a_fs, &a_fr, &a_pfb, &a_N, &a_NB, &a_C, &a_hp, &a_pf};
    launch_ex((const void*)rvq_stage_kernel, dim3(NB), dim3(NT), smem, pdl, args);
}

void rvq_gather(torch::Tensor codes, torch::Tensor cb0, torch::Tensor cb1, int64_t S0, torch::Tensor out) {
    TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kInt64 && codes.dim() == 3, "codes: int64 (B, K, T)");
    TORCH_CHECK(cb0.is_cuda() && cb1.is_cuda() && cb0.dtype() == torch::kFloat32 && cb1.dtype() == torch::kFloat32 && cb0.is_contiguous() && cb1.is_contiguous(), "codebooks: contiguous fp32");
    TORCH_CHECK(cb0.size(2) == D && cb1.size(2) == D && cb0.size(1) == cb1.size(1), "codebooks: (S, C, 256)");
    const int B = (int)codes.size(0), K = (int)codes.size(1), T = (int)codes.size(2);
    const int S1 = K - (int)S0;
    TORCH_CHECK(S0 >= 1 && S1 >= 1 && S0 <= GS_MAX && S1 <= GS_MAX && S0 <= cb0.size(0) && S1 <= cb1.size(0), "codebook counts");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dtype() == torch::kFloat32 && out.dim() == 3 && out.size(0) == B && out.size(1) == 2 * D && out.size(2) == T, "out: contiguous (B, 512, T) fp32");
    const long long* a_codes = reinterpret_cast<const long long*>(codes.data_ptr<int64_t>());
    const float* a_cb0 = cb0.data_ptr<float>(); const float* a_cb1 = cb1.data_ptr<float>(); float* a_out = out.data_ptr<float>();
    int a_T = T, a_S0 = (int)S0, a_S1 = S1, a_C = (int)cb0.size(1);
    long long a_sb = codes.stride(0), a_sk = codes.stride(1), a_st = codes.stride(2), a_osb = (long long)out.stride(0);
    void* args[] = {&a_codes, &a_cb0, &a_cb1, &a_out, &a_T, &a_S0, &a_S1, &a_C, &a_sb, &a_sk, &a_st, &a_osb};
    launch_ex((const void*)rvq_gather_kernel, dim3(B * T, 2), dim3(128), 0, false, args);
}

void rvq_final(torch::Tensor pd, torch::Tensor pi, torch::Tensor idx, torch::Tensor flags, int64_t stage, bool pdl) {
    const int N = (int)pd.size(1), NB = (int)pd.size(2);
    const size_t ps = (size_t)N * NB;
    const float* a_pd = pd.data_ptr<float>() + stage * ps; const int* a_pi = pi.data_ptr<int>() + stage * ps;
    long long* a_idx = reinterpret_cast<long long*>(idx.data_ptr<int64_t>()) + stage * (size_t)N;
    int* a_fr = (stage >= 2) ? flags.data_ptr<int>() + (stage - 1) : nullptr;    // the flag stage `stage` consumed
    int a_N = N, a_NB = NB;
    void* args[] = {&a_pd, &a_pi, &a_idx, &a_fr, &a_N, &a_NB};
    launch_ex((const void*)rvq_final_kernel, dim3((N + NTF / 32 - 1) / (NTF / 32)), dim3(NTF), 0, pdl, args);
}

// register-tiled search: two launches per stage, no PDL/flag chain (at these row counts the stage is long enough
// that the launch pair costs less than the redundant per-block row reads the fused chain pays)
void rvq_tiled(torch::Tensor r_in, torch::Tensor r_out, torch::Tensor cb, int64_t stage, torch::Tensor pd,
               torch::Tensor pi, torch::Tensor idx, int64_t num_q) {
    const int N = (int)r_in.size(0), C = (int)cb.size(1), CB = C / TBC;
    TORCH_CHECK(cb.size(2) == D && C % TBC == 0, "codebook must be (S, C, 256) with C a multiple of 64");
    auto stream = c10::cuda::getCurrentCUDAStream();
    tiled_stage(r_in.data_ptr<float>(), (int)r_in.stride(0), r_out.data_ptr<float>(),
                cb.data_ptr<float>() + (size_t)stage * C * D, pd.data_ptr<float>(), pi.data_ptr<int>(),
                reinterpret_cast<long long*>(idx.data_ptr<int64_t>()) + (size_t)stage * N, N, C, CB, stream);
}
"""
CPP_SRC = """
void rvq_stage(torch::Tensor r_in, torch::Tensor r_out, torch::Tensor cb, int64_t stage, torch::Tensor pd, torch::Tensor pi, torch::Tensor idx, torch::Tensor flags, bool has_prev, bool has_next, torch::Tensor pf_cb, int64_t pf_stage, int64_t pf, bool pdl);
void rvq_final(torch::Tensor pd, torch::Tensor pi, torch::Tensor idx, torch::Tensor flags, int64_t stage, bool pdl);
void rvq_gather(torch::Tensor codes, torch::Tensor cb0, torch::Tensor cb1, int64_t S0, torch::Tensor out);
void rvq_tiled(torch::Tensor r_in, torch::Tensor r_out, torch::Tensor cb, int64_t stage, torch::Tensor pd, torch::Tensor pi, torch::Tensor idx, int64_t num_q);
"""


def _ext():
    """Compile the exact fp32 RVQ search once (torch caches by source hash under .fast-kernel/build; no fast-math)."""
    global _MOD
    if _MOD is None:
        from .._compat import build_dir, ensure_cuda_home
        ensure_cuda_home()                      # must precede the cpp_extension import (it reads CUDA_HOME on import)
        from torch.utils.cpp_extension import load_inline
        build = build_dir(None) / ("fk_rvq_exact" + ("".join(_EXTRA_FLAGS).replace("-D", "_").replace("=", "") if _EXTRA_FLAGS else ""))
        build.mkdir(parents=True, exist_ok=True)
        _MOD = load_inline(name="fk_rvq_exact" + ("".join(_EXTRA_FLAGS).replace("-D", "_").replace("=", "") if _EXTRA_FLAGS else ""), cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                           functions=["rvq_stage", "rvq_final", "rvq_gather", "rvq_tiled"], extra_cuda_cflags=["-O3"] + _EXTRA_FLAGS, build_directory=str(build))
    return _MOD


_CHAIN = {"last": None, "next": {}}     # which codebook stack follows which in the call sequence (semantic -> acoustic)
_FLAGS: dict[tuple, torch.Tensor] = {}   # per device: the "residual rows published" flags; the kernels leave it zeroed


def _flags(device, S: int) -> torch.Tensor:
    key = (str(device), max(S, 64))
    f = _FLAGS.get(key)
    if f is None:
        f = _FLAGS[key] = torch.zeros((max(S, 64),), device=device, dtype=torch.int32)
    return f


def _rows(x: torch.Tensor) -> torch.Tensor:
    """(B, D, T) -> (B*T, D) rows; the stage kernels read strided rows in place (unit column stride, 16-byte aligned
    rows), so the split quantizer's projection rows need no copy."""
    B, D, T = x.shape
    r = x.permute(0, 2, 1).reshape(B * T, D)
    if r.stride(1) == 1 and r.stride(0) % 4 == 0 and r.data_ptr() % 16 == 0:
        return r
    return r.contiguous()


def _note_chain(cb: torch.Tensor):
    """Learn the codebook stack that follows this one (the split quantizer calls the semantic chain, then the
    acoustic one); a wrong guess only wastes a prefetch hint."""
    last = _CHAIN["last"]
    if last is not None and last.data_ptr() != cb.data_ptr():
        _CHAIN["next"][last.data_ptr()] = cb
    _CHAIN["last"] = cb
    nxt = _CHAIN["next"].get(cb.data_ptr())
    if nxt is not None and (nxt.shape[1] != cb.shape[1] or nxt.shape[2] != cb.shape[2] or not nxt.is_contiguous()):
        nxt = None
    return nxt


def rvq_prepare(cb: torch.Tensor) -> None:
    """Weight-prepack hook called once from apply(): nothing to precompute for the exact search; builds the extension
    ahead of the first (warm-up) call so compilation never overlaps a timed run."""
    _ext()


REF_MIN_ROWS = 400   # rows above which the reference cdist formulation is both faster and bit-identical


_AUG: dict[int, torch.Tensor] = {}      # codebook stack -> ATen's `_euclidean_dist` right operand, built once


def _augmented(cb: torch.Tensor) -> torch.Tensor:
    """`cat([embed, 1, ||embed||^2], -1)` per stage -- the right operand `torch.cdist(p=2)` builds internally.

    cdist rebuilds it on every call for every stage: a (C, D + 2) cat plus a pow / sum over the whole codebook,
    ~3 ms per 100 s round trip across the 32 stages.  The codebooks are constant, so it is hoisted here; the GEMM
    then sees the same values in the same layout and the distances stay bit-identical.
    """
    key = cb.data_ptr()
    aug = _AUG.get(key)
    if aug is None or aug.shape[:2] != cb.shape[:2]:
        norm = cb.pow(2).sum(-1, keepdim=True)
        aug = torch.cat([cb, torch.ones_like(norm), norm], -1).contiguous()
        _AUG[key] = aug
    return aug


def _cdist_argmin(rows: torch.Tensor, aug_s: torch.Tensor) -> torch.Tensor:
    """`torch.cdist(rows[None], embed[None], p=2)[0].argmin(-1)` with the right operand prepacked.

    Same expression as ATen's `_euclidean_dist`: cat([-2 x, ||x||^2, 1]) @ cat([e, 1, ||e||^2])^T, then
    clamp_min_(0).sqrt_() -- the sqrt is kept because it can round two distinct squared distances onto the same
    float, and argmin resolves that tie to the lower index exactly as the reference does.
    """
    x1 = rows[None]
    x1_norm = x1.pow(2).sum(-1, keepdim=True)
    x1_ = torch.cat([x1.mul(-2), x1_norm, torch.ones_like(x1_norm)], -1)
    d = x1_.matmul(aug_s[None].transpose(-2, -1))
    return d.clamp_min_(0).sqrt_()[0].argmin(dim=-1)


def rvq_encode_ref(x: torch.Tensor, cb: torch.Tensor, num_q: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """The reference search, stage by stage: `torch.cdist(rows, embed, p=2).argmin(-1)` then `r -= embed[code]`.

    Bit-identical to `transformers`' `MimiResidualVectorQuantizer.encode` (same ops, same shapes, so the same cuBLAS
    kernel and the same rounding), and past a few hundred frames faster than the fused per-stage kernel, which is tuned
    for the tens-of-rows shapes and computes `sum_d (r - e)^2` instead -- a different fp32 expression whose argmin can
    disagree with the reference's on a knife-edge frame (7 codes in 20000 at 50 s).  Rows below `REF_MIN_ROWS` keep the
    fused kernel, where it is several times faster and its codes have always matched.
    """
    B, D, T = x.shape
    N = B * T
    idx = out if out is not None else torch.empty((num_q, N), device=x.device, dtype=torch.int64)
    aug = _augmented(cb)
    residual = x
    for s in range(num_q):
        embed = cb[s]
        rows = residual.permute(0, 2, 1).reshape(N, D)
        ind = _cdist_argmin(rows, aug[s])
        idx[s] = ind
        residual = residual - torch.nn.functional.embedding(ind, embed).view(B, T, D).permute(0, 2, 1)
    return idx.view(num_q, B, T)


TILED_MIN_ROWS = 48   # rows above which the register-tiled kernel beats the fused per-stage chain, 32 stages:
                      # 63 rows 0.394 -> 0.334 ms, 125 0.702 -> 0.485, 250 1.293 -> 0.807, 313 1.596 -> 1.129,
                      # 388 1.955 -> 1.333.  Only the 65-90 window still favours the chain (75 rows 0.456 vs
                      # 0.483), by less than the 63-row win next to it.


def rvq_encode_tiled(x: torch.Tensor, cb: torch.Tensor, num_q: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """The same search as `rvq_encode`'s fused chain -- same 8 x 32 fmaf order, same lowest-index argmin, so the
    codes are bit-identical -- with each thread owning an RT x CT tile of (row, code) instead of one code.

    The fused chain gives every thread a single code and streams the rows past it, so one shared-memory load is
    spent per MAC and every one of its 64 blocks re-reads all N residual rows and re-gathers the previous stage's
    embedding rows.  Both are free at 13 rows (the 2 MB codebook slab dominates) and dominate by 313, which is why
    this path takes over above TILED_MIN_ROWS.
    """
    B, D_, T = x.shape
    S, C, _ = cb.shape
    N = B * T
    ext = _ext()
    CB = C // 64
    r0 = _rows(x)
    rbuf = torch.empty((2, N, D_), device=x.device, dtype=torch.float32)
    pd = torch.empty((N, CB), device=x.device, dtype=torch.float32)
    pi = torch.empty((N, CB), device=x.device, dtype=torch.int32)
    idx = out if out is not None else torch.empty((num_q, N), device=x.device, dtype=torch.int64)
    for s in range(num_q):
        rin = r0 if s == 0 else rbuf[(s - 1) % 2]
        ext.rvq_tiled(rin, rbuf[s % 2], cb, s, pd, pi, idx, num_q)
    return idx.view(num_q, B, T)


def rvq_encode(x: torch.Tensor, cb: torch.Tensor, num_q: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """x: (B, D, T) fp32 residual input; cb: (S, C, D) fp32 codebooks. Returns codes (num_q, B, T) int64
    (written into `out`, a contiguous (num_q, B*T) int64 buffer, when given)."""
    B, D, T = x.shape
    S, C, _ = cb.shape
    N = B * T
    if N >= REF_MIN_ROWS:
        return rvq_encode_ref(x, cb, num_q, out)
    if N >= TILED_MIN_ROWS:
        return rvq_encode_tiled(x, cb, num_q, out)
    ext = _ext()
    NB = C // 32
    r0 = _rows(x)
    rbuf = torch.empty((2, N, D), device=x.device, dtype=torch.float32)
    pd = torch.empty((num_q, N, NB), device=x.device, dtype=torch.float32)
    pi = torch.empty((num_q, N, NB), device=x.device, dtype=torch.int32)
    idx = out if out is not None else torch.empty((num_q, N), device=x.device, dtype=torch.int64)
    flags = _flags(x.device, S)                                          # per-stage "residual rows published" flags (self-resetting)
    nxt = _note_chain(cb) if num_q <= 2 else None      # a short (semantic) chain: prefetch the next chain's first slabs
    for s in range(num_q):
        rin = r0 if s <= 1 else rbuf[(s + 1) % 2]
        pf, pf_cb, pf_stage = 0, cb, min(s + 1, S - 1)
        if PREFETCH:
            if s + 1 < num_q:
                pf = 1                         # this block's slab of the next stage
            elif nxt is not None:
                pf, pf_cb, pf_stage = (3 if nxt.shape[0] >= 2 else 1), nxt, 0   # the chain that follows starts warm
        ext.rvq_stage(rin, rbuf[s % 2], cb, s, pd, pi, idx, flags, s > 0, s + 1 < num_q, pf_cb, pf_stage, pf, PDL and s > 0)
    ext.rvq_final(pd, pi, idx, flags, num_q - 1, PDL)
    return idx.view(num_q, B, T)


@triton.jit
def _rvq_decode_kernel(codes_ptr, cb_ptr, out_ptr, T, S, stride_b, stride_k, stride_t, out_sb,
                       C: tl.constexpr, D: tl.constexpr):
    n = tl.program_id(0)
    b = n // T
    t = n % T
    ds = tl.arange(0, D)
    acc = tl.zeros((D,), dtype=tl.float32)
    for s in range(S):
        code = tl.load(codes_ptr + b * stride_b + s * stride_k + t * stride_t).to(tl.int64)
        acc = acc + tl.load(cb_ptr + (s * C + code) * D + ds)
    tl.store(out_ptr + b * out_sb + ds * T + t, acc)


@triton.jit
def _rvq_decode2_kernel(codes_ptr, cb0_ptr, cb1_ptr, out_ptr, T, S0, S1, stride_b, stride_k, stride_t, out_sb,
                        C: tl.constexpr, D: tl.constexpr):
    """half 0: rows [0, D) of out <- sum over codebooks 0..S0-1 (codes 0..S0-1); half 1: rows [D, 2D) <- codebooks
    of the second set (codes S0..S0+S1-1); the sums start from 0 and add in the reference's sequential order."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    b = n // T
    t = n % T
    ds = tl.arange(0, D)
    acc = tl.zeros((D,), dtype=tl.float32)
    if h == 0:
        for s in range(S0):
            code = tl.load(codes_ptr + b * stride_b + s * stride_k + t * stride_t).to(tl.int64)
            acc = acc + tl.load(cb0_ptr + (s * C + code) * D + ds)
    else:
        for s in range(S1):
            code = tl.load(codes_ptr + b * stride_b + (S0 + s) * stride_k + t * stride_t).to(tl.int64)
            acc = acc + tl.load(cb1_ptr + (s * C + code) * D + ds)
    tl.store(out_ptr + b * out_sb + (h * D + ds) * T + t, acc)


def rvq_decode_split(codes: torch.Tensor, cb_sem: torch.Tensor, cb_ac: torch.Tensor, ns: int, out: torch.Tensor) -> torch.Tensor:
    """codes: (B, K, T) with K = ns + acoustic count; out: contiguous (B, 2D, T): rows [0, D) <- semantic gather sum,
    rows [D, 2D) <- acoustic gather sum, in ONE launch."""
    B, K, T = codes.shape
    _, C, D = cb_sem.shape
    assert out.is_contiguous() and out.shape == (B, 2 * D, T) and K > ns
    if codes.dtype == torch.int64 and D == 256 and ns <= 32 and K - ns <= 32:
        _ext().rvq_gather(codes, cb_sem, cb_ac, ns, out)       # all row loads in flight at once, sequential sum
        return out
    sb, sk, st = codes.stride()
    _rvq_decode2_kernel[(B * T, 2)](codes, cb_sem, cb_ac, out, T, ns, K - ns, sb, sk, st, out.shape[1] * T, C=C, D=D, num_warps=1)
    return out


def rvq_decode(codes: torch.Tensor, cb: torch.Tensor, out: torch.Tensor | None = None, offset: int = 0) -> torch.Tensor:
    """codes: (B, S, T) integer (any strides); cb: (S, C, D). Returns (B, D, T) fp32 sum of embeddings
    (0 + embed[0][c0] + embed[1][c1] + ..., the reference's order).  With `out` (a contiguous (B, DO, T) buffer,
    DO >= offset + D) the rows [offset, offset + D) of `out` receive the result and `out` is returned."""
    B, S, T = codes.shape
    _, C, D = cb.shape
    if out is None:
        out = torch.empty((B, D, T), device=codes.device, dtype=torch.float32)
        dst, out_sb = out, D * T
    else:
        assert out.is_contiguous() and out.shape[0] == B and out.shape[2] == T and out.shape[1] >= offset + D
        dst, out_sb = out[:, offset:offset + D], out.shape[1] * T
    sb, sk, st = codes.stride()
    _rvq_decode_kernel[(B * T,)](codes, cb, dst, T, S, sb, sk, st, out_sb, C=C, D=D, num_warps=1)
    return out
