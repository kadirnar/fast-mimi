#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__device__ __forceinline__ float mimi_elu(float value) {
  return value <= 0.0f ? __expf(value) - 1.0f : value;
}

__device__ __forceinline__ float mimi_tf32_rne(float value) {
  unsigned bits = __float_as_uint(value);
  unsigned exponent = bits & 0x7f800000u;
  if (exponent == 0x7f800000u) {
    return value;
  }
  bits += 0x00000fffu + ((bits >> 13) & 1u);
  return __uint_as_float(bits & 0xffffe000u);
}

template <int Tile, int Warps>
__global__ void decoder12_final_wmma_kernel(
    const half* __restrict__ branch,
    const half* __restrict__ pointwise_weight,
    const float* __restrict__ residual,
    const float* __restrict__ pointwise_bias,
    const float* __restrict__ final_weight,
    const float* __restrict__ final_bias,
    float* __restrict__ output,
    int length) {
  constexpr int K = 32;
  constexpr int N = 64;
  constexpr int MmaTile = 16;
  constexpr int Ntiles = 4;
  constexpr int ActiveWarps = (Tile + 2 + MmaTile - 1) / MmaTile;
  constexpr int TileElements = MmaTile * MmaTile;
  static_assert(ActiveWarps <= Warps);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile_start = int(blockIdx.x) * Tile;
  extern __shared__ unsigned char storage_bytes[];
  half* shared_a = reinterpret_cast<half*>(storage_bytes);
  float* shared_acc = reinterpret_cast<float*>(shared_a + Warps * TileElements);
  float* activated = shared_acc + Warps * TileElements;

  if (warp < ActiveWarps) {
    const int local_row_base = warp * MmaTile;
    const int time_base = tile_start - 2 + local_row_base;
    half* warp_a = shared_a + warp * TileElements;
    float* warp_acc = shared_acc + warp * TileElements;
    wmma::fragment<wmma::accumulator, MmaTile, MmaTile, MmaTile, float>
        accumulators[Ntiles];
#pragma unroll
    for (int tile_n = 0; tile_n < Ntiles; ++tile_n) {
      wmma::fill_fragment(accumulators[tile_n], 0.0f);
    }
#pragma unroll
    for (int k_base = 0; k_base < K; k_base += MmaTile) {
      for (int index = lane; index < TileElements; index += 32) {
        const int time = time_base + index / MmaTile;
        const int k = k_base + index % MmaTile;
        warp_a[index] = (time >= 0 && time < length)
                            ? branch[time * K + k]
                            : __float2half_rn(0.0f);
      }
      __syncwarp();
      wmma::fragment<wmma::matrix_a, MmaTile, MmaTile, MmaTile, half,
                     wmma::row_major>
          fragment_a;
      wmma::load_matrix_sync(fragment_a, warp_a, MmaTile);
#pragma unroll
      for (int tile_n = 0; tile_n < Ntiles; ++tile_n) {
        const int column = tile_n * MmaTile;
        wmma::fragment<wmma::matrix_b, MmaTile, MmaTile, MmaTile, half,
                       wmma::col_major>
            fragment_b;
        wmma::load_matrix_sync(
            fragment_b, pointwise_weight + column * K + k_base, K);
        wmma::mma_sync(
            accumulators[tile_n], fragment_a, fragment_b,
            accumulators[tile_n]);
      }
      __syncwarp();
    }

#pragma unroll
    for (int tile_n = 0; tile_n < Ntiles; ++tile_n) {
      wmma::store_matrix_sync(
          warp_acc, accumulators[tile_n], MmaTile, wmma::mem_row_major);
      __syncwarp();
      for (int index = lane; index < TileElements; index += 32) {
        const int local_row = local_row_base + index / MmaTile;
        const int time = tile_start - 2 + local_row;
        const int column = tile_n * MmaTile + index % MmaTile;
        if (local_row < Tile + 2) {
          float value = 0.0f;
          if (time >= 0 && time < length) {
            value = warp_acc[index] + residual[time * N + column] +
                    pointwise_bias[column];
            value = mimi_elu(value);
          }
          activated[local_row * N + column] = value;
        }
      }
      __syncwarp();
    }
  }
  __syncthreads();

  constexpr int GroupWidth = 16;
  constexpr int GroupsPerWarp = 2;
  constexpr int TermsPerLane = N * 3 / GroupWidth;
  constexpr int ConcurrentOutputs = Warps * GroupsPerWarp;
  const int group = lane / GroupWidth;
  const int group_lane = lane - group * GroupWidth;
  const int output_slot = warp * GroupsPerWarp + group;
  for (int local_time = output_slot; local_time < Tile;
       local_time += ConcurrentOutputs) {
    float accumulator = 0.0f;
#pragma unroll
    for (int term_index = 0; term_index < TermsPerLane; ++term_index) {
      const int reduction_index = group_lane * TermsPerLane + term_index;
      const int channel = reduction_index / 3;
      const int kernel_index = reduction_index - channel * 3;
      const float sample = mimi_tf32_rne(
          activated[(local_time + kernel_index) * N + channel]);
      const float coefficient = mimi_tf32_rne(final_weight[reduction_index]);
      accumulator = fmaf(sample, coefficient, accumulator);
    }
    const unsigned mask = __activemask();
#pragma unroll
    for (int delta = GroupWidth / 2; delta > 0; delta >>= 1) {
      accumulator += __shfl_down_sync(mask, accumulator, delta, GroupWidth);
    }
    const int output_time = tile_start + local_time;
    if (group_lane == 0 && output_time < length) {
      output[output_time] = accumulator + final_bias[0];
    }
  }
}

template <int Tile, int Warps>
int launch_variant(
    const void* branch,
    const void* pointwise_weight,
    const void* residual,
    const void* pointwise_bias,
    const void* final_weight,
    const void* final_bias,
    void* output,
    int length,
    cudaStream_t stream) {
  constexpr int MmaTileElements = 16 * 16;
  const int shared_bytes =
      Warps * MmaTileElements * (sizeof(half) + sizeof(float)) +
      (Tile + 2) * 64 * sizeof(float);
  decoder12_final_wmma_kernel<Tile, Warps>
      <<<(length + Tile - 1) / Tile, Warps * 32, shared_bytes, stream>>>(
          static_cast<const half*>(branch),
          static_cast<const half*>(pointwise_weight),
          static_cast<const float*>(residual),
          static_cast<const float*>(pointwise_bias),
          static_cast<const float*>(final_weight),
          static_cast<const float*>(final_bias),
          static_cast<float*>(output), length);
  cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? 0 : static_cast<int>(error);
}

extern "C" int mimi_decoder12_final_wmma(
    const void* branch,
    const void* pointwise_weight,
    const void* residual,
    const void* pointwise_bias,
    const void* final_weight,
    const void* final_bias,
    void* output,
    int length,
    int config,
    void* stream) {
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  switch (config) {
    case 0:
      return launch_variant<64, 8>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    case 1:
      return launch_variant<32, 4>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    case 2:
      return launch_variant<32, 8>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    case 3:
      return launch_variant<48, 4>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    case 4:
      return launch_variant<48, 8>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    case 5:
      return launch_variant<64, 6>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    case 6:
      return launch_variant<80, 6>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    case 7:
      return launch_variant<80, 8>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    case 8:
      return launch_variant<96, 8>(
          branch, pointwise_weight, residual, pointwise_bias, final_weight,
          final_bias, output, length, cuda_stream);
    default:
      return -1;
  }
}

