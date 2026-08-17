#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__device__ __forceinline__ float mimi_fast_elu(float value) {
  return value >= 0.0f ? value : __expf(value) - 1.0f;
}

__global__ void decoder9_fused_wmma_kernel(
    const half* __restrict__ branch,
    const half* __restrict__ weight,
    const float* __restrict__ residual,
    const float* __restrict__ bias,
    half* __restrict__ output,
    int rows) {
  constexpr int K = 64;
  constexpr int N = 128;
  constexpr int Tile = 16;
  constexpr int Ntiles = 4;
  constexpr int Warps = 2;
  constexpr int TileElements = Tile * Tile;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row_base = (blockIdx.x * Warps + warp) * Tile;
  const int column_group = blockIdx.y * Ntiles * Tile;

  extern __shared__ unsigned char storage_bytes[];
  half* shared_a = reinterpret_cast<half*>(storage_bytes) +
                   warp * TileElements;
  float* shared_acc = reinterpret_cast<float*>(
      reinterpret_cast<half*>(storage_bytes) + Warps * TileElements);
  shared_acc += warp * TileElements;

  wmma::fragment<wmma::accumulator, Tile, Tile, Tile, float>
      accumulators[Ntiles];
#pragma unroll
  for (int tile_n = 0; tile_n < Ntiles; ++tile_n) {
    wmma::fill_fragment(accumulators[tile_n], 0.0f);
  }

#pragma unroll
  for (int k_base = 0; k_base < K; k_base += Tile) {
    for (int index = lane; index < TileElements; index += 32) {
      const int row = row_base + index / Tile;
      const int k = k_base + index % Tile;
      shared_a[index] =
          row < rows ? branch[row * K + k] : __float2half_rn(0.0f);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, Tile, Tile, Tile, half, wmma::row_major>
        fragment_a;
    wmma::load_matrix_sync(fragment_a, shared_a, Tile);
#pragma unroll
    for (int tile_n = 0; tile_n < Ntiles; ++tile_n) {
      const int column = column_group + tile_n * Tile;
      wmma::fragment<wmma::matrix_b, Tile, Tile, Tile, half,
                     wmma::col_major>
          fragment_b;
      wmma::load_matrix_sync(fragment_b, weight + column * K + k_base, K);
      wmma::mma_sync(
          accumulators[tile_n], fragment_a, fragment_b,
          accumulators[tile_n]);
    }
    __syncwarp();
  }

#pragma unroll
  for (int tile_n = 0; tile_n < Ntiles; ++tile_n) {
    wmma::store_matrix_sync(
        shared_acc, accumulators[tile_n], Tile, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < TileElements; index += 32) {
      const int row = row_base + index / Tile;
      const int column = column_group + tile_n * Tile + index % Tile;
      if (row < rows) {
        float value = shared_acc[index] + residual[row * N + column] +
                      bias[column];
        output[row * N + column] = __float2half_rn(mimi_fast_elu(value));
      }
    }
    __syncwarp();
  }
}

extern "C" int mimi_decoder9_fused_wmma(
    const void* branch,
    const void* weight,
    const void* residual,
    const void* bias,
    void* output,
    int rows,
    void* stream) {
  constexpr int Tile = 16;
  constexpr int Ntiles = 4;
  constexpr int Warps = 2;
  dim3 grid((rows + Tile * Warps - 1) / (Tile * Warps),
            128 / (Tile * Ntiles), 1);
  dim3 block(Warps * 32, 1, 1);
  const int shared_bytes =
      Warps * Tile * Tile * (sizeof(half) + sizeof(float));
  decoder9_fused_wmma_kernel<<<grid, block, shared_bytes,
                               static_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(branch),
      static_cast<const half*>(weight),
      static_cast<const float*>(residual),
      static_cast<const float*>(bias),
      static_cast<half*>(output), rows);
  cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? 0 : static_cast<int>(error);
}

