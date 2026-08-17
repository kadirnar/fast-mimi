#include <cuda_runtime.h>

template <int Tile, int Threads, int ReductionOrder, int ExpMode>
__global__ void final_post_kernel(const float* residual, const float* branch,
                                  const float* weight, const float* bias,
                                  float* output, int length) {
  extern __shared__ float activated[];
  constexpr int kChannels = 64;
  constexpr int kKernel = 3;
  constexpr int kWarps = Threads / 32;
  int tile_start = int(blockIdx.x) * Tile;
  int tile_values = (Tile + kKernel - 1) * kChannels;

  for (int offset = int(threadIdx.x); offset < tile_values; offset += Threads) {
    int row = offset / kChannels;
    int channel = offset - row * kChannels;
    int time = tile_start + row - (kKernel - 1);
    float value = 0.0f;
    if (time >= 0 && time < length) {
      int index = time * kChannels + channel;
      value = residual[index] + branch[index];
      if (value <= 0.0f) {
        if constexpr (ExpMode == 0) {
          value = expm1f(value);
        } else if constexpr (ExpMode == 1) {
          value = expf(value) - 1.0f;
        } else {
          value = __expf(value) - 1.0f;
        }
      }
    }
    activated[offset] = value;
  }
  __syncthreads();

  int lane = int(threadIdx.x) & 31;
  int warp = int(threadIdx.x) >> 5;
  for (int local_time = warp; local_time < Tile; local_time += kWarps) {
    int output_time = tile_start + local_time;
    float accumulator = 0.0f;
    if constexpr (ReductionOrder == 0) {
#pragma unroll
      for (int channel_step = 0; channel_step < 2; ++channel_step) {
        int channel = lane + channel_step * 32;
#pragma unroll
        for (int kernel_index = 0; kernel_index < kKernel; ++kernel_index) {
          accumulator = fmaf(
              activated[(local_time + kernel_index) * kChannels + channel],
              weight[channel * kKernel + kernel_index], accumulator);
        }
      }
    } else {
#pragma unroll
      for (int kernel_index = 0; kernel_index < kKernel; ++kernel_index) {
#pragma unroll
        for (int channel_step = 0; channel_step < 2; ++channel_step) {
          int channel = lane + channel_step * 32;
          accumulator = fmaf(
              activated[(local_time + kernel_index) * kChannels + channel],
              weight[channel * kKernel + kernel_index], accumulator);
        }
      }
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      accumulator += __shfl_down_sync(0xffffffffu, accumulator, delta);
    }
    if (lane == 0 && output_time < length) {
      output[output_time] = accumulator + bias[0];
    }
  }
}

template <int Tile, int Threads, int ReductionOrder, int ExpMode>
int launch_kernel(const float* residual, const float* branch,
                  const float* weight, const float* bias, float* output,
                  int length, cudaStream_t stream) {
  auto kernel = final_post_kernel<Tile, Threads, ReductionOrder, ExpMode>;
  int shared_bytes = (Tile + 2) * 64 * int(sizeof(float));
  if (shared_bytes > 48 * 1024) {
    auto error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (error != cudaSuccess) {
      return int(error);
    }
  }
  kernel<<<(length + Tile - 1) / Tile, Threads, shared_bytes, stream>>>(
      residual, branch, weight, bias, output, length);
  return int(cudaGetLastError());
}

template <int Tf32Mode>
__device__ __forceinline__ float multiply_input(float value) {
  if constexpr (Tf32Mode == 0) {
    return value;
  }
  unsigned bits = __float_as_uint(value);
  unsigned exponent = bits & 0x7f800000u;
  if (exponent == 0x7f800000u) {
    return value;
  }
  if constexpr (Tf32Mode == 1) {
    bits += 0x00000fffu + ((bits >> 13) & 1u);
  } else if constexpr (Tf32Mode == 2) {
    bits += 0x00001000u;
  }
  return __uint_as_float(bits & 0xffffe000u);
}

template <int Tile, int Threads, int GroupWidth, bool Strided, int ExpMode,
          int Tf32Mode>
__global__ void final_post_partition_kernel(
    const float* residual, const float* branch, const float* weight,
    const float* bias, float* output, int length) {
  extern __shared__ float activated[];
  constexpr int kChannels = 64;
  constexpr int kKernel = 3;
  constexpr int kWarps = Threads / 32;
  constexpr int kGroupsPerWarp = 32 / GroupWidth;
  constexpr int kTermsPerLane = kChannels * kKernel / GroupWidth;
  int tile_start = int(blockIdx.x) * Tile;
  int tile_values = (Tile + kKernel - 1) * kChannels;

  for (int offset = int(threadIdx.x); offset < tile_values; offset += Threads) {
    int row = offset / kChannels;
    int channel = offset - row * kChannels;
    int time = tile_start + row - (kKernel - 1);
    float value = 0.0f;
    if (time >= 0 && time < length) {
      int index = time * kChannels + channel;
      value = residual[index] + branch[index];
      if (value <= 0.0f) {
        if constexpr (ExpMode == 0) {
          value = expm1f(value);
        } else if constexpr (ExpMode == 1) {
          value = expf(value) - 1.0f;
        } else {
          value = __expf(value) - 1.0f;
        }
      }
    }
    activated[offset] = value;
  }
  __syncthreads();

  int lane = int(threadIdx.x) & 31;
  int warp = int(threadIdx.x) >> 5;
  int group = lane / GroupWidth;
  int group_lane = lane - group * GroupWidth;
  int output_slot = warp * kGroupsPerWarp + group;
  constexpr int kConcurrentOutputs = kWarps * kGroupsPerWarp;
  for (int local_time = output_slot; local_time < Tile;
       local_time += kConcurrentOutputs) {
    float accumulator = 0.0f;
#pragma unroll
    for (int term_index = 0; term_index < kTermsPerLane; ++term_index) {
      int reduction_index =
          Strided ? group_lane + term_index * GroupWidth
                  : group_lane * kTermsPerLane + term_index;
      int channel = reduction_index / kKernel;
      int kernel_index = reduction_index - channel * kKernel;
      float sample = multiply_input<Tf32Mode>(
          activated[(local_time + kernel_index) * kChannels + channel]);
      float coefficient = multiply_input<Tf32Mode>(weight[reduction_index]);
      accumulator = fmaf(sample, coefficient, accumulator);
    }
    unsigned mask = __activemask();
#pragma unroll
    for (int delta = GroupWidth / 2; delta > 0; delta >>= 1) {
      accumulator += __shfl_down_sync(mask, accumulator, delta, GroupWidth);
    }
    int output_time = tile_start + local_time;
    if (group_lane == 0 && output_time < length) {
      output[output_time] = accumulator + bias[0];
    }
  }
}

template <int Tile, int Threads, int GroupWidth, bool Strided, int ExpMode,
          int Tf32Mode>
int launch_partition_kernel(
    const float* residual, const float* branch, const float* weight,
    const float* bias, float* output, int length, cudaStream_t stream) {
  auto kernel = final_post_partition_kernel<
      Tile, Threads, GroupWidth, Strided, ExpMode, Tf32Mode>;
  int shared_bytes = (Tile + 2) * 64 * int(sizeof(float));
  if (shared_bytes > 48 * 1024) {
    auto error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (error != cudaSuccess) {
      return int(error);
    }
  }
  kernel<<<(length + Tile - 1) / Tile, Threads, shared_bytes, stream>>>(
      residual, branch, weight, bias, output, length);
  return int(cudaGetLastError());
}

template <int GroupWidth, bool Strided, int ExpMode, int Tf32Mode>
int dispatch_partition_config(
    int config, const float* residual, const float* branch,
    const float* weight, const float* bias, float* output, int length,
    cudaStream_t stream) {
  switch (config) {
    case 0:
      return launch_partition_kernel<
          64, 128, GroupWidth, Strided, ExpMode, Tf32Mode>(
          residual, branch, weight, bias, output, length, stream);
    case 1:
      return launch_partition_kernel<
          128, 128, GroupWidth, Strided, ExpMode, Tf32Mode>(
          residual, branch, weight, bias, output, length, stream);
    case 2:
      return launch_partition_kernel<
          256, 128, GroupWidth, Strided, ExpMode, Tf32Mode>(
          residual, branch, weight, bias, output, length, stream);
    case 3:
      return launch_partition_kernel<
          256, 256, GroupWidth, Strided, ExpMode, Tf32Mode>(
          residual, branch, weight, bias, output, length, stream);
    default:
      return -2;
  }
}

template <int GroupWidth, bool Strided, int Tf32Mode = 0>
int dispatch_partition_exp(
    int exp_mode, int config, const float* residual, const float* branch,
    const float* weight, const float* bias, float* output, int length,
    cudaStream_t stream) {
  switch (exp_mode) {
    case 0:
      return dispatch_partition_config<GroupWidth, Strided, 0, Tf32Mode>(
          config, residual, branch, weight, bias, output, length, stream);
    case 1:
      return dispatch_partition_config<GroupWidth, Strided, 1, Tf32Mode>(
          config, residual, branch, weight, bias, output, length, stream);
    case 2:
      return dispatch_partition_config<GroupWidth, Strided, 2, Tf32Mode>(
          config, residual, branch, weight, bias, output, length, stream);
    default:
      return -3;
  }
}

template <int Tile, int Threads, int OutputsPerThread, int ReductionOrder,
          int ExpMode>
__global__ void final_post_sequential_kernel(
    const float* residual, const float* branch, const float* weight,
    const float* bias, float* output, int length) {
  static_assert(Tile == Threads * OutputsPerThread);
  extern __shared__ float activated[];
  constexpr int kChannels = 64;
  constexpr int kKernel = 3;
  int tile_start = int(blockIdx.x) * Tile;
  int tile_values = (Tile + kKernel - 1) * kChannels;

  for (int offset = int(threadIdx.x); offset < tile_values; offset += Threads) {
    int row = offset / kChannels;
    int channel = offset - row * kChannels;
    int time = tile_start + row - (kKernel - 1);
    float value = 0.0f;
    if (time >= 0 && time < length) {
      int index = time * kChannels + channel;
      value = residual[index] + branch[index];
      if (value <= 0.0f) {
        if constexpr (ExpMode == 0) {
          value = expm1f(value);
        } else if constexpr (ExpMode == 1) {
          value = expf(value) - 1.0f;
        } else {
          value = __expf(value) - 1.0f;
        }
      }
    }
    activated[offset] = value;
  }
  __syncthreads();

  float accumulator[OutputsPerThread];
#pragma unroll
  for (int output_index = 0; output_index < OutputsPerThread; ++output_index) {
    accumulator[output_index] = 0.0f;
  }
  if constexpr (ReductionOrder == 0) {
#pragma unroll
    for (int channel = 0; channel < kChannels; ++channel) {
#pragma unroll
      for (int kernel_index = 0; kernel_index < kKernel; ++kernel_index) {
        float coefficient = weight[channel * kKernel + kernel_index];
#pragma unroll
        for (int output_index = 0; output_index < OutputsPerThread;
             ++output_index) {
          int local_time = int(threadIdx.x) + output_index * Threads;
          accumulator[output_index] = fmaf(
              activated[(local_time + kernel_index) * kChannels + channel],
              coefficient, accumulator[output_index]);
        }
      }
    }
  } else {
#pragma unroll
    for (int kernel_index = 0; kernel_index < kKernel; ++kernel_index) {
#pragma unroll
      for (int channel = 0; channel < kChannels; ++channel) {
        float coefficient = weight[channel * kKernel + kernel_index];
#pragma unroll
        for (int output_index = 0; output_index < OutputsPerThread;
             ++output_index) {
          int local_time = int(threadIdx.x) + output_index * Threads;
          accumulator[output_index] = fmaf(
              activated[(local_time + kernel_index) * kChannels + channel],
              coefficient, accumulator[output_index]);
        }
      }
    }
  }
#pragma unroll
  for (int output_index = 0; output_index < OutputsPerThread; ++output_index) {
    int local_time = int(threadIdx.x) + output_index * Threads;
    int output_time = tile_start + local_time;
    if (output_time < length) {
      output[output_time] = accumulator[output_index] + bias[0];
    }
  }
}

template <int Tile, int Threads, int OutputsPerThread, int ReductionOrder,
          int ExpMode>
int launch_sequential_kernel(
    const float* residual, const float* branch, const float* weight,
    const float* bias, float* output, int length, cudaStream_t stream) {
  auto kernel = final_post_sequential_kernel<
      Tile, Threads, OutputsPerThread, ReductionOrder, ExpMode>;
  int shared_bytes = (Tile + 2) * 64 * int(sizeof(float));
  if (shared_bytes > 48 * 1024) {
    auto error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (error != cudaSuccess) {
      return int(error);
    }
  }
  kernel<<<(length + Tile - 1) / Tile, Threads, shared_bytes, stream>>>(
      residual, branch, weight, bias, output, length);
  return int(cudaGetLastError());
}

template <int ReductionOrder, int ExpMode>
int dispatch_sequential_config(
    int config, const float* residual, const float* branch,
    const float* weight, const float* bias, float* output, int length,
    cudaStream_t stream) {
  switch (config) {
    case 0:
      return launch_sequential_kernel<128, 128, 1, ReductionOrder, ExpMode>(
          residual, branch, weight, bias, output, length, stream);
    case 1:
      return launch_sequential_kernel<256, 128, 2, ReductionOrder, ExpMode>(
          residual, branch, weight, bias, output, length, stream);
    case 2:
      return launch_sequential_kernel<384, 128, 3, ReductionOrder, ExpMode>(
          residual, branch, weight, bias, output, length, stream);
    case 3:
      return launch_sequential_kernel<128, 64, 2, ReductionOrder, ExpMode>(
          residual, branch, weight, bias, output, length, stream);
    default:
      return -2;
  }
}

template <int ReductionOrder>
int dispatch_sequential_exp(
    int exp_mode, int config, const float* residual, const float* branch,
    const float* weight, const float* bias, float* output, int length,
    cudaStream_t stream) {
  switch (exp_mode) {
    case 0:
      return dispatch_sequential_config<ReductionOrder, 0>(
          config, residual, branch, weight, bias, output, length, stream);
    case 1:
      return dispatch_sequential_config<ReductionOrder, 1>(
          config, residual, branch, weight, bias, output, length, stream);
    case 2:
      return dispatch_sequential_config<ReductionOrder, 2>(
          config, residual, branch, weight, bias, output, length, stream);
    default:
      return -3;
  }
}

template <int ReductionOrder, int ExpMode>
int dispatch_config(int config, const float* residual, const float* branch,
                    const float* weight, const float* bias, float* output,
                    int length, cudaStream_t stream) {
  switch (config) {
    case 0:
      return launch_kernel<64, 128, ReductionOrder, ExpMode>(
          residual, branch, weight, bias, output, length, stream);
    case 1:
      return launch_kernel<128, 128, ReductionOrder, ExpMode>(
          residual, branch, weight, bias, output, length, stream);
    case 2:
      return launch_kernel<256, 128, ReductionOrder, ExpMode>(
          residual, branch, weight, bias, output, length, stream);
    case 3:
      return launch_kernel<256, 256, ReductionOrder, ExpMode>(
          residual, branch, weight, bias, output, length, stream);
    default:
      return -2;
  }
}

template <int ReductionOrder>
int dispatch_exp(int exp_mode, int config, const float* residual,
                 const float* branch, const float* weight, const float* bias,
                 float* output, int length, cudaStream_t stream) {
  switch (exp_mode) {
    case 0:
      return dispatch_config<ReductionOrder, 0>(
          config, residual, branch, weight, bias, output, length, stream);
    case 1:
      return dispatch_config<ReductionOrder, 1>(
          config, residual, branch, weight, bias, output, length, stream);
    case 2:
      return dispatch_config<ReductionOrder, 2>(
          config, residual, branch, weight, bias, output, length, stream);
    default:
      return -3;
  }
}

extern "C" int mimi_final_post_cuda(
    int reduction_order, int exp_mode, int config, void* residual,
    void* branch, void* weight, void* bias, void* output, int length,
    void* stream) {
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (reduction_order == 0) {
    return dispatch_exp<0>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 1) {
    return dispatch_exp<1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 2) {
    return dispatch_sequential_exp<0>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 3) {
    return dispatch_sequential_exp<1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 4) {
    return dispatch_partition_exp<32, false>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 5) {
    return dispatch_partition_exp<32, true>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 6) {
    return dispatch_partition_exp<16, false>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 7) {
    return dispatch_partition_exp<16, true>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 8) {
    return dispatch_partition_exp<8, false>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 9) {
    return dispatch_partition_exp<8, true>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 10) {
    return dispatch_partition_exp<4, false>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 11) {
    return dispatch_partition_exp<4, true>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 12) {
    return dispatch_partition_exp<32, false, 1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 13) {
    return dispatch_partition_exp<32, true, 1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 14) {
    return dispatch_partition_exp<16, false, 1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 15) {
    return dispatch_partition_exp<16, true, 1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 16) {
    return dispatch_partition_exp<8, false, 1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 17) {
    return dispatch_partition_exp<8, true, 1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 18) {
    return dispatch_partition_exp<4, false, 1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 19) {
    return dispatch_partition_exp<4, true, 1>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 20) {
    return dispatch_partition_exp<16, false, 2>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  if (reduction_order == 21) {
    return dispatch_partition_exp<16, false, 3>(
        exp_mode, config, static_cast<const float*>(residual),
        static_cast<const float*>(branch), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output), length,
        cuda_stream);
  }
  return -1;
}

