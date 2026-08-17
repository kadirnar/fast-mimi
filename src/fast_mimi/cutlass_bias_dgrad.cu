#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

using Layout = cutlass::layout::TensorNHWC;
using PlainEpilogue =
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;
using Swizzle =
    cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<4>;

struct BiasKernelBuilder {
  // The stage-2 64x64x32 tile was selected from 22 profiled SM120 variants.
  using BaseKernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
      cutlass::half_t, Layout, cutlass::half_t, Layout, float, Layout, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>, PlainEpilogue, Swizzle, 2,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kAnalytic,
      cutlass::conv::StrideSupport::kStrided, 8, 8>::Kernel;

  using ThreadMap = typename BaseKernel::Epilogue::OutputTileIterator::ThreadMap;

  // CUTLASS 3.x has no TensorOp + strided-dgrad broadcast epilogue. The bias
  // depends only on output channel, so add it in the normal output functor.
  // Fragment order matches PredicatedTileIteratorStridedDgrad::store(), whose
  // column loop is innermost. Mimi layer 11 is identity-swizzled and has one
  // 64-channel N tile; Python guards those frozen dimensions before launch.
  class BiasOutputOp {
   public:
    using ElementOutput = float;
    using ElementSource = float;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;
    using ElementC = float;
    using ElementD = float;

    static int const kCount = 4;
    using FragmentOutput = cutlass::Array<float, kCount>;
    using FragmentSource = cutlass::Array<float, kCount>;
    using FragmentAccumulator = cutlass::Array<float, kCount>;

    struct Params {
      float const* bias;
      int channels;

      CUTLASS_HOST_DEVICE
      Params(float const* bias_ = nullptr, int channels_ = 0)
          : bias(bias_), channels(channels_) {}
    };

   private:
    float const* bias_;
    int channels_;
    int initial_column_;
    mutable int vector_index_;

   public:
    CUTLASS_DEVICE
    explicit BiasOutputOp(Params const& params)
        : bias_(params.bias),
          channels_(params.channels),
          initial_column_(ThreadMap::initial_offset(threadIdx.x).column()),
          vector_index_(0) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return false; }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int, int) {}

    CUTLASS_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
      FragmentOutput output;
      int column = initial_column_ +
                   ThreadMap::Delta::kColumn *
                       (vector_index_ % ThreadMap::Iterations::kColumn);
      ++vector_index_;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        float bias_value =
            (column + i < channels_) ? bias_[column + i] : 0.0f;
        output[i] = accumulator[i] + bias_value;
      }
      return output;
    }

    CUTLASS_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator,
                              FragmentSource const&) const {
      return operator()(accumulator);
    }
  };

  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultEpilogueTensorOpStridedDgrad<
          typename BaseKernel::Epilogue::Shape,
          typename BaseKernel::Epilogue::WarpMmaOperator,
          BaseKernel::Epilogue::kPartitionsK, BiasOutputOp, 4>::Epilogue;

  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionStridedDgrad<
      typename BaseKernel::Mma, Epilogue, Swizzle,
      cutlass::conv::Operator::kDgrad>;
};

int launch(cutlass::half_t* input, cutlass::half_t* weight, float* bias,
           float* output, int input_width, int input_channels,
           int output_channels, int kernel_width, int stride_width,
           int output_width, cudaStream_t stream) {
  using Kernel = BiasKernelBuilder::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
  cutlass::Tensor4DCoord activation(1, 1, output_width, output_channels);
  cutlass::Tensor4DCoord filter(input_channels, 1, kernel_width,
                               output_channels);
  cutlass::Tensor4DCoord padding(0, 0, 0, kernel_width - stride_width);
  cutlass::MatrixCoord stride(1, stride_width);
  cutlass::MatrixCoord dilation(1, 1);
  cutlass::Tensor4DCoord input_shape(1, 1, input_width, input_channels);
  cutlass::conv::Conv2dProblemSize problem(
      activation, filter, padding, stride, dilation, input_shape,
      cutlass::conv::Mode::kCrossCorrelation, 1);
  typename Op::Arguments arguments{
      problem,
      {input, Layout::packed(input_shape)},
      {weight, Layout::packed(filter)},
      {output, Layout::packed(activation)},
      {output, Layout::packed(activation)},
      {bias, output_channels}};
  auto status = Op::can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return int(status);
  }
  Op operation;
  status = operation(arguments, nullptr, stream);
  return status == cutlass::Status::kSuccess ? 0 : int(status);
}

extern "C" int mimi_cutlass_bias_dgrad(
    void* input, void* weight, void* bias, void* output, int input_width,
    int input_channels, int output_channels, int kernel_width,
    int stride_width, int output_width, void* stream) {
  return launch(
      static_cast<cutlass::half_t*>(input),
      static_cast<cutlass::half_t*>(weight), static_cast<float*>(bias),
      static_cast<float*>(output), input_width, input_channels,
      output_channels, kernel_width, stride_width, output_width,
      static_cast<cudaStream_t>(stream));
}
