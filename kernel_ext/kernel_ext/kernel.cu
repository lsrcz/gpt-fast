#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/core/DispatchKey.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/tuple.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/RankLocal.hpp>
#include <torch/extension.h>
#include <utility>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t x) {
  return 1 / (1 + exp(-x));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t swish(scalar_t x) {
  return x * sigmoid(x);
}

template <typename scalar_t>
__global__ void swish_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>
        inputs,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>
        outputs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < inputs.size(0)) {
    outputs[idx] = swish(inputs[idx]);
  }
}

torch::Tensor swish_forward(torch::Tensor inputs) {
  // auto inputs_flat = inputs.view(-1);
  // const int size = inputs_flat.size(0);
  // const int threads = 1024;
  // const dim3 blocks((size + threads - 1) / threads);
  auto outputs = torch::empty_like(inputs);
  // auto outputs_flat = outputs.view(-1);
  auto iter =
      at::TensorIteratorConfig().add_input(inputs).add_output(outputs).build();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, inputs.scalar_type(),
      "swish_forward_kernel", ([&] {
        at::native::gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t x_acc = static_cast<opmath_t>(x);
          return x_acc / (opmath_t(1) + ::exp(-x_acc));
        });
      }));
  return outputs;
}

torch::Tensor __inline__ conditional_feed_forward_forward(
    torch::Tensor x, torch::Tensor expert_indices, torch::Tensor w1,
    torch::Tensor w2, torch::Tensor w3) {
  auto w1_weights = w1.index({expert_indices});
  auto w3_weights = w3.index({expert_indices});
  auto w2_weights = w2.index({expert_indices});
  auto x1 = torch::silu(torch::einsum("ti,taoi -> tao", {x, w1_weights}));
  auto x3 = torch::einsum("ti,taoi -> tao", {x, w3_weights});
  auto expert_outs = torch::einsum("tao,taio -> tai", {x1 * x3, w2_weights});
  return expert_outs;
}

const std::unordered_map<std::string, c10d::ReduceOp> str_to_reduce_op = {
    {"sum", c10d::ReduceOp(c10d::ReduceOp::RedOpType::SUM)},
    {"avg", c10d::ReduceOp(c10d::ReduceOp::RedOpType::AVG)},
    {"product", c10d::ReduceOp(c10d::ReduceOp::RedOpType::PRODUCT)},
    {"min", c10d::ReduceOp(c10d::ReduceOp::RedOpType::MIN)},
    {"max", c10d::ReduceOp(c10d::ReduceOp::RedOpType::MAX)},
    {"band", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BAND)},
    {"bor", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BOR)},
    {"bxor", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BXOR)},
    // TODO: support premul_sum
    // {"premul_sum", c10d::ReduceOp(c10d::ReduceOp::RedOpType::PREMUL_SUM)},
    {"unused", c10d::ReduceOp(c10d::ReduceOp::RedOpType::UNUSED)}};

c10d::ReduceOp to_reduce_op(const std::string &reduce_op) {
  auto it = str_to_reduce_op.find(reduce_op);
  TORCH_CHECK(it != str_to_reduce_op.end(),
              "Unrecognized reduce_op: ", reduce_op);
  return it->second;
}

class WorkRegistry {
public:
  void register_work(const at::Tensor &tensor,
                     const c10::intrusive_ptr<c10d::Work> &work) {
    auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);
    auto [it, inserted] = registry_.try_emplace(std::move(storage), work);
    TORCH_CHECK(inserted || it->second != work,
                "The tensor storage is already associated with another work.");
  }

  c10::intrusive_ptr<c10d::Work> pop_work(const at::Tensor &tensor) {
    const auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);
    auto it = registry_.find(storage);
    if (it == registry_.end()) {
      return nullptr;
    }
    auto work = it->second;
    registry_.erase(it);
    return work;
  }

  ~WorkRegistry() {
    // If there are still unwaited work objects, their corresponding process
    // groups should have already been destroyed at this stage. Any attempts to
    // wait for these work objects or to destroy them will only result in
    // confusing errors. Therefore, we simply issue a warning and intentionally
    // allow the unwaited work objects to leak.
    if (!registry_.empty()) {
      TORCH_WARN(
          "At the time of process termination, there are still ",
          registry_.size(),
          " unwaited c10d_functional collective calls. "
          "Please review your program to ensure c10d_functional.wait_tensor() "
          "is invoked on all tensors returned from c10d_functional collective "
          "ops before they are used.");
    }
    for (auto &it : registry_) {
      it.second.release();
    }
  }

private:
  std::unordered_map<c10::weak_intrusive_ptr<c10::StorageImpl>,
                     c10::intrusive_ptr<c10d::Work>>
      registry_;
  std::mutex lock_;
};

at::Tensor &all_reduce_(at::Tensor &input,
                        // NOLINTNEXTLINE(performance-unnecessary-value-param)
                        std::string reduce_op,
                        // NOLINTNEXTLINE(performance-unnecessary-value-param)
                        std::string group_name) {
  c10d::AllreduceOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);

  std::vector<at::Tensor> inputs{input};
  auto group = c10d::resolve_process_group(group_name);
  auto work = group->allreduce(inputs, opts);
  c10d::RankLocal<WorkRegistry>::get().register_work(input, work);
  return input;
}

at::Tensor all_reduce(const at::Tensor &input, std::string reduce_op,
                      std::string group_name) {
  auto output = input.clone(at::MemoryFormat::Contiguous);
  return all_reduce_(output, std::move(reduce_op), std::move(group_name));
}

torch::Tensor moe_feed_forward_after_expert_weights_indices(
    torch::Tensor x, torch::Tensor expert_weights, torch::Tensor expert_indices,
    torch::Tensor w1, torch::Tensor w2, torch::Tensor w3, std::string group) {
  expert_weights /= expert_weights.sum(-1, true);
  auto expert_outs = all_reduce(
      conditional_feed_forward_forward(x, expert_indices, w1, w2, w3), "sum",
      group);

  return torch::einsum("tai,ta -> ti", {expert_outs, expert_weights});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swish_forward", &swish_forward, "Swish forward (CUDA)");
  m.def("conditional_feed_forward_forward", &conditional_feed_forward_forward,
        "Conditional feed forward forward (CUDA)");
  m.def("moe_feed_forward_after_expert_weights_indices",
        &moe_feed_forward_after_expert_weights_indices,
        "Mixture of experts feed forward after expert weights indices (CUDA)");
}
