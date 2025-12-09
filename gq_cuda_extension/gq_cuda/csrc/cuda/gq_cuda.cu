#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <curand_kernel.h>

namespace extension_cpp {

__global__ void gq_cuda_kernel(
  const float* mu_q,
  const float* std_q,
  const float* noise_q,
  float* result,
  int64_t dim,
  int64_t b,
  int64_t n_samples,
  double beta
) {
  
  int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  if (idx >= n_samples * b) return;

  int64_t bi = idx / n_samples;
  int64_t ni = idx % n_samples;

  // printf("dim %d b %d n %d, idx %d bi %d ni %d\n", dim, b, n_samples, idx, bi, ni);

  float log_w_value = 0.0f;
  for (int i = 0; i < dim; i++) {
      float iv = (noise_q[ni * dim + i] - mu_q[bi * dim + i]) / std_q[bi * dim + i];
      float co = noise_q[ni * dim + i];
      log_w_value -= iv * iv;
      log_w_value += co * co * beta;
  }
  result[idx] = log_w_value;
  return; 
}

__global__ void gq_int_cuda_kernel(
  const float* mu_q,
  const float* std_q,
  const float* noise_q,
  int32_t* result,
  int64_t dim2,
  int64_t b,
  int64_t n_samples,
  double beta
) {
  const int dim = 16;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= b) return;
  int bi = idx;
  float log_best = std::numeric_limits<float>::min();
  int best_j = -1;
  for (int j = 0; j < n_samples; j++){
    float log_j = 0.0f;
    for (int i = 0; i < dim; i++) {
      int nji = noise_q[j * dim + i];
      int mbi = mu_q[bi * dim + i];
      int sbi = std_q[bi * dim + i];
      float iv = (nji - mbi) / sbi;
      float co = nji;
      log_j -= iv * iv;
      log_j += co * co * beta;
    }
    if (log_best < log_j){
      log_best = log_j;
      best_j = j;
    }
  }
  result[bi] = best_j;
}

void gq_cuda(
  const at::Tensor& mu_q,
  const at::Tensor& std_q,
  const at::Tensor& noise_q,
  at::Tensor &result,
  int64_t dim,
  int64_t b,
  int64_t n_samples,
  double beta
) {
  // mu q: (b, dim)
  // std q: (b, dim)
  // noise q: (n_samples, dim)
  // result q: (b, n_samples)
  TORCH_CHECK(mu_q.sizes() == std_q.sizes());
  TORCH_CHECK(mu_q.dtype() == at::kFloat);
  TORCH_CHECK(std_q.dtype() == at::kFloat);
  TORCH_CHECK(noise_q.dtype() == at::kFloat);
  TORCH_CHECK(result.dtype() == at::kFloat);
  TORCH_CHECK(result.is_contiguous());

  TORCH_INTERNAL_ASSERT(mu_q.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(std_q.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(noise_q.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(result.device().type() == at::DeviceType::CUDA);

  at::Tensor a_contig = mu_q.contiguous();
  at::Tensor b_contig = std_q.contiguous();
  at::Tensor noise_contig = noise_q.contiguous();

  const float* mu_ptr = a_contig.data_ptr<float>();
  const float* std_ptr = b_contig.data_ptr<float>();
  const float* noise_ptr = noise_contig.data_ptr<float>();

  float* result_ptr = result.data_ptr<float>();

  // b x n_samples, ...
  int64_t numel = b * n_samples;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gq_cuda_kernel<<<(numel+1024 - 1)/1024, 1024, 0, stream>>>(mu_ptr, std_ptr, noise_ptr, result_ptr, dim, b, n_samples, beta);
  return;
}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("gq", &gq_cuda);
}

}
