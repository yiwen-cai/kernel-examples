// LSP smoke: jump to cute::make_tensor should resolve under cutlass include/.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

__global__ void touch_tensor(float *ptr, int n) {
  using namespace cute;
  Tensor t = make_tensor(make_gmem_ptr(ptr), make_shape(n));
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    t(i) = t(i) + 1.0f;
  }
}

int main() {
  const int n = 256;
  float *d = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d), n * sizeof(float));
  // Launch syntax <<<>>> is CUDA-only; call as a normal function for C++ clangd.
  touch_tensor(d, n);
  cudaDeviceSynchronize();
  cudaFree(d);
  return 0;
}
