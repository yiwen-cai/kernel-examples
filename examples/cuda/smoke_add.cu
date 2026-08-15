// LSP smoke: jump to cudaMalloc / threadIdx should resolve via cuda_runtime.h.
#include <cuda_runtime.h>

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1024;
  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&a), n * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&b), n * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&c), n * sizeof(float));

  // Do not call the kernel: Clang 21 lowers <<<>>> to cudaConfigureCall,
  // which CUDA 12 removed, and a direct __global__ call is also illegal.
  cudaDeviceSynchronize();

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return 0;
}
