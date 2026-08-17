#include <cuda_runtime.h>
#include <math.h>

__device__ float warpReduceSum(float localSum) {
  for (int off = 16; off > 0; off >>= 1) {
    localSum = localSum + __shfl_xor_sync(0xffffffff, localSum, off);
  }
  return localSum;
}

__device__ float blockReduceSum(float localSum) {
  int tid = threadIdx.x;
  int lane = tid & 31;
  int wid = tid / 32;
  int warpNum = (blockDim.x + 31) / 32;

  localSum = warpReduceSum(localSum);
  __shared__ float warpSum[32];
  __shared__ float blockSum;
  if (lane == 0) {
    warpSum[wid] = localSum;
  }
  __syncthreads();

  if (wid == 0) {
    localSum = (lane < warpNum) ? warpSum[lane] : 0.0f;
    localSum = warpReduceSum(localSum);
    if (lane == 0) {
      blockSum = localSum;
    }
  }
  __syncthreads();

  localSum = blockSum;
  return localSum;
}

// per block per token
// per block 1024 threads
// x: [*, N]
// o: [*, N]
// w: [N,]
// b: [N,]
__global__ void layerNorm(float *x, float *o, float *w, float *b, int N,
                          float eps) {
  int tid = threadIdx.x;
  int lane = tid & 31;
  int wid = tid / 32;
  int stride = blockDim.x;
  int row = blockIdx.x;
  float localMean = 0.0f;
  float localVar = 0.0f;
  for (int i = tid; i < N; i += stride) {
    float x_i = x[row * N + i];
    localMean += x_i;
  }
  localMean = blockReduceSum(localMean) / N;
  for (int i = tid; i < N; i += stride) {
    float x_i = x[row * N + i];
    localVar += (x_i - localMean) * (x_i - localMean);
  }
  localVar = blockReduceSum(localVar) / N;

  float rstd = rsqrtf(localVar + eps);

  for (int i = tid; i < N; i += stride) {
    float x_i = x[row * N + i];
    o[row * N + i] = (x_i - localMean) * rstd * w[i] + b[i];
  }
}