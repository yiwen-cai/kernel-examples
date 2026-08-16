#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// per block per row
// n must less than BLOCK_SIZE
__global__ void scan(const float *x, float *o, int n, int isExclusive) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int write = 0, read = 1;

  __shared__ float sum[2][BLOCK_SIZE + 1];
  sum[read][tid] = (tid < n) ? x[row * n + tid] : 0.0f;
  __syncthreads();

  for (int off = 1; off < n; off <<= 1) {
    float v = sum[read][tid];
    if (tid >= off)
      v += sum[read][tid - off];
    sum[write][tid] = v;
    __syncthreads();
    write ^= 1;
    read ^= 1;
  }

  if (tid < n) {
    float val;
    if (isExclusive) {
      val = (tid == 0) ? 0.0f : sum[read][tid - 1];
    } else {
      val = sum[read][tid];
    }
    o[row * n + tid] = val;
  }
}

// n >> BLOCK_SIZE
__global__ void blockScan(const float *x, float *o, int n, float *blockSum) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  int read = 0, write = 1;

  __shared__ float sum[2][BLOCK_SIZE + 1];

  sum[read][tid] = (gid < n) ? x[gid] : 0.0f;
  __syncthreads();

  for (int off = 1; off < BLOCK_SIZE; off <<= 1) {
    float v = sum[read][tid];
    if (tid >= off)
      v += sum[read][tid - off];
    sum[write][tid] = v;
    __syncthreads();
    write ^= 1;
    read ^= 1;
  }

  if (gid < n)
    o[gid] = sum[read][tid];
  if (tid == BLOCK_SIZE - 1)
    blockSum[bid] = sum[read][tid];
}

__global__ void addAllBlocks(float *o, const float *blockOff, int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  __shared__ float offset;
  if (tid == 0) {
    offset = blockOff[bid];
  }
  __syncthreads();

  if (gid < n) {
    o[gid] += offset;
  }
}

void multiBlockScan(const float *x, float *o, int n) {
  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float *blockSum = nullptr;
  float *blockOff = nullptr;
  cudaMalloc(&blockSum, numBlocks * sizeof(float));
  cudaMalloc(&blockOff, numBlocks * sizeof(float));

  blockScan<<<numBlocks, BLOCK_SIZE>>>(x, o, n, blockSum);

  scan<<<1, BLOCK_SIZE>>>(blockSum, blockOff, numBlocks, 1);

  addAllBlocks<<<numBlocks, BLOCK_SIZE>>>(o, blockOff, n);

  cudaFree(blockSum);
  cudaFree(blockOff);
}