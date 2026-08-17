#include <cuda_runtime.h>
#include <math.h>

__device__ float warp_reduce_max(float local_max) {
  int lane = threadIdx.x & 31;
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max =
        fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
  }
  return local_max;
}

// x: [*, N]
// q: [*, N]
// s: [*,]
__global__ void int8_per_token_quant(float *x, int8_t *q, float *s, int N) {
  // one block one row
  // one block 32 warps, 1024 threads
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int wid = tid / 32;
  int lane = tid & 31;
  int BLOCK_SIZE = blockDim.x;

  float local_max = -INFINITY;
  for (int i = tid; i < N; i += BLOCK_SIZE) {
    local_max = fmax(local_max, x[row * N + i]);
  }

  local_max = warp_reduce_max(local_max);

  __shared__ float warp_max[32];
  __shared__ float block_max;

  if (lane == 0) {
    warp_max[wid] = local_max;
  }
  __syncthreads();

  if (wid == 0) {
    local_max = warp_max[lane];
    local_max = warp_reduce_max(local_max);
    if (lane == 0) {
      block_max = local_max;
    }
  }
  __syncthreads();

  float amax = block_max;
  float scale = amax / 127.0;
  float inv = amax > 0.f ? 1.f / scale : 0.f;

  for (int i = tid; i < N; i += BLOCK_SIZE) {
    int v = __float2int_rn(x[row * N + i] * inv);
    q[row * N + i] = (int8_t)fmax(-127, fmin(127, v));
  }

  if (tid == 0) {
    s[row] = scale;
  }
}