#include <cuda_runtime.h>

struct welfordState {
  float mean;
  float m2;
  int count;
};

__device__ __forceinline__ welfordState welfordAdd(welfordState s, float x) {
  int n = s.count + 1;
  float delta = x - s.mean;
  float mean = fmaf(delta, 1.0f / n, s.mean);
  float delta2 = x - mean;
  float m2 = fmaf(delta, delta2, s.m2);

  s.count = n;
  s.mean = mean;
  s.m2 = m2;
  return s;
}

__device__ __forceinline__ welfordState welfordReduce(welfordState a,
                                                      welfordState b) {
  if (a.count == 0)
    return b;
  if (b.count == 0)
    return a;

  int n = a.count + b.count;
  float delta = b.mean - a.mean;
  float wb = float(b.count) / float(n);
  float correction =
      delta * delta * ((float(a.count) * float(b.count)) / float(n));
  a.mean = fmaf(delta, wb, a.mean);
  a.m2 = a.m2 + b.m2 + correction;
  a.count = n;
  return a;
}

__device__ __forceinline__ welfordState warpReduceWelford(welfordState s) {
  int lane = threadIdx.x & 31;
  for (int off = 16; off > 0; off >>= 1) {
    welfordState other;
    other.mean = __shfl_down_sync(0xffffffff, s.mean, off);
    other.m2 = __shfl_down_sync(0xffffffff, s.m2, off);
    other.count = __shfl_down_sync(0xffffffff, s.count, off);
    if (lane + off < 32) {
      s = welfordReduce(s, other);
    }
  }
  return s;
}

__device__ __forceinline__ welfordState blockReduceWelford(welfordState s) {
  int tid = threadIdx.x;
  int lane = tid & 31;
  int wid = tid >> 5;
  int numWarp = (blockDim.x + 31) / 32;
  welfordState bs = welfordState{0.0f, 0.0f, 0};

  s = warpReduceWelford(s);

  __shared__ float warpMean[32];
  __shared__ float warpM2[32];
  __shared__ int warpCount[32];
  __shared__ float blockMean;
  __shared__ float blockM2;
  __shared__ int blockCount;
  if (lane == 0) {
    warpMean[wid] = s.mean;
    warpM2[wid] = s.m2;
    warpCount[wid] = s.count;
  }
  __syncthreads();

  if (wid == 0) {
    bs = (lane < numWarp)
             ? welfordState{warpMean[lane], warpM2[lane], warpCount[lane]}
             : welfordState{0.0f, 0.0f, 0};
    bs = warpReduceWelford(bs);

    if (lane == 0) {
      blockMean = bs.mean;
      blockM2 = bs.m2;
      blockCount = bs.count;
    }
  }
  __syncthreads();

  return welfordState{blockMean, blockM2, blockCount};
}

// x: [*, N]
// o: [*, N]
// w: [N,]
// b: [N,]
__global__ void welfordLayerNorm(float *x, float *o, float *w, float *b,
                                 float eps, int N) {
  int tid = threadIdx.x;
  int stride = blockDim.x;
  int row = blockIdx.x;
  welfordState localState{0.0f, 0.0f, 0};

  for (int i = tid; i < N; i += stride) {
    float x_i = x[row * N + i];
    localState = welfordAdd(localState, x_i);
  }

  localState = blockReduceWelford(localState);

  float rstd = rsqrtf((localState.m2 / N) + eps);

  for (int i = tid; i < N; i += stride) {
    float x_i = x[row * N + i];
    o[row * N + i] = (x_i - localState.mean) * rstd * w[i] + b[i];
  }
}