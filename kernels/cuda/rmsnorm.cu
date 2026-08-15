#include <cuda_runtime.h>
#include <math.h>

__device__ float warpSumReduce(float localSum) {
	for (int off = 16; off > 0; off >>= 1) {
		localSum = localSum + __shfl_xor_sync(0xffffffff, localSum, off);
	}
	return localSum;
}

// per block per row
// 1024 threads per block
__global__ void rmsnorm(float *x, float *o, float *g, float eps, int N) {
	int tid = threadIdx.x;
	int row = blockIdx.x;
	int stride = blockDim.x;
	int lane = tid & 31;
	int wid = tid / 32;
	int numWarps = (blockDim.x + 31) / 32;

	float localSum = 0.0f;
	for (int i = tid; i < N; i += stride) {
		float x_i = x[row * N + i];
		localSum = localSum + x_i * x_i;
	}
	localSum = warpSumReduce(localSum);

	__shared__ float warpSum[32];
	__shared__ float rowSum;
	if (lane == 0) {
		warpSum[wid] = localSum;
	}
	__syncthreads();
	if (wid == 0) {
		localSum = (lane < numWarps) ? warpSum[lane] : 0.0f;
		localSum = warpSumReduce(localSum);
		if (lane == 0) {
			rowSum = localSum;
		}
	}
	__syncthreads();
	localSum = rowSum;
	float rrms = rsqrtf(localSum / N + eps);

	for (int i = tid; i < N; i += stride) {
		o[row * N + i] = x[row * N + i] * rrms * g[i];
	}
}