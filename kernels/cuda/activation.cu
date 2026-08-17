#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoid(float x) {
	return x >= 0 ? 1.0f / (1 + __expf(-x)) : __expf(x) / (1.0f + __expf(x));
}

// x: [N,]
__global__ void silu(float *x, float *o, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (; i < N; i += stride) {
		float x_i = x[i];
		o[i] = x_i * sigmoid(x_i);
	}
}

__global__ void gilu(float *x_1, float *x_2, float *o, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (; i < N; i += stride) {
		float x1_i = x_1[i];
		o[i] = x1_i * sigmoid(x1_i) * x_2[i];
	}
}