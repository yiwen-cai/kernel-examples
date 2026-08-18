#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

struct SoftmaxState {
	float m;
	float l;
};

__device__ __forceinline__ SoftmaxState softmaxStateAdd(SoftmaxState s, float x) {
	float m_new = fmaxf(x, s.m);
	float alpha = __expf(s.m - m_new);
	s.l = s.l * alpha + __expf(x - m_new);
	s.m = m_new;
	return s;
}

__device__ __forceinline__ SoftmaxState softmaxStateReduce(SoftmaxState a, SoftmaxState b) {
	SoftmaxState s{-FLT_MAX, 0.0f};
	float m_new = fmaxf(a.m, b.m);
	float alpha1 = __expf(a.m - m_new);
	float alpha2 = __expf(b.m - m_new);
	s.l = a.l * alpha1 + b.l * alpha2;
	s.m = m_new;
	return s;
}

__device__ __forceinline__ SoftmaxState warpSoftmax(SoftmaxState local) {
	for (int off = 16; off > 0; off >>= 1) {
		SoftmaxState other;
		other.m = __shfl_down_sync(0xffffffff, local.m, off);
		other.l = __shfl_down_sync(0xffffffff, local.l, off);
		local = softmaxStateReduce(local, other);
	}
	return local;
}

// x: [num_rows, N]
// o: [num_rows, N]
__global__ void
softmax(float *x, float *o, int N) {
	int tid = threadIdx.x;
	int stride = blockDim.x;
	int lane = tid & 31;
	int wid = tid / 32;
	int numWarp = (blockDim.x + 31) / 32;
	int row = blockIdx.x;

	SoftmaxState local{-FLT_MAX, 0.0f};
	for (int i = tid; i < N; i += stride) {
		float x_i = x[row * N + i];
		local = softmaxStateAdd(local, x_i);
	}

	local = warpSoftmax(local);

	__shared__ float warpM[32];
	__shared__ float warpL[32];
	__shared__ float blockM;
	__shared__ float blockL;
	if (lane == 0) {
		warpM[wid] = local.m;
		warpL[wid] = local.l;
	}
	__syncthreads();

	if (wid == 0) {
		local.m = (lane < numWarp) ? warpM[lane] : -FLT_MAX;
		local.l = (lane < numWarp) ? warpL[lane] : 0.0f;

		local = warpSoftmax(local);

		if (lane == 0) {
			blockM = local.m;
			blockL = local.l;
		}
	}

	__syncthreads();
	local.m = blockM;
	local.l = blockL;
	for (int i = tid; i < N; i += stride) {
		float x_i = x[row * N + i];
		o[row * N + i] = __expf(x_i - local.m) / local.l;
	}
}