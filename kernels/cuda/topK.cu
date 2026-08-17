#include <cuda_runtime.h>

__device__ __forceinline__ int warpSum(int localSum) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		localSum += __shfl_down_sync(0xffffffff, localSum, offset);
	}
	return localSum;
}

__device__ __forceinline__ int blockSum(int localSum, __shared__ int *warp) {
	int tid = threadIdx.x;
	int lane = tid & 31;
	int wid = tid >> 5;
	int numWarp = (blockDim.x + 31) / 32;

	localSum = warpSum(localSum);
	__shared__ int blockSum;
	if (lane == 0) {
		warp[wid] = localSum;
	}
	__syncthreads();

	if (wid == 0) {
		localSum = (lane < numWarp) ? warp[lane] : 0;
		localSum = warpSum(localSum);

		if (lane == 0) {
			blockSum = localSum;
		}
	}

	__syncthreads();
	localSum = blockSum;
	return localSum;
}

__device__ __forceinline__ uint32_t float2intKey(float x) {
	uint32_t bits = __float_as_uint(x);
	uint32_t mask = (bits & 0x80000000u) ? 0xffffffff : 0x80000000;
	return mask ^ bits;
}

__device__ uint32_t radixSelect(float *x, int N, int K) {
	int tid = threadIdx.x;
	int stride = blockDim.x;
	__shared__ uint32_t mask;
	__shared__ uint32_t prefix;
	__shared__ int warpCnt[32];
	__shared__ int k;
	if (tid == 0) {
		mask = 0;
		prefix = 0;
		k = K;
	}
	__syncthreads();

	for (int bits = 31; bits >= 0; bits--) {
		int local = 0;
		for (int i = tid; i < N; i += stride) {
			uint32_t key = float2intKey(x[i]);
			int inPrefix = (mask & key) == prefix;
			local += inPrefix && ((key >> bits) & 1);
		}
		int cnt = blockSum(local, warpCnt);
		if (tid == 0) {
			mask |= (1u << bits);
			if (cnt >= k) {
				prefix |= (1u << bits);
			} else {
				k -= cnt;
			}
		}
		__syncthreads();
	}

	return prefix;
}

// per block per row
// blockDim.x must be a multiple of 32
// x: [*, N]
// v: [*, K]
// i: [*, K]
__global__ void topK(float *x, float *v, int *i, int N, int K) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	int stride = blockDim.x;

	uint32_t threadholdKey = radixSelect(&x[row * N], N, K);
	__shared__ int cur;
	if (tid == 0) {
		cur = 0;
	}
	__syncthreads();

	for (int col = tid; col < N; col += stride) {
		float x_i = x[row * N + col];
		uint32_t key = float2intKey(x_i);
		if (key > threadholdKey) {
			int slot = atomicAdd(&cur, 1);
			if (slot < K) {
				v[row * K + slot] = x_i;
				i[row * K + slot] = col;
			}
		}
	}
	__syncthreads();

	if (cur < K) {
		for (int col = tid; col < N; col += stride) {
			float x_i = x[row * N + col];
			uint32_t key = float2intKey(x_i);
			if (key == threadholdKey) {
				int slot = atomicAdd(&cur, 1);
				if (slot < K) {
					v[row * K + slot] = x_i;
					i[row * K + slot] = col;
				}
			}
		}
	}
}