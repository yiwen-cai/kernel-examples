#include <cuda_runtime.h>

// x: [*, S, D]
// o: [*, S, D]
// cos: [S, D/2]
// sin: [S, D/2]
__global__ void rope(float *x, float *o, float *cos, float *sin, int S, int D) {
	int tid = threadIdx.x;
	int stride = blockDim.x;
	int row = blockIdx.x;
	int half = D / 2;
	int s = row % S;

	for (int i = tid; i < half; i += stride) {
		float x_1 = x[row * D + i * 2];           // rotate_half: x[row * D + i]
		float x_2 = x[row * D + i * 2 + 1];       // rotate_half: x[row * D + i + half]
		float co = cos[s * half + i];
		float sn = sin[s * half + i];

		o[row * D + i * 2] = x_1 * co - x_2 * sn;           // rotate_half: o[row * D + i]
		o[row * D + i * 2 + 1] = x_1 * sn + x_2 * co;       // rotate_half: o[row * D + i + half]
	}
}