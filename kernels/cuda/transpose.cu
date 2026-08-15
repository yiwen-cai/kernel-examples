#include <cuda_runtime.h>

#define TILE 32

// one block transpose one tile
// per block 1024 threads
__global__ void transpose(float *A, float *B, int M, int N) {
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;

  int row = tile_m * TILE;
  int col = tile_n * TILE;

  __shared__ float sm[TILE][TILE + 1];

  int tid = threadIdx.x;
  int aThreadRow = (tid * 4) / TILE;
  int aThreadCol = (tid * 4) & (TILE - 1);
  float4 temp;

  if (row + aThreadRow < M && col + aThreadCol < N) {
    temp = *reinterpret_cast<float4 *>(
        &A[(row + aThreadRow) * N + col + aThreadCol]);
    sm[aThreadRow][aThreadCol] = temp.x;
    sm[aThreadRow][aThreadCol + 1] = temp.y;
    sm[aThreadRow][aThreadCol + 2] = temp.z;
    sm[aThreadRow][aThreadCol + 3] = temp.w;
  }

  __syncthreads();

  int outRow = tile_n * TILE;
  int outCol = tile_m * TILE;

  temp.x = sm[aThreadCol][aThreadRow];
  temp.y = sm[aThreadCol + 1][aThreadRow];
  temp.z = sm[aThreadCol + 2][aThreadRow];
  temp.w = sm[aThreadCol + 3][aThreadRow];

  if (outRow + aThreadRow < N && outCol + aThreadCol < M) {
    *reinterpret_cast<float4 *>(
        &B[(outRow + aThreadRow) * M + outCol + aThreadCol]) = temp;
  }
}