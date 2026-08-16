#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>

#ifdef __cplusplus
extern "C" {
#endif

// Clang CUDA frontend lowers <<<>>> to cudaConfigureCall on host,
// which was deprecated/removed in CUDA 12 headers.
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0,
                              cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
