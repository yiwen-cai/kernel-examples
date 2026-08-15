// Mac clangd helpers for CuTe files (C++ parse mode, no CUDA frontend).
// Raw CUDA under kernels/cuda and examples/cuda uses -x cuda instead.
//
// Do NOT define __CUDACC__ in .clangd: cuda_runtime.h then pulls
// crt/common_functions.h -> math_functions.h, which conflicts with Apple libc++.
// Without __CUDACC__, cuda_runtime.h still exposes cudaMalloc via
// cuda_runtime_api.h, but skips device_launch_parameters.h — so include it here.

#pragma once

#ifndef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif

#include "crt/host_defines.h"
#include "device_launch_parameters.h"
#include "vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void __syncthreads();
void __syncwarp(unsigned mask = 0xffffffff);
unsigned __ballot_sync(unsigned mask, int predicate);
int __all_sync(unsigned mask, int predicate);
int __any_sync(unsigned mask, int predicate);

// CUDA math (rsqrtf / rsqrt) lives in math_functions.h, gated on __CUDACC__.
float rsqrtf(float x);
double rsqrt(double x);

// Conversion intrinsics live in crt/device_functions.h, gated on __CUDACC__.
int __float2int_rn(float x);
int __float2int_rz(float x);
int __float2int_ru(float x);
int __float2int_rd(float x);
unsigned __float2uint_rn(float x);
unsigned __float2uint_rz(float x);
unsigned __float2uint_ru(float x);
unsigned __float2uint_rd(float x);

#ifdef __cplusplus
}
#endif

// Warp shuffles live in sm_30_intrinsics.h, which is gated on __CUDACC__.
// Overloads cannot be extern "C" (C has no overloading).
#ifdef __cplusplus
int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32);
unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32);
float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32);
double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32);
long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32);
unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane,
                              int width = 32);

int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32);
unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32);
float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32);
double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32);

int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32);
unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32);
float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32);
double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32);

int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32);
unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32);
float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32);
double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32);
#endif
