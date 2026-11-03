#pragma once
#include <stdint.h>

// Row-major A[M,K] * B[K,N] -> C[M,N]. Signed symmetric INT8, INT32 sums.
// Bounds are part of the host/IP contract, not a full-network size limit.
enum { POSE_MAX_M = 512, POSE_MAX_N = 25600, POSE_MAX_K = 4608 };
extern "C" int pose_gemm(const int8_t* a, const int8_t* b, int32_t* c,
                          int m, int n, int k);
