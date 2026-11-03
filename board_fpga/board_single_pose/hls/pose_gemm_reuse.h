#pragma once
#include "pose_gemm.h"
// Same register/memory layout as the byte-port core. DDR base addresses must
// be 8-byte aligned and buffers padded to a multiple of 8 bytes (host uses 64).
extern "C" int pose_gemm_reuse(const uint64_t* a, const uint64_t* b, int32_t* c,
                               int m, int n, int k);
