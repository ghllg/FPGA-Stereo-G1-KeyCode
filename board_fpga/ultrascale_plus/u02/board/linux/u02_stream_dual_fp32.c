/* Reuse the proven FP32 execution, cache ownership, timeout retention and
 * normal recovery code. Only identity, allocation and slot validation differ. */
#include "u02_stream_dual_fp32_api.h"
#define U02_FP32_DEVICE_NAME "u02_stream_dual_fp32"
#define U02_FP32_DEVICE_ABI U02_STREAM_DUAL_FP32_ABI
#define U02_FP32_DMA_BYTES U02_STREAM_DUAL_FP32_ARENA_BYTES
#define U02_FP32_SLOTS_VALID(j) u02_stream_dual_fp32_slots_valid(j)
#include "u02_stream_fp32.c"
