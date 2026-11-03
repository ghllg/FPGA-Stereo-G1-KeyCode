#ifndef U02_STREAM_DUAL_FP32_API_H
#define U02_STREAM_DUAL_FP32_API_H
#include "u02_stream_fp32_api.h"
#include "u02_stream_dual_api.h"

/* Independent ABI8/packet2 candidate. Keep every packet's full 1MiB access
 * envelope and every possible output's 384KiB+guard envelope within its own
 * context. The existing ABI3/4/7 layouts and 8MiB allocations do not change. */
#define U02_STREAM_DUAL_FP32_ABI 8u
#define U02_STREAM_DUAL_FP32_ARENA_BYTES (16u * 1024u * 1024u)
#define U02_STREAM_DUAL_FP32_C_BASE (8u * 1024u * 1024u)
#define U02_STREAM_DUAL_FP32_C_STRIDE (3u * 1024u * 1024u)
#define U02_STREAM_DUAL_FP32_C_OFFSET(ctx) (U02_STREAM_DUAL_FP32_C_BASE + (ctx) * U02_STREAM_DUAL_FP32_C_STRIDE)
#define U02_STREAM_DUAL_FP32_C_POOL_BYTES (2u * 1024u * 1024u)
#if U02_STREAM_DUAL_USED_BYTES > U02_STREAM_DUAL_FP32_C_BASE || \
    U02_STREAM_DUAL_FP32_C_BASE + 2u * U02_STREAM_DUAL_FP32_C_STRIDE > U02_STREAM_DUAL_FP32_ARENA_BYTES || \
    U02_STREAM_DUAL_FP32_C_POOL_BYTES + U02_STREAM_OUTPUT_BYTES + 64u > U02_STREAM_DUAL_FP32_C_STRIDE
#error Dual FP32 slots overlap or lose a hardware access envelope
#endif

static inline int u02_stream_dual_fp32_slots_valid(const struct u02_pose_job *j)
{
    unsigned int ctx = j->a_offset / U02_STREAM_DUAL_A_STRIDE;
    unsigned int local;
    __u64 cbase;
    if (ctx >= 2u) return 0;
    local = j->a_offset - U02_STREAM_DUAL_A_OFFSET(ctx);
    cbase = U02_STREAM_DUAL_FP32_C_OFFSET(ctx);
    return !(local & (U02_STREAM_RESIDENT_ALIGNMENT - 1u)) &&
        local < U02_STREAM_RESIDENT_POOL_BYTES &&
        (__u64)local + U02_STREAM_PACKET_BYTES <= U02_STREAM_DUAL_A_STRIDE &&
        j->b_offset == U02_STREAM_DUAL_B_OFFSET(ctx) &&
        !(j->c_offset & (U02_STREAM_FP32_C_ALIGNMENT - 1u)) &&
        (__u64)j->c_offset >= cbase &&
        (__u64)j->c_offset < cbase + U02_STREAM_DUAL_FP32_C_POOL_BYTES &&
        (__u64)j->c_offset + j->c_bytes <= cbase + U02_STREAM_DUAL_FP32_C_POOL_BYTES &&
        (__u64)j->c_offset + U02_STREAM_OUTPUT_BYTES + 64u <= cbase + U02_STREAM_DUAL_FP32_C_STRIDE;
}
#endif
