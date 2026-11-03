#ifndef U02_STREAM_DUAL_API_H
#define U02_STREAM_DUAL_API_H
#include "u02_stream_resident_api.h"

/* Separate ABI/device, same packet protocol and one physical PL core.
 * One process owns the mapping; its two contexts use disjoint cache lines.
 * Each 2 MiB packet pool retains a complete 1 MiB hardware-access guard.
 * No input/output capacity of the existing hardware contract is reduced. */
// ABI5/6 are already allocated to the frozen region-fusion experiments.
#define U02_STREAM_DUAL_ABI 7u
#define U02_STREAM_DUAL_CONTEXTS 2u
#define U02_STREAM_DUAL_A_STRIDE (3u * 1024u * 1024u)
#define U02_STREAM_DUAL_B_BASE (6u * 1024u * 1024u)
#define U02_STREAM_DUAL_B_BYTES (128u * 1024u)
#define U02_STREAM_DUAL_C_BYTES (512u * 1024u)
#define U02_STREAM_DUAL_IO_STRIDE (U02_STREAM_DUAL_B_BYTES + U02_STREAM_DUAL_C_BYTES)
#define U02_STREAM_DUAL_A_OFFSET(ctx) ((ctx) * U02_STREAM_DUAL_A_STRIDE)
#define U02_STREAM_DUAL_B_OFFSET(ctx) (U02_STREAM_DUAL_B_BASE + (ctx) * U02_STREAM_DUAL_IO_STRIDE)
#define U02_STREAM_DUAL_C_OFFSET(ctx) (U02_STREAM_DUAL_B_OFFSET(ctx) + U02_STREAM_DUAL_B_BYTES)
#define U02_STREAM_DUAL_USED_BYTES (U02_STREAM_DUAL_B_BASE + U02_STREAM_DUAL_CONTEXTS * U02_STREAM_DUAL_IO_STRIDE)
#if U02_STREAM_DUAL_USED_BYTES > U02_POSE_ARENA_BYTES || \
    U02_STREAM_INPUT_BYTES > U02_STREAM_DUAL_B_BYTES || \
    U02_STREAM_OUTPUT_BYTES + 64 > U02_STREAM_DUAL_C_BYTES
#error Dual-context slots do not preserve the guarded hardware contract
#endif

/* Validate the context before touching packet metadata. The job wire format
 * is unchanged; all three addresses must identify the same context. */
static inline int u02_stream_dual_slots_valid(const struct u02_pose_job *j)
{
    unsigned int ctx = j->a_offset / U02_STREAM_DUAL_A_STRIDE;
    unsigned int local;
    if (ctx >= U02_STREAM_DUAL_CONTEXTS) return 0;
    local = j->a_offset - U02_STREAM_DUAL_A_OFFSET(ctx);
    return !(local & (U02_STREAM_RESIDENT_ALIGNMENT - 1)) &&
           local < U02_STREAM_RESIDENT_POOL_BYTES &&
           local + U02_STREAM_PACKET_BYTES <= U02_STREAM_DUAL_A_STRIDE &&
           j->b_offset == U02_STREAM_DUAL_B_OFFSET(ctx) &&
           j->c_offset == U02_STREAM_DUAL_C_OFFSET(ctx);
}
#endif
