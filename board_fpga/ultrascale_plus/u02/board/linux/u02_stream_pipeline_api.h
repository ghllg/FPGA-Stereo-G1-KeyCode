#ifndef U02_STREAM_PIPELINE_API_H
#define U02_STREAM_PIPELINE_API_H
#include "u02_stream_resident_api.h"

/* ABI9: one shared model pool, four private B/C pairs, one physical PL core.
 * A prefix must remain locked until synchronous RUN completes. The existing
 * 1MiB hardware packet envelope and all B/C guards remain reserved. */
#define U02_STREAM_PIPELINE_ABI 9u
#define U02_STREAM_PIPELINE_CONTEXTS 4u
#define U02_STREAM_PIPELINE_B_BASE (3u * 1024u * 1024u)
#define U02_STREAM_PIPELINE_B_BYTES (128u * 1024u)
#define U02_STREAM_PIPELINE_C_BYTES (512u * 1024u)
#define U02_STREAM_PIPELINE_IO_STRIDE (U02_STREAM_PIPELINE_B_BYTES + U02_STREAM_PIPELINE_C_BYTES)
#define U02_STREAM_PIPELINE_B_OFFSET(ctx) (U02_STREAM_PIPELINE_B_BASE + (ctx) * U02_STREAM_PIPELINE_IO_STRIDE)
#define U02_STREAM_PIPELINE_C_OFFSET(ctx) (U02_STREAM_PIPELINE_B_OFFSET(ctx) + U02_STREAM_PIPELINE_B_BYTES)
#define U02_STREAM_PIPELINE_USED_BYTES (U02_STREAM_PIPELINE_B_BASE + U02_STREAM_PIPELINE_CONTEXTS * U02_STREAM_PIPELINE_IO_STRIDE)
#if U02_STREAM_PIPELINE_USED_BYTES > U02_POSE_ARENA_BYTES || \
    U02_STREAM_INPUT_BYTES > U02_STREAM_PIPELINE_B_BYTES || \
    U02_STREAM_OUTPUT_BYTES + 64 > U02_STREAM_PIPELINE_C_BYTES
#error Pipeline slots do not preserve the guarded DMA contract
#endif

static inline int u02_stream_pipeline_slots_valid(const struct u02_pose_job *j)
{
    unsigned int relative, ctx;
    if (j->b_offset < U02_STREAM_PIPELINE_B_BASE) return 0;
    relative = j->b_offset - U02_STREAM_PIPELINE_B_BASE;
    if (relative % U02_STREAM_PIPELINE_IO_STRIDE) return 0;
    ctx = relative / U02_STREAM_PIPELINE_IO_STRIDE;
    return ctx < U02_STREAM_PIPELINE_CONTEXTS &&
           j->c_offset == U02_STREAM_PIPELINE_C_OFFSET(ctx) &&
           !(j->a_offset & (U02_STREAM_RESIDENT_ALIGNMENT - 1)) &&
           j->a_offset < U02_STREAM_RESIDENT_POOL_BYTES &&
           j->a_offset <= U02_STREAM_RESIDENT_A_REGION_BYTES - U02_STREAM_PACKET_BYTES;
}
#endif
