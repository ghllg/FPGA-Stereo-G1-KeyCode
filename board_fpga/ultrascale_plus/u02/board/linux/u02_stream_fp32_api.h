#ifndef U02_STREAM_FP32_API_H
#define U02_STREAM_FP32_API_H
#include "u02_stream_resident_api.h"

/* Packet2 and device ABI4 are separate from accepted integer packet1/ABI3.
 * Output offsets support the framework arena, but remain inside owned DMA.
 * A full maximum hardware output envelope and one cache line are reserved
 * after every permitted start, even if writable metadata changes later. */
#define U02_STREAM_FP32_ABI 4u
#define U02_STREAM_FP32_VERSION 2u
#define U02_STREAM_FP32_C_BASE U02_STREAM_RESIDENT_C_OFFSET
#define U02_STREAM_FP32_C_POOL_BYTES (2u * 1024u * 1024u)
#define U02_STREAM_FP32_C_ALIGNMENT 64u
#if U02_STREAM_FP32_C_BASE + U02_STREAM_FP32_C_POOL_BYTES + U02_STREAM_OUTPUT_BYTES + 64u > U02_POSE_ARENA_BYTES
#error FP32 output starts exceed kernel-owned DMA allocation
#endif

/* Integer bit-pattern validation avoids any floating point in the kernel.
 * Scales: positive [2^-40,2^40]. Bias: either signed zero or magnitude
 * [2^-80,2^80]. NaN, infinity, subnormals and reserved bits are rejected. */
struct u02_stream_fp32_affine { __u32 incoming, weight, bias, reserved; };
static inline int u02_stream_fp32_affine_valid(const struct u02_stream_fp32_affine *q)
{
    __u32 magnitude = q->bias & 0x7fffffffu;
    return q->reserved == 0 &&
        q->incoming >= 0x2b800000u && q->incoming <= 0x53800000u &&
        q->weight >= 0x2b800000u && q->weight <= 0x53800000u &&
        (!magnitude || (magnitude >= 0x17800000u && magnitude <= 0x67800000u));
}
#endif
