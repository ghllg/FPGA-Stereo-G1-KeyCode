#ifndef U02_STREAM_RESIDENT_API_H
#define U02_STREAM_RESIDENT_API_H
#include "u02_stream_api.h"

/* Separate device/ABI; the accepted ABI 2 and its binaries stay unchanged.
 * Each packet retains the existing <=1 MiB hardware access contract. A start
 * within the 2 MiB pool therefore has a full 1 MiB guard before input/output.
 * All addresses still refer exclusively to the driver's 8 MiB DMA allocation.
 */
#define U02_STREAM_RESIDENT_ABI 3u
#define U02_STREAM_RESIDENT_POOL_BYTES (2u * 1024u * 1024u)
#define U02_STREAM_RESIDENT_A_REGION_BYTES (3u * 1024u * 1024u)
#define U02_STREAM_RESIDENT_ALIGNMENT 4096u
#define U02_STREAM_RESIDENT_B_OFFSET U02_STREAM_RESIDENT_A_REGION_BYTES
#define U02_STREAM_RESIDENT_C_OFFSET (U02_STREAM_RESIDENT_B_OFFSET + U02_STREAM_INPUT_BYTES)
#if U02_STREAM_RESIDENT_C_OFFSET + U02_STREAM_OUTPUT_BYTES > U02_POSE_ARENA_BYTES
#error Resident streaming regions exceed the kernel-owned DMA allocation
#endif
#endif
