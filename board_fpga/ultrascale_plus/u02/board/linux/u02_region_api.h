#ifndef U02_REGION_API_H
#define U02_REGION_API_H
#ifdef U02_REGION_PORTABLE_TEST
#include <stdint.h>
typedef uint32_t __u32;
typedef int32_t __s32;
typedef uint64_t __u64;
#else
#include <linux/types.h>
#include <linux/ioctl.h>
#endif

/* Separate region ABI. Current u02_stream_fp32 does not implement these.
 * INSTALL copies metadata/model from userspace into private kernel-owned DMA,
 * validates them and publishes a generation token only after success.
 * RUN names that token, not physical addresses or userspace command buffers.
 * mmap exposes only owned scratch. Caller must not touch it while in flight.
 * A timeout pins DMA/configuration until RECOVER observes the actual IP idle. */
#define U02_REGION_ABI 5u
#define U02_REGION_METADATA_CAPACITY 16384u
#define U02_REGION_MODEL_CAPACITY 1048576u
#define U02_REGION_SCRATCH_CAPACITY 2097152u
#define U02_REGION_TIMEOUT_MAX_US 30000000u

struct u02_region_info {
    __u32 abi, metadata_version, model_version, flags;
    __u32 metadata_capacity, model_capacity, scratch_capacity, active_arena_bytes;
    __u64 token, reserved[3];
};
struct u02_region_install {
    __u64 metadata_ptr, model_ptr;
    __u32 metadata_bytes, model_bytes, flags, reserved0;
    __u64 token, reserved[3];
};
struct u02_region_job {
    __u64 token;
    __u32 timeout_us, flags;
    __s32 status;
    __u32 polls;
    __u64 elapsed_ns, reserved[4];
};
#ifndef U02_REGION_PORTABLE_TEST
#define U02_REGION_INFO _IOR('R', 0x70, struct u02_region_info)
#define U02_REGION_INSTALL _IOWR('R', 0x71, struct u02_region_install)
#define U02_REGION_RUN _IOWR('R', 0x72, struct u02_region_job)
#define U02_REGION_RECOVER _IO('R', 0x73)
#endif

static inline int u02_region_install_valid(const struct u02_region_install *request)
{
    unsigned i;
    if (!request->metadata_ptr || !request->model_ptr ||
        request->metadata_bytes<64 || request->metadata_bytes>U02_REGION_METADATA_CAPACITY || request->metadata_bytes%64 ||
        request->model_bytes<128 || request->model_bytes>U02_REGION_MODEL_CAPACITY || request->model_bytes%64 ||
        request->flags || request->reserved0 || request->token) return 0;
    for (i=0;i<3;++i) if (request->reserved[i]) return 0;
    return 1;
}
static inline int u02_region_job_valid(const struct u02_region_job *request, __u64 active_token)
{
    unsigned i;
    if (!active_token || request->token!=active_token || !request->timeout_us ||
        request->timeout_us>U02_REGION_TIMEOUT_MAX_US || request->flags ||
        request->status || request->polls || request->elapsed_ns) return 0;
    for (i=0;i<4;++i) if (request->reserved[i]) return 0;
    return 1;
}
#endif
