#ifndef U02_POSE_API_H
#define U02_POSE_API_H
#include <linux/types.h>
#include <linux/ioctl.h>
#define U02_POSE_ABI 1
#define U02_POSE_ARENA_BYTES (8u * 1024u * 1024u)
struct u02_pose_info {
    __u32 abi, arena_bytes;
    __u64 dma_address;
};
struct u02_pose_job {
    __u32 a_offset, a_bytes, b_offset, b_bytes, c_offset, c_bytes;
    __u32 m, n, k, timeout_us;
    __s32 status;
    __u32 polls;
    __u64 elapsed_ns;
};
#define U02_POSE_INFO _IOR('P', 0x20, struct u02_pose_info)
#define U02_POSE_RUN _IOWR('P', 0x21, struct u02_pose_job)
#define U02_POSE_RECOVER _IO('P', 0x22)
#endif
