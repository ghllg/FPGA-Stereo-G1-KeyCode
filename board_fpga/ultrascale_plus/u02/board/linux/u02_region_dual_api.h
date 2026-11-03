#ifndef U02_REGION_DUAL_API_H
#define U02_REGION_DUAL_API_H
#include "u02_region_api.h"
/* Separate candidate ABI. Region-only ABI5 and formal ABI4 cannot implement
 * this ordinary-group command. INFO must advertise ABI6 before its use.
 * Region INSTALL/RUN/RECOVER retain their structures and command numbers.
 * Packet and input are copied to private owned DMA before validation/start;
 * output is an offset within the existing owned 2 MiB scratch mapping.
 * Caller must not touch mapped scratch while either mode is device-owned. */
#define U02_REGION_DUAL_ABI 6u
#define U02_REGION_DUAL_PACKET_CAPACITY 1048576u
#define U02_REGION_DUAL_INPUT_CAPACITY 98304u
#define U02_REGION_DUAL_OUTPUT_CAPACITY 393216u
struct u02_region_dual_job {
    __u64 packet_ptr,input_ptr;
    __u32 packet_bytes,input_bytes,output_bytes,output_offset;
    __u32 operators,timeout_us;
    __s32 status;
    __u32 polls;
    __u64 elapsed_ns,reserved;
};
#ifndef U02_REGION_PORTABLE_TEST
#define U02_REGION_DUAL_LEGACY_RUN _IOWR('R',0x74,struct u02_region_dual_job)
#endif
static inline int u02_region_dual_job_valid(const struct u02_region_dual_job *job)
{
    return job->packet_ptr && job->input_ptr && job->packet_bytes>=64 &&
        job->packet_bytes<=U02_REGION_DUAL_PACKET_CAPACITY && !(job->packet_bytes%64) &&
        job->input_bytes && job->input_bytes<=U02_REGION_DUAL_INPUT_CAPACITY &&
        job->output_bytes && job->output_bytes<=U02_REGION_DUAL_OUTPUT_CAPACITY &&
        !(job->output_bytes%8) && !(job->output_offset%64) &&
        (__u64)job->output_offset+U02_REGION_DUAL_OUTPUT_CAPACITY+64<=U02_REGION_SCRATCH_CAPACITY &&
        job->operators>=1 && job->operators<=6 && job->timeout_us &&
        job->timeout_us<=U02_REGION_TIMEOUT_MAX_US && !job->status && !job->polls &&
        !job->elapsed_ns && !job->reserved;
}
#endif
