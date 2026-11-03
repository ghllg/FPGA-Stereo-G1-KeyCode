#ifndef U02_REGION_DUAL_LEGACY_H
#define U02_REGION_DUAL_LEGACY_H
#include "u02_region_dual_api.h"
#include "u02_region_dual_packet_validation.h"
/* Included after the candidate driver's ownership and MMIO helpers.
 * Private staging is never mapped. All calls hold that driver's mutex. */
static struct u02_region_buffer dual_packet,dual_input;
static void dual_release_private(void)
{
    if (dual_input.cpu) u02_region_dma_release(&pdev->dev,&dual_input);
    if (dual_packet.cpu) u02_region_dma_release(&pdev->dev,&dual_packet);
}
static int dual_allocate_private(void)
{
    int status=u02_region_dma_allocate(&pdev->dev,U02_REGION_DUAL_PACKET_CAPACITY,&dual_packet);
    if (!status) status=u02_region_dma_allocate(&pdev->dev,U02_REGION_DUAL_INPUT_CAPACITY,&dual_input);
    if (status) dual_release_private();
    return status;
}
static long dual_legacy_ioctl(unsigned long argument)
{
    struct u02_region_dual_job job;
    u64 start,deadline;
    u32 control;
    bool done=false;
    int status=0;
    unsigned i;
    if (session.inflight) return -EBUSY;
    if (copy_from_user(&job,(void __user *)argument,sizeof(job))) return -EFAULT;
    if (!u02_region_dual_job_valid(&job)) return -EINVAL;
    if (!ip_idle(readl(regs))) return -EBUSY;
    if (copy_from_user(dual_packet.cpu,u64_to_user_ptr(job.packet_ptr),job.packet_bytes)) return -EFAULT;
    status=u02_dual_packet_validate(dual_packet.cpu,job.packet_bytes,job.input_bytes,job.output_bytes,job.operators);
    if (status) return status;
    memset(dual_input.cpu,0,ALIGN(job.input_bytes,8));
    if (copy_from_user(dual_input.cpu,u64_to_user_ptr(job.input_ptr),job.input_bytes)) return -EFAULT;
    if (!try_module_get(THIS_MODULE)) return -ENODEV;
    /* The shared flag pins private staging, configuration and mapped scratch.
     * RECOVER may clear it only after independently observing actual idle. */
    session.inflight=1;
    memset((u8 *)scratch+job.output_offset+job.output_bytes,0xa5,64);
    writel(0,regs+4);
    set_address(U02_REGION_REG_METADATA,dual_packet.dma);
    set_address(U02_REGION_REG_MODEL,dual_input.dma);
    set_address(U02_REGION_REG_ARENA,(u64)scratch_dma+job.output_offset);
    writel(job.operators,regs+U02_REGION_REG_METADATA_BYTES);
    writel(job.packet_bytes,regs+U02_REGION_REG_MODEL_BYTES);
    writel(0x55303253u,regs+U02_REGION_REG_ARENA_BYTES);
    dma_wmb();start=ktime_get_ns();deadline=start+(u64)job.timeout_us*1000;
    writel(1,regs);
    for (;;) {
        control=readl(regs);++job.polls;done|=!!(control&2);
        if (done && ip_idle(control)) break;
        if (ktime_get_ns()>=deadline) {status=-ETIMEDOUT;break;}
        if (job.timeout_us>=1000 && job.polls>64) usleep_range(10,20);
        else cpu_relax();
    }
    job.elapsed_ns=ktime_get_ns()-start;
    if (!status) {
        dma_rmb();job.status=(s32)readl(regs+U02_REGION_REG_RETURN);
        for (i=0;i<64;++i)
            if (*((u8 *)scratch+job.output_offset+job.output_bytes+i)!=0xa5) {status=-EOVERFLOW;break;}
        return_cpu_ownership();
    }
    if (copy_to_user((void __user *)argument,&job,sizeof(job))) status=-EFAULT;
    return status;
}
#endif
