/* Separate candidate device. Never load until fresh bitstream/boot-bound
 * JTAG clock/reset/AXI proof and whole-design acceptance exist. */
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/ioport.h>
#include <linux/dma-mapping.h>
#include <linux/mutex.h>
#include <linux/uaccess.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include "u02_region_linux_memory.h"
#include "u02_region_hw.h"

#define U02_REGION_BASE 0xa0000000ULL
#define U02_REGION_MMIO_BYTES 0x10000
static struct platform_device *pdev;
static void __iomem *regs;
static void *scratch;
static dma_addr_t scratch_dma;
static struct u02_region_session session;
static struct u02_region_validation_workspace *workspace;
static DEFINE_MUTEX(lock);
static atomic_t opened=ATOMIC_INIT(0);
static bool bridge_checked;
module_param(bridge_checked,bool,0444);
MODULE_PARM_DESC(bridge_checked,"Require fresh candidate/boot-bound independent JTAG clock, reset and AXI proof");

static bool ip_idle(u32 control)
{
    return !(control&0x81) && (control&4);
}
static void return_cpu_ownership(void)
{
    if (!session.inflight) return;
    dma_rmb();
    u02_region_observed_idle(&session,1);
    module_put(THIS_MODULE);
}
static void set_address(unsigned offset,u64 address)
{
    writel(lower_32_bits(address),regs+offset);
    writel(upper_32_bits(address),regs+offset+4);
}
static long region_ioctl(struct file *file,unsigned command,unsigned long argument)
{
    int status=0;
    u32 control;
    if (command!=U02_REGION_INFO && command!=U02_REGION_INSTALL &&
        command!=U02_REGION_RUN && command!=U02_REGION_RECOVER) return -ENOTTY;
    if (mutex_lock_interruptible(&lock)) return -ERESTARTSYS;
    if (command==U02_REGION_INFO) {
        struct u02_region_info info={
            .abi=U02_REGION_ABI,.metadata_version=1,.model_version=1,
            .metadata_capacity=U02_REGION_METADATA_CAPACITY,.model_capacity=U02_REGION_MODEL_CAPACITY,
            .scratch_capacity=U02_REGION_SCRATCH_CAPACITY,.active_arena_bytes=session.active.arena_bytes,
            .token=session.active.token,
        };
        status=copy_to_user((void __user *)argument,&info,sizeof(info)) ? -EFAULT : 0;
    } else if (command==U02_REGION_RECOVER) {
        control=readl(regs);
        if (!ip_idle(control)) status=-EBUSY;
        else return_cpu_ownership();
    } else if (command==U02_REGION_INSTALL) {
        struct u02_region_install request;
        if (session.inflight) {status=-EBUSY;goto out;}
        if (copy_from_user(&request,(void __user *)argument,sizeof(request))) {status=-EFAULT;goto out;}
        if (!u02_region_install_valid(&request)) {status=-EINVAL;goto out;}
        if (!ip_idle(readl(regs))) {status=-EBUSY;goto out;}
        status=u02_region_install_config(&session,&request,&u02_region_linux_memory_ops,&pdev->dev,workspace);
        if (!status) {
            dma_wmb();
            /* INFO remains authoritative if publication to userspace fails. */
            if (copy_to_user((void __user *)argument,&request,sizeof(request))) status=-EFAULT;
        }
    } else {
        struct u02_region_job job;
        u64 start,deadline,now;
        bool done=false;
        if (session.inflight) {status=-EBUSY;goto out;}
        if (copy_from_user(&job,(void __user *)argument,sizeof(job))) {status=-EFAULT;goto out;}
        if (!u02_region_job_valid(&job,session.active.token)) {status=-EINVAL;goto out;}
        control=readl(regs); /* clears any previous ap_done before starting */
        if (!ip_idle(control)) {status=-EBUSY;goto out;}
        if (!try_module_get(THIS_MODULE)) {status=-ENODEV;goto out;}
        status=u02_region_begin(&session,&job);
        if (status) {module_put(THIS_MODULE);goto out;}
        writel(0,regs+4); /* auto_restart and interrupts are not used */
        set_address(U02_REGION_REG_METADATA,session.active.metadata.dma);
        set_address(U02_REGION_REG_MODEL,session.active.model.dma);
        set_address(U02_REGION_REG_ARENA,(u64)scratch_dma);
        writel(session.active.metadata_bytes,regs+U02_REGION_REG_METADATA_BYTES);
        writel(session.active.model_bytes,regs+U02_REGION_REG_MODEL_BYTES);
        writel(session.active.arena_bytes,regs+U02_REGION_REG_ARENA_BYTES);
        dma_wmb();
        start=ktime_get_ns();deadline=start+(u64)job.timeout_us*1000;
        writel(1,regs);
        for (;;) {
            control=readl(regs);++job.polls;done|=!!(control&2);
            /* ap_done is clear-on-read; latch it until idle is also observed. */
            if (done && ip_idle(control)) break;
            now=ktime_get_ns();
            if (now>=deadline) {status=-ETIMEDOUT;break;}
            if (job.timeout_us>=1000 && job.polls>64) usleep_range(10,20);
            else cpu_relax();
        }
        job.elapsed_ns=ktime_get_ns()-start;
        if (!status) {
            dma_rmb();job.status=(s32)readl(regs+U02_REGION_REG_RETURN);
            return_cpu_ownership();
        }
        /* Timeout leaves config, scratch and a module reference pinned. */
        if (copy_to_user((void __user *)argument,&job,sizeof(job))) status=-EFAULT;
    }
out:
    mutex_unlock(&lock);
    return status;
}
static int region_open(struct inode *inode,struct file *file)
{
    return atomic_cmpxchg(&opened,0,1) ? -EBUSY : 0;
}
static int region_release(struct inode *inode,struct file *file)
{
    /* The extra in-flight module reference retains all DMA after close. */
    atomic_set(&opened,0);return 0;
}
static int region_mmap(struct file *file,struct vm_area_struct *vma)
{
    int status;
    if (vma->vm_pgoff || vma->vm_end-vma->vm_start!=U02_REGION_SCRATCH_CAPACITY) return -EINVAL;
    if (mutex_lock_interruptible(&lock)) return -ERESTARTSYS;
    if (session.inflight) status=-EBUSY;
    else status=dma_mmap_coherent(&pdev->dev,vma,scratch,scratch_dma,U02_REGION_SCRATCH_CAPACITY);
    mutex_unlock(&lock);return status;
}
static const struct file_operations fops={
    .owner=THIS_MODULE,.open=region_open,.release=region_release,
    .unlocked_ioctl=region_ioctl,.mmap=region_mmap,.llseek=no_llseek,
};
static struct miscdevice device={.minor=MISC_DYNAMIC_MINOR,.name="u02_region",.fops=&fops,.mode=0600};
static int __init region_init(void)
{
    int status;
    if (!bridge_checked) return -EPERM;
    if (!request_mem_region(U02_REGION_BASE,U02_REGION_MMIO_BYTES,"u02_region")) return -EBUSY;
    pdev=platform_device_register_simple("u02_region",-1,NULL,0);
    if (IS_ERR(pdev)) {status=PTR_ERR(pdev);goto region;}
    status=dma_coerce_mask_and_coherent(&pdev->dev,DMA_BIT_MASK(49));if (status) goto platform;
    workspace=kzalloc(sizeof(*workspace),GFP_KERNEL);
    if (!workspace) {status=-ENOMEM;goto platform;}
    scratch=dma_alloc_coherent(&pdev->dev,U02_REGION_SCRATCH_CAPACITY,&scratch_dma,GFP_KERNEL);
    if (!scratch) {status=-ENOMEM;goto workspace;}
    regs=ioremap(U02_REGION_BASE,U02_REGION_MMIO_BYTES);
    if (!regs) {status=-ENOMEM;goto dma;}
    device.parent=&pdev->dev;status=misc_register(&device);if (status) goto mapping;
    pr_info("u02_region: private configuration ABI5, owned scratch bytes=%u\n",U02_REGION_SCRATCH_CAPACITY);
    return 0;
mapping:iounmap(regs);
dma:dma_free_coherent(&pdev->dev,U02_REGION_SCRATCH_CAPACITY,scratch,scratch_dma);
workspace:kfree(workspace);
platform:platform_device_unregister(pdev);
region:release_mem_region(U02_REGION_BASE,U02_REGION_MMIO_BYTES);
    return status;
}
static void __exit region_exit(void)
{
    /* Cannot be entered while inflight holds its additional module reference. */
    misc_deregister(&device);
    u02_region_close_config(&session,&u02_region_linux_memory_ops,&pdev->dev);
    iounmap(regs);dma_free_coherent(&pdev->dev,U02_REGION_SCRATCH_CAPACITY,scratch,scratch_dma);
    kfree(workspace);platform_device_unregister(pdev);
    release_mem_region(U02_REGION_BASE,U02_REGION_MMIO_BYTES);
}
module_init(region_init);
module_exit(region_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Candidate TinyPose region dispatcher with private owned commands and timeout quarantine");
