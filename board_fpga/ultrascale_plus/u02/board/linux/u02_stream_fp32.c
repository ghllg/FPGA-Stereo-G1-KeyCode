// V06f whole-network backend: kernel-owned DMA, no raw user MMIO or physical RAM.
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
#include "u02_stream_fp32_api.h"

#ifndef U02_FP32_DEVICE_NAME
#define U02_FP32_DEVICE_NAME "u02_stream_fp32"
#endif
#ifndef U02_FP32_DEVICE_ABI
#define U02_FP32_DEVICE_ABI U02_STREAM_FP32_ABI
#endif
#ifndef U02_FP32_DMA_BYTES
#define U02_FP32_DMA_BYTES U02_POSE_ARENA_BYTES
#endif

#define BASE 0xa0000000ULL
#define REGION_BYTES 0x10000
#define CONV_FLAG 0x40000000u
#define DW_FLAG 0x20000000u
static struct platform_device *pdev;
static void __iomem *regs;
static void *arena;
static dma_addr_t dma_address;
static DEFINE_MUTEX(lock);
static atomic_t opened = ATOMIC_INIT(0);
static bool inflight, bridge_checked, cached_dma;
static unsigned int poll_spin_us;
static struct u02_pose_job active_job;
module_param(bridge_checked, bool, 0444);
MODULE_PARM_DESC(bridge_checked, "Require current bitstream/boot-bound independent JTAG clock, reset and AXI proof");
module_param(cached_dma, bool, 0444);
MODULE_PARM_DESC(cached_dma, "Opt-in cached kernel-owned DMA pages with explicit CPU/device ownership transfers");
module_param(poll_spin_us, uint, 0444);
MODULE_PARM_DESC(poll_spin_us, "Optional initial active-poll window, 0..1000 us; deadline and timeout ownership unchanged");

static void sync_job(bool for_device)
{
    const u32 offsets[] = {active_job.a_offset, active_job.b_offset, active_job.c_offset};
    /* Include the reserved output guard so cached reads cannot conceal a
     * device overwrite beyond the declared result. C + maximum result +
     * cache-line padding remains well inside the owned 8 MiB allocation. */
    const u32 lengths[] = {active_job.a_bytes, active_job.b_bytes, active_job.c_bytes + 64};
    unsigned int i;
    if (!cached_dma) return;
    for (i = 0; i < ARRAY_SIZE(offsets); ++i) {
        /* Packet/input bases are page aligned; output starts are cache-line aligned. Cover the
         * complete last cache line, including AXI word padding. */
        size_t bytes = ALIGN(lengths[i], dma_get_cache_alignment());
        if (for_device)
            dma_sync_single_range_for_device(&pdev->dev, dma_address, offsets[i], bytes, DMA_BIDIRECTIONAL);
        else
            dma_sync_single_range_for_cpu(&pdev->dev, dma_address, offsets[i], bytes, DMA_BIDIRECTIONAL);
    }
}

static void finish(void)
{
    if (inflight) {
        /* Called only after completion or independently observed idle.
         * A timed-out transaction retains device ownership until RECOVER. */
        sync_job(false);
        dma_rmb();
        inflight = false;
        module_put(THIS_MODULE);
    }
}

#include "u02_stream_fp32_validation.h"

static long pose_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct u02_pose_job j;
    u32 ctrl;
    int rc = 0;
    u64 start, deadline, spin_deadline, now;
    if (cmd == U02_POSE_INFO) {
        struct u02_pose_info info = {U02_FP32_DEVICE_ABI, U02_FP32_DMA_BYTES, dma_address};
        return copy_to_user((void __user *)arg, &info, sizeof(info)) ? -EFAULT : 0;
    }
    if (cmd != U02_POSE_RUN && cmd != U02_POSE_RECOVER) return -ENOTTY;
    if (mutex_lock_interruptible(&lock)) return -ERESTARTSYS;
    if (cmd == U02_POSE_RECOVER) {
        ctrl = readl(regs);
        if ((ctrl & 1) || !(ctrl & 4)) rc = -EBUSY;
        else { dma_rmb(); finish(); }
        goto out;
    }
    if (inflight) { rc = -EBUSY; goto out; }
    if (copy_from_user(&j, (void __user *)arg, sizeof(j))) { rc = -EFAULT; goto out; }
    rc = validate_job(&j);
    if (rc) goto out;
    ctrl = readl(regs);
    if ((ctrl & 1) || !(ctrl & 4)) { rc = -EBUSY; goto out; }
    dma_wmb();
    writel(0, regs + 4);
    writel(lower_32_bits(dma_address + j.a_offset), regs + U02_STREAM_REG_A);
    writel(upper_32_bits(dma_address + j.a_offset), regs + U02_STREAM_REG_A + 4);
    writel(lower_32_bits(dma_address + j.b_offset), regs + U02_STREAM_REG_B);
    writel(upper_32_bits(dma_address + j.b_offset), regs + U02_STREAM_REG_B + 4);
    writel(lower_32_bits(dma_address + j.c_offset), regs + U02_STREAM_REG_C);
    writel(upper_32_bits(dma_address + j.c_offset), regs + U02_STREAM_REG_C + 4);
    writel(j.m, regs + U02_STREAM_REG_M); writel(j.n, regs + U02_STREAM_REG_N); writel(j.k, regs + U02_STREAM_REG_K);
    if (!try_module_get(THIS_MODULE)) { rc = -ENODEV; goto out; }
    inflight = true;
    active_job = j;
    sync_job(true);
    dma_wmb();
    j.polls = 0; j.status = 0;
    start = ktime_get_ns(); deadline = start + (u64)j.timeout_us * 1000;
    spin_deadline = min(deadline, start + (u64)poll_spin_us * 1000);
    writel(1, regs);
    for (;;) {
        ctrl = readl(regs); ++j.polls;
        if (ctrl & 2) break;
        now = ktime_get_ns();
        if (now >= deadline) { rc = -ETIMEDOUT; break; }
        if (j.timeout_us >= 1000 && j.polls > 64 && now >= spin_deadline)
            usleep_range(10, 20);
        else cpu_relax();
    }
    j.elapsed_ns = ktime_get_ns() - start;
    if (!rc) { dma_rmb(); j.status = (s32)readl(regs + U02_STREAM_REG_RETURN); finish(); }
    if (copy_to_user((void __user *)arg, &j, sizeof(j))) rc = -EFAULT;
out:
    mutex_unlock(&lock);
    return rc;
}

static int pose_open(struct inode *inode, struct file *file)
{
    return atomic_cmpxchg(&opened, 0, 1) ? -EBUSY : 0;
}
static int pose_release(struct inode *inode, struct file *file)
{
    atomic_set(&opened, 0);
    return 0;
}
static int pose_mmap(struct file *file, struct vm_area_struct *vma)
{
    int rc;
    if (vma->vm_pgoff || vma->vm_end - vma->vm_start != U02_FP32_DMA_BYTES)
        return -EINVAL;
    if (mutex_lock_interruptible(&lock)) return -ERESTARTSYS;
    /* A replacement process can open for RECOVER after a timeout, but it
     * cannot acquire writable DMA storage while PL still owns the arena. */
    if (inflight) rc = -EBUSY;
    else if (cached_dma)
        rc = dma_mmap_pages(&pdev->dev, vma, U02_FP32_DMA_BYTES, virt_to_page(arena));
    else
        rc = dma_mmap_coherent(&pdev->dev, vma, arena, dma_address, U02_FP32_DMA_BYTES);
    mutex_unlock(&lock);
    return rc;
}
static const struct file_operations fops = {
    .owner = THIS_MODULE, .open = pose_open, .release = pose_release,
    .unlocked_ioctl = pose_ioctl, .mmap = pose_mmap, .llseek = no_llseek,
};
static struct miscdevice device = {.minor = MISC_DYNAMIC_MINOR, .name = U02_FP32_DEVICE_NAME, .fops = &fops, .mode = 0600};
static int __init pose_init(void)
{
    int rc;
    if (!bridge_checked) return -EPERM;
    if (poll_spin_us > 1000) return -EINVAL;
    if (!request_mem_region(BASE, REGION_BYTES, U02_FP32_DEVICE_NAME)) return -EBUSY;
    pdev = platform_device_register_simple(U02_FP32_DEVICE_NAME, -1, NULL, 0);
    if (IS_ERR(pdev)) { rc = PTR_ERR(pdev); goto region; }
    rc = dma_coerce_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(49));
    if (rc) goto platform;
    if (cached_dma)
        arena = dma_alloc_noncoherent(&pdev->dev, U02_FP32_DMA_BYTES, &dma_address, DMA_BIDIRECTIONAL, GFP_KERNEL);
    else
        arena = dma_alloc_coherent(&pdev->dev, U02_FP32_DMA_BYTES, &dma_address, GFP_KERNEL);
    if (!arena) { rc = -ENOMEM; goto platform; }
    if (cached_dma)
        dma_sync_single_for_cpu(&pdev->dev, dma_address, U02_FP32_DMA_BYTES, DMA_BIDIRECTIONAL);
    regs = ioremap(BASE, REGION_BYTES);
    if (!regs) { rc = -ENOMEM; goto dma; }
    device.parent = &pdev->dev;
    rc = misc_register(&device);
    if (rc) goto mapping;
    pr_info("u02_stream_fp32: guarded packet-pool ABI, %s DMA=%pad bytes=%u\n",
            cached_dma ? "cached with explicit sync" : "coherent", &dma_address, U02_FP32_DMA_BYTES);
    return 0;
mapping: iounmap(regs);
dma:
    if (cached_dma)
        dma_free_noncoherent(&pdev->dev, U02_FP32_DMA_BYTES, arena, dma_address, DMA_BIDIRECTIONAL);
    else
        dma_free_coherent(&pdev->dev, U02_FP32_DMA_BYTES, arena, dma_address);
platform: platform_device_unregister(pdev);
region: release_mem_region(BASE, REGION_BYTES);
    return rc;
}
static void __exit pose_exit(void)
{
    misc_deregister(&device); iounmap(regs);
    if (cached_dma)
        dma_free_noncoherent(&pdev->dev, U02_FP32_DMA_BYTES, arena, dma_address, DMA_BIDIRECTIONAL);
    else
        dma_free_coherent(&pdev->dev, U02_FP32_DMA_BYTES, arena, dma_address);
    platform_device_unregister(pdev); release_mem_region(BASE, REGION_BYTES);
}
module_init(pose_init);
module_exit(pose_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("U02 packet2 FP32-output TinyPose transport with guarded resident packets and output starts in kernel-owned DMA");
