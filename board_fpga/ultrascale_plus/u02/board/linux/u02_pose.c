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
#include "u02_pose_api.h"

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
static bool inflight, bridge_checked;
module_param(bridge_checked, bool, 0444);
MODULE_PARM_DESC(bridge_checked, "Require current bitstream/boot-bound independent JTAG clock, reset and AXI proof");

static void finish(void)
{
    if (inflight) { inflight = false; module_put(THIS_MODULE); }
}

static bool range_ok(u32 offset, u32 bytes)
{
    return !(offset & 63) && bytes && offset < U02_POSE_ARENA_BYTES &&
           bytes <= U02_POSE_ARENA_BYTES - offset;
}

/* First deployment uses GEMM, direct INT8 convolution and depthwise. The
 * full model's remaining operators stay on this board's ARM. Reject other
 * descriptor modes until their DMA footprints are separately validated. */
static int validate_job(const struct u02_pose_job *j)
{
    u32 k = j->k & ~(CONV_FLAG | DW_FLAG), flags = j->k & (CONV_FLAG | DW_FLAG);
    u64 as, bs, cs;
    u32 d[16];
    if (!range_ok(j->a_offset, j->a_bytes) || !range_ok(j->b_offset, j->b_bytes) ||
        !range_ok(j->c_offset, j->c_bytes) || j->a_offset + j->a_bytes > j->b_offset ||
        j->b_offset + j->b_bytes > j->c_offset ||
        !j->m || j->m > 512 || !j->n || j->n > 25600 || !k || k > 4608 ||
        !j->timeout_us || j->timeout_us > 30000000 || flags == (CONV_FLAG | DW_FLAG))
        return -EINVAL;
    as = (u64)j->m * k; cs = (u64)j->m * j->n * 4;
    bs = (u64)k * j->n;
    if (flags) {
        if (j->b_bytes < sizeof(d)) return -EINVAL;
        memcpy(d, (u8 *)arena + j->b_offset, sizeof(d));
        if (d[0] != 0x43564e31 || !d[1] || d[1] > 320 || !d[2] || d[2] > 320 ||
            !d[3] || d[3] > 512 || (d[4] != 1 && d[4] != 3) ||
            (d[5] != 1 && d[5] != 3) || (d[6] != 1 && d[6] != 2) ||
            (d[7] != 1 && d[7] != 2) || d[8] > 1 || d[9] > 1 ||
            !d[10] || d[10] > 320 || !d[11] || d[11] > 320 ||
            d[12] > d[10] * d[11] || j->n > d[10] * d[11] - d[12] ||
            d[13] || d[14] || d[15]) return -EINVAL;
        bs = 64 + (u64)d[1] * d[2] * d[3];
        if (flags == DW_FLAG) {
            if (j->m != d[3] || k != d[4] * d[5] ||
                d[1] * d[2] > 3072 || j->n > 3072) return -EINVAL;
        } else if (k != d[3] * d[4] * d[5]) return -EINVAL;
    }
    /* AXI packed reads may fetch the last padded 64-bit word. */
    if (as > j->a_bytes || bs > j->b_bytes || cs > j->c_bytes ||
        ALIGN(as, 8) > j->b_offset - j->a_offset ||
        ALIGN(bs, 8) > j->c_offset - j->b_offset)
        return -EINVAL;
    return 0;
}

static long pose_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct u02_pose_job j;
    u32 ctrl;
    int rc = 0;
    u64 start, deadline;
    if (cmd == U02_POSE_INFO) {
        struct u02_pose_info info = {U02_POSE_ABI, U02_POSE_ARENA_BYTES, dma_address};
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
    writel(lower_32_bits(dma_address + j.a_offset), regs + 0x10);
    writel(upper_32_bits(dma_address + j.a_offset), regs + 0x14);
    writel(lower_32_bits(dma_address + j.b_offset), regs + 0x1c);
    writel(upper_32_bits(dma_address + j.b_offset), regs + 0x20);
    writel(lower_32_bits(dma_address + j.c_offset), regs + 0x28);
    writel(upper_32_bits(dma_address + j.c_offset), regs + 0x2c);
    writel(j.m, regs + 0x34); writel(j.n, regs + 0x3c); writel(j.k, regs + 0x44);
    if (!try_module_get(THIS_MODULE)) { rc = -ENODEV; goto out; }
    inflight = true;
    j.polls = 0; j.status = 0;
    start = ktime_get_ns(); deadline = start + (u64)j.timeout_us * 1000;
    writel(1, regs);
    for (;;) {
        ctrl = readl(regs); ++j.polls;
        if (ctrl & 2) break;
        if (ktime_get_ns() >= deadline) { rc = -ETIMEDOUT; break; }
        if (j.timeout_us >= 1000 && j.polls > 64) usleep_range(10, 20);
        else cpu_relax();
    }
    j.elapsed_ns = ktime_get_ns() - start;
    if (!rc) { dma_rmb(); j.status = (s32)readl(regs + 0x4c); finish(); }
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
    if (vma->vm_pgoff || vma->vm_end - vma->vm_start != U02_POSE_ARENA_BYTES)
        return -EINVAL;
    if (mutex_lock_interruptible(&lock)) return -ERESTARTSYS;
    /* A replacement process can open for RECOVER after a timeout, but it
     * cannot acquire writable DMA storage while PL still owns the arena. */
    rc = inflight ? -EBUSY : dma_mmap_coherent(&pdev->dev, vma, arena, dma_address, U02_POSE_ARENA_BYTES);
    mutex_unlock(&lock);
    return rc;
}
static const struct file_operations fops = {
    .owner = THIS_MODULE, .open = pose_open, .release = pose_release,
    .unlocked_ioctl = pose_ioctl, .mmap = pose_mmap, .llseek = no_llseek,
};
static struct miscdevice device = {.minor = MISC_DYNAMIC_MINOR, .name = "u02_pose", .fops = &fops, .mode = 0600};
static int __init pose_init(void)
{
    int rc;
    if (!bridge_checked) return -EPERM;
    if (!request_mem_region(BASE, REGION_BYTES, "u02_pose")) return -EBUSY;
    pdev = platform_device_register_simple("u02_pose", -1, NULL, 0);
    if (IS_ERR(pdev)) { rc = PTR_ERR(pdev); goto region; }
    rc = dma_coerce_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(49));
    if (rc) goto platform;
    arena = dma_alloc_coherent(&pdev->dev, U02_POSE_ARENA_BYTES, &dma_address, GFP_KERNEL);
    if (!arena) { rc = -ENOMEM; goto platform; }
    regs = ioremap(BASE, REGION_BYTES);
    if (!regs) { rc = -ENOMEM; goto dma; }
    device.parent = &pdev->dev;
    rc = misc_register(&device);
    if (rc) goto mapping;
    pr_info("u02_pose: 64-bit V06f ABI, coherent DMA=%pad bytes=%u\n", &dma_address, U02_POSE_ARENA_BYTES);
    return 0;
mapping: iounmap(regs);
dma: dma_free_coherent(&pdev->dev, U02_POSE_ARENA_BYTES, arena, dma_address);
platform: platform_device_unregister(pdev);
region: release_mem_region(BASE, REGION_BYTES);
    return rc;
}
static void __exit pose_exit(void)
{
    misc_deregister(&device); iounmap(regs);
    dma_free_coherent(&pdev->dev, U02_POSE_ARENA_BYTES, arena, dma_address);
    platform_device_unregister(pdev); release_mem_region(BASE, REGION_BYTES);
}
module_init(pose_init);
module_exit(pose_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("U02 V06f full TinyPose convolution transport using coherent DMA");
