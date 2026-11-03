// Known U02 bitstream only; kernel DMA API owns all DDR buffers.
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/ioport.h>
#include <linux/iopoll.h>
#include <linux/dma-mapping.h>
#include <linux/mutex.h>
#include <linux/uaccess.h>
#include <linux/ktime.h>
#include "u02_test_api.h"

// Derived from create_compute_project.tcl and generated HLS register map.
#define U02_BASE 0xa0000000ULL
#define U02_SIZE 0x10000
#define WORDS 4096
#define DMA_BYTES (WORDS * 2 * sizeof(u32))
static struct platform_device *pdev;
static void __iomem *regs;
static u32 *buffer;
static dma_addr_t dma_address;
static DEFINE_MUTEX(lock);
static bool inflight;
static bool bridge_checked;
module_param(bridge_checked, bool, 0444);
MODULE_PARM_DESC(bridge_checked, "Set only after known bitstream, PL0 clock and AFI state are independently verified");

static void finish_inflight(void)
{
    if (inflight) {
        inflight = false;
        module_put(THIS_MODULE);
    }
}

static long u02_ioctl(struct file *file, unsigned int command, unsigned long arg)
{
    struct u02_test request;
    u32 control;
    unsigned int i;
    u64 start;
    int result;
    if (command != U02_RUN && command != U02_RECOVER) return -ENOTTY;
    if (mutex_lock_interruptible(&lock)) return -ERESTARTSYS;
    if (command == U02_RECOVER) {
        control = readl(regs);
        if (inflight && ((control & BIT(0)) || !(control & (BIT(1) | BIT(2))))) result = -EBUSY;
        else { dma_rmb(); finish_inflight(); result = 0; }
        goto out;
    }
    if (inflight) { result = -EBUSY; goto out; }
    if (copy_from_user(&request, (void __user *)arg, sizeof(request))) {
        result = -EFAULT; goto out;
    }
    if (request.count > WORDS + 1 || request.timeout_us < 1 || request.timeout_us > 1000000) {
        result = -EINVAL; goto out;
    }
    control = readl(regs);
    if (!(control & BIT(2)) || (control & BIT(0))) { result = -EBUSY; goto out; }
    for (i = 0; i < WORDS; ++i) {
        buffer[i] = (i * 0x10203041u) ^ request.seed;
        buffer[WORDS + i] = 0xa5a5a5a5;
    }
    // dma_alloc_coherent provides the CPU/device mapping contract; barriers order ownership.
    dma_wmb();
    writel(0, regs + 0x04);
    writel(lower_32_bits(dma_address), regs + 0x18);
    writel(upper_32_bits(dma_address), regs + 0x1c);
    writel(lower_32_bits(dma_address + WORDS * sizeof(u32)), regs + 0x24);
    writel(upper_32_bits(dma_address + WORDS * sizeof(u32)), regs + 0x28);
    writel(request.count, regs + 0x30);
    writel(request.multiplier, regs + 0x38);
    writel(request.addend, regs + 0x40);
    if (readl(regs + 0x30) != request.count ||
        readl(regs + 0x38) != request.multiplier ||
        readl(regs + 0x40) != request.addend ||
        readl(regs + 0x18) != lower_32_bits(dma_address) ||
        readl(regs + 0x1c) != upper_32_bits(dma_address) ||
        readl(regs + 0x24) != lower_32_bits(dma_address + WORDS * sizeof(u32)) ||
        readl(regs + 0x28) != upper_32_bits(dma_address + WORDS * sizeof(u32))) {
        result = -EIO; goto out;
    }
    // Hold a module reference until PL is quiescent, including after timeout/close.
    if (!try_module_get(THIS_MODULE)) { result = -ENODEV; goto out; }
    inflight = true;
    start = ktime_get_ns();
    writel(1, regs);
    if (request.timeout_us < 1000) {
        u64 deadline = start + (u64)request.timeout_us * 1000;
        /* Use elapsed time: the atomic helper's iteration budget is not a
         * wall-clock deadline when MMIO reads themselves take substantial time.
         * On expiration leave done unread and retain the DMA ownership pin. */
        result = -ETIMEDOUT;
        while (ktime_get_ns() < deadline) {
            control = readl(regs);
            if (control & BIT(1)) { result = 0; break; }
            cpu_relax();
        }
    } else
        result = readl_poll_timeout(regs, control, control & BIT(1), 1, request.timeout_us);
    request.elapsed_ns = ktime_get_ns() - start;
    request.dma_address = dma_address;
    request.errors = 0;
    request.status = result ? 0 : readl(regs + 0x10);
    request.first_output = 0;
    if (!result) {
        dma_rmb();
        request.first_output = buffer[WORDS];
        for (i = 0; i < WORDS; ++i) {
            u32 expected = request.count && request.count <= WORDS && i < request.count
                ? buffer[i] * request.multiplier + request.addend : 0xa5a5a5a5;
            if (buffer[WORDS + i] != expected) ++request.errors;
        }
        if (request.status != ((request.count && request.count <= WORDS) ? 0x55303201u : 0xbad00001u))
            ++request.errors;
        finish_inflight();
    }
    // On timeout, keep buffers and the module pinned until explicit recovery sees idle/done.
    if (copy_to_user((void __user *)arg, &request, sizeof(request))) result = -EFAULT;
out:
    mutex_unlock(&lock);
    return result;
}

static const struct file_operations operations = {
    .owner = THIS_MODULE, .unlocked_ioctl = u02_ioctl, .llseek = no_llseek,
};
static struct miscdevice device = {
    .minor = MISC_DYNAMIC_MINOR, .name = "u02_ddr_test", .fops = &operations, .mode = 0600,
};

static int __init u02_init(void)
{
    int result;
    // Existing firmware queries stalled on this board. Require independent JTAG evidence.
    // The gate is checked before mapping or changing hardware.
    if (!bridge_checked) return -EPERM;
    if (!request_mem_region(U02_BASE, U02_SIZE, "u02_ddr_test")) return -EBUSY;
    // Fabric startup reset is provided by the bitstream's local POR counter.
    // No PS GPIO writes or firmware queries are required.
    pdev = platform_device_register_simple("u02_ddr_test", -1, NULL, 0);
    if (IS_ERR(pdev)) { result = PTR_ERR(pdev); goto region; }
    result = dma_coerce_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(49));
    if (result) goto platform;
    buffer = dma_alloc_coherent(&pdev->dev, DMA_BYTES, &dma_address, GFP_KERNEL);
    if (!buffer) { result = -ENOMEM; goto platform; }
    regs = ioremap(U02_BASE, U02_SIZE);
    if (!regs) { result = -ENOMEM; goto dma; }
    device.parent = &pdev->dev;
    result = misc_register(&device);
    if (result) goto mapping;
    pr_info("u02_ddr_test: CSR=%llx DMA=%pad bytes=%u coherent, 49-bit HP aperture\n",
            U02_BASE, &dma_address, (unsigned int)DMA_BYTES);
    return 0;
mapping: iounmap(regs);
dma: dma_free_coherent(&pdev->dev, DMA_BYTES, buffer, dma_address);
platform: platform_device_unregister(pdev);
region: release_mem_region(U02_BASE, U02_SIZE);
    return result;
}

static void __exit u02_exit(void)
{
    misc_deregister(&device);
    iounmap(regs);
    dma_free_coherent(&pdev->dev, DMA_BYTES, buffer, dma_address);
    platform_device_unregister(pdev);
    release_mem_region(U02_BASE, U02_SIZE);
}
module_init(u02_init);
module_exit(u02_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("U02 known-CSR PL compute test using kernel-owned coherent DMA");
